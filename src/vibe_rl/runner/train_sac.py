"""Hybrid SAC training loop.

Same design as the DQN runner (Python outer loop + jitted inner step)
since SAC is off-policy and needs a mutable replay buffer.

Usage::

    from vibe_rl.algorithms.sac import SACConfig
    from vibe_rl.env import make
    from vibe_rl.env.wrappers import AutoResetWrapper
    from vibe_rl.runner import RunnerConfig, train_sac

    env, env_params = make("Pendulum-v1")  # continuous-action env
    env = AutoResetWrapper(env)
    sac_config = SACConfig()
    runner_config = RunnerConfig(total_timesteps=100_000)

    result = train_sac(
        env, env_params,
        sac_config=sac_config,
        runner_config=runner_config,
        action_dim=1,
    )
"""

from __future__ import annotations

import logging
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from vibe_rl.algorithms.sac.agent import SAC
from vibe_rl.algorithms.sac.config import SACConfig
from vibe_rl.algorithms.sac.types import SACState
from vibe_rl.env.base import EnvParams, EnvState, Environment
from vibe_rl.metrics import MetricsLogger, TrainingProgress
from vibe_rl.run_dir import RunDir
from vibe_rl.runner.config import RunnerConfig
from vibe_rl.types import Transition

logger = logging.getLogger(__name__)


class _ContinuousReplayBuffer:
    """Replay buffer supporting continuous (multi-dim) actions.

    The base ``ReplayBuffer`` uses int32 actions; SAC needs float32
    vectors.  This is a minimal variant for the runner.
    """

    def __init__(
        self, capacity: int, obs_shape: tuple[int, ...], action_dim: int,
    ) -> None:
        self.capacity = capacity
        self._size = 0
        self._ptr = 0
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self._ptr
        self._obs[idx] = obs
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_obs[idx] = next_obs
        self._dones[idx] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        indices = np.random.randint(0, self._size, size=batch_size)
        return Transition(
            obs=jnp.asarray(self._obs[indices]),
            action=jnp.asarray(self._actions[indices]),
            reward=jnp.asarray(self._rewards[indices]),
            next_obs=jnp.asarray(self._next_obs[indices]),
            done=jnp.asarray(self._dones[indices]),
        )

    def __len__(self) -> int:
        return self._size


class SACTrainResult(NamedTuple):
    """Return value from ``train_sac``."""

    agent_state: SACState
    episode_returns: list[float]
    metrics_log: list[dict[str, float]]


@partial(jax.jit, static_argnames=("env_step_fn", "config"))
def _act_and_step(
    agent_state: SACState,
    obs: chex.Array,
    env_state: EnvState,
    env_step_fn: callable,
    env_params: EnvParams,
    *,
    config: SACConfig,
) -> tuple[SACState, chex.Array, EnvState, chex.Array, chex.Array, chex.Array]:
    """JIT-compiled: sample action + step environment."""
    action, agent_state = SAC.act(agent_state, obs, config=config, explore=True)
    rng, step_key = jax.random.split(agent_state.rng)
    agent_state = agent_state._replace(rng=rng)
    next_obs, new_env_state, reward, done, _info = env_step_fn(
        step_key, env_state, action, env_params,
    )
    return agent_state, next_obs, new_env_state, action, reward, done


def train_sac(
    env: Environment,
    env_params: EnvParams,
    *,
    sac_config: SACConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    action_dim: int | None = None,
    callback: callable | None = None,
    run_dir: RunDir | None = None,
) -> SACTrainResult:
    """Train SAC with a hybrid Python/JAX loop.

    Supports checkpoint-based resume: when ``runner_config.resume`` is
    ``True`` and a checkpoint exists at ``runner_config.checkpoint_dir``,
    training resumes from the saved step with restored optimizer state.

    Args:
        env: Pure-JAX environment (should auto-reset on done).
        env_params: Environment parameters.
        sac_config: SAC algorithm hyperparameters.
        runner_config: Outer-loop settings.
        obs_shape: Observation shape. Inferred from env if ``None``.
        action_dim: Action dimensionality. Inferred from env if ``None``.
        callback: Optional ``callback(step, agent_state, metrics_dict)``.
        run_dir: Optional :class:`~vibe_rl.run_dir.RunDir` for JSONL
            metrics logging.  When provided, metrics are written to
            ``<run_dir>/logs/metrics.jsonl``.

    Returns:
        ``SACTrainResult`` with final agent state, episode returns, and
        training metrics log.
    """
    if obs_shape is None:
        obs_space = env.observation_space(env_params)
        obs_shape = obs_space.shape
    if action_dim is None:
        act_space = env.action_space(env_params)
        action_dim = act_space.shape[0]

    rng = jax.random.PRNGKey(runner_config.seed)
    rng, agent_key, env_key = jax.random.split(rng, 3)

    agent_state = SAC.init(agent_key, obs_shape, action_dim, sac_config)
    obs, env_state = env.reset(env_key, env_params)
    buffer = _ContinuousReplayBuffer(
        capacity=runner_config.buffer_size,
        obs_shape=obs_shape,
        action_dim=action_dim,
    )

    # --- Checkpoint setup ---
    ckpt_mgr = None
    start_step = 1

    if runner_config.checkpoint_dir is not None:
        from vibe_rl.checkpoint import initialize_checkpoint_dir

        ckpt_mgr, resuming = initialize_checkpoint_dir(
            runner_config.checkpoint_dir,
            keep_period=runner_config.keep_period,
            overwrite=runner_config.overwrite,
            resume=runner_config.resume,
            max_to_keep=runner_config.max_checkpoints,
            save_interval_steps=runner_config.checkpoint_interval,
        )

        if resuming:
            restored_step = ckpt_mgr.latest_step()
            agent_state = ckpt_mgr.restore(restored_step, agent_state)
            start_step = restored_step + 1
            logger.info("Resumed SAC training from step %d", restored_step)

    # --- Metrics logger ---
    metrics_logger = (
        MetricsLogger(run_dir.log_path()) if run_dir is not None else None
    )

    # --- Progress bar ---
    progress = (
        TrainingProgress(total=runner_config.total_timesteps, prefix="SAC")
        if runner_config.progress_bar
        else None
    )

    episode_returns: list[float] = []
    metrics_log: list[dict[str, float]] = []
    ep_return = 0.0
    ep_length = 0
    _last_log_record: dict[str, float] = {}

    try:
        for step in range(start_step, runner_config.total_timesteps + 1):
            agent_state, next_obs, env_state, action, reward, done = _act_and_step(
                agent_state, obs, env_state, env.step, env_params, config=sac_config,
            )

            buffer.push(
                np.asarray(obs), np.asarray(action), float(reward),
                np.asarray(next_obs), bool(done),
            )
            ep_return += float(reward)
            ep_length += 1
            obs = next_obs

            if done:
                episode_returns.append(ep_return)
                if metrics_logger is not None:
                    metrics_logger.write({
                        "step": step,
                        "episode_return": ep_return,
                        "episode_length": ep_length,
                    })
                ep_return = 0.0
                ep_length = 0

            if len(buffer) >= runner_config.warmup_steps:
                batch = buffer.sample(sac_config.batch_size)
                agent_state, metrics = SAC.update(
                    agent_state, batch, config=sac_config,
                )

                if step % runner_config.log_interval == 0:
                    record = {
                        "step": step,
                        "actor_loss": float(metrics.actor_loss),
                        "critic_loss": float(metrics.critic_loss),
                        "alpha": float(metrics.alpha),
                        "entropy": float(metrics.entropy),
                        "q_mean": float(metrics.q_mean),
                    }
                    metrics_log.append(record)
                    _last_log_record = record
                    if metrics_logger is not None:
                        metrics_logger.write(record)
                    if callback is not None:
                        callback(step, agent_state, record)

            if progress is not None and step % runner_config.log_interval == 0:
                progress.update(step, _last_log_record)

            # Periodic checkpointing
            if ckpt_mgr is not None:
                ckpt_mgr.save(step, agent_state)
    finally:
        if progress is not None:
            progress.close()
        if metrics_logger is not None:
            metrics_logger.close()
        if ckpt_mgr is not None:
            ckpt_mgr.wait()
            ckpt_mgr.close()

    return SACTrainResult(
        agent_state=agent_state,
        episode_returns=episode_returns,
        metrics_log=metrics_log,
    )
