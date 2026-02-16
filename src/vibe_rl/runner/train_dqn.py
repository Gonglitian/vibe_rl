"""Hybrid DQN training loop.

Off-policy algorithms (DQN, SAC) use a mutable replay buffer that cannot
live inside ``lax.scan``.  The design here is:

- **Outer loop**: Python ``for`` — pushes transitions into the buffer and
  handles eval/logging callbacks.
- **Inner step**: ``jax.jit``-compiled ``env.step`` + ``DQN.act`` +
  ``DQN.update`` — no Python overhead on the hot path.

Usage::

    from vibe_rl.algorithms.dqn import DQNConfig
    from vibe_rl.env import make
    from vibe_rl.env.wrappers import AutoResetWrapper
    from vibe_rl.runner import RunnerConfig, train_dqn

    env, env_params = make("CartPole-v1")
    env = AutoResetWrapper(env)
    dqn_config = DQNConfig()
    runner_config = RunnerConfig(total_timesteps=50_000)

    final_state, metrics = train_dqn(
        env, env_params,
        dqn_config=dqn_config,
        runner_config=runner_config,
    )
"""

from __future__ import annotations

import logging
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from vibe_rl.algorithms.dqn.agent import DQN, DQNMetrics
from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.dqn.types import DQNState
from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer
from vibe_rl.env.base import EnvParams, EnvState, Environment
from vibe_rl.runner.config import RunnerConfig

logger = logging.getLogger(__name__)


class _StepCarry(NamedTuple):
    """Carry for the jitted env-step + act combo."""

    agent_state: DQNState
    obs: chex.Array
    env_state: EnvState


class DQNTrainResult(NamedTuple):
    """Return value from ``train_dqn``."""

    agent_state: DQNState
    episode_returns: list[float]
    metrics_log: list[dict[str, float]]


@partial(jax.jit, static_argnames=("env_step_fn", "config"))
def _act_and_step(
    agent_state: DQNState,
    obs: chex.Array,
    env_state: EnvState,
    env_step_fn: callable,
    env_params: EnvParams,
    *,
    config: DQNConfig,
) -> tuple[DQNState, chex.Array, EnvState, chex.Array, chex.Array, chex.Array]:
    """JIT-compiled: select action + step environment.

    Returns (new_agent_state, next_obs, new_env_state, action, reward, done).
    """
    action, agent_state = DQN.act(agent_state, obs, config=config, explore=True)
    rng, step_key = jax.random.split(agent_state.rng)
    agent_state = agent_state._replace(rng=rng)
    next_obs, new_env_state, reward, done, _info = env_step_fn(
        step_key, env_state, action, env_params,
    )
    return agent_state, next_obs, new_env_state, action, reward, done


def train_dqn(
    env: Environment,
    env_params: EnvParams,
    *,
    dqn_config: DQNConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    n_actions: int | None = None,
    callback: callable | None = None,
) -> DQNTrainResult:
    """Train DQN with a hybrid Python/JAX loop.

    The replay buffer lives in Python (numpy); everything else is jitted.

    Supports checkpoint-based resume: when ``runner_config.resume`` is
    ``True`` and a checkpoint exists at ``runner_config.checkpoint_dir``,
    training resumes from the saved step with restored optimizer state.

    Args:
        env: Pure-JAX environment (should auto-reset on done).
        env_params: Environment parameters.
        dqn_config: DQN algorithm hyperparameters.
        runner_config: Outer-loop settings.
        obs_shape: Observation shape. Inferred from env if ``None``.
        n_actions: Number of discrete actions. Inferred from env if ``None``.
        callback: Optional ``callback(step, agent_state, metrics_dict)``
            called every ``runner_config.log_interval`` steps.

    Returns:
        ``DQNTrainResult`` containing final agent state, episode returns,
        and a log of training metrics dicts.
    """
    if obs_shape is None:
        obs_space = env.observation_space(env_params)
        obs_shape = obs_space.shape
    if n_actions is None:
        act_space = env.action_space(env_params)
        n_actions = act_space.n

    rng = jax.random.PRNGKey(runner_config.seed)
    rng, agent_key, env_key = jax.random.split(rng, 3)

    agent_state = DQN.init(agent_key, obs_shape, n_actions, dqn_config)
    obs, env_state = env.reset(env_key, env_params)
    buffer = ReplayBuffer(capacity=runner_config.buffer_size, obs_shape=obs_shape)

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
            logger.info("Resumed DQN training from step %d", restored_step)

    episode_returns: list[float] = []
    metrics_log: list[dict[str, float]] = []
    ep_return = 0.0

    try:
        for step in range(start_step, runner_config.total_timesteps + 1):
            agent_state, next_obs, env_state, action, reward, done = _act_and_step(
                agent_state, obs, env_state, env.step, env_params, config=dqn_config,
            )

            # Push to replay buffer (transfers to numpy)
            buffer.push(obs, int(action), float(reward), next_obs, bool(done))
            ep_return += float(reward)
            obs = next_obs

            if done:
                episode_returns.append(ep_return)
                ep_return = 0.0

            # Train once buffer has enough data
            if len(buffer) >= runner_config.warmup_steps:
                batch = buffer.sample(dqn_config.batch_size)
                agent_state, metrics = DQN.update(
                    agent_state, batch, config=dqn_config,
                )

                if step % runner_config.log_interval == 0:
                    record = {
                        "step": step,
                        "loss": float(metrics.loss),
                        "q_mean": float(metrics.q_mean),
                        "epsilon": float(metrics.epsilon),
                    }
                    metrics_log.append(record)
                    if callback is not None:
                        callback(step, agent_state, record)

            # Periodic checkpointing
            if ckpt_mgr is not None:
                ckpt_mgr.save(step, agent_state)
    finally:
        if ckpt_mgr is not None:
            ckpt_mgr.wait()
            ckpt_mgr.close()

    return DQNTrainResult(
        agent_state=agent_state,
        episode_returns=episode_returns,
        metrics_log=metrics_log,
    )
