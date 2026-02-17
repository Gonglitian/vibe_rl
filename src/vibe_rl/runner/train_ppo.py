"""PureJaxRL-style PPO training loop.

The entire training loop — collect rollout, compute GAE, run PPO updates —
is compiled into a single ``jax.lax.scan``.  One JIT compilation up-front,
then the hardware runs at full speed with zero Python overhead.

Supports vectorized parallel environments via ``jax.vmap`` when
``ppo_config.num_envs > 1``.  The core pattern::

    batch_reset = jax.vmap(env.reset, in_axes=(0, None))
    batch_step  = jax.vmap(env.step,  in_axes=(0, 0, 0, None))

Usage (single env)::

    from vibe_rl.algorithms.ppo import PPOConfig
    from vibe_rl.env import make
    from vibe_rl.env.wrappers import AutoResetWrapper
    from vibe_rl.runner import RunnerConfig, train_ppo

    env, env_params = make("CartPole-v1")
    env = AutoResetWrapper(env)  # required for lax.scan loop
    ppo_config = PPOConfig(n_steps=128, hidden_sizes=(64, 64))
    runner_config = RunnerConfig(total_timesteps=100_000)

    train_state, metrics = train_ppo(
        env, env_params,
        ppo_config=ppo_config,
        runner_config=runner_config,
    )

Usage (vectorized, 8 parallel envs)::

    ppo_config = PPOConfig(n_steps=128, num_envs=8, hidden_sizes=(64, 64))
    runner_config = RunnerConfig(total_timesteps=100_000)

    train_state, metrics = train_ppo(
        env, env_params,
        ppo_config=ppo_config,
        runner_config=runner_config,
    )
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from vibe_rl.algorithms.ppo.agent import PPO, PPOMetrics
from vibe_rl.algorithms.ppo.config import PPOConfig
from vibe_rl.algorithms.ppo.types import PPOState
from vibe_rl.env.base import EnvParams, EnvState, Environment
from vibe_rl.metrics import MetricsLogger
from vibe_rl.run_dir import RunDir
from vibe_rl.runner.config import RunnerConfig


class PPOTrainState(NamedTuple):
    """Full training state threaded through the scan loop."""

    agent_state: PPOState
    env_obs: chex.Array
    env_state: EnvState
    rng: chex.PRNGKey
    ep_return_sum: chex.Array  # running sum of rewards in current episode(s)


class PPOMetricsHistory(NamedTuple):
    """Per-update metrics collected across the entire training run.

    Each field has shape ``(n_updates,)`` (or ``(n_updates, ...)`` for
    structured fields).
    """

    total_loss: chex.Array
    actor_loss: chex.Array
    critic_loss: chex.Array
    entropy: chex.Array
    approx_kl: chex.Array
    episode_return: chex.Array


def train_ppo(
    env: Environment,
    env_params: EnvParams,
    *,
    ppo_config: PPOConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    n_actions: int | None = None,
    run_dir: RunDir | None = None,
) -> tuple[PPOTrainState, PPOMetricsHistory]:
    """Train PPO from scratch using a single-JIT ``lax.scan`` loop.

    The environment **must** be wrapped with ``AutoResetWrapper`` (or
    equivalent) so that episodes auto-reset inside the scan.

    When ``ppo_config.num_envs > 1``, the training loop uses
    ``jax.vmap``-vectorized environments for parallel rollout collection.
    The total timesteps per update becomes ``n_steps * num_envs``.

    Args:
        env: Pure-JAX environment (must auto-reset on done).
        env_params: Environment parameters.
        ppo_config: PPO algorithm hyperparameters.
        runner_config: Outer-loop settings (total_timesteps, seed, ...).
        obs_shape: Observation shape. Inferred from env if ``None``.
        n_actions: Number of discrete actions. Inferred from env if ``None``.
        run_dir: Optional :class:`~vibe_rl.run_dir.RunDir` for JSONL
            metrics logging.  Since PPO runs entirely inside
            ``lax.scan``, metrics are batch-written after training
            completes.

    Returns:
        ``(final_train_state, metrics_history)`` where *metrics_history*
        fields have shape ``(n_updates,)``.
    """
    if obs_shape is None:
        obs_space = env.observation_space(env_params)
        obs_shape = obs_space.shape
    if n_actions is None:
        act_space = env.action_space(env_params)
        n_actions = act_space.n

    num_envs = ppo_config.num_envs
    steps_per_update = ppo_config.n_steps * num_envs
    n_updates = runner_config.total_timesteps // steps_per_update

    if num_envs > 1:
        train_state, history = _train_ppo_vectorized(
            env, env_params,
            ppo_config=ppo_config,
            obs_shape=obs_shape,
            n_actions=n_actions,
            n_updates=n_updates,
            seed=runner_config.seed,
        )
    else:
        train_state, history = _train_ppo_single(
            env, env_params,
            ppo_config=ppo_config,
            obs_shape=obs_shape,
            n_actions=n_actions,
            n_updates=n_updates,
            seed=runner_config.seed,
        )

    # Batch-write metrics to JSONL after training (scan prevents I/O during)
    if run_dir is not None:
        _write_ppo_metrics(run_dir, history, steps_per_update)

    return train_state, history


def _episode_returns_from_rollout(
    rewards: chex.Array,
    dones: chex.Array,
    ep_return_sum: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Compute mean episode return from a rollout's rewards and done flags.

    Accumulates rewards step-by-step.  Whenever ``done=True``, the
    accumulated sum is recorded as a completed episode return.

    Args:
        rewards: shape ``(T,)`` or ``(T, N)`` for vectorized envs.
        dones: shape ``(T,)`` or ``(T, N)``.
        ep_return_sum: running episode return carried from previous
            rollout; shape ``()`` or ``(N,)``.

    Returns:
        ``(mean_episode_return, n_episodes, new_ep_return_sum)`` where
        *mean_episode_return* is the average over completed episodes
        (0.0 if none completed), and *new_ep_return_sum* is the
        carry-over for the next rollout.
    """

    def _scan_fn(carry, step):
        running_sum, total_return, n_eps = carry
        reward, done = step
        running_sum = running_sum + reward
        # On done: add completed return and reset accumulator
        total_return = total_return + running_sum * done
        n_eps = n_eps + done
        running_sum = running_sum * (1.0 - done)
        return (running_sum, total_return, n_eps), None

    zero = jnp.zeros_like(ep_return_sum)
    init = (ep_return_sum, zero, zero)
    (new_ep_return_sum, total_return, n_eps), _ = jax.lax.scan(
        _scan_fn, init, (rewards, dones),
    )

    # Sum across envs for vectorized case (N,) -> ()
    total_return_scalar = total_return.sum()
    n_eps_scalar = n_eps.sum()

    mean_return = jnp.where(
        n_eps_scalar > 0,
        total_return_scalar / n_eps_scalar,
        jnp.float32(0.0),
    )
    return mean_return, n_eps_scalar, new_ep_return_sum


def _write_ppo_metrics(
    run_dir: RunDir,
    history: PPOMetricsHistory,
    steps_per_update: int,
) -> None:
    """Batch-write PPO metrics history to JSONL."""
    with MetricsLogger(run_dir.log_path()) as logger:
        n_updates = int(history.total_loss.shape[0])
        for i in range(n_updates):
            record: dict[str, float] = {
                "step": (i + 1) * steps_per_update,
                "total_loss": float(history.total_loss[i]),
                "actor_loss": float(history.actor_loss[i]),
                "critic_loss": float(history.critic_loss[i]),
                "entropy": float(history.entropy[i]),
                "approx_kl": float(history.approx_kl[i]),
                "episode_return": float(history.episode_return[i]),
            }
            logger.write(record)


def _train_ppo_single(
    env: Environment,
    env_params: EnvParams,
    *,
    ppo_config: PPOConfig,
    obs_shape: tuple[int, ...],
    n_actions: int,
    n_updates: int,
    seed: int,
) -> tuple[PPOTrainState, PPOMetricsHistory]:
    """Single-environment training path (original implementation)."""

    @partial(jax.jit, static_argnames=("_ppo_config", "_n_updates"))
    def _train(
        rng: chex.PRNGKey,
        _ppo_config: PPOConfig,
        _n_updates: int,
    ) -> tuple[PPOTrainState, PPOMetricsHistory]:
        rng, agent_key, env_key = jax.random.split(rng, 3)

        # Initialise agent and environment
        agent_state = PPO.init(agent_key, obs_shape, n_actions, _ppo_config)
        env_obs, env_state = env.reset(env_key, env_params)

        init_state = PPOTrainState(
            agent_state=agent_state,
            env_obs=env_obs,
            env_state=env_state,
            rng=rng,
            ep_return_sum=jnp.float32(0.0),
        )

        def _scan_body(
            train_state: PPOTrainState, _: None,
        ) -> tuple[PPOTrainState, PPOMetricsHistory]:
            rng, _ = jax.random.split(train_state.rng)

            # Collect a full rollout
            agent_state, trajectories, final_obs, final_env_state, last_value = (
                PPO.collect_rollout(
                    train_state.agent_state,
                    train_state.env_obs,
                    train_state.env_state,
                    env.step,
                    env_params,
                    config=_ppo_config,
                )
            )

            # Compute episode returns from this rollout
            mean_ep_return, _n_eps, new_ep_return_sum = _episode_returns_from_rollout(
                trajectories.reward, trajectories.done, train_state.ep_return_sum,
            )

            # PPO update
            agent_state, metrics = PPO.update(
                agent_state, trajectories, last_value, config=_ppo_config,
            )

            new_train_state = PPOTrainState(
                agent_state=agent_state,
                env_obs=final_obs,
                env_state=final_env_state,
                rng=rng,
                ep_return_sum=new_ep_return_sum,
            )
            update_metrics = PPOMetricsHistory(
                total_loss=metrics.total_loss,
                actor_loss=metrics.actor_loss,
                critic_loss=metrics.critic_loss,
                entropy=metrics.entropy,
                approx_kl=metrics.approx_kl,
                episode_return=mean_ep_return,
            )
            return new_train_state, update_metrics

        final_state, history = jax.lax.scan(
            _scan_body, init_state, None, length=_n_updates,
        )

        return final_state, history

    rng = jax.random.PRNGKey(seed)
    return _train(rng, ppo_config, n_updates)


def _train_ppo_vectorized(
    env: Environment,
    env_params: EnvParams,
    *,
    ppo_config: PPOConfig,
    obs_shape: tuple[int, ...],
    n_actions: int,
    n_updates: int,
    seed: int,
) -> tuple[PPOTrainState, PPOMetricsHistory]:
    """Vectorized multi-environment training path using jax.vmap.

    Resets ``num_envs`` environments in parallel and collects rollouts
    from all of them simultaneously at each update step.
    """
    num_envs = ppo_config.num_envs

    @partial(jax.jit, static_argnames=("_ppo_config", "_n_updates", "_num_envs"))
    def _train(
        rng: chex.PRNGKey,
        _ppo_config: PPOConfig,
        _n_updates: int,
        _num_envs: int,
    ) -> tuple[PPOTrainState, PPOMetricsHistory]:
        rng, agent_key, env_key = jax.random.split(rng, 3)

        # Initialise agent (shared across all envs)
        agent_state = PPO.init(agent_key, obs_shape, n_actions, _ppo_config)

        # Initialise N parallel environments via vmap
        env_keys = jax.random.split(env_key, _num_envs)
        batch_reset = jax.vmap(env.reset, in_axes=(0, None))
        env_obs, env_states = batch_reset(env_keys, env_params)
        # env_obs: (num_envs, *obs_shape), env_states: batched EnvState

        init_state = PPOTrainState(
            agent_state=agent_state,
            env_obs=env_obs,
            env_state=env_states,
            rng=rng,
            ep_return_sum=jnp.zeros(_num_envs, dtype=jnp.float32),
        )

        def _scan_body(
            train_state: PPOTrainState, _: None,
        ) -> tuple[PPOTrainState, PPOMetricsHistory]:
            rng, _ = jax.random.split(train_state.rng)

            # Collect rollouts from N parallel environments
            agent_state, trajectories, final_obs, final_env_states, last_values = (
                PPO.collect_rollout_batch(
                    train_state.agent_state,
                    train_state.env_obs,
                    train_state.env_state,
                    env.step,
                    env_params,
                    config=_ppo_config,
                )
            )

            # Compute episode returns from this rollout
            mean_ep_return, _n_eps, new_ep_return_sum = _episode_returns_from_rollout(
                trajectories.reward, trajectories.done, train_state.ep_return_sum,
            )

            # PPO update — handles (T, N, ...) shaped trajectories
            agent_state, metrics = PPO.update(
                agent_state, trajectories, last_values, config=_ppo_config,
            )

            new_train_state = PPOTrainState(
                agent_state=agent_state,
                env_obs=final_obs,
                env_state=final_env_states,
                rng=rng,
                ep_return_sum=new_ep_return_sum,
            )
            update_metrics = PPOMetricsHistory(
                total_loss=metrics.total_loss,
                actor_loss=metrics.actor_loss,
                critic_loss=metrics.critic_loss,
                entropy=metrics.entropy,
                approx_kl=metrics.approx_kl,
                episode_return=mean_ep_return,
            )
            return new_train_state, update_metrics

        final_state, history = jax.lax.scan(
            _scan_body, init_state, None, length=_n_updates,
        )

        return final_state, history

    rng = jax.random.PRNGKey(seed)
    return _train(rng, ppo_config, n_updates, num_envs)
