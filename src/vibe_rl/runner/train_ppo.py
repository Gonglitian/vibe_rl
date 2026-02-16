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
from vibe_rl.runner.config import RunnerConfig


class PPOTrainState(NamedTuple):
    """Full training state threaded through the scan loop."""

    agent_state: PPOState
    env_obs: chex.Array
    env_state: EnvState
    rng: chex.PRNGKey


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


def train_ppo(
    env: Environment,
    env_params: EnvParams,
    *,
    ppo_config: PPOConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    n_actions: int | None = None,
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
        return _train_ppo_vectorized(
            env, env_params,
            ppo_config=ppo_config,
            obs_shape=obs_shape,
            n_actions=n_actions,
            n_updates=n_updates,
            seed=runner_config.seed,
        )
    else:
        return _train_ppo_single(
            env, env_params,
            ppo_config=ppo_config,
            obs_shape=obs_shape,
            n_actions=n_actions,
            n_updates=n_updates,
            seed=runner_config.seed,
        )


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
        )

        def _scan_body(
            train_state: PPOTrainState, _: None,
        ) -> tuple[PPOTrainState, PPOMetrics]:
            rng, collect_key = jax.random.split(train_state.rng)

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

            # PPO update
            agent_state, metrics = PPO.update(
                agent_state, trajectories, last_value, config=_ppo_config,
            )

            new_train_state = PPOTrainState(
                agent_state=agent_state,
                env_obs=final_obs,
                env_state=final_env_state,
                rng=rng,
            )
            return new_train_state, metrics

        final_state, metrics_history = jax.lax.scan(
            _scan_body, init_state, None, length=_n_updates,
        )

        history = PPOMetricsHistory(
            total_loss=metrics_history.total_loss,
            actor_loss=metrics_history.actor_loss,
            critic_loss=metrics_history.critic_loss,
            entropy=metrics_history.entropy,
            approx_kl=metrics_history.approx_kl,
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
        )

        def _scan_body(
            train_state: PPOTrainState, _: None,
        ) -> tuple[PPOTrainState, PPOMetrics]:
            rng, collect_key = jax.random.split(train_state.rng)

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

            # PPO update — handles (T, N, ...) shaped trajectories
            agent_state, metrics = PPO.update(
                agent_state, trajectories, last_values, config=_ppo_config,
            )

            new_train_state = PPOTrainState(
                agent_state=agent_state,
                env_obs=final_obs,
                env_state=final_env_states,
                rng=rng,
            )
            return new_train_state, metrics

        final_state, metrics_history = jax.lax.scan(
            _scan_body, init_state, None, length=_n_updates,
        )

        history = PPOMetricsHistory(
            total_loss=metrics_history.total_loss,
            actor_loss=metrics_history.actor_loss,
            critic_loss=metrics_history.critic_loss,
            entropy=metrics_history.entropy,
            approx_kl=metrics_history.approx_kl,
        )
        return final_state, history

    rng = jax.random.PRNGKey(seed)
    return _train(rng, ppo_config, n_updates, num_envs)
