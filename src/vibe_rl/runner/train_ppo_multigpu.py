"""Multi-GPU PPO training via ``jax.jit`` + ``NamedSharding``.

Data-parallel PPO across multiple devices using JAX's SPMD
partitioning with ``shard_map`` for explicit per-device code.
Data (env states, observations, RNG keys) is sharded across devices
via ``NamedSharding`` on a 2-D mesh ``(batch, fsdp)``.  Model
parameters are replicated.  Gradients are averaged across devices
via ``jax.lax.pmean`` inside the ``shard_map`` context.

The data shape convention is::

    (n_devices, num_envs, *feature_dims)

Each device runs ``num_envs`` parallel environments. The sharded
outer axis distributes across devices, and inside each device
``vmap`` vectorises across environments.

Single-GPU / multi-GPU switching is config-driven — set
``RunnerConfig(num_devices=1)`` (or ``None`` for auto-detect) to
run on one GPU with the same code path.

Usage::

    from vibe_rl.algorithms.ppo import PPOConfig
    from vibe_rl.env import make
    from vibe_rl.env.wrappers import AutoResetWrapper
    from vibe_rl.runner import RunnerConfig, train_ppo_multigpu

    env, env_params = make("CartPole-v1")
    env = AutoResetWrapper(env)
    ppo_config = PPOConfig(n_steps=128, hidden_sizes=(64, 64))
    runner_config = RunnerConfig(total_timesteps=100_000, num_envs=4)

    train_state, metrics = train_ppo_multigpu(
        env, env_params,
        ppo_config=ppo_config,
        runner_config=runner_config,
    )
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from vibe_rl.algorithms.ppo.agent import PPO, PPOMetrics, compute_gae
from vibe_rl.algorithms.ppo.config import PPOConfig
from vibe_rl.algorithms.ppo.network import ActorCategorical, ActorCriticShared, Critic
from vibe_rl.algorithms.ppo.types import ActorCriticParams, PPOState
from vibe_rl.dataprotocol.transition import PPOTransition
from vibe_rl.env.base import EnvParams, EnvState, Environment
from vibe_rl.runner.config import RunnerConfig
from vibe_rl.runner.device_utils import (
    get_num_devices,
    replicate,
    split_key_across_devices,
    unreplicate,
)
from vibe_rl.runner.train_ppo import PPOMetricsHistory, PPOTrainState
from vibe_rl.sharding import BATCH_AXIS, make_mesh


def train_ppo_multigpu(
    env: Environment,
    env_params: EnvParams,
    *,
    ppo_config: PPOConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    n_actions: int | None = None,
) -> tuple[PPOTrainState, PPOMetricsHistory]:
    """Train PPO with data-parallel ``jit`` + ``NamedSharding`` across devices.

    The environment **must** be wrapped with ``AutoResetWrapper`` (or
    equivalent) so that episodes auto-reset inside the scan.

    Data is shaped as ``(n_devices, num_envs, *feature_dims)``. Gradients
    are synchronised across devices via ``lax.pmean`` inside a
    ``shard_map`` context.  Within each device, the minibatch loss
    already averages over all samples (including those from different
    vmapped environments).

    Args:
        env: Pure-JAX environment (must auto-reset on done).
        env_params: Environment parameters.
        ppo_config: PPO algorithm hyperparameters.
        runner_config: Outer-loop settings (total_timesteps, seed, ...).
        obs_shape: Observation shape. Inferred from env if ``None``.
        n_actions: Number of discrete actions. Inferred from env if ``None``.

    Returns:
        ``(final_train_state, metrics_history)`` where *final_train_state*
        has the leading device dimension (use ``unreplicate`` to get a
        single copy), and *metrics_history* fields have shape
        ``(n_devices, n_updates)``.
    """
    if obs_shape is None:
        obs_space = env.observation_space(env_params)
        obs_shape = obs_space.shape
    if n_actions is None:
        act_space = env.action_space(env_params)
        n_actions = act_space.n

    n_devices = get_num_devices(runner_config.num_devices)
    num_envs = runner_config.num_envs

    # Total timesteps per update = n_devices * num_envs * n_steps
    steps_per_update = n_devices * num_envs * ppo_config.n_steps
    n_updates = runner_config.total_timesteps // steps_per_update

    if n_updates == 0:
        raise ValueError(
            f"total_timesteps ({runner_config.total_timesteps}) is less than "
            f"one update worth of steps ({steps_per_update} = "
            f"{n_devices} devices * {num_envs} envs * {ppo_config.n_steps} steps)."
        )

    import math
    obs_dim = math.prod(obs_shape)

    # ------------------------------------------------------------------
    # Create mesh
    # ------------------------------------------------------------------
    mesh = make_mesh(
        num_fsdp_devices=runner_config.fsdp_devices,
        num_devices=n_devices,
    )

    # ------------------------------------------------------------------
    # Build the per-device learner function
    # ------------------------------------------------------------------

    def _learner_fn(
        device_rng: chex.PRNGKey,
        shared_agent_key: chex.PRNGKey,
        _ppo_config: PPOConfig,
        _n_updates: int,
        _num_envs: int,
    ) -> tuple[PPOTrainState, PPOMetrics]:
        """Single-device learner: init + scan over updates.

        This function is mapped over the device axis via shard_map.
        ``shared_agent_key`` is the same on every device so that
        initial params are identical (required for pmean sync).

        Note: shard_map preserves rank (unlike pmap which removes the
        mapped axis). The device_rng arrives as shape (1, 2), so we
        squeeze out the leading shard dimension.
        """
        # shard_map slices but preserves rank: (1, 2) -> squeeze to (2,)
        device_rng = device_rng.squeeze(axis=0)
        rng, env_key = jax.random.split(device_rng)

        # Initialise agent — shared_agent_key is identical across devices
        # so all replicas start with the same params.
        agent_state = PPO.init(shared_agent_key, obs_shape, n_actions, _ppo_config)

        # Initialise num_envs parallel environments on this device
        env_keys = jax.random.split(env_key, _num_envs)
        env_obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(
            env_keys, env_params,
        )
        # env_obs: (num_envs, *obs_shape), env_states: vmapped EnvState

        init_state = PPOTrainState(
            agent_state=agent_state,
            env_obs=env_obs,
            env_state=env_states,
            rng=rng,
        )

        # Vectorized env step
        batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        def _collect_rollout_batch(
            agent_state: PPOState,
            env_obs: chex.Array,
            env_states,
        ) -> tuple[PPOState, PPOTransition, chex.Array, chex.ArrayTree, chex.Array]:
            """Collect n_steps from num_envs parallel environments."""

            def _step(carry, _):
                a_state, obs, e_states = carry
                # Batched action selection
                actions, log_probs, values, a_state = PPO.act_batch(
                    a_state, obs, config=_ppo_config,
                )
                rng_step, step_key = jax.random.split(a_state.rng)
                a_state = a_state._replace(rng=rng_step)

                step_keys = jax.random.split(step_key, _num_envs)
                next_obs, new_e_states, rewards, dones, _infos = batch_step(
                    step_keys, e_states, actions, env_params,
                )

                transition = PPOTransition(
                    obs=obs,
                    action=actions,
                    reward=rewards,
                    next_obs=next_obs,
                    done=dones,
                    log_prob=log_probs,
                    value=values,
                )
                return (a_state, next_obs, new_e_states), transition

            (agent_state, final_obs, final_env_states), trajectories = jax.lax.scan(
                _step, (agent_state, env_obs, env_states), None, _ppo_config.n_steps,
            )

            last_values = PPO.get_value_batch(
                agent_state, final_obs, config=_ppo_config,
            )

            return agent_state, trajectories, final_obs, final_env_states, last_values

        def _update_step(
            state: PPOState,
            trajectories: PPOTransition,
            last_value: chex.Array,
        ) -> tuple[PPOState, PPOMetrics]:
            """PPO update with cross-device gradient synchronisation.

            Trajectories have shape (T, num_envs, ...).
            """
            # Compute GAE — works with (T, N) shapes
            advantages, returns = compute_gae(
                rewards=trajectories.reward,
                values=trajectories.value,
                dones=trajectories.done,
                last_value=last_value,
                gamma=_ppo_config.gamma,
                gae_lambda=_ppo_config.gae_lambda,
            )

            # Flatten time and env dims for mini-batch SGD
            batch_size = advantages.size
            flat_traj = jax.tree.map(
                lambda x: x.reshape(batch_size, *x.shape[len(advantages.shape):]),
                trajectories,
            )
            flat_advantages = advantages.reshape(batch_size)
            flat_returns = returns.reshape(batch_size)

            optimizer = _ppo_config.make_optimizer()

            def _epoch(carry, _):
                params, opt_state, rng = carry
                rng, shuffle_key = jax.random.split(rng)

                perm = jax.random.permutation(shuffle_key, batch_size)
                mb_size = batch_size // _ppo_config.n_minibatches

                def _minibatch_step(carry, start_idx):
                    params, opt_state = carry
                    mb_idx = jax.lax.dynamic_slice(perm, (start_idx,), (mb_size,))

                    mb_obs = flat_traj.obs[mb_idx]
                    mb_actions = flat_traj.action[mb_idx]
                    mb_old_log_probs = flat_traj.log_prob[mb_idx]
                    mb_old_values = flat_traj.value[mb_idx]
                    mb_advantages = flat_advantages[mb_idx]
                    mb_returns = flat_returns[mb_idx]

                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                    def loss_fn(params):
                        log_probs, values, entropy = PPO.evaluate_actions(
                            params, mb_obs, mb_actions, config=_ppo_config,
                        )

                        ratio = jnp.exp(log_probs - mb_old_log_probs)
                        surr1 = ratio * mb_advantages
                        surr2 = (
                            jnp.clip(
                                ratio,
                                1.0 - _ppo_config.clip_eps,
                                1.0 + _ppo_config.clip_eps,
                            )
                            * mb_advantages
                        )
                        actor_loss = -jnp.minimum(surr1, surr2).mean()

                        value_pred_clipped = mb_old_values + jnp.clip(
                            values - mb_old_values,
                            -_ppo_config.clip_eps,
                            _ppo_config.clip_eps,
                        )
                        vf_loss1 = (values - mb_returns) ** 2
                        vf_loss2 = (value_pred_clipped - mb_returns) ** 2
                        critic_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2).mean()

                        entropy_mean = entropy.mean()
                        total_loss = (
                            actor_loss
                            + _ppo_config.vf_coef * critic_loss
                            - _ppo_config.ent_coef * entropy_mean
                        )
                        approx_kl = (mb_old_log_probs - log_probs).mean()

                        return total_loss, (actor_loss, critic_loss, entropy_mean, approx_kl)

                    (total_loss, (actor_loss, critic_loss, entropy_mean, approx_kl)), grads = (
                        eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
                    )

                    # Average gradients across devices (shard_map axis)
                    grads = jax.lax.pmean(grads, axis_name=BATCH_AXIS)

                    updates, new_opt_state = optimizer.update(
                        grads, opt_state, eqx.filter(params, eqx.is_array),
                    )
                    new_params = eqx.apply_updates(params, updates)

                    metrics = PPOMetrics(
                        total_loss=total_loss,
                        actor_loss=actor_loss,
                        critic_loss=critic_loss,
                        entropy=entropy_mean,
                        approx_kl=approx_kl,
                    )
                    return (new_params, new_opt_state), metrics

                start_indices = jnp.arange(_ppo_config.n_minibatches) * mb_size
                (params, opt_state), mb_metrics = jax.lax.scan(
                    _minibatch_step, (params, opt_state), start_indices,
                )

                return (params, opt_state, rng), mb_metrics

            (new_params, new_opt_state, new_rng), epoch_metrics = jax.lax.scan(
                _epoch,
                (state.params, state.opt_state, state.rng),
                None,
                _ppo_config.n_epochs,
            )

            avg_metrics = jax.tree.map(lambda x: x.mean(), epoch_metrics)

            new_state = PPOState(
                params=new_params,
                opt_state=new_opt_state,
                step=state.step + 1,
                rng=new_rng,
            )

            return new_state, avg_metrics

        # ---- Main scan loop over updates ----
        def _scan_body(
            train_state: PPOTrainState, _: None,
        ) -> tuple[PPOTrainState, PPOMetrics]:
            rng, _ = jax.random.split(train_state.rng)

            agent_state, trajectories, final_obs, final_env_state, last_value = (
                _collect_rollout_batch(
                    train_state.agent_state,
                    train_state.env_obs,
                    train_state.env_state,
                )
            )

            agent_state, metrics = _update_step(
                agent_state, trajectories, last_value,
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

        # shard_map preserves rank: add leading shard dim so concatenation
        # across devices produces shape (n_devices, ...).
        final_state = jax.tree.map(lambda x: x[None, ...], final_state)
        history = jax.tree.map(lambda x: x[None, ...], history)

        return final_state, history

    # ------------------------------------------------------------------
    # shard_map the learner across devices (replaces pmap)
    # ------------------------------------------------------------------
    # device_keys: sharded across batch axis (each device gets its own key)
    # shared_agent_key: replicated (same on all devices, P() = no sharding)
    sharded_learner = jax.shard_map(
        partial(
            _learner_fn,
            _ppo_config=ppo_config,
            _n_updates=n_updates,
            _num_envs=num_envs,
        ),
        mesh=mesh,
        in_specs=(P(BATCH_AXIS), P()),
        out_specs=(P(BATCH_AXIS), P(BATCH_AXIS)),
        check_vma=False,
    )

    # Wrap in jit for compilation
    jitted_learner = jax.jit(sharded_learner)

    # Split RNG: shared agent key (same on all devices) + per-device keys
    rng = jax.random.PRNGKey(runner_config.seed)
    rng, agent_key = jax.random.split(rng)
    device_keys = split_key_across_devices(rng, n_devices)

    return jitted_learner(device_keys, agent_key)
