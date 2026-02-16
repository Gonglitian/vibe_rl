"""Multi-GPU PPO training via ``jax.jit`` + ``NamedSharding`` + FSDP.

Data-parallel PPO across multiple devices using JAX's GSPMD
partitioning with explicit ``in_shardings``/``out_shardings`` on
``jax.jit``.  Data (env states, observations, RNG keys) is sharded
across the ``(batch, fsdp)`` mesh axes.  Model parameters are
sharded per-parameter via :func:`~vibe_rl.sharding.fsdp_sharding`
(large 2-D+ params across the FSDP axis, small params replicated).
Gradient reduction is handled implicitly by JAX's GSPMD — no
manual ``pmean`` is needed.

The data shape convention is::

    (n_devices, num_envs, *feature_dims)

Each device shard runs ``num_envs`` parallel environments.  The
sharded outer axis distributes across devices, and inside each
shard ``vmap`` vectorises across environments.

When ``fsdp_devices=1`` (default), the FSDP axis is trivial and
every parameter is replicated — equivalent to pure data-parallelism.

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

import math
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
from vibe_rl.sharding import (
    BATCH_AXIS,
    DATA_AXIS,
    FSDP_AXIS,
    data_sharding,
    fsdp_sharding,
    make_mesh,
    replicate_sharding,
)


def train_ppo_multigpu(
    env: Environment,
    env_params: EnvParams,
    *,
    ppo_config: PPOConfig,
    runner_config: RunnerConfig,
    obs_shape: tuple[int, ...] | None = None,
    n_actions: int | None = None,
) -> tuple[PPOTrainState, PPOMetricsHistory]:
    """Train PPO with data-parallel ``jit`` + FSDP ``NamedSharding``.

    The environment **must** be wrapped with ``AutoResetWrapper`` (or
    equivalent) so that episodes auto-reset inside the scan.

    Data is shaped as ``(n_devices, num_envs, *feature_dims)``.
    Gradient reduction is handled implicitly by JAX's GSPMD through
    the declared ``in_shardings``/``out_shardings`` on ``jax.jit``.

    When ``runner_config.fsdp_devices > 1``, large model parameters
    (>= 4 MB, 2-D+) are sharded across the FSDP axis to reduce
    per-device memory.  Small parameters are replicated.

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

    obs_dim = math.prod(obs_shape)

    # ------------------------------------------------------------------
    # Create mesh
    # ------------------------------------------------------------------
    mesh = make_mesh(
        num_fsdp_devices=runner_config.fsdp_devices,
        num_devices=n_devices,
    )

    # ------------------------------------------------------------------
    # Sharding specs
    # ------------------------------------------------------------------
    data_spec = data_sharding(mesh)        # leading axis over (batch, fsdp)
    replicated_spec = replicate_sharding(mesh)  # fully replicated

    # ------------------------------------------------------------------
    # Initialise agent state and compute FSDP param shardings
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(runner_config.seed)
    rng, agent_key = jax.random.split(rng)

    agent_state = PPO.init(agent_key, obs_shape, n_actions, ppo_config)

    # Compute per-parameter FSDP shardings via eval_shape (no
    # materialisation of large arrays).
    params_abstract = jax.eval_shape(
        lambda: eqx.filter(agent_state.params, eqx.is_array),
    )
    param_shardings = fsdp_sharding(params_abstract, mesh)

    # Build full params sharding tree (non-array eqx leaves → None).
    full_param_shardings = jax.tree.map(
        lambda leaf, spec: spec if eqx.is_array(leaf) else None,
        agent_state.params,
        param_shardings,
        is_leaf=lambda x: x is None,
    )

    # Optimizer state shardings mirror param structure.
    opt_state_abstract = jax.eval_shape(lambda: agent_state.opt_state)
    opt_state_shardings = fsdp_sharding(opt_state_abstract, mesh)

    # ------------------------------------------------------------------
    # Initialise n_devices * num_envs environments
    # ------------------------------------------------------------------
    device_keys = split_key_across_devices(rng, n_devices)
    env_rngs = jax.vmap(lambda k: jax.random.split(k, num_envs))(device_keys)
    # env_rngs: (n_devices, num_envs, 2)

    env_obs, env_states = jax.vmap(
        jax.vmap(env.reset, in_axes=(0, None)),
        in_axes=(0, None),
    )(env_rngs, env_params)
    # env_obs: (n_devices, num_envs, *obs_shape)

    # Single replicated RNG for the outer scan loop (GSPMD handles the
    # single-program view; per-device randomness comes from the agent
    # RNG and sharded env keys).
    rng, loop_key = jax.random.split(rng)

    init_state = PPOTrainState(
        agent_state=agent_state,
        env_obs=env_obs,
        env_state=env_states,
        rng=loop_key,
    )

    # ------------------------------------------------------------------
    # Compute train-state shardings for jit
    # ------------------------------------------------------------------
    agent_state_shardings = PPOState(
        params=full_param_shardings,
        opt_state=opt_state_shardings,
        step=replicated_spec,
        rng=replicated_spec,
    )

    env_state_shardings = jax.tree.map(lambda _: data_spec, env_states)

    train_state_shardings = PPOTrainState(
        agent_state=agent_state_shardings,
        env_obs=data_spec,
        env_state=env_state_shardings,
        rng=replicated_spec,
    )

    # Metrics are per-update scalars from the scan → (n_updates,), replicated.
    metrics_shardings = PPOMetricsHistory(
        total_loss=replicated_spec,
        actor_loss=replicated_spec,
        critic_loss=replicated_spec,
        entropy=replicated_spec,
        approx_kl=replicated_spec,
    )

    # ------------------------------------------------------------------
    # Place initial state on devices
    # ------------------------------------------------------------------
    init_state = jax.device_put(init_state, train_state_shardings)

    # ------------------------------------------------------------------
    # Build the training function
    # ------------------------------------------------------------------

    # Double-vmapped env step: (n_devices, num_envs) parallel envs
    batch_step = jax.vmap(
        jax.vmap(env.step, in_axes=(0, 0, 0, None)),
        in_axes=(0, 0, 0, None),
    )

    def _train_loop(
        train_state: PPOTrainState,
    ) -> tuple[PPOTrainState, PPOMetricsHistory]:
        """Full training loop: scan over n_updates.

        Env data has shape (n_devices, num_envs, ...) — sharded across
        the batch axis.  Model params are FSDP-sharded (or replicated
        for small params).  Rollout collection vmaps over both device
        and env axes.  The PPO update averages the loss over all
        n_devices * num_envs samples, and GSPMD handles cross-device
        gradient reduction.
        """

        def _collect_rollout_batch(
            agent_state: PPOState,
            env_obs: chex.Array,
            env_states,
        ):
            """Collect n_steps from (n_devices, num_envs) envs.

            obs shape: (n_devices, num_envs, *obs_shape)
            Returns trajectories with shape (n_steps, n_devices, num_envs, ...).
            """

            def _step(carry, _):
                a_state, obs, e_states = carry

                # act_batch expects (N, *obs_shape) — reshape to
                # (n_devices * num_envs, *obs_shape), act, reshape back.
                flat_obs = obs.reshape(n_devices * num_envs, *obs_shape)

                actions, log_probs, values, a_state = PPO.act_batch(
                    a_state, flat_obs, config=ppo_config,
                )
                # Reshape back to (n_devices, num_envs, ...)
                actions = actions.reshape(n_devices, num_envs)
                log_probs = log_probs.reshape(n_devices, num_envs)
                values = values.reshape(n_devices, num_envs)

                rng_step, step_key = jax.random.split(a_state.rng)
                a_state = a_state._replace(rng=rng_step)

                # Per-env step keys: (n_devices, num_envs, 2)
                step_keys = jax.random.split(step_key, n_devices * num_envs)
                step_keys = step_keys.reshape(n_devices, num_envs, 2)

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
                _step, (agent_state, env_obs, env_states), None, ppo_config.n_steps,
            )
            # trajectories fields: (n_steps, n_devices, num_envs, ...)

            # Last values for GAE bootstrap
            flat_final_obs = final_obs.reshape(n_devices * num_envs, *obs_shape)
            last_values = PPO.get_value_batch(
                agent_state, flat_final_obs, config=ppo_config,
            )
            last_values = last_values.reshape(n_devices, num_envs)

            return agent_state, trajectories, final_obs, final_env_states, last_values

        def _update_step(
            state: PPOState,
            trajectories: PPOTransition,
            last_value: chex.Array,
        ) -> tuple[PPOState, PPOMetrics]:
            """PPO update — gradient sync is implicit via GSPMD.

            Trajectories: (n_steps, n_devices, num_envs, ...).
            All samples are flattened into one batch for mini-batch SGD.
            Since data is sharded and params are replicated/FSDP-sharded,
            GSPMD automatically reduces gradients across devices.
            """
            # GAE: works on (T, D, N) shapes
            advantages, returns = compute_gae(
                rewards=trajectories.reward,
                values=trajectories.value,
                dones=trajectories.done,
                last_value=last_value,
                gamma=ppo_config.gamma,
                gae_lambda=ppo_config.gae_lambda,
            )

            # Flatten all dims for mini-batch SGD
            batch_size = advantages.size
            flat_traj = jax.tree.map(
                lambda x: x.reshape(batch_size, *x.shape[len(advantages.shape):]),
                trajectories,
            )
            flat_advantages = advantages.reshape(batch_size)
            flat_returns = returns.reshape(batch_size)

            optimizer = ppo_config.make_optimizer()

            def _epoch(carry, _):
                params, opt_state, epoch_rng = carry
                epoch_rng, shuffle_key = jax.random.split(epoch_rng)

                perm = jax.random.permutation(shuffle_key, batch_size)
                mb_size = batch_size // ppo_config.n_minibatches

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
                            params, mb_obs, mb_actions, config=ppo_config,
                        )

                        ratio = jnp.exp(log_probs - mb_old_log_probs)
                        surr1 = ratio * mb_advantages
                        surr2 = (
                            jnp.clip(
                                ratio,
                                1.0 - ppo_config.clip_eps,
                                1.0 + ppo_config.clip_eps,
                            )
                            * mb_advantages
                        )
                        actor_loss = -jnp.minimum(surr1, surr2).mean()

                        value_pred_clipped = mb_old_values + jnp.clip(
                            values - mb_old_values,
                            -ppo_config.clip_eps,
                            ppo_config.clip_eps,
                        )
                        vf_loss1 = (values - mb_returns) ** 2
                        vf_loss2 = (value_pred_clipped - mb_returns) ** 2
                        critic_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2).mean()

                        entropy_mean = entropy.mean()
                        total_loss = (
                            actor_loss
                            + ppo_config.vf_coef * critic_loss
                            - ppo_config.ent_coef * entropy_mean
                        )
                        approx_kl = (mb_old_log_probs - log_probs).mean()

                        return total_loss, (actor_loss, critic_loss, entropy_mean, approx_kl)

                    (total_loss, (actor_loss, critic_loss, entropy_mean, approx_kl)), grads = (
                        eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
                    )

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

                start_indices = jnp.arange(ppo_config.n_minibatches) * mb_size
                (params, opt_state), mb_metrics = jax.lax.scan(
                    _minibatch_step, (params, opt_state), start_indices,
                )

                return (params, opt_state, epoch_rng), mb_metrics

            (new_params, new_opt_state, new_rng), epoch_metrics = jax.lax.scan(
                _epoch,
                (state.params, state.opt_state, state.rng),
                None,
                ppo_config.n_epochs,
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
            loop_rng, _ = jax.random.split(train_state.rng)

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
                rng=loop_rng,
            )
            return new_train_state, metrics

        final_state, metrics_history = jax.lax.scan(
            _scan_body, train_state, None, length=n_updates,
        )

        history = PPOMetricsHistory(
            total_loss=metrics_history.total_loss,
            actor_loss=metrics_history.actor_loss,
            critic_loss=metrics_history.critic_loss,
            entropy=metrics_history.entropy,
            approx_kl=metrics_history.approx_kl,
        )

        return final_state, history

    # ------------------------------------------------------------------
    # JIT with explicit shardings (GSPMD handles gradient reduction)
    # ------------------------------------------------------------------
    jitted_train = jax.jit(
        _train_loop,
        in_shardings=(train_state_shardings,),
        out_shardings=(train_state_shardings, metrics_shardings),
        donate_argnums=(0,),
    )

    return jitted_train(init_state)
