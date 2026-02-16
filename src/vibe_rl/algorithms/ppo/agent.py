"""Pure-functional PPO agent.

All methods are static pure functions — no mutable state anywhere.
State is threaded explicitly through ``PPOState``.

Supports both discrete (Categorical) action spaces.
The entire collect + update loop is jit-compatible via ``jax.lax.scan``.

Usage::

    config = PPOConfig()
    state = PPO.init(rng, obs_shape=(4,), n_actions=2, config=config)
    action, log_prob, value, state = PPO.act(state, obs, config=config)
    state, metrics = PPO.update(state, trajectories, config=config)
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.algorithms.ppo.config import PPOConfig
from vibe_rl.algorithms.ppo.network import ActorCategorical, ActorCriticShared, Critic
from vibe_rl.algorithms.ppo.types import ActorCriticParams, PPOState
from vibe_rl.dataprotocol.transition import PPOTransition


class PPOMetrics(NamedTuple):
    total_loss: chex.Array
    actor_loss: chex.Array
    critic_loss: chex.Array
    entropy: chex.Array
    approx_kl: chex.Array


# ---------------------------------------------------------------------------
# GAE (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: chex.Array,
    values: chex.Array,
    dones: chex.Array,
    last_value: chex.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[chex.Array, chex.Array]:
    """Compute GAE advantages and value targets.

    Args:
        rewards: shape ``(T,)`` or ``(T, N)`` for vectorized envs.
        values: shape ``(T,)`` or ``(T, N)``.
        dones: shape ``(T,)`` or ``(T, N)``.
        last_value: shape ``()`` or ``(N,)`` — bootstrap value for final step.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        ``(advantages, returns)`` each with same shape as ``rewards``.
    """

    def _scan_fn(carry, transition):
        gae, next_value = carry
        reward, value, done = transition
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, value), gae

    init_carry = (jnp.zeros_like(last_value), last_value)
    # Scan in reverse time order
    _, advantages = jax.lax.scan(
        _scan_fn,
        init_carry,
        (rewards, values, dones),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO agent namespace
# ---------------------------------------------------------------------------

class PPO:
    """Namespace for PPO pure functions.

    Not instantiated — all methods are static.
    """

    @staticmethod
    def init(
        rng: chex.PRNGKey,
        obs_shape: tuple[int, ...],
        n_actions: int,
        config: PPOConfig,
    ) -> PPOState:
        """Create initial PPO state."""
        import math

        obs_dim = math.prod(obs_shape)
        k1, k2, k3 = jax.random.split(rng, 3)

        if config.shared_backbone:
            params = ActorCriticShared(
                obs_dim, n_actions, config.hidden_sizes, key=k1,
            )
        else:
            actor = ActorCategorical(
                obs_dim, n_actions, config.hidden_sizes, key=k1,
            )
            critic = Critic(obs_dim, config.hidden_sizes, key=k2)
            params = ActorCriticParams(actor=actor, critic=critic)

        optimizer = config.make_optimizer()
        opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

        return PPOState(
            params=params,
            opt_state=opt_state,
            step=jnp.zeros((), dtype=jnp.int32),
            rng=k3,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("config",))
    def act(
        state: PPOState,
        obs: chex.Array,
        *,
        config: PPOConfig,
    ) -> tuple[chex.Array, chex.Array, chex.Array, PPOState]:
        """Select action, returning ``(action, log_prob, value, new_state)``.

        Args:
            state: Current PPO state.
            obs: Single observation, shape ``(*obs_shape,)``.
            config: PPO hyperparameters (static).

        Returns:
            ``(action, log_prob, value, new_state)`` tuple.
        """
        rng, key = jax.random.split(state.rng)

        if config.shared_backbone:
            logits, value = state.params(obs)
        else:
            logits = state.params.actor(obs)
            value = state.params.critic(obs)

        # Categorical sampling
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[action]

        new_state = state._replace(rng=rng)
        return action, log_prob, value, new_state

    @staticmethod
    @partial(jax.jit, static_argnames=("config",))
    def act_batch(
        state: PPOState,
        obs: chex.Array,
        *,
        config: PPOConfig,
    ) -> tuple[chex.Array, chex.Array, chex.Array, PPOState]:
        """Vectorized action selection for a batch of observations.

        Args:
            state: Current PPO state.
            obs: Batch of observations, shape ``(N, *obs_shape)``.
            config: PPO hyperparameters (static).

        Returns:
            ``(actions, log_probs, values, new_state)`` each with leading dim N.
        """
        rng, key = jax.random.split(state.rng)
        n_envs = obs.shape[0]
        keys = jax.random.split(key, n_envs)

        if config.shared_backbone:
            logits, values = jax.vmap(state.params)(obs)
        else:
            logits = jax.vmap(state.params.actor)(obs)
            values = jax.vmap(state.params.critic)(obs)

        actions = jax.vmap(jax.random.categorical)(keys, logits)
        log_softmax = jax.nn.log_softmax(logits)
        log_probs = log_softmax[jnp.arange(n_envs), actions]

        new_state = state._replace(rng=rng)
        return actions, log_probs, values, new_state

    @staticmethod
    @partial(jax.jit, static_argnames=("config",))
    def evaluate_actions(
        params: chex.ArrayTree,
        obs: chex.Array,
        actions: chex.Array,
        *,
        config: PPOConfig,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        """Re-evaluate log_prob, value, entropy for a batch of (obs, action).

        Used inside the update step to compute losses with current params.

        Returns:
            ``(log_probs, values, entropy)`` each shape ``(B,)``.
        """
        if config.shared_backbone:
            logits, values = jax.vmap(params)(obs)
        else:
            logits = jax.vmap(params.actor)(obs)
            values = jax.vmap(params.critic)(obs)

        log_softmax = jax.nn.log_softmax(logits)
        log_probs = log_softmax[jnp.arange(obs.shape[0]), actions.astype(jnp.int32)]

        # Entropy of categorical distribution
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_softmax, axis=-1)

        return log_probs, values, entropy

    @staticmethod
    @partial(jax.jit, static_argnames=("config",))
    def get_value(
        state: PPOState,
        obs: chex.Array,
        *,
        config: PPOConfig,
    ) -> chex.Array:
        """Compute value estimate for a single observation."""
        if config.shared_backbone:
            _, value = state.params(obs)
        else:
            value = state.params.critic(obs)
        return value

    @staticmethod
    @partial(jax.jit, static_argnames=("config",))
    def get_value_batch(
        state: PPOState,
        obs: chex.Array,
        *,
        config: PPOConfig,
    ) -> chex.Array:
        """Compute value estimates for a batch of observations."""
        if config.shared_backbone:
            _, values = jax.vmap(state.params)(obs)
        else:
            values = jax.vmap(state.params.critic)(obs)
        return values

    @staticmethod
    @partial(jax.jit, static_argnames=("config",))
    def update(
        state: PPOState,
        trajectories: PPOTransition,
        last_value: chex.Array,
        *,
        config: PPOConfig,
    ) -> tuple[PPOState, PPOMetrics]:
        """PPO update: compute GAE, then run multiple epochs of mini-batch SGD.

        Args:
            state: Current PPO state.
            trajectories: Collected rollout data, each field shape ``(T, ...)``.
                ``T = config.n_steps``. For vectorized envs: ``(T, N, ...)``.
            last_value: Bootstrap value for the final observation.
            config: PPO hyperparameters (static).

        Returns:
            ``(new_state, metrics)`` tuple.
        """
        # Compute advantages with GAE
        advantages, returns = compute_gae(
            rewards=trajectories.reward,
            values=trajectories.value,
            dones=trajectories.done,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # Flatten time (and env) dimensions for mini-batch SGD
        # trajectories fields: (T, ...) or (T, N, ...)
        batch_size = advantages.size
        flat_traj = jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[len(advantages.shape):]), trajectories)
        flat_advantages = advantages.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)

        optimizer = config.make_optimizer()

        def _epoch(carry, _):
            params, opt_state, rng = carry
            rng, shuffle_key = jax.random.split(rng)

            # Shuffle and split into minibatches
            perm = jax.random.permutation(shuffle_key, batch_size)
            mb_size = batch_size // config.n_minibatches

            def _minibatch_step(carry, start_idx):
                params, opt_state = carry
                mb_idx = jax.lax.dynamic_slice(perm, (start_idx,), (mb_size,))

                mb_obs = flat_traj.obs[mb_idx]
                mb_actions = flat_traj.action[mb_idx]
                mb_old_log_probs = flat_traj.log_prob[mb_idx]
                mb_old_values = flat_traj.value[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]

                # Normalize advantages per minibatch
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                def loss_fn(params):
                    log_probs, values, entropy = PPO.evaluate_actions(
                        params, mb_obs, mb_actions, config=config,
                    )

                    # PPO clipped surrogate objective
                    ratio = jnp.exp(log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * mb_advantages
                    actor_loss = -jnp.minimum(surr1, surr2).mean()

                    # Clipped value loss
                    value_pred_clipped = mb_old_values + jnp.clip(
                        values - mb_old_values, -config.clip_eps, config.clip_eps,
                    )
                    vf_loss1 = (values - mb_returns) ** 2
                    vf_loss2 = (value_pred_clipped - mb_returns) ** 2
                    critic_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2).mean()

                    entropy_mean = entropy.mean()

                    total_loss = actor_loss + config.vf_coef * critic_loss - config.ent_coef * entropy_mean

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

            start_indices = jnp.arange(config.n_minibatches) * mb_size
            (params, opt_state), mb_metrics = jax.lax.scan(
                _minibatch_step, (params, opt_state), start_indices,
            )

            return (params, opt_state, rng), mb_metrics

        (new_params, new_opt_state, new_rng), epoch_metrics = jax.lax.scan(
            _epoch, (state.params, state.opt_state, state.rng), None, config.n_epochs,
        )

        # Average metrics across all epochs and minibatches
        avg_metrics = jax.tree.map(lambda x: x.mean(), epoch_metrics)

        new_state = PPOState(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1,
            rng=new_rng,
        )

        return new_state, avg_metrics

    @staticmethod
    @partial(jax.jit, static_argnames=("env_step_fn", "config"))
    def collect_rollout(
        state: PPOState,
        env_obs: chex.Array,
        env_state: chex.ArrayTree,
        env_step_fn: callable,
        env_params: chex.ArrayTree,
        *,
        config: PPOConfig,
    ) -> tuple[PPOState, PPOTransition, chex.Array, chex.ArrayTree, chex.Array]:
        """Collect a fixed-length rollout using ``lax.scan``.

        This performs ``config.n_steps`` environment transitions, storing
        each as a ``PPOTransition``. After collection, the buffer is used
        for a single PPO update and then discarded.

        Args:
            state: Current PPO agent state.
            env_obs: Current observation, shape ``(*obs_shape,)``.
            env_state: Current environment state (PyTree).
            env_step_fn: ``step(key, state, action, params) -> (obs, state, reward, done, info)``
            env_params: Environment parameters.
            config: PPO hyperparameters (static).

        Returns:
            ``(new_agent_state, trajectories, final_obs, final_env_state, last_value)``
        """

        def _step(carry, _):
            agent_state, obs, e_state = carry
            action, log_prob, value, agent_state = PPO.act(
                agent_state, obs, config=config,
            )
            rng, step_key = jax.random.split(agent_state.rng)
            agent_state = agent_state._replace(rng=rng)

            next_obs, new_e_state, reward, done, _info = env_step_fn(
                step_key, e_state, action, env_params,
            )

            transition = PPOTransition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                log_prob=log_prob,
                value=value,
            )
            return (agent_state, next_obs, new_e_state), transition

        (state, final_obs, final_env_state), trajectories = jax.lax.scan(
            _step, (state, env_obs, env_state), None, config.n_steps,
        )

        last_value = PPO.get_value(state, final_obs, config=config)

        return state, trajectories, final_obs, final_env_state, last_value

    @staticmethod
    @partial(jax.jit, static_argnames=("env_step_fn", "config"))
    def collect_rollout_batch(
        state: PPOState,
        env_obs: chex.Array,
        env_states: chex.ArrayTree,
        env_step_fn: callable,
        env_params: chex.ArrayTree,
        *,
        config: PPOConfig,
    ) -> tuple[PPOState, PPOTransition, chex.Array, chex.ArrayTree, chex.Array]:
        """Collect rollouts from N parallel environments using vmap.

        Args:
            state: Current PPO agent state (shared across envs).
            env_obs: Observations shape ``(N, *obs_shape)``.
            env_states: Batched env states.
            env_step_fn: Single-env step function.
            env_params: Env parameters (shared, not batched).
            config: PPO hyperparameters (static).

        Returns:
            ``(new_state, trajectories, final_obs, final_env_states, last_values)``
            where trajectories fields have shape ``(T, N, ...)``.
        """
        batch_step = jax.vmap(env_step_fn, in_axes=(0, 0, 0, None))

        def _step(carry, _):
            agent_state, obs, e_states = carry
            actions, log_probs, values, agent_state = PPO.act_batch(
                agent_state, obs, config=config,
            )
            rng, step_key = jax.random.split(agent_state.rng)
            agent_state = agent_state._replace(rng=rng)

            n_envs = obs.shape[0]
            step_keys = jax.random.split(step_key, n_envs)

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
            return (agent_state, next_obs, new_e_states), transition

        (state, final_obs, final_env_states), trajectories = jax.lax.scan(
            _step, (state, env_obs, env_states), None, config.n_steps,
        )

        last_values = PPO.get_value_batch(state, final_obs, config=config)

        return state, trajectories, final_obs, final_env_states, last_values
