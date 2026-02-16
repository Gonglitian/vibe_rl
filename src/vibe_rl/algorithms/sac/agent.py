"""Pure-functional SAC (Soft Actor-Critic) agent.

All methods are static pure functions — no mutable state anywhere.
State is threaded explicitly through ``SACState``.

Implements:
  - Reparameterized Gaussian policy with tanh squashing
  - Clipped double-Q learning (twin Q-networks)
  - Automatic temperature (alpha) tuning (optional)
  - Soft target network updates (Polyak averaging)

Usage::

    config = SACConfig()
    state = SAC.init(rng, obs_shape=(3,), action_dim=1, config=config)
    action, state = SAC.act(state, obs, config=config, explore=True)
    state, metrics = SAC.update(state, batch, config=config)
"""

from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from vibe_rl.algorithms.sac.config import SACConfig
# Note: optax is still imported for optax.apply_updates in the alpha update path.
from vibe_rl.algorithms.sac.network import GaussianActor, TwinQNetwork
from vibe_rl.algorithms.sac.types import SACState
from vibe_rl.types import Transition

# Numerical stability constant for log computations
_LOG_EPS = 1e-6


class SACMetrics(NamedTuple):
    actor_loss: chex.Array
    critic_loss: chex.Array
    alpha_loss: chex.Array
    alpha: chex.Array
    entropy: chex.Array
    q_mean: chex.Array


def _sample_action(
    actor_params: GaussianActor,
    obs: jax.Array,
    key: chex.PRNGKey,
    config: SACConfig,
) -> tuple[jax.Array, jax.Array]:
    """Reparameterized sample from the squashed Gaussian policy.

    Returns:
        (action, log_prob) where action is in [action_low, action_high]
        and log_prob is the corrected log-probability (accounting for
        tanh squashing).
    """
    mean, log_std = actor_params(obs)
    log_std = jnp.clip(log_std, config.log_std_min, config.log_std_max)
    std = jnp.exp(log_std)

    # Reparameterization trick: z = mean + std * eps
    eps = jax.random.normal(key, shape=mean.shape)
    z = mean + std * eps

    # Tanh squashing -> [-1, 1]
    action_tanh = jnp.tanh(z)

    # Log-prob with tanh correction: log pi(a|s) = log N(z) - sum log(1 - tanh(z)^2)
    log_prob = -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_std + eps**2)
    log_prob = jnp.sum(log_prob, axis=-1)
    # Tanh squashing correction
    log_prob = log_prob - jnp.sum(jnp.log(1 - action_tanh**2 + _LOG_EPS), axis=-1)

    # Rescale from [-1, 1] to [action_low, action_high]
    low, high = config.action_low, config.action_high
    action = low + 0.5 * (action_tanh + 1.0) * (high - low)

    return action, log_prob


def _deterministic_action(
    actor_params: GaussianActor,
    obs: jax.Array,
    config: SACConfig,
) -> jax.Array:
    """Deterministic action: tanh(mean) rescaled to action bounds."""
    mean, _log_std = actor_params(obs)
    action_tanh = jnp.tanh(mean)
    low, high = config.action_low, config.action_high
    return low + 0.5 * (action_tanh + 1.0) * (high - low)


class SAC:
    """Namespace for SAC pure functions.

    Not instantiated — all methods are static.
    """

    @staticmethod
    def init(
        rng: chex.PRNGKey,
        obs_shape: tuple[int, ...],
        action_dim: int,
        config: SACConfig,
    ) -> SACState:
        """Create initial SAC state."""
        obs_dim = math.prod(obs_shape)
        k_actor, k_critic, k_target, k_state = jax.random.split(rng, 4)

        actor = GaussianActor(
            obs_dim, action_dim, config.hidden_sizes, key=k_actor
        )
        critic = TwinQNetwork(
            obs_dim, action_dim, config.hidden_sizes, key=k_critic
        )
        # Target critic initialised as copy of critic
        target_critic = TwinQNetwork(
            obs_dim, action_dim, config.hidden_sizes, key=k_critic
        )

        actor_optimizer = config.make_actor_optimizer()
        critic_optimizer = config.make_critic_optimizer()
        alpha_optimizer = config.make_alpha_optimizer()

        actor_opt_state = actor_optimizer.init(
            eqx.filter(actor, eqx.is_array)
        )
        critic_opt_state = critic_optimizer.init(
            eqx.filter(critic, eqx.is_array)
        )

        log_alpha = jnp.log(jnp.array(config.init_alpha, dtype=jnp.float32))
        alpha_opt_state = alpha_optimizer.init(log_alpha)

        return SACState(
            actor_params=actor,
            critic_params=critic,
            target_critic_params=target_critic,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
            step=jnp.zeros((), dtype=jnp.int32),
            rng=k_state,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("config", "explore"))
    def act(
        state: SACState,
        obs: chex.Array,
        *,
        config: SACConfig,
        explore: bool = True,
    ) -> tuple[chex.Array, SACState]:
        """Select action from the policy (pure function).

        Args:
            state: Current SAC state.
            obs: Single observation, shape ``(*obs_shape,)``.
            config: SAC hyperparameters (static).
            explore: If True, sample stochastically. If False, use mean.

        Returns:
            (action, new_state) — action shape ``(action_dim,)``.
        """
        rng, key = jax.random.split(state.rng)

        action = jax.lax.cond(
            explore,
            lambda: _sample_action(state.actor_params, obs, key, config)[0],
            lambda: _deterministic_action(state.actor_params, obs, config),
        )

        new_state = state._replace(rng=rng)
        return action, new_state

    @staticmethod
    @partial(jax.jit, static_argnames=("config",))
    def update(
        state: SACState,
        batch: Transition,
        *,
        config: SACConfig,
    ) -> tuple[SACState, SACMetrics]:
        """One gradient step on a batch of transitions (pure function).

        Performs three updates in sequence:
        1. Critic (twin Q) update via soft Bellman residual
        2. Actor update via maximum entropy objective
        3. Temperature (alpha) update (if autotune_alpha)

        Args:
            state: Current SAC state.
            batch: Batched transitions, each field has shape ``(B, ...)``.
            config: SAC hyperparameters (static).

        Returns:
            (new_state, metrics) tuple.
        """
        rng, key_critic, key_actor, key_alpha = jax.random.split(state.rng, 4)
        alpha = jnp.exp(state.log_alpha)
        action_dim = batch.action.shape[-1]
        target_entropy = -config.target_entropy_scale * action_dim

        # --- Critic update ---
        critic_optimizer = config.make_critic_optimizer()

        def critic_loss_fn(critic_params):
            # Current Q-values for actions in the batch
            q1, q2 = jax.vmap(
                lambda o, a: critic_params(o, a)
            )(batch.obs, batch.action)

            # Target: r + gamma * (min Q_target(s', a') - alpha * log pi(a'|s'))
            next_keys = jax.random.split(key_critic, batch.obs.shape[0])

            def _next_value(next_obs, k):
                next_action, next_log_prob = _sample_action(
                    state.actor_params, next_obs, k, config
                )
                next_q1, next_q2 = state.target_critic_params(
                    next_obs, next_action
                )
                next_q_min = jnp.minimum(next_q1, next_q2)
                return next_q_min - alpha * next_log_prob

            next_v = jax.vmap(_next_value)(batch.next_obs, next_keys)
            targets = batch.reward + config.gamma * next_v * (1.0 - batch.done)
            targets = jax.lax.stop_gradient(targets)

            loss_q1 = jnp.mean((q1 - targets) ** 2)
            loss_q2 = jnp.mean((q2 - targets) ** 2)
            critic_loss = 0.5 * (loss_q1 + loss_q2)
            return critic_loss, jnp.mean(jnp.stack([q1, q2]))

        (critic_loss, q_mean), critic_grads = eqx.filter_value_and_grad(
            critic_loss_fn, has_aux=True
        )(state.critic_params)

        critic_updates, new_critic_opt_state = critic_optimizer.update(
            critic_grads,
            state.critic_opt_state,
            eqx.filter(state.critic_params, eqx.is_array),
        )
        new_critic_params = eqx.apply_updates(state.critic_params, critic_updates)

        # --- Actor update ---
        actor_optimizer = config.make_actor_optimizer()

        def actor_loss_fn(actor_params):
            actor_keys = jax.random.split(key_actor, batch.obs.shape[0])

            def _per_sample(obs, k):
                action, log_prob = _sample_action(actor_params, obs, k, config)
                q1, q2 = new_critic_params(obs, action)
                q_min = jnp.minimum(q1, q2)
                return alpha * log_prob - q_min, log_prob

            losses, log_probs = jax.vmap(_per_sample)(batch.obs, actor_keys)
            return jnp.mean(losses), jnp.mean(-log_probs)

        (actor_loss, entropy), actor_grads = eqx.filter_value_and_grad(
            actor_loss_fn, has_aux=True
        )(state.actor_params)

        actor_updates, new_actor_opt_state = actor_optimizer.update(
            actor_grads,
            state.actor_opt_state,
            eqx.filter(state.actor_params, eqx.is_array),
        )
        new_actor_params = eqx.apply_updates(state.actor_params, actor_updates)

        # --- Alpha (temperature) update ---
        alpha_optimizer = config.make_alpha_optimizer()

        def alpha_loss_fn(log_alpha_val):
            alpha_keys = jax.random.split(key_alpha, batch.obs.shape[0])

            def _per_sample(obs, k):
                _action, log_prob = _sample_action(
                    new_actor_params, obs, k, config
                )
                return log_prob

            log_probs = jax.vmap(_per_sample)(batch.obs, alpha_keys)
            # Eq. 18 from SAC paper (arXiv:1812.05905)
            loss = -jnp.exp(log_alpha_val) * jnp.mean(
                jax.lax.stop_gradient(log_probs) + target_entropy
            )
            return loss

        if config.autotune_alpha:
            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(
                state.log_alpha
            )
            alpha_updates, new_alpha_opt_state = alpha_optimizer.update(
                alpha_grads, state.alpha_opt_state
            )
            new_log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)
        else:
            alpha_loss = jnp.array(0.0)
            new_log_alpha = state.log_alpha
            new_alpha_opt_state = state.alpha_opt_state

        # --- Soft target update (Polyak averaging) ---
        new_target_critic_params = jax.tree.map(
            lambda target, online: (1.0 - config.tau) * target + config.tau * online,
            eqx.filter(state.target_critic_params, eqx.is_array),
            eqx.filter(new_critic_params, eqx.is_array),
        )
        new_target_critic = eqx.combine(
            new_target_critic_params,
            eqx.filter(state.target_critic_params, lambda x: not eqx.is_array(x)),
        )

        new_state = SACState(
            actor_params=new_actor_params,
            critic_params=new_critic_params,
            target_critic_params=new_target_critic,
            actor_opt_state=new_actor_opt_state,
            critic_opt_state=new_critic_opt_state,
            log_alpha=new_log_alpha,
            alpha_opt_state=new_alpha_opt_state,
            step=state.step + 1,
            rng=rng,
        )

        metrics = SACMetrics(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            alpha_loss=alpha_loss,
            alpha=jnp.exp(new_log_alpha),
            entropy=entropy,
            q_mean=q_mean,
        )

        return new_state, metrics
