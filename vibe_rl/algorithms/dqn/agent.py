"""Pure-functional DQN agent (Mnih et al., 2015) in JAX.

All state is explicit; every function is jit-compatible.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.dqn.network import QNetwork
from vibe_rl.types import Transition


# ---------------------------------------------------------------------------
# State & metrics
# ---------------------------------------------------------------------------


class DQNState(NamedTuple):
    """Immutable training state for DQN."""

    params: eqx.Module  # online Q-network
    target_params: eqx.Module  # target Q-network
    opt_state: optax.OptState
    rng: jax.Array
    step: jax.Array  # scalar int32


class DQNMetrics(NamedTuple):
    """Metrics returned by a single update step."""

    loss: jax.Array
    q_mean: jax.Array
    epsilon: jax.Array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _epsilon_from_step(step: jax.Array, config: DQNConfig) -> jax.Array:
    """Linear epsilon schedule derived from step count."""
    fraction = jnp.minimum(step / jnp.maximum(config.epsilon_decay_steps, 1), 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def _copy_params(model: eqx.Module) -> eqx.Module:
    """Deep-copy array leaves of an equinox module."""
    arrays = eqx.filter(model, eqx.is_array)
    copied = jax.tree.map(lambda a: jnp.array(a), arrays)
    static = eqx.filter(model, lambda x: not eqx.is_array(x))
    return eqx.combine(copied, static)


# ---------------------------------------------------------------------------
# DQN namespace
# ---------------------------------------------------------------------------


class DQN:
    """Functional DQN agent — a namespace of pure static methods."""

    @staticmethod
    def init(
        rng: jax.Array,
        obs_shape: tuple[int, ...],
        n_actions: int,
        config: DQNConfig,
    ) -> DQNState:
        """Create initial ``DQNState`` from config."""
        obs_dim = int(jnp.prod(jnp.array(obs_shape)))
        k1, k2 = jax.random.split(rng)

        params = QNetwork(obs_dim, n_actions, config.hidden_sizes, key=k1)
        target_params = _copy_params(params)

        tx = optax.adam(config.lr)
        opt_state = tx.init(eqx.filter(params, eqx.is_array))

        return DQNState(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            rng=k2,
            step=jnp.array(0, dtype=jnp.int32),
        )

    @staticmethod
    @eqx.filter_jit
    def act(
        state: DQNState,
        obs: jax.Array,
        *,
        config: DQNConfig,
        explore: bool = True,
    ) -> tuple[jax.Array, DQNState]:
        """Select an action via epsilon-greedy policy."""
        k1, k2, new_rng = jax.random.split(state.rng, 3)

        # Greedy action from online network
        q_values = state.params(obs)
        greedy_action = jnp.argmax(q_values)

        if explore:
            epsilon = _epsilon_from_step(state.step, config)
            n_actions = q_values.shape[0]
            random_action = jax.random.randint(k1, shape=(), minval=0, maxval=n_actions)
            do_explore = jax.random.uniform(k2) < epsilon
            action = jnp.where(do_explore, random_action, greedy_action)
        else:
            action = greedy_action

        new_state = state._replace(rng=new_rng)
        return action, new_state

    @staticmethod
    @eqx.filter_jit
    def update(
        state: DQNState,
        batch: Transition,
        *,
        config: DQNConfig,
    ) -> tuple[DQNState, DQNMetrics]:
        """One gradient step on a batch of transitions."""
        tx = optax.adam(config.lr)

        def loss_fn(params, target_params, batch):
            q_all = jax.vmap(params)(batch.obs)  # (B, n_actions)
            q_sa = q_all[jnp.arange(q_all.shape[0]), batch.action]

            next_q = jax.vmap(target_params)(batch.next_obs)  # (B, n_actions)
            next_q_max = jnp.max(next_q, axis=-1)
            target = batch.reward + config.gamma * next_q_max * (1.0 - batch.done)

            loss = jnp.mean((q_sa - jax.lax.stop_gradient(target)) ** 2)
            return loss, q_sa

        (loss, q_sa), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            state.params, state.target_params, batch
        )

        # Apply optimizer updates to array leaves only
        params_arrays = eqx.filter(state.params, eqx.is_array)
        grad_arrays = eqx.filter(grads, eqx.is_array)
        updates, new_opt_state = tx.update(grad_arrays, state.opt_state, params_arrays)
        new_params = eqx.apply_updates(state.params, updates)

        new_step = state.step + 1

        # Hard target update at target_update_freq
        do_target_update = new_step % config.target_update_freq == 0
        new_target_arrays = jax.tree.map(
            lambda online, target: jnp.where(do_target_update, online, target),
            eqx.filter(new_params, eqx.is_array),
            eqx.filter(state.target_params, eqx.is_array),
        )
        new_target_params = eqx.combine(
            new_target_arrays,
            eqx.filter(state.target_params, lambda x: not eqx.is_array(x)),
        )

        epsilon = _epsilon_from_step(new_step, config)

        new_state = DQNState(
            params=new_params,
            target_params=new_target_params,
            opt_state=new_opt_state,
            rng=state.rng,
            step=new_step,
        )

        metrics = DQNMetrics(
            loss=loss,
            q_mean=jnp.mean(q_sa),
            epsilon=epsilon,
        )
        return new_state, metrics
