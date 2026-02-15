"""Pure-functional DQN agent.

All methods are static pure functions — no mutable state anywhere.
State is threaded explicitly through ``DQNState``.

Usage::

    config = DQNConfig()
    state = DQN.init(rng, obs_shape=(4,), n_actions=2, config=config)
    action, state = DQN.act(state, obs, explore=True)
    state, metrics = DQN.update(state, batch)
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from vibe_rl.algorithms.dqn.config import DQNConfig
from vibe_rl.algorithms.dqn.network import QNetwork
from vibe_rl.algorithms.dqn.types import DQNState
from vibe_rl.types import Transition


class DQNMetrics(NamedTuple):
    loss: chex.Array
    q_mean: chex.Array
    epsilon: chex.Array


class DQN:
    """Namespace for DQN pure functions.

    Not instantiated — all methods are static.
    Satisfies the ``Agent`` protocol.
    """

    @staticmethod
    def init(
        rng: chex.PRNGKey,
        obs_shape: tuple[int, ...],
        n_actions: int,
        config: DQNConfig,
    ) -> DQNState:
        """Create initial DQN state."""
        import math

        obs_dim = math.prod(obs_shape)
        k1, k2 = jax.random.split(rng)

        q_net = QNetwork(obs_dim, n_actions, config.hidden_sizes, key=k1)
        target_net = QNetwork(obs_dim, n_actions, config.hidden_sizes, key=k1)

        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr),
        )
        opt_state = optimizer.init(eqx.filter(q_net, eqx.is_array))

        return DQNState(
            params=q_net,
            target_params=target_net,
            opt_state=opt_state,
            step=jnp.zeros((), dtype=jnp.int32),
            rng=k2,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("config", "explore"))
    def act(
        state: DQNState,
        obs: chex.Array,
        *,
        config: DQNConfig,
        explore: bool = True,
    ) -> tuple[chex.Array, DQNState]:
        """Select action via epsilon-greedy policy (pure function).

        Args:
            state: Current DQN state.
            obs: Single observation, shape ``(*obs_shape,)``.
            config: DQN hyperparameters (static, for epsilon schedule).
            explore: If True, use epsilon-greedy. If False, greedy.

        Returns:
            (action, new_state) — action is a scalar int array.
        """
        rng, key_eps, key_rand = jax.random.split(state.rng, 3)

        # Epsilon schedule: linear decay
        frac = jnp.clip(state.step / config.epsilon_decay_steps, 0.0, 1.0)
        epsilon = config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)

        # Greedy action from Q-network
        q_values = state.params(obs)
        greedy_action = jnp.argmax(q_values)

        # Random action
        n_actions = q_values.shape[-1]
        random_action = jax.random.randint(key_rand, (), 0, n_actions)

        # Epsilon-greedy selection (when explore=True)
        use_random = jax.random.uniform(key_eps) < epsilon
        action = jax.lax.cond(
            explore & use_random,
            lambda: random_action,
            lambda: greedy_action,
        )

        new_state = state._replace(rng=rng)
        return action, new_state

    @staticmethod
    @partial(jax.jit, static_argnames=("config",))
    def update(
        state: DQNState,
        batch: Transition,
        *,
        config: DQNConfig,
    ) -> tuple[DQNState, DQNMetrics]:
        """One gradient step on a batch of transitions (pure function).

        Args:
            state: Current DQN state.
            batch: Batched transitions, each field has shape ``(B, ...)``.
            config: DQN hyperparameters (static).

        Returns:
            (new_state, metrics) tuple.
        """
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr),
        )

        def loss_fn(params):
            # Q(s, a) for chosen actions
            q_all = jax.vmap(params)(batch.obs)  # (B, n_actions)
            q_values = q_all[jnp.arange(q_all.shape[0]), batch.action.astype(jnp.int32)]

            # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
            next_q_all = jax.vmap(state.target_params)(batch.next_obs)
            next_q_max = jnp.max(next_q_all, axis=-1)
            targets = batch.reward + config.gamma * next_q_max * (1.0 - batch.done)

            loss = jnp.mean((q_values - jax.lax.stop_gradient(targets)) ** 2)
            return loss, q_values

        (loss, q_values), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            state.params
        )

        # Apply gradients
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, eqx.filter(state.params, eqx.is_array)
        )
        new_params = eqx.apply_updates(state.params, updates)

        # Periodically update target network
        new_step = state.step + 1
        new_target_params = jax.lax.cond(
            new_step % config.target_update_freq == 0,
            lambda: new_params,
            lambda: state.target_params,
        )

        new_state = DQNState(
            params=new_params,
            target_params=new_target_params,
            opt_state=new_opt_state,
            step=new_step,
            rng=state.rng,
        )

        frac = jnp.clip(new_step / config.epsilon_decay_steps, 0.0, 1.0)
        epsilon = config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)

        metrics = DQNMetrics(
            loss=loss,
            q_mean=jnp.mean(q_values),
            epsilon=epsilon,
        )

        return new_state, metrics
