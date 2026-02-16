"""JAX-native core types used across the framework."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    """A single or batched (s, a, r, s', done) experience tuple.

    All fields are JAX arrays, making this a valid pytree that works
    with ``jax.vmap``, ``jax.lax.scan``, and ``jax.tree`` utilities.
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array


class Metrics(NamedTuple):
    """Training metrics returned by an agent's update step."""

    loss: jax.Array
    # Algorithms may extend this via subclassing.


class AgentState(NamedTuple):
    """Minimal mutable state carried by any agent.

    Algorithm-specific states (e.g. ``DQNState``) extend this with
    extra fields while remaining valid pytrees.
    """

    params: Any  # Equinox module or param dict
    opt_state: Any  # optax optimizer state
    rng: jax.Array  # PRNG key
    step: jax.Array  # scalar int32

    @staticmethod
    def initial(
        params: Any,
        opt_state: Any,
        rng: jax.Array,
    ) -> AgentState:
        return AgentState(
            params=params,
            opt_state=opt_state,
            rng=rng,
            step=jnp.array(0, dtype=jnp.int32),
        )
