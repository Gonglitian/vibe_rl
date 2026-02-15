"""Transition containers for RL experience data.

All containers are NamedTuples — immutable, zero-overhead PyTrees that
compose naturally with jax.jit, jax.vmap, and jax.lax.scan.

A single Transition holds scalar/1-D fields.  A *batched* Transition
(fields with a leading batch dim) serves as the Batch type — no
separate class needed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class Transition(NamedTuple):
    """A single (s, a, r, s', done) experience tuple.

    All fields are jax Arrays. For batched usage, each field carries a
    leading batch dimension — the type is the same, just higher-rank.

    Fields:
        obs:      Observation.          scalar: (*obs_shape,)  batched: (B, *obs_shape)
        action:   Action taken.         scalar: ()  or (A,)    batched: (B,) or (B, A)
        reward:   Scalar reward.        scalar: ()             batched: (B,)
        next_obs: Next observation.     scalar: (*obs_shape,)  batched: (B, *obs_shape)
        done:     Episode termination.  scalar: ()             batched: (B,)
    """

    obs: Array
    action: Array
    reward: Array
    next_obs: Array
    done: Array


class PPOTransition(NamedTuple):
    """Extended transition for on-policy algorithms (PPO).

    Includes value estimates and action log-probabilities needed for
    GAE computation and the PPO surrogate objective.
    """

    obs: Array
    action: Array
    reward: Array
    next_obs: Array
    done: Array
    log_prob: Array
    value: Array


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------
# A "Batch" is simply a Transition (or PPOTransition) whose fields
# have a leading batch dimension.  Using the same type avoids an
# unnecessary conversion step and keeps the code jit/vmap-friendly.
Batch = Transition
PPOBatch = PPOTransition


def make_dummy_transition(obs_shape: tuple[int, ...]) -> Transition:
    """Create a zero-filled Transition (useful as a pytree template)."""
    return Transition(
        obs=jnp.zeros(obs_shape),
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros(()),
        next_obs=jnp.zeros(obs_shape),
        done=jnp.zeros((), dtype=jnp.bool_),
    )
