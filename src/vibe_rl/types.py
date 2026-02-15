"""Core type definitions for vibe_rl.

All state containers are NamedTuples for zero-overhead JAX pytree compatibility.
Type aliases follow the Stoix convention for clarity in function signatures.
"""

from __future__ import annotations

from typing import Any, NamedTuple, TypeAlias

import chex
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Scalar / array type aliases
# ---------------------------------------------------------------------------
Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Reward: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
LogProb: TypeAlias = chex.Array

# Generic pytree aliases
Params: TypeAlias = Any  # network parameter pytree
OptState: TypeAlias = Any  # optax optimizer state pytree


# ---------------------------------------------------------------------------
# Transition containers (immutable NamedTuples, auto-registered as pytrees)
# ---------------------------------------------------------------------------
class Transition(NamedTuple):
    """A single (s, a, r, s', done) experience tuple.

    All fields are JAX arrays. For batched transitions the leading
    dimension is the batch size.
    """

    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    done: chex.Array


class OnPolicyTransition(NamedTuple):
    """Extended transition for on-policy algorithms (PPO, etc.).

    Includes log_prob and value needed for advantage estimation.
    """

    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    log_prob: LogProb
    value: Value


# ---------------------------------------------------------------------------
# Agent state containers
# ---------------------------------------------------------------------------
class AgentState(NamedTuple):
    """Minimal agent state shared across all algorithms.

    Algorithm-specific states extend this by composing their own
    NamedTuples (e.g., DQNState includes target_params and buffer_state).

    Fields:
        params: Network parameter pytree (Equinox model or nested dict).
        opt_state: Optax optimizer state.
        step: Scalar training step counter.
        rng: PRNG key for stochastic operations.
    """

    params: Params
    opt_state: OptState
    step: chex.Array
    rng: chex.PRNGKey

    @staticmethod
    def initial(params: Params, opt_state: OptState, rng: chex.PRNGKey) -> AgentState:
        """Create an AgentState at step 0."""
        return AgentState(
            params=params,
            opt_state=opt_state,
            step=jnp.zeros((), dtype=jnp.int32),
            rng=rng,
        )


class Metrics(NamedTuple):
    """Training metrics returned from an update step.

    Algorithms can return additional metrics by defining their own
    NamedTuple that includes these base fields.
    """

    loss: chex.Array
