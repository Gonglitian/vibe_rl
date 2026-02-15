"""DQN-specific state container."""

from __future__ import annotations

from typing import NamedTuple

import chex

from vibe_rl.types import OptState, Params


class DQNState(NamedTuple):
    """DQN agent state.

    Extends the base AgentState concept with a target network.
    All fields are JAX pytree-compatible.

    Fields:
        params: Online Q-network parameters (Equinox model).
        target_params: Target Q-network parameters (for stable TD targets).
        opt_state: Optax optimizer state.
        step: Scalar training step counter.
        rng: PRNG key.
    """

    params: Params
    target_params: Params
    opt_state: OptState
    step: chex.Array
    rng: chex.PRNGKey
