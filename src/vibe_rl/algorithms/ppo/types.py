"""PPO-specific state containers."""

from __future__ import annotations

from typing import NamedTuple

import chex

from vibe_rl.types import OptState, Params


class PPOState(NamedTuple):
    """PPO agent state.

    For the separate-network variant, ``params`` is an
    ``ActorCriticParams`` NamedTuple holding actor and critic models.
    For the shared-backbone variant, ``params`` is a single
    ``ActorCriticShared`` Equinox model.

    Fields:
        params: Network parameters (actor+critic).
        opt_state: Optax optimizer state.
        step: Scalar training step counter.
        rng: PRNG key.
    """

    params: Params
    opt_state: OptState
    step: chex.Array
    rng: chex.PRNGKey


class ActorCriticParams(NamedTuple):
    """Container for separate actor and critic parameters."""

    actor: Params
    critic: Params
