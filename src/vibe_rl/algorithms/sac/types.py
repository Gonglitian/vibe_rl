"""SAC-specific state container."""

from __future__ import annotations

from typing import NamedTuple

import chex

from vibe_rl.types import OptState, Params


class SACState(NamedTuple):
    """SAC agent state.

    Fields:
        actor_params: Gaussian actor network (Equinox model).
        critic_params: Twin Q-network (Equinox model).
        target_critic_params: Target twin Q-network (for stable TD targets).
        actor_opt_state: Optax optimizer state for the actor.
        critic_opt_state: Optax optimizer state for the critic.
        log_alpha: Log temperature parameter (scalar array).
        alpha_opt_state: Optax optimizer state for alpha.
        step: Scalar training step counter.
        rng: PRNG key.
    """

    actor_params: Params
    critic_params: Params
    target_critic_params: Params
    actor_opt_state: OptState
    critic_opt_state: OptState
    log_alpha: chex.Array
    alpha_opt_state: OptState
    step: chex.Array
    rng: chex.PRNGKey
