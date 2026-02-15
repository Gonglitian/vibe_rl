"""Immutable training state containers.

All states are NamedTuples — immutable PyTrees compatible with
jax.jit, jax.lax.scan, and eqx.tree_serialise_leaves.

Use ``state._replace(step=state.step + 1)`` for functional updates.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import optax
from jax import Array


# ---------------------------------------------------------------------------
# Type aliases for clarity
# ---------------------------------------------------------------------------
Params = Any  # Equinox model (itself a PyTree)
OptState = Any  # optax optimizer state


class TrainState(NamedTuple):
    """Core training state shared by all algorithms.

    Fields:
        params:    Network parameters (Equinox model PyTree).
        opt_state: Optax optimizer state.
        step:      Scalar training step counter.
    """

    params: Params
    opt_state: OptState
    step: Array  # scalar int


class DQNTrainState(NamedTuple):
    """Training state for DQN-family algorithms.

    Extends TrainState with a target network and epsilon schedule.
    """

    params: Params
    target_params: Params
    opt_state: OptState
    step: Array
    epsilon: Array  # current epsilon for exploration


class ActorCriticTrainState(NamedTuple):
    """Training state for actor-critic algorithms (PPO, A2C).

    Single optimizer over joint (actor, critic) params.
    """

    params: Params  # typically an ActorCriticParams NamedTuple
    opt_state: OptState
    step: Array


class SACTrainState(NamedTuple):
    """Training state for SAC with separate actor/critic optimizers.

    Fields:
        actor_params / critic_params: Separate network PyTrees.
        target_critic_params: Polyak-averaged target.
        actor_opt_state / critic_opt_state: Per-network optimizer states.
        alpha_params / alpha_opt_state: Temperature (log_alpha) parameter.
        step: Scalar step counter.
    """

    actor_params: Params
    critic_params: Params
    target_critic_params: Params
    actor_opt_state: OptState
    critic_opt_state: OptState
    alpha_params: Params
    alpha_opt_state: OptState
    step: Array


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_train_state(
    params: Params,
    tx: optax.GradientTransformation,
) -> TrainState:
    """Initialize a basic TrainState from params and an optax optimizer."""
    return TrainState(
        params=params,
        opt_state=tx.init(params),
        step=jnp.zeros((), dtype=jnp.int32),
    )


def create_dqn_train_state(
    params: Params,
    tx: optax.GradientTransformation,
    *,
    epsilon_start: float = 1.0,
) -> DQNTrainState:
    """Initialize a DQN TrainState with a copy of params as target."""
    return DQNTrainState(
        params=params,
        target_params=params,  # same PyTree structure, shared initially
        opt_state=tx.init(params),
        step=jnp.zeros((), dtype=jnp.int32),
        epsilon=jnp.array(epsilon_start),
    )
