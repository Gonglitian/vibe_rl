"""Immutable training-state containers (JAX NamedTuples)."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


class TrainState(NamedTuple):
    """Generic training state: params + optimizer state + step counter."""

    params: Any
    opt_state: Any
    step: jax.Array


class DQNTrainState(NamedTuple):
    """DQN-specific training state with target network and epsilon."""

    params: Any
    target_params: Any
    opt_state: Any
    step: jax.Array
    epsilon: jax.Array


def create_train_state(
    params: Any,
    tx: optax.GradientTransformation,
) -> TrainState:
    """Create a fresh ``TrainState`` at step 0."""
    return TrainState(
        params=params,
        opt_state=tx.init(params),
        step=jnp.array(0, dtype=jnp.int32),
    )


def create_dqn_train_state(
    params: Any,
    tx: optax.GradientTransformation,
    epsilon_start: float = 1.0,
) -> DQNTrainState:
    """Create a fresh ``DQNTrainState`` with target params copied from online."""
    target_params = jax.tree.map(lambda p: p.copy(), params)
    return DQNTrainState(
        params=params,
        target_params=target_params,
        opt_state=tx.init(params),
        step=jnp.array(0, dtype=jnp.int32),
        epsilon=jnp.array(epsilon_start, dtype=jnp.float32),
    )
