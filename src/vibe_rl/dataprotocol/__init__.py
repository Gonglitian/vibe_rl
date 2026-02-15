"""Data structures for JAX-based RL.

Core types:
    - Transition / PPOTransition: immutable NamedTuple experience containers
    - TrainState variants: immutable NamedTuple training state
    - ReplayBuffer: numpy-backed buffer with jax.Array sampling
"""

from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer
from vibe_rl.dataprotocol.train_state import (
    ActorCriticTrainState,
    DQNTrainState,
    SACTrainState,
    TrainState,
    create_dqn_train_state,
    create_train_state,
)
from vibe_rl.dataprotocol.transition import (
    Batch,
    PPOBatch,
    PPOTransition,
    Transition,
    make_dummy_transition,
)

__all__ = [
    # Transitions
    "Transition",
    "PPOTransition",
    "Batch",
    "PPOBatch",
    "make_dummy_transition",
    # Train states
    "TrainState",
    "DQNTrainState",
    "ActorCriticTrainState",
    "SACTrainState",
    "create_train_state",
    "create_dqn_train_state",
    # Buffers
    "ReplayBuffer",
]
