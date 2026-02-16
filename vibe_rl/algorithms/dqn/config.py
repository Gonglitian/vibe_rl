from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DQNConfig:
    """All DQN hyperparameters in one place."""

    # Network
    hidden_sizes: tuple[int, ...] = (128, 128)

    # Optimization
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64

    # Replay buffer
    buffer_capacity: int = 100_000
    min_buffer_size: int = 1_000

    # Target network
    target_update_freq: int = 1_000

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50_000
