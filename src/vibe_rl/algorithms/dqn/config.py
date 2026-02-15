"""DQN hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DQNConfig:
    """All DQN hyperparameters in one place.

    Frozen dataclass — safe to pass into jitted functions as a static
    argument (via ``functools.partial`` or ``jax.jit(..., static_argnums=...)``).
    """

    # Network
    hidden_sizes: tuple[int, ...] = (128, 128)

    # Optimization
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    max_grad_norm: float = 10.0

    # Target network
    target_update_freq: int = 1_000

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50_000
