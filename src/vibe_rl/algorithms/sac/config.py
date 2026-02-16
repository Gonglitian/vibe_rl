"""SAC hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SACConfig:
    """All SAC hyperparameters in one place.

    Frozen dataclass — safe to pass into jitted functions as a static
    argument (via ``functools.partial`` or ``jax.jit(..., static_argnums=...)``).
    """

    # Network
    hidden_sizes: tuple[int, ...] = (256, 256)

    # Optimization
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256
    max_grad_norm: float = 10.0

    # Target network (Polyak averaging)
    tau: float = 0.005

    # Entropy
    init_alpha: float = 1.0
    autotune_alpha: bool = True
    target_entropy_scale: float = 1.0  # target_entropy = -scale * action_dim

    # Action bounds (used for tanh squashing rescale)
    action_low: float = -1.0
    action_high: float = 1.0

    # Log-std clamps for the actor
    log_std_min: float = -20.0
    log_std_max: float = 2.0
