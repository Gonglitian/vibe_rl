"""SAC hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass

import optax


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

    def make_actor_optimizer(self) -> optax.GradientTransformation:
        """Build the actor optimizer chain."""
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.actor_lr),
        )

    def make_critic_optimizer(self) -> optax.GradientTransformation:
        """Build the critic optimizer chain."""
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.critic_lr),
        )

    def make_alpha_optimizer(self) -> optax.GradientTransformation:
        """Build the alpha (temperature) optimizer chain."""
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.alpha_lr),
        )
