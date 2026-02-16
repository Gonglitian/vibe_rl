"""PPO hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass

import optax


@dataclass(frozen=True)
class PPOConfig:
    """All PPO hyperparameters in one place.

    Frozen dataclass — safe to pass into jitted functions as a static
    argument (via ``functools.partial`` or ``jax.jit(..., static_argnums=...)``).
    """

    # Network
    hidden_sizes: tuple[int, ...] = (64, 64)

    # Optimization
    lr: float = 2.5e-4
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO clipping
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01

    # Rollout
    n_steps: int = 128
    n_minibatches: int = 4
    n_epochs: int = 4

    # Vectorized environments
    num_envs: int = 1

    # Architecture
    shared_backbone: bool = False

    def make_optimizer(self) -> optax.GradientTransformation:
        """Build the optax optimizer chain for this config.

        Returns a single ``GradientTransformation`` that should be
        constructed **once** and reused for every ``update()`` call so
        that stateful transforms (e.g. learning-rate schedules) work
        correctly.
        """
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.lr, eps=1e-5),
        )
