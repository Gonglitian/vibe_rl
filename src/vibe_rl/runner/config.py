"""Runner configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunnerConfig:
    """Shared hyperparameters for training runners.

    Controls the outer training loop, evaluation schedule, and logging.
    Algorithm-specific settings live in the algorithm's own config
    (e.g., ``PPOConfig``, ``DQNConfig``).
    """

    # Training budget
    total_timesteps: int = 100_000

    # Evaluation
    eval_every: int = 5_000
    eval_episodes: int = 10

    # Logging
    log_interval: int = 1_000

    # Off-policy specific
    buffer_size: int = 100_000
    warmup_steps: int = 1_000

    # Seeding
    seed: int = 0
