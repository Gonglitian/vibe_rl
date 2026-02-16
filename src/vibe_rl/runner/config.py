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

    # Multi-device (pmap)
    num_devices: int | None = None  # auto-detect if None
    num_envs: int = 1  # parallel envs per device

    # Seeding
    seed: int = 0

    # Checkpointing
    checkpoint_dir: str | None = None  # None = no checkpointing
    checkpoint_interval: int = 5_000  # save every N steps
    max_checkpoints: int = 5  # recent checkpoints to retain
    keep_period: int | None = None  # permanently keep every N steps
    resume: bool = False  # resume from existing checkpoint
    overwrite: bool = False  # wipe existing checkpoints
