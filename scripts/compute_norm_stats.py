#!/usr/bin/env python3
"""Compute and save normalization statistics for a LeRobot dataset.

Usage::

    python scripts/compute_norm_stats.py --repo-id lerobot/aloha_sim_insertion_human
    python scripts/compute_norm_stats.py --repo-id lerobot/aloha_sim_insertion_human \
        --keys obs action --output norm_stats.json
    python scripts/compute_norm_stats.py --repo-id lerobot/aloha_sim_insertion_human \
        --max-samples 1000
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tyro

from vibe_rl.data.lerobot_dataset import LeRobotDatasetAdapter, LeRobotDatasetConfig
from vibe_rl.data.normalize import compute_norm_stats, save_norm_stats


@dataclass(frozen=True)
class ComputeNormStatsArgs:
    """Configuration for computing normalization statistics."""

    # Dataset
    repo_id: str = tyro.MISSING
    """HuggingFace repo id for the LeRobot dataset."""

    root: str | None = None
    """Local cache directory (None = default HF cache)."""

    episodes: list[int] | None = None
    """Filter to specific episode indices."""

    # Statistics computation
    keys: list[str] = field(default_factory=lambda: ["obs", "action"])
    """Transition keys to compute statistics for."""

    max_samples: int | None = None
    """Maximum number of samples to use (None = all)."""

    # Output
    output: str = "norm_stats.json"
    """Path to save the computed statistics."""


def main(args: ComputeNormStatsArgs) -> None:
    print(f"Loading dataset: {args.repo_id}")
    config = LeRobotDatasetConfig(
        repo_id=args.repo_id,
        root=args.root,
        episodes=args.episodes,
    )
    dataset = LeRobotDatasetAdapter(config)
    print(f"Dataset loaded: {len(dataset)} samples")

    print(f"Computing normalization stats for keys: {args.keys}")
    stats = compute_norm_stats(
        dataset,
        keys=args.keys,
        max_samples=args.max_samples,
    )

    for key, s in stats.items():
        print(f"  {key}: mean_norm={float(s.mean.mean()):.4f}, "
              f"std_mean={float(s.std.mean()):.4f}, "
              f"q01_mean={float(s.q01.mean()):.4f}, "
              f"q99_mean={float(s.q99.mean()):.4f}")

    save_norm_stats(stats, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main(tyro.cli(ComputeNormStatsArgs))
