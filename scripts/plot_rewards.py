#!/usr/bin/env python3
"""Plot reward curves from training run directories.

Usage::

    # Plot by experiment name (auto-discovers all matching runs)
    python scripts/plot_rewards.py --experiment-name cartpole_ppo

    # Plot specific run directories
    python scripts/plot_rewards.py --run-dirs runs/ppo_seed0 runs/ppo_seed1

    # Compare multiple algorithms
    python scripts/plot_rewards.py --compare \\
        --run-dirs runs/ppo_run runs/dqn_run runs/sac_run

    # Customise output
    python scripts/plot_rewards.py --experiment-name cartpole_ppo \\
        --y-key total_loss --smooth-radius 20 --output loss_curve.png
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import tyro

from vibe_rl.plotting.config import PlotConfig
from vibe_rl.run_dir import find_runs


@dataclasses.dataclass(frozen=True)
class PlotArgs:
    """CLI arguments for reward curve plotting."""

    # --- Run selection (at least one required) ---
    run_dirs: list[str] | None = None
    """Explicit run directories to plot."""

    experiment_name: str | None = None
    """Auto-discover runs matching this experiment name."""

    base_dir: str = "runs"
    """Base directory for run discovery (used with --experiment-name)."""

    # --- Data ---
    x_key: str = "step"
    """JSONL key for the x-axis."""

    y_key: str = "episode_return"
    """JSONL key for the y-axis."""

    # --- Output ---
    output: str | None = None
    """Output file path. Defaults to <first_run>/artifacts/reward_curve.png."""

    compare: bool = False
    """Multi-algorithm comparison mode (group_by='auto' vs 'none')."""

    # --- PlotConfig fields ---
    smooth_radius: int = 10
    """Half-window size for smoothing."""

    smooth_mode: str = "window"
    """Smoothing mode: 'window' or 'ema'."""

    shaded: str = "std"
    """Shading: 'std', 'stderr', or 'none'."""

    figsize: tuple[float, float] = (8, 6)
    """Figure size (width, height) in inches."""

    dpi: int = 150
    """DPI for raster output."""

    style: str = "seaborn-v0_8-darkgrid"
    """Matplotlib style."""

    title: str | None = None
    """Figure title."""


def main(args: PlotArgs | None = None) -> None:
    if args is None:
        args = tyro.cli(PlotArgs)

    # Collect run directories
    dirs: list[Path] = []

    if args.run_dirs:
        dirs.extend(Path(d) for d in args.run_dirs)

    if args.experiment_name:
        found = find_runs(args.base_dir, args.experiment_name)
        dirs.extend(rd.root for rd in found)

    if not dirs:
        print(
            "Error: no runs found. Provide --run-dirs or --experiment-name.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build PlotConfig from CLI fields
    plot_config = PlotConfig(
        smooth_radius=args.smooth_radius,
        smooth_mode=args.smooth_mode,
        shaded=args.shaded,
        figsize=args.figsize,
        dpi=args.dpi,
        style=args.style,
        title=args.title,
        xlabel=args.x_key.replace("_", " ").title(),
        ylabel=args.y_key.replace("_", " ").title(),
    )

    # Determine output path
    output = args.output
    if output is None:
        artifacts = dirs[0] / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        output = str(artifacts / "reward_curve.png")

    group_by = "auto" if args.compare else "none"

    # Import here so matplotlib is only loaded when plotting
    from vibe_rl.plotting import plot_reward_curve

    fig = plot_reward_curve(
        dirs,
        x_key=args.x_key,
        y_key=args.y_key,
        group_by=group_by,
        config=plot_config,
        save_path=output,
    )

    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"Saved reward curve to: {output}")


if __name__ == "__main__":
    main()
