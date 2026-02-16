"""Core reward-curve plotting for RL experiments.

Reads JSONL metrics produced by :class:`~vibe_rl.metrics.MetricsLogger`,
applies smoothing and multi-seed aggregation, and renders publication-
quality reward curves using matplotlib.

Usage::

    from vibe_rl.plotting import plot_reward_curve
    fig = plot_reward_curve(
        ["runs/ppo_seed0", "runs/ppo_seed1"],
        y_key="episode_return",
        smooth=10,
    )
    fig.savefig("reward.png")
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vibe_rl.plotting.colors import color_for
from vibe_rl.plotting.config import PlotConfig

# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------


def smooth_window(y: NDArray[np.floating], radius: int) -> NDArray[np.floating]:
    """Symmetric moving-average smoothing (SpinningUp-style).

    Each output sample ``y'[i]`` is the mean of
    ``y[max(0, i-radius) : i+radius+1]``.
    """
    if radius <= 0:
        return y.copy()
    kernel = np.ones(2 * radius + 1, dtype=np.float64)
    # Pad edges to avoid boundary artifacts.
    padded = np.pad(y.astype(np.float64), radius, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")[radius:-radius] / np.convolve(
        np.ones_like(padded), kernel, mode="same"
    )[radius:-radius]
    return smoothed


def smooth_ema(y: NDArray[np.floating], span: int) -> NDArray[np.floating]:
    """Exponential moving-average smoothing (rl-plotter style).

    Uses ``alpha = 2 / (span + 1)`` which matches pandas EWM semantics.
    """
    if span <= 1:
        return y.copy()
    alpha = 2.0 / (span + 1)
    out = np.empty_like(y, dtype=np.float64)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def _smooth(y: NDArray[np.floating], config: PlotConfig) -> NDArray[np.floating]:
    if config.smooth_mode == "ema":
        return smooth_ema(y, config.smooth_radius)
    return smooth_window(y, config.smooth_radius)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL metrics file."""
    records: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return records
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _find_metrics_file(run_dir: str | Path) -> Path:
    """Locate the metrics JSONL inside a run directory.

    Tries ``logs/metrics.jsonl`` first, then falls back to the path
    itself (in case the caller passed the file directly).
    """
    p = Path(run_dir)
    candidate = p / "logs" / "metrics.jsonl"
    if candidate.exists():
        return candidate
    if p.is_file():
        return p
    return candidate  # will produce empty data


def _extract_series(
    records: list[dict[str, Any]], x_key: str, y_key: str
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Pull (x, y) arrays from records that contain both keys."""
    xs, ys = [], []
    for r in records:
        if x_key in r and y_key in r:
            xs.append(float(r[x_key]))
            ys.append(float(r[y_key]))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _detect_algo_name(run_dir: str | Path) -> str:
    """Attempt to read algorithm name from config.json, else use dir name."""
    p = Path(run_dir)
    config_path = p / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            for key in ("algorithm", "algo", "algo_name"):
                if key in cfg:
                    return str(cfg[key])
        except (json.JSONDecodeError, KeyError):
            pass
    # Fallback: use the directory name (strip timestamp suffix).
    name = p.name
    parts = name.rsplit("_", 2)
    if len(parts) >= 3:
        # e.g. "ppo_cartpole_20260215_143022" → "ppo_cartpole"
        try:
            int(parts[-1])
            int(parts[-2])
            return "_".join(parts[:-2])
        except ValueError:
            pass
    return name


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------


def _interpolate_to_common_x(
    all_x: list[NDArray[np.floating]],
    all_y: list[NDArray[np.floating]],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Interpolate multiple (x, y) series onto a shared x-grid.

    Returns ``(common_x, y_matrix)`` where ``y_matrix`` has shape
    ``(n_seeds, n_points)``.
    """
    x_min = max(xs[0] for xs in all_x)
    x_max = min(xs[-1] for xs in all_x)
    n_points = max(len(xs) for xs in all_x)
    common_x = np.linspace(x_min, x_max, n_points)

    y_matrix = np.empty((len(all_x), n_points), dtype=np.float64)
    for i, (xs, ys) in enumerate(zip(all_x, all_y, strict=True)):
        y_matrix[i] = np.interp(common_x, xs, ys)
    return common_x, y_matrix


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------


def plot_reward_curve(
    run_dirs: str | Path | Sequence[str | Path],
    *,
    x_key: str = "step",
    y_key: str = "episode_return",
    smooth: int | None = None,
    style: str | None = None,
    group_by: str = "auto",
    config: PlotConfig | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Generate an RL-style reward curve from JSONL metrics files.

    Parameters
    ----------
    run_dirs:
        One or more run directories (or direct JSONL file paths).
        Multiple directories with the same algorithm name are treated
        as different seeds and aggregated.
    x_key:
        JSON key for the x-axis (default ``"step"``).
    y_key:
        JSON key for the y-axis (default ``"episode_return"``).
    smooth:
        Override ``PlotConfig.smooth_radius``.
    style:
        Override ``PlotConfig.style``.
    group_by:
        ``"auto"`` groups by detected algorithm name. ``"none"``
        treats every run as its own line.
    config:
        Full plot configuration. Other keyword overrides are applied
        on top.
    save_path:
        If given, save the figure to this path. Format is inferred
        from the extension (or ``config.save_format``).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install 'vibe-rl[plotting]'"
        ) from None

    cfg = config or PlotConfig()
    if smooth is not None:
        cfg = dataclasses.replace(cfg, smooth_radius=smooth)
    if style is not None:
        cfg = dataclasses.replace(cfg, style=style)

    # Normalize to list
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    run_dirs = [Path(d) for d in run_dirs]

    # Group runs by algorithm name
    groups: dict[str, list[Path]] = {}
    for rd in run_dirs:
        label = rd.name if group_by == "none" else _detect_algo_name(rd)
        groups.setdefault(label, []).append(rd)

    with plt.style.context(cfg.style):
        fig, ax = plt.subplots(figsize=cfg.figsize)

        for color_idx, (label, paths) in enumerate(sorted(groups.items())):
            color = color_for(color_idx)

            # Load and smooth each seed
            all_x: list[NDArray[np.floating]] = []
            all_y: list[NDArray[np.floating]] = []
            for p in paths:
                metrics_file = _find_metrics_file(p)
                records = _load_jsonl(metrics_file)
                x, y = _extract_series(records, x_key, y_key)
                if len(x) == 0:
                    continue
                y = _smooth(y, cfg)
                all_x.append(x)
                all_y.append(y)

            if not all_x:
                continue

            if len(all_x) == 1:
                # Single seed — just plot the line.
                ax.plot(all_x[0], all_y[0], color=color, label=label, linewidth=1.5)
            else:
                # Multi-seed — aggregate with shading.
                common_x, y_matrix = _interpolate_to_common_x(all_x, all_y)
                mean = np.mean(y_matrix, axis=0)
                ax.plot(common_x, mean, color=color, label=label, linewidth=1.5)

                if cfg.shaded == "std":
                    std = np.std(y_matrix, axis=0)
                    ax.fill_between(
                        common_x, mean - std, mean + std, color=color, alpha=0.2
                    )
                elif cfg.shaded == "stderr":
                    stderr = np.std(y_matrix, axis=0) / np.sqrt(len(all_x))
                    ax.fill_between(
                        common_x, mean - stderr, mean + stderr, color=color, alpha=0.2
                    )

        ax.set_xlabel(cfg.xlabel)
        ax.set_ylabel(cfg.ylabel)
        if cfg.title:
            ax.set_title(cfg.title)
        ax.legend()

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(str(save_path), dpi=cfg.dpi)

    return fig
