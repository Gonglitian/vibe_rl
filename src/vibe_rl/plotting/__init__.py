"""Plotting utilities for RL reward curves.

Requires the ``[plotting]`` extra::

    pip install 'vibe-rl[plotting]'
"""

from vibe_rl.plotting.colors import (
    DEEPMIND,
    color_for,
    get_colors,
    reset_palette,
    set_palette,
)
from vibe_rl.plotting.config import PlotConfig
from vibe_rl.plotting.plot import (
    plot_reward_curve,
    smooth_ema,
    smooth_window,
)

__all__ = [
    "DEEPMIND",
    "PlotConfig",
    "color_for",
    "get_colors",
    "plot_reward_curve",
    "reset_palette",
    "set_palette",
    "smooth_ema",
    "smooth_window",
]
