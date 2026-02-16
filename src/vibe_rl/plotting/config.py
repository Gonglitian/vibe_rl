"""Plot configuration dataclass."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PlotConfig:
    """Immutable configuration for reward curve plots.

    Parameters
    ----------
    smooth_radius:
        Half-window size for window averaging, or span for EMA.
    smooth_mode:
        ``"window"`` for symmetric moving-average, ``"ema"`` for
        exponential moving-average.
    shaded:
        ``"std"`` for mean +/- 1 standard deviation, ``"stderr"`` for
        mean +/- standard error, ``"none"`` for no shading.
    figsize:
        Matplotlib figure size ``(width, height)`` in inches.
    dpi:
        Resolution for raster outputs (PNG).
    style:
        Matplotlib style name applied via ``plt.style.context()``.
    save_format:
        Default file extension when saving (``"png"``, ``"pdf"``,
        ``"svg"``).
    title:
        Optional figure title.
    xlabel:
        X-axis label.
    ylabel:
        Y-axis label.
    """

    smooth_radius: int = 10
    smooth_mode: str = "window"
    shaded: str = "std"
    figsize: tuple[float, float] = (8, 6)
    dpi: int = 150
    style: str = "seaborn-v0_8-darkgrid"
    save_format: str = "png"
    title: str | None = None
    xlabel: str = "Step"
    ylabel: str = "Episode Return"
