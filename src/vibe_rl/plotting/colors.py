"""DeepMind-style color palettes for RL reward curve plots.

Default palette is adapted from DeepMind's publications. Users can
supply a custom list of hex strings via :func:`set_palette`.
"""

from __future__ import annotations

# DeepMind-style palette — 10 distinct, colorblind-friendly colors.
DEEPMIND: list[str] = [
    "#0077BB",  # blue
    "#EE7733",  # orange
    "#009988",  # teal
    "#CC3311",  # red
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
    "#AA3377",  # purple
    "#DDCC77",  # sand
    "#44BB99",  # mint
]

# Module-level default — callers can swap via ``set_palette()``.
_current: list[str] = list(DEEPMIND)


def get_colors() -> list[str]:
    """Return the current color palette (list of hex strings)."""
    return list(_current)


def set_palette(colors: list[str]) -> None:
    """Override the default color palette.

    Parameters
    ----------
    colors:
        List of hex color strings (e.g. ``["#FF0000", "#00FF00"]``).
    """
    global _current  # noqa: PLW0603
    _current = list(colors)


def reset_palette() -> None:
    """Restore the default DeepMind palette."""
    global _current  # noqa: PLW0603
    _current = list(DEEPMIND)


def color_for(index: int) -> str:
    """Return a color by index, cycling if necessary."""
    return _current[index % len(_current)]
