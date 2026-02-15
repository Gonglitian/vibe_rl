from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Transition:
    """A single (s, a, r, s', terminated) experience tuple."""

    state: np.ndarray
    action: int | np.ndarray
    reward: float
    next_state: np.ndarray
    terminated: bool
