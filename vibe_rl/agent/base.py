from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAgent(ABC):
    """
    Abstract agent interface.

    An agent encapsulates:
      - A policy: state -> action   (the `act` method)
      - A learning rule              (the `learn` method)
      - Serialization                (save / load)
    """

    @abstractmethod
    def act(self, state: np.ndarray, *, explore: bool = True) -> int | np.ndarray:
        """
        Select an action given the current state.

        Args:
            state: Current observation.
            explore: If True, use exploration strategy (e.g., epsilon-greedy).
                     If False, use greedy policy (evaluation mode).
        """
        ...

    @abstractmethod
    def learn(self) -> dict[str, float]:
        """
        Perform one learning update.

        Returns:
            A dict of metrics (e.g., {"loss": 0.023, "q_mean": 1.45}).
        """
        ...

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save agent parameters to disk."""
        ...

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Load agent parameters from disk."""
        ...
