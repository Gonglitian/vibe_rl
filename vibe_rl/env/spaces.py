from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Space(ABC):
    """Abstract base for all spaces."""

    @abstractmethod
    def sample(self) -> Any:
        """Draw a uniform random sample from this space."""
        ...

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Return True if x is a valid member of this space."""
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        ...


class Discrete(Space):
    """A space of integers {0, 1, ..., n-1}."""

    def __init__(self, n: int) -> None:
        assert n > 0
        self.n = n

    def sample(self) -> int:
        return int(np.random.randint(self.n))

    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n

    @property
    def shape(self) -> tuple[int, ...]:
        return ()


class Box(Space):
    """A bounded n-dimensional continuous space."""

    def __init__(
        self,
        low: float | np.ndarray,
        high: float | np.ndarray,
        shape: tuple[int, ...] | None = None,
        dtype: type = np.float32,
    ) -> None:
        if shape is not None:
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
        else:
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
        self._shape = self.low.shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        x = np.asarray(x)
        return bool(
            x.shape == self._shape and np.all(x >= self.low) and np.all(x <= self.high)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
