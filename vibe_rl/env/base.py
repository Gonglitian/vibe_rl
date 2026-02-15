from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from vibe_rl.env.spaces import Space


class BaseEnv(ABC):
    """
    Abstract base class for all environments.

    Lifecycle:
        obs = env.reset()
        while not done:
            obs, reward, terminated, truncated, info = env.step(action)
        env.close()
    """

    observation_space: Space
    action_space: Space

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one timestep.

        Returns:
            observation, reward, terminated, truncated, info
        """
        ...

    def close(self) -> None:
        """Optional cleanup."""
        pass

    def render(self) -> str | None:
        """Optional text rendering for debugging."""
        return None
