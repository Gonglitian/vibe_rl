from __future__ import annotations

from typing import Any

import numpy as np

from vibe_rl.env.base import BaseEnv


class EnvWrapper(BaseEnv):
    """Base wrapper that delegates everything to the wrapped env."""

    def __init__(self, env: BaseEnv) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        return self.env.reset(seed=seed)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()

    def render(self) -> str | None:
        return self.env.render()


class RewardScaleWrapper(EnvWrapper):
    """Multiplies rewards by a constant factor."""

    def __init__(self, env: BaseEnv, scale: float) -> None:
        super().__init__(env)
        self.scale = scale

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward * self.scale, terminated, truncated, info
