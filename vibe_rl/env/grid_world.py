from __future__ import annotations

import numpy as np

from vibe_rl.env.base import BaseEnv
from vibe_rl.env.spaces import Box, Discrete


class GridWorldEnv(BaseEnv):
    """
    Simple NxN grid world.

    Actions: 0=up, 1=right, 2=down, 3=left
    Observation: flat array of length 2 representing (row, col), normalized to [0,1].
    Reward: -0.01 per step, +1.0 on reaching goal.
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def __init__(self, size: int = 5, max_steps: int = 100) -> None:
        self.size = size
        self.max_steps = max_steps
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,))
        self._agent_pos: tuple[int, int] = (0, 0)
        self._goal_pos: tuple[int, int] = (size - 1, size - 1)
        self._step_count = 0

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self._agent_pos = (0, 0)
        self._step_count = 0
        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        dr, dc = self.ACTIONS[action]
        r = max(0, min(self.size - 1, self._agent_pos[0] + dr))
        c = max(0, min(self.size - 1, self._agent_pos[1] + dc))
        self._agent_pos = (r, c)
        self._step_count += 1

        terminated = self._agent_pos == self._goal_pos
        truncated = self._step_count >= self.max_steps
        reward = 1.0 if terminated else -0.01
        info = {"steps": self._step_count}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self._agent_pos[0] / max(1, self.size - 1),
                self._agent_pos[1] / max(1, self.size - 1),
            ],
            dtype=np.float32,
        )

    def render(self) -> str:
        rows = []
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) == self._agent_pos:
                    row += "A "
                elif (r, c) == self._goal_pos:
                    row += "G "
                else:
                    row += ". "
            rows.append(row)
        return "\n".join(rows)
