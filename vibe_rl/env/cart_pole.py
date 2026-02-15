from __future__ import annotations

import math

import numpy as np

from vibe_rl.env.base import BaseEnv
from vibe_rl.env.spaces import Box, Discrete


class CartPoleEnv(BaseEnv):
    """
    Classic CartPole balancing environment (pure numpy, no gym dependency).
    Physics constants match the original Barto, Sutton, Anderson (1983) formulation.
    """

    def __init__(self, max_steps: int = 500) -> None:
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.theta_threshold = 12 * 2 * math.pi / 360  # ~0.2094 rad
        self.x_threshold = 2.4
        self.max_steps = max_steps

        high = np.array(
            [self.x_threshold * 2, np.finfo(np.float32).max,
             self.theta_threshold * 2, np.finfo(np.float32).max],
            dtype=np.float32,
        )
        self.observation_space = Box(low=-high, high=high)
        self.action_space = Discrete(2)
        self.state: np.ndarray | None = None
        self._step_count = 0

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self._step_count = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.state is not None, "Call reset() before step()"
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sin_th) / self.total_mass
        theta_acc = (self.gravity * sin_th - cos_th * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos_th**2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_th / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._step_count += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )
        truncated = self._step_count >= self.max_steps
        reward = 1.0

        return self.state.copy(), reward, terminated, truncated, {}
