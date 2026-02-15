from __future__ import annotations

import numpy as np

from vibe_rl.dataprotocol.batch import Batch
from vibe_rl.dataprotocol.transition import Transition


class ReplayBuffer:
    """
    Fixed-size circular buffer with uniform random sampling.

    Uses pre-allocated numpy arrays for memory efficiency.
    A single numpy index operation replaces Python-level loops for sampling.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self._size = 0
        self._ptr = 0

        self._states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=np.bool_)

    def push(self, transition: Transition) -> None:
        idx = self._ptr
        self._states[idx] = transition.state
        self._actions[idx] = transition.action
        self._rewards[idx] = transition.reward
        self._next_states[idx] = transition.next_state
        self._terminated[idx] = transition.terminated
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        indices = np.random.randint(0, self._size, size=batch_size)
        return Batch(
            states=self._states[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_states=self._next_states[indices],
            terminated=self._terminated[indices],
            indices=indices,
        )

    def __len__(self) -> int:
        return self._size
