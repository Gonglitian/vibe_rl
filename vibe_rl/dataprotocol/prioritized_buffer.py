from __future__ import annotations

import numpy as np

from vibe_rl.dataprotocol.batch import Batch
from vibe_rl.dataprotocol.transition import Transition


class SumTree:
    """Binary tree where parent = sum of children. Enables O(log n) proportional sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def update(self, data_idx: int, priority: float) -> None:
        tree_idx = data_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get(self, value: float) -> int:
        """Retrieve the leaf data index whose cumulative sum covers `value`."""
        idx = 0
        while idx < self.capacity - 1:  # while not a leaf
            left = 2 * idx + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - (self.capacity - 1)

    @property
    def total(self) -> float:
        return float(self.tree[0])


class PrioritizedReplayBuffer:
    """
    Replay buffer with proportional prioritization (Schaul et al., 2015).

    New transitions are inserted with max priority so they are sampled at least once.
    Importance-sampling weights correct for the non-uniform sampling bias.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self._frame = 0

        self._tree = SumTree(capacity)
        self._states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=np.bool_)
        self._max_priority = 1.0
        self._size = 0
        self._ptr = 0

    def push(self, transition: Transition) -> None:
        idx = self._ptr
        self._states[idx] = transition.state
        self._actions[idx] = transition.action
        self._rewards[idx] = transition.reward
        self._next_states[idx] = transition.next_state
        self._terminated[idx] = transition.terminated
        self._tree.update(idx, self._max_priority**self.alpha)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        self._frame += 1
        beta = min(
            1.0, self.beta_start + self._frame * (1.0 - self.beta_start) / self.beta_frames
        )

        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        segment = self._tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            idx = self._tree.get(value)
            indices[i] = idx
            priorities[i] = self._tree.tree[idx + self._tree.capacity - 1]

        probs = priorities / self._tree.total
        weights = (self._size * probs) ** (-beta)
        weights /= weights.max()

        return Batch(
            states=self._states[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_states=self._next_states[indices],
            terminated=self._terminated[indices],
            indices=indices,
            weights=weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, p in zip(indices, priorities):
            self._tree.update(int(idx), float(p))
            self._max_priority = max(self._max_priority, float(p))

    def __len__(self) -> int:
        return self._size
