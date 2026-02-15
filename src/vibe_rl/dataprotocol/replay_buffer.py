"""Replay buffers for off-policy RL algorithms.

Design choice (方案B): numpy arrays for storage + mutation, jax.Array
output on sample().  This avoids the complexity of fully-jittable
buffers while keeping the hot path (gradient computation on sampled
batches) in JAX.

The buffer itself is NOT jit-compatible — it lives outside the
compiled training step.  Typical usage::

    # Python loop (not jitted)
    for step in range(total_steps):
        action = jit_select_action(state, obs, rng)
        next_obs, reward, done = env.step(action)
        buffer.push(obs, action, reward, next_obs, done)
        if len(buffer) >= min_buffer:
            batch = buffer.sample(batch_size)  # returns Transition of jax arrays
            state = jit_update_step(state, batch)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from vibe_rl.dataprotocol.transition import Transition


class ReplayBuffer:
    """Fixed-size circular buffer with uniform random sampling.

    Storage uses pre-allocated numpy arrays for O(1) insertion.
    Sampling returns a ``Transition`` of jax arrays, ready for jit.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self._size = 0
        self._ptr = 0

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)

    def push(
        self,
        obs: np.ndarray,
        action: int | np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        idx = self._ptr
        self._obs[idx] = obs
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_obs[idx] = next_obs
        self._dones[idx] = done
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_transition(self, t: Transition) -> None:
        """Store a Transition (convenience wrapper that accepts jax arrays)."""
        self.push(
            obs=np.asarray(t.obs),
            action=np.asarray(t.action),
            reward=np.asarray(t.reward),
            next_obs=np.asarray(t.next_obs),
            done=np.asarray(t.done),
        )

    def sample(self, batch_size: int) -> Transition:
        """Uniformly sample a batch and return as jax arrays."""
        indices = np.random.randint(0, self._size, size=batch_size)
        return Transition(
            obs=jnp.asarray(self._obs[indices]),
            action=jnp.asarray(self._actions[indices]),
            reward=jnp.asarray(self._rewards[indices]),
            next_obs=jnp.asarray(self._next_obs[indices]),
            done=jnp.asarray(self._dones[indices]),
        )

    def __len__(self) -> int:
        return self._size
