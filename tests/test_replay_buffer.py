"""Tests for ReplayBuffer."""

import jax
import jax.numpy as jnp
import numpy as np

from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer
from vibe_rl.dataprotocol.transition import Transition


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(4,))
        assert len(buf) == 0
        buf.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
        assert len(buf) == 1

    def test_circular_overwrite(self):
        buf = ReplayBuffer(capacity=3, obs_shape=(2,))
        for i in range(5):
            buf.push(np.full(2, float(i)), 0, 0.0, np.zeros(2), False)
        assert len(buf) == 3
        # Buffer should contain the last 3 entries (indices 2, 3, 4)
        # After 5 pushes into capacity=3: ptr=2, data at [0]=3.0, [1]=4.0, [2]=2.0
        assert buf._obs[0, 0] == 3.0
        assert buf._obs[1, 0] == 4.0
        assert buf._obs[2, 0] == 2.0

    def test_sample_returns_jax_arrays(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(4,))
        for _ in range(20):
            buf.push(np.random.randn(4), 0, 1.0, np.random.randn(4), False)

        batch = buf.sample(8)
        assert isinstance(batch, Transition)
        assert isinstance(batch.obs, jax.Array)
        assert isinstance(batch.action, jax.Array)
        assert isinstance(batch.reward, jax.Array)
        assert isinstance(batch.next_obs, jax.Array)
        assert isinstance(batch.done, jax.Array)

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(4,))
        for _ in range(20):
            buf.push(np.random.randn(4), 1, 0.5, np.random.randn(4), True)

        batch = buf.sample(8)
        assert batch.obs.shape == (8, 4)
        assert batch.action.shape == (8,)
        assert batch.reward.shape == (8,)
        assert batch.next_obs.shape == (8, 4)
        assert batch.done.shape == (8,)

    def test_sample_jit_compatible(self):
        """Sampled batch can be consumed inside jit."""
        buf = ReplayBuffer(capacity=100, obs_shape=(4,))
        for _ in range(20):
            buf.push(np.random.randn(4), 0, 1.0, np.random.randn(4), False)

        batch = buf.sample(8)

        @jax.jit
        def compute_mean_reward(t: Transition) -> jax.Array:
            return jnp.mean(t.reward)

        result = compute_mean_reward(batch)
        assert result.shape == ()

    def test_push_transition(self):
        """push_transition accepts jax arrays."""
        buf = ReplayBuffer(capacity=10, obs_shape=(3,))
        t = Transition(
            obs=jnp.array([1.0, 2.0, 3.0]),
            action=jnp.array(1, dtype=jnp.int32),
            reward=jnp.array(0.5),
            next_obs=jnp.array([4.0, 5.0, 6.0]),
            done=jnp.array(True),
        )
        buf.push_transition(t)
        assert len(buf) == 1

        batch = buf.sample(1)
        assert jnp.allclose(batch.obs[0], jnp.array([1.0, 2.0, 3.0]))
        assert int(batch.action[0]) == 1
        assert bool(batch.done[0]) is True

    def test_dtypes(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(2,))
        buf.push(np.array([1.0, 2.0]), 3, -1.0, np.array([3.0, 4.0]), True)
        batch = buf.sample(1)
        assert batch.obs.dtype == jnp.float32
        assert batch.action.dtype == jnp.int32
        assert batch.reward.dtype == jnp.float32
        assert batch.done.dtype == jnp.bool_
