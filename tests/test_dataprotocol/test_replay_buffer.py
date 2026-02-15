import numpy as np

from vibe_rl.dataprotocol.replay_buffer import ReplayBuffer
from vibe_rl.dataprotocol.transition import Transition


def _make_transition(val: float) -> Transition:
    return Transition(
        state=np.array([val, val], dtype=np.float32),
        action=int(val) % 4,
        reward=val,
        next_state=np.array([val + 1, val + 1], dtype=np.float32),
        terminated=False,
    )


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(2,))
        assert len(buf) == 0
        buf.push(_make_transition(1.0))
        assert len(buf) == 1
        buf.push(_make_transition(2.0))
        assert len(buf) == 2

    def test_capacity_wraps(self):
        buf = ReplayBuffer(capacity=3, obs_shape=(2,))
        for i in range(5):
            buf.push(_make_transition(float(i)))
        assert len(buf) == 3  # capped at capacity

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(2,))
        for i in range(20):
            buf.push(_make_transition(float(i)))
        batch = buf.sample(batch_size=8)
        assert batch.states.shape == (8, 2)
        assert batch.actions.shape == (8,)
        assert batch.rewards.shape == (8,)
        assert batch.next_states.shape == (8, 2)
        assert batch.terminated.shape == (8,)

    def test_sample_values(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(2,))
        for i in range(50):
            buf.push(_make_transition(float(i)))
        batch = buf.sample(batch_size=16)
        # All sampled rewards should be in [0, 49]
        assert np.all(batch.rewards >= 0)
        assert np.all(batch.rewards < 50)

    def test_to_torch(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(2,))
        for i in range(20):
            buf.push(_make_transition(float(i)))
        batch = buf.sample(batch_size=4)
        torch_batch = batch.to_torch()
        assert torch_batch.states.shape == (4, 2)
        assert torch_batch.actions.dtype.is_floating_point is False  # long
