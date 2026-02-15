import numpy as np

from vibe_rl.dataprotocol.transition import Transition


class TestTransition:
    def test_creation(self):
        t = Transition(
            state=np.array([1.0, 2.0]),
            action=0,
            reward=1.0,
            next_state=np.array([3.0, 4.0]),
            terminated=False,
        )
        assert t.action == 0
        assert t.reward == 1.0
        assert not t.terminated

    def test_frozen(self):
        t = Transition(
            state=np.array([1.0]),
            action=0,
            reward=0.0,
            next_state=np.array([2.0]),
            terminated=True,
        )
        try:
            t.action = 1
            assert False, "Should not be able to set attribute on frozen dataclass"
        except AttributeError:
            pass
