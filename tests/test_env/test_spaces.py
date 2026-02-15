import numpy as np

from vibe_rl.env.spaces import Box, Discrete


class TestDiscrete:
    def test_contains(self):
        space = Discrete(4)
        assert space.contains(0)
        assert space.contains(3)
        assert not space.contains(4)
        assert not space.contains(-1)

    def test_sample(self):
        space = Discrete(4)
        for _ in range(100):
            s = space.sample()
            assert space.contains(s)

    def test_shape(self):
        space = Discrete(5)
        assert space.shape == ()

    def test_n(self):
        space = Discrete(10)
        assert space.n == 10


class TestBox:
    def test_contains(self):
        space = Box(low=-1.0, high=1.0, shape=(3,))
        assert space.contains(np.array([0.0, 0.0, 0.0]))
        assert space.contains(np.array([-1.0, 1.0, 0.5]))
        assert not space.contains(np.array([2.0, 0.0, 0.0]))

    def test_sample(self):
        space = Box(low=0.0, high=1.0, shape=(2,))
        for _ in range(100):
            s = space.sample()
            assert space.contains(s)

    def test_shape(self):
        space = Box(low=0.0, high=1.0, shape=(4,))
        assert space.shape == (4,)

    def test_from_arrays(self):
        low = np.array([0.0, -1.0])
        high = np.array([1.0, 1.0])
        space = Box(low=low, high=high)
        assert space.shape == (2,)
        assert space.contains(np.array([0.5, 0.0]))
