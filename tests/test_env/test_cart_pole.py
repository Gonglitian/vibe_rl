import numpy as np

from vibe_rl.env.cart_pole import CartPoleEnv


class TestCartPole:
    def test_reset(self):
        env = CartPoleEnv()
        obs = env.reset(seed=42)
        assert obs.shape == (4,)
        assert np.all(np.abs(obs) < 0.1)  # initial state is near zero

    def test_step(self):
        env = CartPoleEnv()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (4,)
        assert reward == 1.0
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_terminates_on_angle(self):
        env = CartPoleEnv()
        env.reset(seed=42)
        # Push in one direction repeatedly until pole falls
        terminated = False
        for _ in range(1000):
            _, _, terminated, truncated, _ = env.step(1)
            if terminated or truncated:
                break
        # Should eventually terminate (pole falls or truncation)
        assert terminated or truncated

    def test_observation_space(self):
        env = CartPoleEnv()
        obs = env.reset(seed=42)
        assert env.observation_space.shape == (4,)
        assert env.observation_space.contains(obs)

    def test_action_space(self):
        env = CartPoleEnv()
        assert env.action_space.n == 2
        assert env.action_space.contains(0)
        assert env.action_space.contains(1)
        assert not env.action_space.contains(2)

    def test_truncation(self):
        env = CartPoleEnv(max_steps=5)
        env.reset(seed=42)
        truncated = False
        for _ in range(10):
            # Alternate actions to keep pole balanced
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
            _, _, terminated, truncated, _ = env.step(1)
            if terminated or truncated:
                break
        assert truncated or terminated
