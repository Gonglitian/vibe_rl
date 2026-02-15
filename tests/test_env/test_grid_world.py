import numpy as np

from vibe_rl.env.grid_world import GridWorldEnv


class TestGridWorld:
    def test_reset(self):
        env = GridWorldEnv(size=5)
        obs = env.reset(seed=42)
        assert obs.shape == (2,)
        assert np.allclose(obs, [0.0, 0.0])  # starts at (0, 0)

    def test_step_right(self):
        env = GridWorldEnv(size=5)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(1)  # right
        assert np.allclose(obs, [0.0, 0.25])  # (0, 1) normalized
        assert reward == -0.01
        assert not terminated

    def test_reach_goal(self):
        env = GridWorldEnv(size=2)
        env.reset(seed=42)
        # From (0,0), go right to (0,1), then down to (1,1) = goal
        env.step(1)  # right -> (0,1)
        obs, reward, terminated, truncated, info = env.step(2)  # down -> (1,1)
        assert terminated
        assert reward == 1.0

    def test_boundary_clipping(self):
        env = GridWorldEnv(size=3)
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(0)  # up from (0,0) should stay at (0,0)
        assert np.allclose(obs, [0.0, 0.0])

    def test_truncation(self):
        env = GridWorldEnv(size=5, max_steps=3)
        env.reset(seed=42)
        env.step(0)
        env.step(0)
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated
        assert not terminated

    def test_render(self):
        env = GridWorldEnv(size=3)
        env.reset(seed=42)
        text = env.render()
        assert "A" in text
        assert "G" in text

    def test_observation_space(self):
        env = GridWorldEnv(size=5)
        obs = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_action_space(self):
        env = GridWorldEnv(size=5)
        assert env.action_space.n == 4
        for a in range(4):
            assert env.action_space.contains(a)
