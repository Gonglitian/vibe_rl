"""Tests for the PixelGridWorld vision environment."""

import jax
import jax.numpy as jnp
import pytest

from vibe_rl.env import PixelGridWorld, PixelGridWorldParams
from vibe_rl.env.spaces import Image


@pytest.fixture
def env():
    return PixelGridWorld()


@pytest.fixture
def params():
    return PixelGridWorldParams()


class TestPixelGridWorldBasic:
    def test_reset_obs_shape(self, env, params):
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        expected_size = params.size * params.cell_px  # 5 * 8 = 40
        assert obs.shape == (expected_size, expected_size, 3)
        assert obs.dtype == jnp.uint8

    def test_observation_space_is_image(self, env, params):
        space = env.observation_space(params)
        assert isinstance(space, Image)
        assert space.shape == (40, 40, 3)

    def test_reset_agent_at_origin(self, env, params):
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        # Agent at (0, 0) → top-left cell_px block should be white
        cell = obs[0:params.cell_px, 0:params.cell_px]
        assert jnp.all(cell == 255)

    def test_goal_rendered_green(self, env, params):
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        # Goal at (4, 4) → bottom-right cell_px block should be green
        s = (params.size - 1) * params.cell_px
        e = params.size * params.cell_px
        goal_cell = obs[s:e, s:e]
        # Green = (0, 255, 0), but agent at (0,0) so goal cell is pure green
        assert jnp.all(goal_cell[:, :, 0] == 0)
        assert jnp.all(goal_cell[:, :, 1] == 255)
        assert jnp.all(goal_cell[:, :, 2] == 0)

    def test_step_changes_obs(self, env, params):
        key = jax.random.PRNGKey(0)
        obs1, state = env.reset(key, params)
        obs2, state, reward, done, info = env.step(key, state, jnp.int32(1), params)  # right
        assert not jnp.array_equal(obs1, obs2)
        assert reward == jnp.float32(-0.01)
        assert not done

    def test_goal_reward(self, env, params):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        for _ in range(4):
            obs, state, r, done, info = env.step(key, state, jnp.int32(1), params)
        for _ in range(4):
            obs, state, r, done, info = env.step(key, state, jnp.int32(2), params)
        assert r == jnp.float32(1.0)
        assert done

    def test_make_registry(self):
        from vibe_rl.env import make
        env, params = make("PixelGridWorld-v0")
        assert isinstance(env, PixelGridWorld)


class TestPixelGridWorldJAXCompat:
    def test_jit(self, env, params):
        jit_step = jax.jit(env.step)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        obs, state, r, d, info = jit_step(key, state, jnp.int32(1), params)
        expected_size = params.size * params.cell_px
        assert obs.shape == (expected_size, expected_size, 3)

    def test_vmap(self, env, params):
        batch_reset = jax.vmap(env.reset, in_axes=(0, None))
        n = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        obs, states = batch_reset(keys, params)
        expected_size = params.size * params.cell_px
        assert obs.shape == (n, expected_size, expected_size, 3)
