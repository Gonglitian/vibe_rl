"""Tests for the pure-JAX GridWorld environment."""

import jax
import jax.numpy as jnp
import pytest

from vibe_rl.env import GridWorld, GridWorldParams


@pytest.fixture
def env():
    return GridWorld()


@pytest.fixture
def params():
    return GridWorldParams()


class TestGridWorldBasic:
    def test_reset_obs_shape(self, env, params):
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        assert obs.shape == (2,)
        assert obs.dtype == jnp.float32

    def test_reset_starts_at_origin(self, env, params):
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        assert state.row == 0
        assert state.col == 0
        assert jnp.allclose(obs, jnp.array([0.0, 0.0]))

    def test_step_right(self, env, params):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        obs, state, reward, done, info = env.step(key, state, jnp.int32(1), params)  # right
        assert state.col == 1
        assert state.row == 0
        assert reward == jnp.float32(-0.01)
        assert not done

    def test_step_down(self, env, params):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        obs, state, reward, done, info = env.step(key, state, jnp.int32(2), params)  # down
        assert state.row == 1
        assert state.col == 0

    def test_wall_clipping(self, env, params):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        # Try to go up from origin — should clip to 0
        obs, state, _, _, _ = env.step(key, state, jnp.int32(0), params)
        assert state.row == 0

    def test_goal_reward(self, env, params):
        """Navigate to goal and check reward/done."""
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        # Go right 4 times, then down 4 times to reach (4,4) in a 5x5 grid
        for _ in range(4):
            obs, state, r, done, info = env.step(key, state, jnp.int32(1), params)
        for _ in range(4):
            obs, state, r, done, info = env.step(key, state, jnp.int32(2), params)

        assert state.row == params.size - 1
        assert state.col == params.size - 1
        assert r == jnp.float32(1.0)
        assert done

    def test_truncation(self, env):
        params = GridWorldParams(max_steps=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        for _ in range(5):
            obs, state, r, done, info = env.step(key, state, jnp.int32(0), params)
        assert done
        assert info["truncated"]

    def test_spaces(self, env, params):
        assert env.observation_space(params).shape == (2,)
        assert env.action_space(params).n == 4


class TestGridWorldJAXCompat:
    def test_jit(self, env, params):
        jit_step = jax.jit(env.step)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        obs, state, r, d, info = jit_step(key, state, jnp.int32(1), params)
        assert obs.shape == (2,)

    def test_vmap(self, env, params):
        batch_reset = jax.vmap(env.reset, in_axes=(0, None))
        batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        n = 6
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        obs, states = batch_reset(keys, params)
        assert obs.shape == (n, 2)
        actions = jnp.ones(n, dtype=jnp.int32)
        keys2 = jax.random.split(jax.random.PRNGKey(1), n)
        obs, states, rewards, dones, infos = batch_step(keys2, states, actions, params)
        assert obs.shape == (n, 2)

    def test_lax_scan(self, env, params):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)

        def scan_step(carry, _):
            state, key = carry
            key, subkey = jax.random.split(key)
            obs, state, reward, done, info = env.step(subkey, state, jnp.int32(2), params)
            return (state, key), reward

        (final_state, _), rewards = jax.lax.scan(
            scan_step, (state, key), None, length=5,
        )
        assert rewards.shape == (5,)
