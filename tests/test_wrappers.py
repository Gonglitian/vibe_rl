"""Tests for environment wrappers."""

import jax
import jax.numpy as jnp
import pytest

from vibe_rl.env import (
    AutoResetWrapper,
    CartPole,
    CartPoleParams,
    GridWorld,
    GridWorldParams,
    ObsNormWrapper,
    RewardScaleWrapper,
    make,
)


class TestAutoResetWrapper:
    def test_auto_reset_on_done(self):
        env = AutoResetWrapper(GridWorld())
        params = GridWorldParams(size=2, max_steps=100)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)

        # In a 2x2 grid, goal is at (1,1). Go right then down.
        obs, state, r, done, info = env.step(key, state, jnp.int32(1), params)  # right → (0,1)
        assert not done
        obs, state, r, done, info = env.step(key, state, jnp.int32(2), params)  # down → (1,1) = goal
        assert done

        # After auto-reset, state should be back at origin
        assert state.inner.row == 0
        assert state.inner.col == 0
        # terminal_obs should contain the goal observation
        assert "terminal_obs" in info

    def test_auto_reset_jit(self):
        env = AutoResetWrapper(CartPole())
        params = CartPoleParams()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        jit_step = jax.jit(env.step)
        obs, state, r, d, info = jit_step(key, state, jnp.int32(0), params)
        assert obs.shape == (4,)

    def test_auto_reset_vmap(self):
        env = AutoResetWrapper(CartPole())
        params = CartPoleParams()
        n = 4
        batch_reset = jax.vmap(env.reset, in_axes=(0, None))
        batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        obs, states = batch_reset(keys, params)
        actions = jnp.zeros(n, dtype=jnp.int32)
        keys2 = jax.random.split(jax.random.PRNGKey(1), n)
        obs, states, rewards, dones, infos = batch_step(keys2, states, actions, params)
        assert obs.shape == (n, 4)

    def test_auto_reset_lax_scan(self):
        """Full training-style loop: lax.scan with auto-reset."""
        env = AutoResetWrapper(GridWorld())
        params = GridWorldParams(size=3, max_steps=10)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)

        def scan_step(carry, _):
            state, key = carry
            key, subkey = jax.random.split(key)
            action = jnp.int32(1)  # always go right
            obs, state, reward, done, info = env.step(subkey, state, action, params)
            return (state, key), (reward, done)

        (final_state, _), (rewards, dones) = jax.lax.scan(
            scan_step, (state, key), None, length=30,
        )
        assert rewards.shape == (30,)
        # With max_steps=10 and always going right in a 3x3 grid,
        # we should see some resets.
        assert jnp.any(dones)


class TestRewardScaleWrapper:
    def test_scales_reward(self):
        env = RewardScaleWrapper(CartPole(), scale=0.5)
        params = env.default_params()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        _, _, reward, _, _ = env.step(key, state, jnp.int32(0), params)
        assert reward == jnp.float32(0.5)  # CartPole gives +1, scaled to 0.5

    def test_jit_compatible(self):
        env = RewardScaleWrapper(CartPole(), scale=2.0)
        params = env.default_params()
        jit_step = jax.jit(env.step)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        _, _, reward, _, _ = jit_step(key, state, jnp.int32(0), params)
        assert reward == jnp.float32(2.0)


class TestObsNormWrapper:
    def test_normalised_obs_shape(self):
        env = ObsNormWrapper(CartPole())
        params = env.default_params()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        assert obs.shape == (4,)

    def test_step_updates_statistics(self):
        env = ObsNormWrapper(CartPole())
        params = env.default_params()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        initial_count = state.count
        obs, state, _, _, _ = env.step(key, state, jnp.int32(0), params)
        assert state.count > initial_count

    def test_jit_vmap(self):
        env = ObsNormWrapper(CartPole())
        params = env.default_params()
        n = 4
        batch_reset = jax.vmap(env.reset, in_axes=(0, None))
        batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        obs, states = batch_reset(keys, params)
        assert obs.shape == (n, 4)
        actions = jnp.ones(n, dtype=jnp.int32)
        keys2 = jax.random.split(jax.random.PRNGKey(1), n)
        obs, states, rewards, dones, infos = batch_step(keys2, states, actions, params)
        assert obs.shape == (n, 4)


class TestRegistry:
    def test_make_cartpole(self):
        env, params = make("CartPole-v1")
        assert isinstance(env, CartPole)

    def test_make_gridworld(self):
        env, params = make("GridWorld-v0")
        assert isinstance(env, GridWorld)

    def test_make_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown environment"):
            make("NonExistent-v0")
