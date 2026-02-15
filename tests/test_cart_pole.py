"""Tests for the pure-JAX CartPole environment."""

import jax
import jax.numpy as jnp
import pytest

from vibe_rl.env import CartPole, CartPoleParams


@pytest.fixture
def env():
    return CartPole()


@pytest.fixture
def params():
    return CartPoleParams()


class TestCartPoleBasic:
    def test_reset_obs_shape(self, env, params):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        assert obs.shape == (4,)
        assert obs.dtype == jnp.float32

    def test_reset_state_time_zero(self, env, params):
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        assert state.time == 0

    def test_step_returns_correct_types(self, env, params):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        obs, state, reward, done, info = env.step(key, state, jnp.int32(1), params)
        assert obs.shape == (4,)
        assert reward.dtype == jnp.float32
        assert done.dtype == jnp.bool_
        assert "terminated" in info
        assert "truncated" in info

    def test_step_increments_time(self, env, params):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        _, state, _, _, _ = env.step(key, state, jnp.int32(0), params)
        assert state.time == 1

    def test_terminates_on_out_of_bounds(self, env, params):
        """Running many steps with constant action should eventually terminate."""
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key, params)
        done = jnp.bool_(False)
        for i in range(600):
            key, subkey = jax.random.split(key)
            obs, state, reward, done, info = env.step(subkey, state, jnp.int32(1), params)
            if done:
                break
        assert done

    def test_spaces(self, env, params):
        obs_sp = env.observation_space(params)
        act_sp = env.action_space(params)
        assert obs_sp.shape == (4,)
        assert act_sp.n == 2


class TestCartPoleJAXCompat:
    def test_jit_reset_step(self, env, params):
        jit_reset = jax.jit(env.reset, static_argnums=())
        jit_step = jax.jit(env.step, static_argnums=())

        key = jax.random.PRNGKey(0)
        obs, state = jit_reset(key, params)
        assert obs.shape == (4,)

        obs, state, reward, done, info = jit_step(key, state, jnp.int32(0), params)
        assert obs.shape == (4,)

    def test_vmap_reset(self, env, params):
        """vmap across 8 parallel environments."""
        batch_reset = jax.vmap(env.reset, in_axes=(0, None))
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        obs_batch, state_batch = batch_reset(keys, params)
        assert obs_batch.shape == (8, 4)
        assert state_batch.time.shape == (8,)

    def test_vmap_step(self, env, params):
        batch_reset = jax.vmap(env.reset, in_axes=(0, None))
        batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        n = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        obs, states = batch_reset(keys, params)
        actions = jnp.zeros(n, dtype=jnp.int32)
        keys2 = jax.random.split(jax.random.PRNGKey(1), n)
        obs, states, rewards, dones, infos = batch_step(keys2, states, actions, params)
        assert obs.shape == (n, 4)
        assert rewards.shape == (n,)
        assert dones.shape == (n,)

    def test_lax_scan(self, env, params):
        """Run a short rollout with lax.scan."""
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)

        def scan_step(carry, _):
            state, key = carry
            key, subkey = jax.random.split(key)
            action = jnp.int32(1)
            obs, state, reward, done, info = env.step(subkey, state, action, params)
            return (state, key), (obs, reward, done)

        (final_state, _), (obs_traj, reward_traj, done_traj) = jax.lax.scan(
            scan_step, (state, key), None, length=10,
        )
        assert obs_traj.shape == (10, 4)
        assert reward_traj.shape == (10,)

    def test_deterministic(self, env, params):
        """Same key produces same results."""
        key = jax.random.PRNGKey(123)
        obs1, state1 = env.reset(key, params)
        obs2, state2 = env.reset(key, params)
        assert jnp.allclose(obs1, obs2)
