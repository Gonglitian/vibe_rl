"""Tests for the pure-JAX Pendulum environment."""

import jax
import jax.numpy as jnp

from vibe_rl.env import make
from vibe_rl.env.pendulum import Pendulum, PendulumParams, PendulumState


class TestPendulum:
    def test_reset_shapes(self):
        env = Pendulum()
        params = env.default_params()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        assert obs.shape == (3,)
        assert isinstance(state, PendulumState)

    def test_step_shapes(self):
        env = Pendulum()
        params = env.default_params()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        action = jnp.array([0.5])
        k1, k2 = jax.random.split(key)
        next_obs, new_state, reward, done, info = env.step(k1, state, action, params)
        assert next_obs.shape == (3,)
        assert reward.shape == ()
        assert done.shape == ()

    def test_obs_range(self):
        env = Pendulum()
        params = env.default_params()
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key, params)
        # cos and sin should be in [-1, 1]
        assert jnp.abs(obs[0]) <= 1.0 + 1e-6
        assert jnp.abs(obs[1]) <= 1.0 + 1e-6

    def test_terminates_after_max_steps(self):
        env = Pendulum()
        params = PendulumParams(max_steps=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        action = jnp.array([0.0])
        done = False
        for i in range(10):
            key, step_key = jax.random.split(key)
            obs, state, reward, done, info = env.step(step_key, state, action, params)
            if bool(done):
                break
        assert bool(done)
        assert int(state.time) == 5

    def test_registry(self):
        env, params = make("Pendulum-v1")
        assert isinstance(env, Pendulum)
        assert isinstance(params, PendulumParams)

    def test_action_space(self):
        env = Pendulum()
        params = env.default_params()
        space = env.action_space(params)
        assert space.shape == (1,)

    def test_observation_space(self):
        env = Pendulum()
        params = env.default_params()
        space = env.observation_space(params)
        assert space.shape == (3,)

    def test_vmap_compatible(self):
        env = Pendulum()
        params = env.default_params()
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        batch_reset = jax.vmap(env.reset, in_axes=(0, None))
        obs_batch, states = batch_reset(keys, params)
        assert obs_batch.shape == (4, 3)

    def test_jit_compatible(self):
        env = Pendulum()
        params = env.default_params()
        key = jax.random.PRNGKey(0)

        @jax.jit
        def _reset_and_step(key):
            k1, k2 = jax.random.split(key)
            obs, state = env.reset(k1, params)
            action = jnp.array([1.0])
            next_obs, state, reward, done, info = env.step(k2, state, action, params)
            return next_obs, reward

        next_obs, reward = _reset_and_step(key)
        assert next_obs.shape == (3,)
        assert jnp.isfinite(reward)
