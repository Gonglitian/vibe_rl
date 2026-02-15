"""Tests for vibe_rl.env.spaces."""

import jax
import jax.numpy as jnp
import pytest

from vibe_rl.env.spaces import Box, Discrete, MultiBinary


class TestDiscrete:
    def test_sample_in_range(self):
        sp = Discrete(n=5)
        key = jax.random.PRNGKey(0)
        samples = jax.vmap(sp.sample)(jax.random.split(key, 100))
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < 5)

    def test_contains(self):
        sp = Discrete(n=3)
        assert sp.contains(jnp.int32(0))
        assert sp.contains(jnp.int32(2))
        assert not sp.contains(jnp.int32(3))
        assert not sp.contains(jnp.int32(-1))

    def test_shape_dtype(self):
        sp = Discrete(n=4)
        assert sp.shape == ()
        assert sp.dtype == jnp.int32

    def test_jit_sample(self):
        sp = Discrete(n=10)
        jit_sample = jax.jit(sp.sample)
        result = jit_sample(jax.random.PRNGKey(42))
        assert result.shape == ()


class TestBox:
    def test_sample_in_bounds(self):
        sp = Box(low=-1.0, high=1.0, shape=(3,))
        key = jax.random.PRNGKey(0)
        samples = jax.vmap(sp.sample)(jax.random.split(key, 100))
        assert jnp.all(samples >= -1.0)
        assert jnp.all(samples <= 1.0)
        assert samples.shape == (100, 3)

    def test_contains(self):
        sp = Box(low=0.0, high=1.0, shape=(2,))
        assert sp.contains(jnp.array([0.5, 0.5]))
        assert not sp.contains(jnp.array([1.5, 0.5]))

    def test_shape_from_arrays(self):
        low = jnp.array([-1.0, -2.0])
        high = jnp.array([1.0, 2.0])
        sp = Box(low=low, high=high)
        assert sp.shape == (2,)

    def test_jit_sample(self):
        sp = Box(low=-1.0, high=1.0, shape=(4,))
        jit_sample = jax.jit(sp.sample)
        result = jit_sample(jax.random.PRNGKey(0))
        assert result.shape == (4,)


class TestMultiBinary:
    def test_sample_binary(self):
        sp = MultiBinary(n=5)
        key = jax.random.PRNGKey(0)
        samples = jax.vmap(sp.sample)(jax.random.split(key, 100))
        assert jnp.all((samples == 0) | (samples == 1))
        assert samples.shape == (100, 5)

    def test_contains(self):
        sp = MultiBinary(n=3)
        assert sp.contains(jnp.array([0, 1, 0]))
        assert not sp.contains(jnp.array([0, 2, 0]))

    def test_shape_dtype(self):
        sp = MultiBinary(n=4)
        assert sp.shape == (4,)
        assert sp.dtype == jnp.int32
