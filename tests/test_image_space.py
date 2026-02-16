"""Tests for the Image observation space."""

import jax
import jax.numpy as jnp

from vibe_rl.env.spaces import Image


class TestImage:
    def test_sample_shape(self):
        sp = Image(height=84, width=84, channels=3)
        sample = sp.sample(jax.random.PRNGKey(0))
        assert sample.shape == (84, 84, 3)
        assert sample.dtype == jnp.uint8

    def test_sample_range(self):
        sp = Image(height=32, width=32, channels=1)
        samples = jax.vmap(sp.sample)(jax.random.split(jax.random.PRNGKey(0), 50))
        assert jnp.all(samples >= 0)
        assert jnp.all(samples <= 255)

    def test_contains(self):
        sp = Image(height=4, width=4, channels=3)
        valid = jnp.zeros((4, 4, 3), dtype=jnp.uint8)
        assert sp.contains(valid)
        wrong_shape = jnp.zeros((3, 4, 3), dtype=jnp.uint8)
        assert not sp.contains(wrong_shape)

    def test_shape_dtype(self):
        sp = Image(height=64, width=48, channels=1)
        assert sp.shape == (64, 48, 1)
        assert sp.dtype == jnp.uint8

    def test_jit_sample(self):
        sp = Image(height=16, width=16, channels=3)
        jit_sample = jax.jit(sp.sample)
        result = jit_sample(jax.random.PRNGKey(42))
        assert result.shape == (16, 16, 3)
