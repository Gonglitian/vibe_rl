"""Tests for the CNN encoder."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from vibe_rl.networks.cnn import CNNEncoder


class TestCNNEncoder:
    def test_output_shape(self):
        enc = CNNEncoder(84, 84, channels=3, key=jax.random.PRNGKey(0))
        x = jnp.ones((84, 84, 3), dtype=jnp.float32)
        out = enc(x)
        assert out.shape == (enc.output_dim,)
        assert out.shape == (512,)

    def test_custom_config(self):
        enc = CNNEncoder(
            48, 48, channels=1,
            channel_sizes=(16, 32),
            kernel_sizes=(4, 3),
            strides=(2, 1),
            mlp_hidden=128,
            key=jax.random.PRNGKey(1),
        )
        x = jnp.ones((48, 48, 1), dtype=jnp.float32)
        out = enc(x)
        assert out.shape == (128,)
        assert enc.output_dim == 128

    def test_jit_compile(self):
        enc = CNNEncoder(84, 84, channels=3, key=jax.random.PRNGKey(0))
        x = jnp.ones((84, 84, 3), dtype=jnp.float32)
        jit_fn = eqx.filter_jit(enc)
        out = jit_fn(x)
        assert out.shape == (512,)

    def test_vmap(self):
        enc = CNNEncoder(84, 84, channels=3, key=jax.random.PRNGKey(0))
        batch = jnp.ones((4, 84, 84, 3), dtype=jnp.float32)
        out = jax.vmap(enc)(batch)
        assert out.shape == (4, 512)

    def test_output_not_all_zero(self):
        enc = CNNEncoder(84, 84, channels=3, key=jax.random.PRNGKey(42))
        x = jax.random.uniform(jax.random.PRNGKey(1), (84, 84, 3))
        out = enc(x)
        assert jnp.any(out != 0.0)
