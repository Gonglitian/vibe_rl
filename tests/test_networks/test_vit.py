"""Tests for the ViT encoder."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from vibe_rl.networks.vit import ViTEncoder


class TestViTEncoder:
    def test_output_shape(self):
        enc = ViTEncoder(
            32, 32, channels=3,
            patch_size=8, embed_dim=64, depth=2, num_heads=4, output_dim=128,
            key=jax.random.PRNGKey(0),
        )
        x = jnp.ones((32, 32, 3), dtype=jnp.float32)
        out = enc(x)
        assert out.shape == (128,)
        assert enc.output_dim == 128

    def test_jit_compile(self):
        enc = ViTEncoder(
            32, 32, channels=3,
            patch_size=8, embed_dim=64, depth=2, num_heads=4, output_dim=128,
            key=jax.random.PRNGKey(0),
        )
        x = jnp.ones((32, 32, 3), dtype=jnp.float32)
        jit_fn = eqx.filter_jit(enc)
        out = jit_fn(x)
        assert out.shape == (128,)

    def test_vmap(self):
        enc = ViTEncoder(
            32, 32, channels=3,
            patch_size=8, embed_dim=64, depth=2, num_heads=4, output_dim=128,
            key=jax.random.PRNGKey(0),
        )
        batch = jnp.ones((4, 32, 32, 3), dtype=jnp.float32)
        out = jax.vmap(enc)(batch)
        assert out.shape == (4, 128)

    def test_patch_size_mismatch(self):
        with pytest.raises(AssertionError):
            ViTEncoder(30, 30, patch_size=8, key=jax.random.PRNGKey(0))

    def test_output_not_all_zero(self):
        enc = ViTEncoder(
            32, 32, channels=3,
            patch_size=8, embed_dim=64, depth=2, num_heads=4, output_dim=128,
            key=jax.random.PRNGKey(42),
        )
        x = jax.random.uniform(jax.random.PRNGKey(1), (32, 32, 3))
        out = enc(x)
        assert jnp.any(out != 0.0)
