"""Tests for the unified encoder interface and factory."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from vibe_rl.networks.encoder import Encoder, MLPEncoder, make_encoder
from vibe_rl.networks.cnn import CNNEncoder
from vibe_rl.networks.vit import ViTEncoder


class TestMLPEncoder:
    def test_output_shape(self):
        enc = MLPEncoder(10, hidden_sizes=(32, 16), key=jax.random.PRNGKey(0))
        x = jnp.ones(10)
        out = enc(x)
        assert out.shape == (16,)
        assert enc.output_dim == 16

    def test_jit(self):
        enc = MLPEncoder(4, key=jax.random.PRNGKey(0))
        out = eqx.filter_jit(enc)(jnp.ones(4))
        assert out.shape == (64,)


class TestEncoderProtocol:
    def test_cnn_satisfies_protocol(self):
        enc = CNNEncoder(84, 84, key=jax.random.PRNGKey(0))
        assert isinstance(enc, Encoder)

    def test_vit_satisfies_protocol(self):
        enc = ViTEncoder(32, 32, patch_size=8, embed_dim=64, depth=2, num_heads=4, output_dim=128, key=jax.random.PRNGKey(0))
        assert isinstance(enc, Encoder)

    def test_mlp_satisfies_protocol(self):
        enc = MLPEncoder(10, key=jax.random.PRNGKey(0))
        assert isinstance(enc, Encoder)


class TestMakeEncoder:
    def test_make_mlp(self):
        enc = make_encoder("mlp", key=jax.random.PRNGKey(0), input_dim=10)
        assert isinstance(enc, MLPEncoder)
        assert enc.output_dim == 64

    def test_make_cnn(self):
        enc = make_encoder("cnn", key=jax.random.PRNGKey(0), height=84, width=84)
        assert isinstance(enc, CNNEncoder)
        assert enc.output_dim == 512

    def test_make_vit(self):
        enc = make_encoder(
            "vit", key=jax.random.PRNGKey(0),
            height=32, width=32, patch_size=8, embed_dim=64,
            depth=2, num_heads=4, output_dim=128,
        )
        assert isinstance(enc, ViTEncoder)
        assert enc.output_dim == 128

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown encoder kind"):
            make_encoder("rnn", key=jax.random.PRNGKey(0))

    def test_mlp_missing_input_dim(self):
        with pytest.raises(ValueError, match="input_dim"):
            make_encoder("mlp", key=jax.random.PRNGKey(0))

    def test_cnn_missing_dims(self):
        with pytest.raises(ValueError, match="height and width"):
            make_encoder("cnn", key=jax.random.PRNGKey(0))
