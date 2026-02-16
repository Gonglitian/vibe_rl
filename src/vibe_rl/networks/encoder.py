"""Unified encoder interface for MLP, CNN, and ViT backbones.

The ``Encoder`` protocol defines the minimal contract that any encoder
must satisfy: it is a callable ``eqx.Module`` with an ``output_dim``
attribute.  ``make_encoder`` is a convenience factory that selects the
right encoder class from a string tag.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from vibe_rl.networks.cnn import CNNEncoder
from vibe_rl.networks.vit import ViTEncoder


@runtime_checkable
class Encoder(Protocol):
    """Protocol for observation encoders.

    Any ``eqx.Module`` that maps an observation array to a flat feature
    vector and exposes its ``output_dim`` satisfies this protocol.
    """

    output_dim: int

    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode a single observation into a feature vector."""
        ...


class MLPEncoder(eqx.Module):
    """Simple MLP encoder for flat (vector) observations.

    Mirrors the hidden-layer pattern used in the existing algorithm
    networks, providing a drop-in encoder for non-vision observations.

    Args:
        input_dim: Dimensionality of the flat observation.
        hidden_sizes: Widths of hidden layers.
        key: PRNG key for initialization.
    """

    layers: list
    output_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        *,
        key: jax.Array,
    ) -> None:
        dims = [input_dim, *hidden_sizes]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for d_in, d_out, k in zip(dims[:-1], dims[1:], keys)
        ]
        self.output_dim = hidden_sizes[-1]

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = jax.nn.tanh(layer(x))
        return x


def make_encoder(
    kind: str,
    *,
    key: jax.Array,
    # MLP args
    input_dim: int | None = None,
    hidden_sizes: tuple[int, ...] = (64, 64),
    # CNN / ViT shared args
    height: int | None = None,
    width: int | None = None,
    channels: int = 3,
    # CNN-specific
    channel_sizes: tuple[int, ...] = (32, 64, 64),
    kernel_sizes: tuple[int, ...] = (8, 4, 3),
    strides: tuple[int, ...] = (4, 2, 1),
    mlp_hidden: int = 512,
    # ViT-specific
    patch_size: int = 8,
    embed_dim: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    mlp_ratio: float = 4.0,
    output_dim: int = 256,
) -> CNNEncoder | ViTEncoder | MLPEncoder:
    """Factory to create an encoder by name.

    Args:
        kind: One of ``"mlp"``, ``"cnn"``, or ``"vit"``.
        key: PRNG key for initialization.
        **kwargs: Forwarded to the chosen encoder constructor.

    Returns:
        An encoder instance satisfying the :class:`Encoder` protocol.
    """
    kind = kind.lower()
    if kind == "mlp":
        if input_dim is None:
            raise ValueError("input_dim is required for MLPEncoder")
        return MLPEncoder(input_dim, hidden_sizes, key=key)
    elif kind == "cnn":
        if height is None or width is None:
            raise ValueError("height and width are required for CNNEncoder")
        return CNNEncoder(
            height, width, channels,
            channel_sizes, kernel_sizes, strides, mlp_hidden,
            key=key,
        )
    elif kind == "vit":
        if height is None or width is None:
            raise ValueError("height and width are required for ViTEncoder")
        return ViTEncoder(
            height, width, channels,
            patch_size, embed_dim, depth, num_heads, mlp_ratio, output_dim,
            key=key,
        )
    else:
        raise ValueError(f"Unknown encoder kind {kind!r}. Choose from 'mlp', 'cnn', 'vit'.")
