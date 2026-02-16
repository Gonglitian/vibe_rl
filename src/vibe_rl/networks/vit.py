"""Lightweight Vision Transformer (ViT) encoder for image observations.

Implements a minimal ViT following the architecture of Dosovitskiy et al.
(2020) with design choices inspired by SigLIP (Zhai et al. 2023):

  1. Patch embedding via Conv2d (non-overlapping patches)
  2. Learnable positional embedding (additive)
  3. Stacked Transformer blocks (pre-norm with LayerNorm)
  4. Mean-pool over patch tokens -> Linear projection

This is intentionally *lightweight* — no CLS token, no dropout — to keep
things fast under ``jax.jit`` for RL workloads.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp


class _TransformerBlock(eqx.Module):
    """Pre-norm Transformer block: LayerNorm -> MHA -> residual -> LayerNorm -> MLP -> residual."""

    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp_up: eqx.nn.Linear
    mlp_down: eqx.nn.Linear

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, *, key: jax.Array) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            key=k1,
        )
        self.norm2 = eqx.nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp_up = eqx.nn.Linear(embed_dim, mlp_hidden, key=k2)
        self.mlp_down = eqx.nn.Linear(mlp_hidden, embed_dim, key=k3)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass. x: (num_patches, embed_dim)."""
        # Self-attention with pre-norm
        h = jax.vmap(self.norm1)(x)
        h = self.attn(h, h, h)
        x = x + h

        # MLP with pre-norm
        h = jax.vmap(self.norm2)(x)
        h = jax.vmap(self.mlp_up)(h)
        h = jax.nn.gelu(h)
        h = jax.vmap(self.mlp_down)(h)
        return x + h


class ViTEncoder(eqx.Module):
    """Vision Transformer encoder: image (H, W, C) -> flat feature vector.

    Input images are expected in ``float32`` with values in ``[0, 1]``.

    Args:
        height: Image height (must be divisible by ``patch_size``).
        width: Image width (must be divisible by ``patch_size``).
        channels: Number of input channels.
        patch_size: Side length of each square patch.
        embed_dim: Embedding dimension for patch tokens.
        depth: Number of Transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim = ``embed_dim * mlp_ratio``.
        output_dim: Size of the final projection (encoder output).
        key: PRNG key for initialization.
    """

    patch_embed: eqx.nn.Conv2d
    pos_embed: jax.Array
    blocks: list
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    output_dim: int = eqx.field(static=True)
    _num_patches: int = eqx.field(static=True)

    def __init__(
        self,
        height: int,
        width: int,
        channels: int = 3,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        output_dim: int = 256,
        *,
        key: jax.Array,
    ) -> None:
        assert height % patch_size == 0, f"height {height} must be divisible by patch_size {patch_size}"
        assert width % patch_size == 0, f"width {width} must be divisible by patch_size {patch_size}"

        n_patches_h = height // patch_size
        n_patches_w = width // patch_size
        self._num_patches = n_patches_h * n_patches_w

        keys = jax.random.split(key, depth + 3)

        # Patch embedding: non-overlapping convolution
        self.patch_embed = eqx.nn.Conv2d(
            channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            key=keys[0],
        )

        # Learnable positional embedding
        self.pos_embed = jax.random.normal(keys[1], (self._num_patches, embed_dim)) * 0.02

        # Transformer blocks
        self.blocks = [
            _TransformerBlock(embed_dim, num_heads, mlp_ratio, key=keys[2 + i])
            for i in range(depth)
        ]

        self.norm = eqx.nn.LayerNorm(embed_dim)
        self.head = eqx.nn.Linear(embed_dim, output_dim, key=keys[2 + depth])
        self.output_dim = output_dim

    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode an image observation.

        Args:
            x: Image array of shape ``(H, W, C)`` in float32 [0, 1].

        Returns:
            Feature vector of shape ``(output_dim,)``.
        """
        # Patch embed: (H, W, C) -> channels-first -> Conv2d -> (embed_dim, nH, nW)
        x = jnp.transpose(x, (2, 0, 1))
        x = self.patch_embed(x)  # (embed_dim, nH, nW)

        # Reshape to sequence: (num_patches, embed_dim)
        x = x.reshape(x.shape[0], -1).T  # (num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Post-norm + mean pool + projection
        x = jax.vmap(self.norm)(x)
        x = jnp.mean(x, axis=0)  # (embed_dim,)
        return self.head(x)
