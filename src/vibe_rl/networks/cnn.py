"""CNN encoder for image observations (Nature DQN style).

The default architecture follows the Nature DQN paper (Mnih et al. 2015):
  Conv2d(32, 8x8, stride 4) -> ReLU
  Conv2d(64, 4x4, stride 2) -> ReLU
  Conv2d(64, 3x3, stride 1) -> ReLU
  Flatten -> Linear(512) -> ReLU

All layers are configurable via constructor arguments.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp


class CNNEncoder(eqx.Module):
    """Convolutional encoder: image (H, W, C) -> flat feature vector.

    Input images are expected in ``float32`` with values in ``[0, 1]``.
    If your observations are ``uint8`` in ``[0, 255]``, divide by 255
    before passing them through the encoder.

    Args:
        height: Image height.
        width: Image width.
        channels: Number of input channels.
        channel_sizes: Output channels for each conv layer.
        kernel_sizes: Kernel size for each conv layer.
        strides: Stride for each conv layer.
        mlp_hidden: Size of the fully-connected layer after flatten.
        key: PRNG key for parameter initialization.
    """

    conv_layers: list
    fc: eqx.nn.Linear
    output_dim: int = eqx.field(static=True)

    def __init__(
        self,
        height: int,
        width: int,
        channels: int = 3,
        channel_sizes: tuple[int, ...] = (32, 64, 64),
        kernel_sizes: tuple[int, ...] = (8, 4, 3),
        strides: tuple[int, ...] = (4, 2, 1),
        mlp_hidden: int = 512,
        *,
        key: jax.Array,
    ) -> None:
        assert len(channel_sizes) == len(kernel_sizes) == len(strides)

        n_conv = len(channel_sizes)
        keys = jax.random.split(key, n_conv + 1)

        # Build conv stack (channels-last convention: input is (H, W, C))
        in_channels = channels
        self.conv_layers = []
        for i, (out_ch, ks, st) in enumerate(
            zip(channel_sizes, kernel_sizes, strides)
        ):
            self.conv_layers.append(
                eqx.nn.Conv2d(
                    in_channels, out_ch, kernel_size=ks, stride=st, key=keys[i],
                )
            )
            in_channels = out_ch

        # Compute flattened feature size by tracing spatial dims
        h, w = height, width
        for ks, st in zip(kernel_sizes, strides):
            h = (h - ks) // st + 1
            w = (w - ks) // st + 1
        flat_dim = h * w * channel_sizes[-1]

        self.fc = eqx.nn.Linear(flat_dim, mlp_hidden, key=keys[n_conv])
        self.output_dim = mlp_hidden

    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode an image observation.

        Args:
            x: Image array of shape ``(H, W, C)`` in float32 [0, 1].

        Returns:
            Feature vector of shape ``(output_dim,)``.
        """
        # Equinox Conv2d expects channels-first: (C, H, W)
        x = jnp.transpose(x, (2, 0, 1))

        for conv in self.conv_layers:
            x = jax.nn.relu(conv(x))

        x = x.reshape(-1)  # flatten
        return jax.nn.relu(self.fc(x))
