"""Composable data transform pipeline.

Provides a ``Transform`` protocol and a ``TransformGroup`` combiner so
that data processing steps (resize, normalize, tokenize, pad) can be
freely composed into a pipeline applied to each sample *before* batching.

Usage::

    from vibe_rl.data.transforms import TransformGroup, Resize, Pad

    pipeline = TransformGroup([
        Resize(keys=["obs", "next_obs"], height=64, width=64),
        Pad(keys=["action"], max_len=10, pad_value=0.0),
    ])
    sample = pipeline(sample)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

# A sample is a plain dict mapping string keys to arrays or scalars.
Sample = dict[str, Any]


@runtime_checkable
class Transform(Protocol):
    """Minimal interface for a data transform.

    A transform takes a sample dict and returns a (possibly modified) sample
    dict.  Transforms are expected to be pure functions of their input — any
    configuration is captured at construction time.
    """

    def __call__(self, sample: Sample) -> Sample: ...


# ---------------------------------------------------------------------------
# TransformGroup — compose multiple transforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformGroup:
    """Apply a sequence of transforms in order.

    Parameters:
        transforms: Ordered sequence of ``Transform`` callables.
    """

    transforms: Sequence[Transform] = field(default_factory=list)

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resize:
    """Resize image arrays to ``(height, width)``.

    Expects arrays with shape ``(..., H, W, C)`` (channels-last) or
    ``(..., C, H, W)`` (channels-first, when ``channels_first=True``).

    Only keys listed in ``keys`` are affected; missing keys are silently
    skipped so the same transform works for datasets with/without images.

    Parameters:
        keys: Sample keys to resize.
        height: Target height.
        width: Target width.
        channels_first: If ``True``, interpret input as ``(..., C, H, W)``.
    """

    keys: Sequence[str]
    height: int
    width: int
    channels_first: bool = False

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        for key in self.keys:
            if key not in sample:
                continue
            arr = np.asarray(sample[key])
            sample[key] = _resize_array(
                arr, self.height, self.width, self.channels_first
            )
        return sample


def _resize_array(
    arr: np.ndarray, height: int, width: int, channels_first: bool
) -> np.ndarray:
    """Resize a single image array using nearest-neighbor interpolation.

    This is a pure-numpy implementation (no PIL/cv2 dependency) using
    index-based nearest-neighbor resampling.
    """
    if channels_first:
        # (..., C, H_in, W_in) → transpose last 3 dims to channels-last
        arr = np.moveaxis(arr, -3, -1)

    h_in, w_in = arr.shape[-3], arr.shape[-2]
    if h_in == height and w_in == width:
        result = arr
    else:
        row_idx = (np.arange(height) * h_in / height).astype(int)
        col_idx = (np.arange(width) * w_in / width).astype(int)
        result = arr[..., row_idx[:, None], col_idx[None, :], :]

    if channels_first:
        result = np.moveaxis(result, -1, -3)
    return result


@dataclass(frozen=True)
class Normalize:
    """Per-element affine normalization: ``(x - loc) / scale``.

    Typically used with pre-computed normalization statistics.

    Parameters:
        keys: Sample keys to normalize.
        loc: Offset (e.g. mean or q01). Broadcastable to the array shape.
        scale: Scale (e.g. std or q99-q01). Broadcastable to the array shape.
        eps: Small constant to avoid division by zero.
    """

    keys: Sequence[str]
    loc: np.ndarray | float
    scale: np.ndarray | float
    eps: float = 1e-8

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        loc = np.asarray(self.loc)
        scale = np.asarray(self.scale)
        for key in self.keys:
            if key not in sample:
                continue
            arr = np.asarray(sample[key], dtype=np.float32)
            sample[key] = (arr - loc) / np.maximum(scale, self.eps)
        return sample


@dataclass(frozen=True)
class Tokenize:
    """Discretize continuous values into integer tokens.

    Maps ``[vmin, vmax]`` linearly into ``[0, num_tokens - 1]``, then
    clips to valid range.

    Parameters:
        keys: Sample keys to tokenize.
        num_tokens: Number of discrete bins.
        vmin: Lower bound of the continuous range.
        vmax: Upper bound of the continuous range.
    """

    keys: Sequence[str]
    num_tokens: int = 256
    vmin: float = -1.0
    vmax: float = 1.0

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        for key in self.keys:
            if key not in sample:
                continue
            arr = np.asarray(sample[key], dtype=np.float32)
            scaled = (arr - self.vmin) / (self.vmax - self.vmin)
            tokens = np.clip(
                np.round(scaled * (self.num_tokens - 1)).astype(np.int32),
                0,
                self.num_tokens - 1,
            )
            sample[key] = tokens
        return sample


@dataclass(frozen=True)
class Pad:
    """Pad arrays along the first axis to ``max_len``.

    If the array is already at least ``max_len`` along axis 0 it is
    truncated to exactly ``max_len``.

    Parameters:
        keys: Sample keys to pad.
        max_len: Target length along axis 0.
        pad_value: Value used for padding.
    """

    keys: Sequence[str]
    max_len: int
    pad_value: float = 0.0

    def __call__(self, sample: Sample) -> Sample:
        sample = dict(sample)
        for key in self.keys:
            if key not in sample:
                continue
            arr = np.asarray(sample[key])
            current_len = arr.shape[0]
            if current_len >= self.max_len:
                sample[key] = arr[: self.max_len]
            else:
                pad_width = [(0, self.max_len - current_len)] + [
                    (0, 0)
                ] * (arr.ndim - 1)
                sample[key] = np.pad(
                    arr, pad_width, mode="constant", constant_values=self.pad_value
                )
        return sample


@dataclass(frozen=True)
class LambdaTransform:
    """Wrap an arbitrary callable as a ``Transform``.

    Parameters:
        fn: A callable ``(sample) -> sample``.
    """

    fn: Callable[[Sample], Sample]

    def __call__(self, sample: Sample) -> Sample:
        return self.fn(sample)
