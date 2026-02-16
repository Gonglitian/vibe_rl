"""Policy wrapper for trained model inference.

Wraps trained model parameters with input/output transforms and provides
a unified ``infer(observation) -> action`` interface with JIT compilation.

The ``Policy`` class handles:
- Input normalization (z-score or quantile) for state observations
- Image resizing for vision inputs
- Action denormalization (inverse of training-time normalization)
- JIT compilation of the full inference pipeline
- Single and batch inference via the same ``infer`` method

Usage::

    from vibe_rl.policies.policy import Policy

    policy = Policy(
        model=trained_model,
        infer_fn=my_infer_fn,
        input_transform=my_input_transform,
        output_transform=my_output_transform,
    )
    action = policy.infer(obs)          # single observation
    actions = policy.infer(obs_batch)   # batch of observations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# Type alias for observation: either a flat array or a dict of arrays
# (for mixed image+state observations).
Observation = jax.Array | dict[str, jax.Array]


@runtime_checkable
class InferFn(Protocol):
    """Protocol for the raw inference function.

    Takes model parameters and a single (preprocessed) observation,
    returns an action array.
    """

    def __call__(self, model: Any, obs: Observation) -> jax.Array: ...


@runtime_checkable
class InputTransform(Protocol):
    """Protocol for input transforms applied before inference."""

    def __call__(self, obs: Observation) -> Observation: ...


@runtime_checkable
class OutputTransform(Protocol):
    """Protocol for output transforms applied after inference."""

    def __call__(self, action: jax.Array) -> jax.Array: ...


@dataclass(frozen=True)
class _Identity:
    """No-op transform."""

    def __call__(self, x: Any) -> Any:
        return x


@dataclass
class Policy:
    """Inference wrapper for a trained RL model.

    Combines a model, an inference function, and optional input/output
    transforms into a single callable with JIT compilation.

    Parameters
    ----------
    model:
        Trained model parameters (Equinox module or any JAX pytree).
    infer_fn:
        Pure function ``(model, obs) -> action`` that performs a single
        forward pass.  Must be JIT-compatible.
    input_transform:
        Transform applied to observations before inference.  Typically
        normalizes state vectors or resizes images.
    output_transform:
        Transform applied to actions after inference.  Typically
        denormalizes actions back to the original scale.
    """

    model: Any
    infer_fn: InferFn
    input_transform: InputTransform | _Identity = field(default_factory=_Identity)
    output_transform: OutputTransform | _Identity = field(default_factory=_Identity)
    _jitted_infer: Any = field(default=None, init=False, repr=False)
    _jitted_infer_batch: Any = field(default=None, init=False, repr=False)

    def infer(self, obs: Observation) -> jax.Array:
        """Run inference on an observation or batch of observations.

        Automatically detects whether ``obs`` is a single observation or a
        batch based on the array rank.  For dict observations, batching is
        detected on the first value.

        Parameters
        ----------
        obs:
            A single observation or a batch of observations.
            - Array: ``(*obs_shape)`` for single, ``(B, *obs_shape)`` for batch.
            - Dict: each value is ``(*shape)`` for single, ``(B, *shape)`` for batch.

        Returns
        -------
        Action array.  Shape ``(action_dim,)`` for single, ``(B, action_dim,)``
        for batch (or scalar action shapes for discrete policies).
        """
        is_batched = _is_batched(obs)
        if is_batched:
            return self._infer_batch(obs)
        return self._infer_single(obs)

    def _infer_single(self, obs: Observation) -> jax.Array:
        """JIT-compiled single-observation inference."""
        if self._jitted_infer is None:
            object.__setattr__(
                self, "_jitted_infer", jax.jit(self._raw_infer_single)
            )
        return self._jitted_infer(obs)

    def _infer_batch(self, obs: Observation) -> jax.Array:
        """JIT-compiled batch inference via vmap."""
        if self._jitted_infer_batch is None:
            object.__setattr__(
                self, "_jitted_infer_batch", jax.jit(self._raw_infer_batch)
            )
        return self._jitted_infer_batch(obs)

    def _raw_infer_single(self, obs: Observation) -> jax.Array:
        """Single inference: transform -> forward -> untransform."""
        obs = self.input_transform(obs)
        action = self.infer_fn(self.model, obs)
        return self.output_transform(action)

    def _raw_infer_batch(self, obs: Observation) -> jax.Array:
        """Batch inference via vmap over the single path."""
        vmapped = jax.vmap(self._raw_infer_single)
        return vmapped(obs)


def _is_batched(obs: Observation) -> bool:
    """Heuristic: treat 2-D+ flat arrays or 4-D+ images as batched.

    For dict observations, checks the first value.

    Convention:
    - Flat state: single = (D,), batch = (B, D)
    - Image: single = (H, W, C), batch = (B, H, W, C)
    - Dict: check first value
    """
    if isinstance(obs, dict):
        first_val = next(iter(obs.values()))
        return first_val.ndim >= 2
    return obs.ndim >= 2


# ---------------------------------------------------------------------------
# Pre-built input / output transforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizeInput:
    """Z-score normalize state observations using stored statistics.

    For dict observations, applies normalization to the specified key.
    For flat array observations, normalizes directly.

    Parameters
    ----------
    mean:
        Per-feature mean, shape ``(D,)``.
    std:
        Per-feature std, shape ``(D,)``.
    key:
        When working with dict observations, which key to normalize.
        If ``None``, treats the observation as a flat array.
    eps:
        Small constant to prevent division by zero.
    """

    mean: jax.Array
    std: jax.Array
    key: str | None = None
    eps: float = 1e-8

    def __call__(self, obs: Observation) -> Observation:
        if self.key is not None and isinstance(obs, dict):
            obs = dict(obs)
            x = obs[self.key].astype(jnp.float32)
            obs[self.key] = (x - self.mean) / jnp.maximum(self.std, self.eps)
            return obs
        x = obs.astype(jnp.float32)
        return (x - self.mean) / jnp.maximum(self.std, self.eps)


@dataclass(frozen=True)
class UnnormalizeOutput:
    """Denormalize actions from z-score normalized space.

    Parameters
    ----------
    mean:
        Per-feature action mean, shape ``(action_dim,)``.
    std:
        Per-feature action std, shape ``(action_dim,)``.
    eps:
        Small constant to prevent near-zero scaling.
    """

    mean: jax.Array
    std: jax.Array
    eps: float = 1e-8

    def __call__(self, action: jax.Array) -> jax.Array:
        return action * jnp.maximum(self.std, self.eps) + self.mean


@dataclass(frozen=True)
class ResizeImageInput:
    """Resize image observations to a target resolution.

    For dict observations, resizes the specified key.
    For flat array observations, resizes directly.

    Uses JAX-native ``jax.image.resize`` for JIT compatibility.

    Parameters
    ----------
    height:
        Target height.
    width:
        Target width.
    key:
        When working with dict observations, which key to resize.
        If ``None``, treats the observation as an image array.
    """

    height: int
    width: int
    key: str | None = None

    def __call__(self, obs: Observation) -> Observation:
        if self.key is not None and isinstance(obs, dict):
            obs = dict(obs)
            obs[self.key] = self._resize(obs[self.key])
            return obs
        return self._resize(obs)

    def _resize(self, img: jax.Array) -> jax.Array:
        """Resize (H, W, C) image using bilinear interpolation."""
        channels = img.shape[-1]
        return jax.image.resize(
            img.astype(jnp.float32),
            (self.height, self.width, channels),
            method="bilinear",
        )


@dataclass(frozen=True)
class ComposeTransforms:
    """Apply a sequence of transforms in order.

    Parameters
    ----------
    transforms:
        Ordered sequence of transform callables.
    """

    transforms: tuple[Any, ...] = ()

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x
