"""Mesh and sharding utilities for data-parallel and FSDP training.

Provides a 2-D device mesh ``(batch, fsdp)`` and helpers for placing
data and parameters on the correct devices.  When ``fsdp_devices=1``
the mesh collapses to pure data-parallel, equivalent to the old
``pmap``-based approach.

The mesh is created once and threaded through the training loop via
the ``set_mesh`` context manager or explicit ``make_mesh`` call.

Usage::

    from vibe_rl.sharding import make_mesh, data_sharding, replicate_sharding

    mesh = make_mesh()  # pure data-parallel by default
    data_spec = data_sharding(mesh)       # shard leading axis over batch
    param_spec = replicate_sharding(mesh) # replicate everywhere

    # FSDP: per-parameter sharding based on size/shape
    param_shardings = fsdp_sharding(params, mesh)

Reference: openpi/src/openpi/training/sharding.py
"""

from __future__ import annotations

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)


def make_mesh(
    num_fsdp_devices: int = 1,
    num_devices: int | None = None,
) -> Mesh:
    """Create a 2-D device mesh ``(batch, fsdp)``.

    When ``num_fsdp_devices=1`` the fsdp axis is trivial and the mesh
    is equivalent to pure data-parallelism across all devices.

    Args:
        num_fsdp_devices: Number of devices for the FSDP axis.
            Must evenly divide *num_devices*.
        num_devices: Total number of devices to include in the mesh.
            If ``None``, uses all available devices.

    Returns:
        A ``jax.sharding.Mesh`` with axes ``("batch", "fsdp")``.

    Raises:
        ValueError: If *num_devices* is not divisible by *num_fsdp_devices*,
            or exceeds the number of available devices.
    """
    all_devices = jax.devices()
    if num_devices is None:
        num_devices = len(all_devices)
    if num_devices > len(all_devices):
        raise ValueError(
            f"Requested {num_devices} devices but only "
            f"{len(all_devices)} available."
        )
    devices = all_devices[:num_devices]

    if num_devices % num_fsdp_devices != 0:
        raise ValueError(
            f"num_devices ({num_devices}) must be divisible by "
            f"num_fsdp_devices ({num_fsdp_devices})."
        )

    n_batch = num_devices // num_fsdp_devices
    device_grid = np.array(devices).reshape(n_batch, num_fsdp_devices)
    return Mesh(device_grid, axis_names=(BATCH_AXIS, FSDP_AXIS))


@contextmanager
def set_mesh(num_fsdp_devices: int = 1, num_devices: int | None = None):
    """Context manager that creates and activates a device mesh.

    Usage::

        with set_mesh(num_fsdp_devices=1) as mesh:
            jitted_fn = jax.jit(fn, in_shardings=..., out_shardings=...)
            result = jitted_fn(data)
    """
    mesh = make_mesh(num_fsdp_devices, num_devices=num_devices)
    yield mesh


def data_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding spec for data: leading axis partitioned over ``(batch, fsdp)``.

    For a tensor of shape ``(N, ...)``, the leading dimension is split
    across both mesh axes.
    """
    return NamedSharding(mesh, PartitionSpec(DATA_AXIS))


def replicate_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding spec for replicated data (e.g. model parameters).

    Every device gets a full copy.
    """
    return NamedSharding(mesh, PartitionSpec())


def fsdp_sharding(pytree, mesh: Mesh, *, min_size_mbytes: float = 4) -> object:
    """Compute per-parameter FSDP sharding for a pytree.

    Decides for each array in *pytree* whether to shard it across the
    ``fsdp`` mesh axis or fully replicate, based on size and shape:

    - Arrays < *min_size_mbytes* → replicated (communication overhead
      exceeds the memory savings from sharding).
    - Scalars and 1-D arrays → replicated.
    - 2-D+ arrays >= threshold → sharded along the largest dimension
      that is evenly divisible by the FSDP axis size.
    - If no dimension is divisible → replicated (fallback).

    When ``mesh.shape[FSDP_AXIS] == 1`` every parameter is replicated,
    making the result equivalent to pure data-parallelism.

    Args:
        pytree: A JAX pytree of arrays (e.g. model params from
            ``jax.eval_shape``).  May contain ``jax.ShapeDtypeStruct``
            or concrete arrays.
        mesh: The device mesh (must have an ``"fsdp"`` axis).
        min_size_mbytes: Minimum array size in MB to consider for
            sharding.  Default is 4 MB.

    Returns:
        A pytree with the same structure as *pytree*, where each leaf
        is replaced by a ``NamedSharding`` specifying how that
        parameter should be distributed.
    """
    fsdp_size = mesh.shape[FSDP_AXIS]

    def _get_sharding(array):
        # Size in MB (works for both concrete arrays and ShapeDtypeStruct)
        size_mb = np.prod(array.shape) * np.dtype(array.dtype).itemsize / (1024 * 1024)

        # Rule 1: small arrays → replicate
        if size_mb < min_size_mbytes:
            return NamedSharding(mesh, PartitionSpec())

        # Rule 2: scalars and 1-D → replicate
        if len(array.shape) < 2:
            return NamedSharding(mesh, PartitionSpec())

        # Rule 3: shard along the largest divisible dimension
        axes_by_size = np.argsort(array.shape)[::-1]  # largest dim first
        spec = [None] * len(array.shape)
        for i in axes_by_size:
            if array.shape[i] % fsdp_size == 0:
                spec[i] = FSDP_AXIS
                return NamedSharding(mesh, PartitionSpec(*spec))

        # Fallback: replicate
        return NamedSharding(mesh, PartitionSpec())

    return jax.tree.map(_get_sharding, pytree)


def activation_sharding_constraint(x, mesh: Mesh):
    """Apply a sharding constraint to an activation tensor.

    Places a ``with_sharding_constraint`` on *x* so that the leading
    axis is sharded over the data axes.  Use inside ``jit``-compiled
    functions to guide the compiler's sharding decisions.
    """
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, PartitionSpec(DATA_AXIS))
    )
