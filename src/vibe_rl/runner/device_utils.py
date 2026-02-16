"""Multi-device utilities for sharding-based data-parallel training.

Provides helpers for replicating state across devices and unreplicating
back to a single copy (e.g. for checkpointing or evaluation).  Works
with both the new ``jit`` + ``NamedSharding`` approach and plain pytrees.

Usage::

    from vibe_rl.runner.device_utils import get_num_devices, replicate, unreplicate

    n_devices = get_num_devices()
    replicated_state = replicate(state, n_devices)
    single_state = unreplicate(replicated_state)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def get_num_devices(requested: int | None = None) -> int:
    """Return the number of devices to use.

    Args:
        requested: Explicit device count. If ``None``, returns the
            number of available JAX devices (GPUs/TPUs/CPUs).

    Returns:
        Number of devices.

    Raises:
        ValueError: If *requested* exceeds available devices.
    """
    available = jax.local_device_count()
    if requested is None:
        return available
    if requested > available:
        raise ValueError(
            f"Requested {requested} devices but only {available} available."
        )
    return requested


def replicate(pytree, n_devices: int):
    """Replicate a pytree across devices by adding a leading axis.

    Each leaf is broadcast to shape ``(n_devices, *original_shape)``.

    Args:
        pytree: Any JAX pytree (NamedTuple, dict, Equinox module, etc.).
        n_devices: Number of devices.

    Returns:
        Replicated pytree with each leaf having an extra leading dimension.
    """
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n_devices, *x.shape)),
        pytree,
    )


def unreplicate(pytree):
    """Take the first replica from a replicated pytree.

    Inverse of :func:`replicate` — removes the leading device dimension
    by taking ``tree[0]``.

    Args:
        pytree: A pytree where each leaf has a leading device dimension.

    Returns:
        Single-device pytree.
    """
    return jax.tree.map(lambda x: x[0], pytree)


def split_key_across_devices(rng, n_devices: int):
    """Split a PRNG key into per-device keys.

    Args:
        rng: A single PRNG key.
        n_devices: Number of devices.

    Returns:
        Array of shape ``(n_devices, 2)`` — one key per device.
    """
    return jax.random.split(rng, n_devices)


def shard_pytree(pytree, mesh: Mesh):
    """Place a pytree on devices with data-sharding on the leading axis.

    The leading dimension of each leaf is split across the mesh's data
    axes ``("batch", "fsdp")``.  Use this for batched data (observations,
    actions, etc.) that should be distributed across devices.

    Args:
        pytree: A pytree whose leaves have a leading batch dimension.
        mesh: The device mesh.

    Returns:
        Pytree with each leaf sharded across the mesh.
    """
    sharding = NamedSharding(mesh, PartitionSpec(("batch", "fsdp")))
    return jax.tree.map(lambda x: jax.device_put(x, sharding), pytree)


def replicate_on_mesh(pytree, mesh: Mesh):
    """Replicate a pytree across all devices in a mesh.

    Every device gets a full copy. Use this for model parameters and
    optimizer state.

    Args:
        pytree: Any JAX pytree.
        mesh: The device mesh.

    Returns:
        Pytree with each leaf replicated across the mesh.
    """
    sharding = NamedSharding(mesh, PartitionSpec())
    return jax.tree.map(lambda x: jax.device_put(x, sharding), pytree)
