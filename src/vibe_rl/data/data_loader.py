"""JAX-compatible DataLoader backed by torch DataLoader.

Uses torch's multi-process data loading machinery (num_workers, prefetch)
and converts each batch to JAX arrays.  Supports sharding across
multiple JAX devices for distributed training.

Usage::

    from vibe_rl.data import LeRobotDatasetAdapter, JaxDataLoader

    ds = LeRobotDatasetAdapter("lerobot/aloha_sim_insertion_human")
    loader = JaxDataLoader(ds, batch_size=64, num_workers=4)
    for batch in loader:
        # batch is a Transition with jax arrays of shape (B, ...)
        train_step(state, batch)

Sharding::

    import jax

    devices = jax.devices()
    loader = JaxDataLoader(ds, batch_size=64, num_workers=4, devices=devices)
    for batch in loader:
        # batch.obs has shape (num_devices, B // num_devices, ...)
        pmap_train_step(state, batch)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from vibe_rl.dataprotocol.transition import Transition

if TYPE_CHECKING:
    pass


def _require_torch():
    """Import torch with a clear error message."""
    try:
        import torch.utils.data  # noqa: F401 — verify real torch, not namespace stub

        return torch
    except (ImportError, AttributeError) as e:
        raise ImportError(
            "PyTorch is required for the DataLoader. "
            "Install with: pip install 'vibe-rl[data]'"
        ) from e


def _transition_collate_fn(samples: list[Transition]) -> Transition:
    """Collate a list of Transitions into a batched Transition of numpy arrays.

    This runs in the torch DataLoader worker process. We stack into
    numpy arrays (not jax) to avoid JAX device transfers in subprocesses.
    """
    return Transition(
        obs=np.stack([np.asarray(s.obs) for s in samples]),
        action=np.stack([np.asarray(s.action) for s in samples]),
        reward=np.stack([np.asarray(s.reward) for s in samples]),
        next_obs=np.stack([np.asarray(s.next_obs) for s in samples]),
        done=np.stack([np.asarray(s.done) for s in samples]),
    )


def _numpy_to_jax(batch: Transition) -> Transition:
    """Convert a Transition of numpy arrays to jax arrays."""
    return Transition(
        obs=jnp.asarray(batch.obs),
        action=jnp.asarray(batch.action),
        reward=jnp.asarray(batch.reward),
        next_obs=jnp.asarray(batch.next_obs),
        done=jnp.asarray(batch.done),
    )


def _shard_batch(
    batch: Transition, devices: list[jax.Device]
) -> Transition:
    """Reshape and shard a batch across devices.

    Input shape:  ``(B, ...)``
    Output shape: ``(num_devices, B // num_devices, ...)``

    Each shard is placed on the corresponding device via ``jax.device_put``.
    """
    n = len(devices)
    mesh = jax.sharding.Mesh(np.array(devices), axis_names=("device",))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("device")
    )

    def _shard_array(arr: jnp.ndarray) -> jnp.ndarray:
        leading = arr.shape[0]
        if leading % n != 0:
            raise ValueError(
                f"Batch size {leading} is not divisible by "
                f"number of devices {n}."
            )
        reshaped = arr.reshape(n, leading // n, *arr.shape[1:])
        return jax.device_put(reshaped, sharding)

    return Transition(
        obs=_shard_array(batch.obs),
        action=_shard_array(batch.action),
        reward=_shard_array(batch.reward),
        next_obs=_shard_array(batch.next_obs),
        done=_shard_array(batch.done),
    )


class JaxDataLoader:
    """Multi-process DataLoader that yields JAX ``Transition`` batches.

    Wraps ``torch.utils.data.DataLoader`` for multi-process prefetching,
    then converts each batch to jax arrays on the main process.

    Parameters:
        dataset: Any object implementing ``__len__`` and ``__getitem__``
            returning ``Transition`` (e.g. ``LeRobotDatasetAdapter``).
        batch_size: Number of transitions per batch.
        num_workers: Number of parallel data-loading workers (0 = main process).
        shuffle: Whether to shuffle indices each epoch.
        drop_last: Drop the last incomplete batch if the dataset size
            is not divisible by ``batch_size``.
        devices: List of JAX devices for sharding. When provided, each
            batch is reshaped to ``(num_devices, B // num_devices, ...)``
            and placed on the corresponding device. ``batch_size`` must
            be divisible by ``len(devices)``.
        prefetch_factor: Number of batches to prefetch per worker.
            Only used when ``num_workers > 0``.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 256,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        devices: list[jax.Device] | None = None,
        prefetch_factor: int | None = 2,
    ) -> None:
        torch = _require_torch()

        self.batch_size = batch_size
        self.devices = devices

        loader_kwargs: dict = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "num_workers": num_workers,
            "collate_fn": _transition_collate_fn,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        self._loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    def __iter__(self) -> Iterator[Transition]:
        for np_batch in self._loader:
            jax_batch = _numpy_to_jax(np_batch)
            if self.devices is not None:
                jax_batch = _shard_batch(jax_batch, self.devices)
            yield jax_batch

    def __len__(self) -> int:
        return len(self._loader)
