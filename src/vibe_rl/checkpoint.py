"""Orbax-based checkpointing for Equinox models and RL train states.

Provides ``save_checkpoint`` / ``load_checkpoint`` for one-shot saves,
and ``CheckpointManager`` for periodic / best-model checkpointing during
training loops.

All functions work with arbitrary JAX pytrees (NamedTuples, Equinox
modules, optax optimizer states, etc.).

Usage::

    from vibe_rl.checkpoint import save_checkpoint, load_checkpoint

    # One-shot save / load
    save_checkpoint(path, train_state)
    restored = load_checkpoint(path, train_state)

    # Managed checkpointing during training
    from vibe_rl.checkpoint import CheckpointManager, initialize_checkpoint_dir

    mgr, resuming = initialize_checkpoint_dir(
        ckpt_dir, keep_period=500, overwrite=False, resume=True,
    )
    with mgr:
        for step in range(start_step, num_steps):
            state = train_step(state)
            mgr.save(step, state)
        mgr.wait()

Requires the ``checkpoint`` optional dependency::

    pip install vibe-rl[checkpoint]
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, TypeVar

import equinox as eqx
import jax

T = TypeVar("T")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Orbax import — fails gracefully if not installed
# ---------------------------------------------------------------------------

_ocp = None


def _get_ocp():
    global _ocp
    if _ocp is None:
        try:
            import orbax.checkpoint as ocp

            _ocp = ocp
        except ImportError:
            raise ImportError(
                "orbax-checkpoint is required for checkpointing. "
                "Install it with: pip install vibe-rl[checkpoint]"
            ) from None
    return _ocp


# ---------------------------------------------------------------------------
# Low-level: Equinox serialization (no Orbax dependency)
# ---------------------------------------------------------------------------


def save_eqx(path: str | Path, pytree: Any) -> Path:
    """Save a pytree using Equinox's built-in serialization.

    Works with Equinox models, NamedTuples, optax states, and any
    JAX pytree.  Does not require Orbax.

    Parameters
    ----------
    path:
        File path for the checkpoint (conventionally ``*.eqx``).
    pytree:
        The pytree to save.

    Returns
    -------
    The resolved path that was written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(p), pytree)
    return p


def load_eqx(path: str | Path, like: T) -> T:
    """Load a pytree saved with :func:`save_eqx`.

    Parameters
    ----------
    path:
        Path to the checkpoint file.
    like:
        A pytree with the same structure (shapes, dtypes) as the saved
        data.  Typically a freshly-initialized copy of the state.  Use
        ``eqx.filter_eval_shape`` to create a zero-memory skeleton.

    Returns
    -------
    The restored pytree with loaded array data.
    """
    return eqx.tree_deserialise_leaves(str(path), like)


# ---------------------------------------------------------------------------
# High-level: Orbax checkpoint manager
# ---------------------------------------------------------------------------


def save_checkpoint(
    directory: str | Path,
    pytree: Any,
    *,
    step: int | None = None,
    metadata: dict[str, Any] | None = None,
    unreplicate: bool = False,
) -> Path:
    """Save a single checkpoint using Orbax.

    Creates ``directory/`` (or ``directory/step_N/`` if *step* is given)
    containing the serialized pytree and optional JSON metadata.

    Parameters
    ----------
    directory:
        Root directory for the checkpoint.
    pytree:
        JAX pytree to save.
    step:
        If given, creates a ``step_{step}/`` subdirectory.
    metadata:
        Optional dict of JSON-serializable metadata to save alongside.
    unreplicate:
        If ``True``, strip the leading device dimension (take the first
        replica) before saving. Use this when saving pmap-replicated
        state so the checkpoint is device-count agnostic.

    Returns
    -------
    Path to the checkpoint directory.
    """
    if unreplicate:
        pytree = jax.tree.map(lambda x: x[0], pytree)

    d = Path(directory)
    if step is not None:
        d = d / f"step_{step}"
    d.mkdir(parents=True, exist_ok=True)

    # Save pytree with Equinox serialization (most compatible with our
    # Equinox-based models)
    eqx_path = d / "state.eqx"
    eqx.tree_serialise_leaves(str(eqx_path), pytree)

    # Save metadata as JSON
    if metadata is not None:
        meta_path = d / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    return d


def load_checkpoint(
    directory: str | Path,
    like: T,
    *,
    step: int | None = None,
    replicate_to: int | None = None,
) -> T:
    """Load a checkpoint saved with :func:`save_checkpoint`.

    Parameters
    ----------
    directory:
        Root checkpoint directory.
    like:
        A pytree template with the same structure as the saved state.
    step:
        If given, load from ``step_{step}/`` subdirectory.
    replicate_to:
        If given, broadcast each leaf to ``(replicate_to, *shape)``
        to prepare for pmap. Use this to restore a single-device
        checkpoint into a multi-device training setup.

    Returns
    -------
    The restored pytree.
    """
    d = Path(directory)
    if step is not None:
        d = d / f"step_{step}"

    eqx_path = d / "state.eqx"
    restored = eqx.tree_deserialise_leaves(str(eqx_path), like)

    if replicate_to is not None:
        import jax.numpy as jnp

        restored = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (replicate_to, *x.shape)),
            restored,
        )

    return restored


def load_metadata(
    directory: str | Path,
    *,
    step: int | None = None,
) -> dict[str, Any] | None:
    """Load metadata from a checkpoint directory.

    Returns ``None`` if no metadata file exists.
    """
    d = Path(directory)
    if step is not None:
        d = d / f"step_{step}"

    meta_path = d / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


# ---------------------------------------------------------------------------
# initialize_checkpoint_dir — openpi-style directory management
# ---------------------------------------------------------------------------


def initialize_checkpoint_dir(
    directory: str | Path,
    *,
    keep_period: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
    max_to_keep: int = 1,
    save_interval_steps: int = 1,
    async_timeout_secs: int = 7200,
    best_fn: Any | None = None,
    best_mode: str = "min",
) -> tuple[CheckpointManager, bool]:
    """Initialize a checkpoint directory with resume/overwrite semantics.

    Follows the openpi convention:

    - Existing checkpoints + ``resume=True`` -> resume training.
    - Existing checkpoints + no flag -> raise ``FileExistsError``.
    - ``overwrite=True`` -> wipe the directory and start fresh.

    Parameters
    ----------
    directory:
        Root directory for checkpoints.
    keep_period:
        If set, permanently retain every checkpoint whose step is a
        multiple of this value (never pruned by ``max_to_keep``).
    overwrite:
        If ``True``, delete existing checkpoints and start fresh.
    resume:
        If ``True``, allow resuming from existing checkpoints.
    max_to_keep:
        Number of recent checkpoints to retain.
    save_interval_steps:
        Only save when step is a multiple of this value.
    async_timeout_secs:
        Timeout for async checkpoint writes (default 7200s = 2 hours).
    best_fn:
        A function ``metrics -> scalar`` for best-model tracking.
    best_mode:
        ``'min'`` or ``'max'`` for best-model ranking.

    Returns
    -------
    A ``(CheckpointManager, resuming)`` tuple.  ``resuming`` is ``True``
    only when valid checkpoints were found and ``resume=True``.

    Raises
    ------
    FileExistsError:
        If checkpoints exist but neither ``resume`` nor ``overwrite`` is set.
    """
    d = Path(directory).resolve()
    resuming = False

    if d.exists() and any(d.iterdir()):
        if overwrite:
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            logger.info("Wiped checkpoint directory %s", d)
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {d} already exists. "
                f"Pass resume=True to continue training, or "
                f"overwrite=True to start fresh."
            )

    mgr = CheckpointManager(
        d,
        max_to_keep=max_to_keep,
        keep_period=keep_period,
        save_interval_steps=save_interval_steps,
        async_timeout_secs=async_timeout_secs,
        best_fn=best_fn,
        best_mode=best_mode,
    )

    # If the directory existed but has no real checkpoints, don't resume
    if resuming and not mgr.all_steps():
        logger.info(
            "Checkpoint directory exists but contains no checkpoints; "
            "starting fresh."
        )
        resuming = False

    return mgr, resuming


# ---------------------------------------------------------------------------
# CheckpointManager — periodic + best-model checkpointing
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages periodic and best-model checkpointing during training.

    Wraps Orbax's ``CheckpointManager`` with Equinox-friendly
    serialization.  Supports async checkpointing, max-to-keep
    retention, ``keep_period`` for permanent snapshots, and best-model
    tracking.

    Parameters
    ----------
    directory:
        Root directory for all checkpoints.
    max_to_keep:
        Number of recent checkpoints to retain.
    keep_period:
        If set, permanently retain every checkpoint whose step is a
        multiple of this value (e.g. ``keep_period=500`` keeps steps
        500, 1000, 1500, ... forever).
    save_interval_steps:
        If set, only save when ``step`` is a multiple of this value.
    async_timeout_secs:
        Timeout in seconds for async checkpoint writes.  Set to
        ``None`` to disable async checkpointing entirely.
    best_fn:
        A function ``metrics -> scalar`` for best-model tracking.
    best_mode:
        ``'min'`` or ``'max'`` — how to rank the ``best_fn`` output.

    Usage::

        with CheckpointManager(ckpt_dir, max_to_keep=5, keep_period=500) as mgr:
            for step in range(num_steps):
                state = train_step(state)
                mgr.save(step, state, metrics={"loss": float(loss)})
            mgr.wait()
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        max_to_keep: int = 5,
        keep_period: int | None = None,
        save_interval_steps: int = 1,
        async_timeout_secs: int | None = 7200,
        best_fn: Any | None = None,
        best_mode: str = "min",
    ) -> None:
        ocp = _get_ocp()

        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._save_interval = save_interval_steps
        self._max_to_keep = max_to_keep
        self._keep_period = keep_period

        # Build async options
        async_options = None
        if async_timeout_secs is not None:
            async_options = ocp.AsyncOptions(timeout_secs=async_timeout_secs)

        options = ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            async_options=async_options,
            best_fn=best_fn,
            best_mode=best_mode,
            keep_checkpoints_without_metrics=(best_fn is not None),
        )

        self._mgr = ocp.CheckpointManager(
            self._directory,
            options=options,
        )

    def save(
        self,
        step: int,
        pytree: Any,
        *,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Save a checkpoint at *step*.

        The manager respects ``save_interval_steps`` — if *step* is not
        on the interval, this is a no-op and returns ``False``.

        Parameters
        ----------
        step:
            Training step number.
        pytree:
            State pytree to checkpoint.
        metrics:
            Optional metrics dict for best-model ranking.
        metadata:
            Optional JSON metadata saved alongside the pytree.

        Returns
        -------
        ``True`` if a checkpoint was actually written.
        """
        ocp = _get_ocp()

        # Use Orbax StandardSave for the pytree
        # Convert Equinox modules to a plain pytree of arrays
        arrays, treedef = jax.tree.flatten(pytree)
        state_dict = {f"leaf_{i}": a for i, a in enumerate(arrays)}

        saved = self._mgr.save(
            step,
            args=ocp.args.StandardSave(state_dict),
            metrics=metrics,
        )

        # Save metadata as a sidecar JSON if requested
        if saved and metadata is not None:
            ckpt_dir = self._directory / str(step)
            if ckpt_dir.exists():
                meta_path = ckpt_dir / "metadata.json"
                meta_path.write_text(
                    json.dumps(metadata, indent=2, default=str) + "\n"
                )

        return saved

    def restore(self, step: int | None, like: T) -> T:
        """Restore a checkpoint.

        Parameters
        ----------
        step:
            Step to restore.  If ``None``, restores the latest.
        like:
            Template pytree for structure / shapes / dtypes.

        Returns
        -------
        The restored pytree.
        """
        ocp = _get_ocp()

        if step is None:
            step = self._mgr.latest_step()
            if step is None:
                raise FileNotFoundError(
                    f"No checkpoints found in {self._directory}"
                )

        arrays, treedef = jax.tree.flatten(like)
        abstract_dict = {
            f"leaf_{i}": jax.ShapeDtypeStruct(a.shape, a.dtype)
            if hasattr(a, "shape")
            else a
            for i, a in enumerate(arrays)
        }

        restored_dict = self._mgr.restore(
            step,
            args=ocp.args.StandardRestore(abstract_dict),
        )

        restored_leaves = [restored_dict[f"leaf_{i}"] for i in range(len(arrays))]
        return jax.tree.unflatten(treedef, restored_leaves)

    def latest_step(self) -> int | None:
        """Return the latest checkpointed step, or ``None``."""
        return self._mgr.latest_step()

    def all_steps(self) -> list[int]:
        """Return all available checkpoint steps."""
        return sorted(self._mgr.all_steps())

    def wait(self) -> None:
        """Block until any async saves complete."""
        self._mgr.wait_until_finished()

    def close(self) -> None:
        """Wait for async saves and close the manager."""
        self._mgr.wait_until_finished()
        self._mgr.close()

    def __enter__(self) -> CheckpointManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"CheckpointManager({self._directory}, "
            f"max_to_keep={self._max_to_keep}, "
            f"keep_period={self._keep_period})"
        )
