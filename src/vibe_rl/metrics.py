"""Structured JSONL metrics logger with optional backend integrations.

Writes one JSON object per line to ``logs/metrics.jsonl`` inside a
:class:`~vibe_rl.run_dir.RunDir`. Each line is self-describing — fields
can vary between entries.

Optional backends (WandB, TensorBoard) can be attached to fan-out
every ``write()`` call to external services.

Usage::

    from vibe_rl.run_dir import RunDir
    from vibe_rl.metrics import MetricsLogger

    run = RunDir("dqn_cartpole")
    logger = MetricsLogger(run)
    logger.write({"step": 1000, "loss": 0.42, "reward": 195.0})
    logger.close()

    # With WandB (requires ``pip install wandb``):
    logger = MetricsLogger(path, backends=[WandbBackend(project="rl")])

    # With TensorBoard (requires ``pip install tensorboardX``):
    logger = MetricsLogger(path, backends=[TensorBoardBackend(log_dir)])
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import IO, Any, Protocol

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Backend protocol — implement this to add a new logging sink
# ---------------------------------------------------------------------------


class LogBackend(Protocol):
    """Protocol for optional logging backends (WandB, TensorBoard, etc.)."""

    def log(self, record: dict[str, Any], step: int | None = None) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# WandB backend
# ---------------------------------------------------------------------------


class WandbBackend:
    """Fan-out metrics to Weights & Biases.

    Requires ``pip install wandb``.

    Parameters
    ----------
    project:
        W&B project name.
    entity:
        W&B entity (user or team). ``None`` uses the default.
    config:
        Experiment config dict to log at init time.
    **init_kwargs:
        Extra keyword arguments passed to ``wandb.init()``.
    """

    def __init__(
        self,
        project: str = "vibe_rl",
        entity: str | None = None,
        config: dict[str, Any] | None = None,
        **init_kwargs: Any,
    ) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is required for WandbBackend. "
                "Install it with: pip install wandb"
            ) from None
        self._wandb = wandb
        self._run = wandb.init(
            project=project, entity=entity, config=config, **init_kwargs
        )

    def log(self, record: dict[str, Any], step: int | None = None) -> None:
        self._wandb.log(record, step=step)

    def close(self) -> None:
        self._run.finish()


# ---------------------------------------------------------------------------
# TensorBoard backend
# ---------------------------------------------------------------------------


class TensorBoardBackend:
    """Fan-out scalar metrics to TensorBoard.

    Requires ``pip install tensorboardX`` (or ``tensorboard``).

    Parameters
    ----------
    log_dir:
        Directory for TensorBoard event files.
    """

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    "tensorboardX (or torch.utils.tensorboard) is required "
                    "for TensorBoardBackend. "
                    "Install it with: pip install tensorboardX"
                ) from None
        self._writer = SummaryWriter(str(log_dir))

    def log(self, record: dict[str, Any], step: int | None = None) -> None:
        for k, v in record.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, v, global_step=step)

    def close(self) -> None:
        self._writer.close()


# ---------------------------------------------------------------------------
# Core MetricsLogger
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Append-only JSONL logger with optional backend fan-out.

    Parameters
    ----------
    path:
        Path to the JSONL file.  Parent directories are created
        automatically.
    backends:
        Optional list of :class:`LogBackend` instances (e.g.
        ``WandbBackend``, ``TensorBoardBackend``).  Each ``write()``
        call is forwarded to all backends.
    """

    def __init__(
        self,
        path: str | Path,
        backends: list[LogBackend] | None = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = open(self._path, "a")  # noqa: SIM115
        self._start_time = time.monotonic()
        self._backends: list[LogBackend] = backends or []

    def write(self, record: dict[str, Any]) -> None:
        """Write a single metrics record as one JSON line.

        Automatically adds ``wall_time`` (seconds since logger creation)
        if not already present.  JAX/numpy scalars are converted to
        Python floats.  The record is also forwarded to all attached
        backends.
        """
        row = {k: _to_python(v) for k, v in record.items()}
        if "wall_time" not in row:
            row["wall_time"] = round(time.monotonic() - self._start_time, 3)
        self._file.write(json.dumps(row, default=str) + "\n")
        self._file.flush()

        # Fan-out to backends
        step = row.get("step")
        for backend in self._backends:
            backend.log(row, step=step)

    def close(self) -> None:
        self._file.close()
        for backend in self._backends:
            backend.close()

    @property
    def path(self) -> Path:
        return self._path

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"MetricsLogger({self._path})"


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL metrics file."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _to_python(val: Any) -> Any:
    """Convert JAX/numpy scalars to plain Python types for JSON."""
    if isinstance(val, (jnp.ndarray, np.ndarray)):
        return val.item()
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val
