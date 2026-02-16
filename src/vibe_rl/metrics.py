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

    # Resume a WandB run after interruption:
    backend = resume_wandb(run.root, project="rl")
    logger = MetricsLogger(path, backends=[backend])
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import IO, Any, Protocol

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Backend protocol — implement this to add a new logging sink
# ---------------------------------------------------------------------------

_WANDB_ID_FILE = "wandb_id.txt"


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
    run_dir:
        If provided, the W&B run ID is persisted to
        ``<run_dir>/wandb_id.txt`` so that training can be resumed
        after interruption with :func:`resume_wandb`.
    **init_kwargs:
        Extra keyword arguments passed to ``wandb.init()``.
    """

    def __init__(
        self,
        project: str = "vibe_rl",
        entity: str | None = None,
        config: dict[str, Any] | None = None,
        run_dir: str | Path | None = None,
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
        if run_dir is not None:
            _save_wandb_id(run_dir, self._run.id)

    @property
    def run_id(self) -> str:
        """The W&B run ID for this backend."""
        return self._run.id

    def log(self, record: dict[str, Any], step: int | None = None) -> None:
        self._wandb.log(record, step=step)

    def close(self) -> None:
        self._run.finish()


def _save_wandb_id(run_dir: str | Path, run_id: str) -> Path:
    """Persist *run_id* to ``<run_dir>/wandb_id.txt``."""
    p = Path(run_dir) / _WANDB_ID_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(run_id + "\n")
    return p


def _load_wandb_id(run_dir: str | Path) -> str | None:
    """Read a previously saved W&B run ID, or ``None``."""
    p = Path(run_dir) / _WANDB_ID_FILE
    if not p.exists():
        return None
    text = p.read_text().strip()
    return text or None


def resume_wandb(
    run_dir: str | Path,
    *,
    project: str = "vibe_rl",
    entity: str | None = None,
    config: dict[str, Any] | None = None,
    **init_kwargs: Any,
) -> WandbBackend:
    """Create a :class:`WandbBackend` that resumes a previous W&B run.

    Reads the run ID from ``<run_dir>/wandb_id.txt`` (written
    automatically when ``run_dir`` is passed to :class:`WandbBackend`).
    If no ID file exists a fresh run is started instead.

    Parameters
    ----------
    run_dir:
        The experiment's root directory (e.g. ``RunDir.root``).
    project, entity, config, **init_kwargs:
        Forwarded to :class:`WandbBackend` / ``wandb.init()``.

    Returns
    -------
    WandbBackend
        A backend instance attached to the resumed (or new) run.
    """
    wandb_id = _load_wandb_id(run_dir)
    if wandb_id is not None:
        init_kwargs.setdefault("id", wandb_id)
        init_kwargs.setdefault("resume", "must")
    return WandbBackend(
        project=project,
        entity=entity,
        config=config,
        run_dir=run_dir,
        **init_kwargs,
    )


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
# Structured console logging
# ---------------------------------------------------------------------------

_LEVEL_ABBREV = {
    logging.DEBUG: "D",
    logging.INFO: "I",
    logging.WARNING: "W",
    logging.ERROR: "E",
    logging.CRITICAL: "C",
}


class _TrainFormatter(logging.Formatter):
    """Compact formatter: abbreviated level + millisecond timestamp.

    Example output::

        I 2026-02-15 14:30:22.123 [vibe_rl] Starting training
    """

    def format(self, record: logging.LogRecord) -> str:
        lvl = _LEVEL_ABBREV.get(record.levelno, "?")
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        ms = int(record.msecs)
        msg = record.getMessage()
        return f"{lvl} {ts}.{ms:03d} [{record.name}] {msg}"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the ``vibe_rl`` logger with compact formatting.

    Installs a :class:`~logging.StreamHandler` on the ``"vibe_rl"``
    logger with abbreviated level names (D/I/W/E/C) and millisecond
    timestamps.  Safe to call multiple times — existing handlers are
    replaced.
    """
    logger = logging.getLogger("vibe_rl")
    logger.setLevel(level)

    # Remove previous handlers to avoid duplicates on repeated calls.
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(_TrainFormatter())
    logger.addHandler(handler)
    logger.propagate = False


def log_step_progress(
    step: int,
    total_steps: int,
    metrics: dict[str, Any] | None = None,
    logger_name: str = "vibe_rl",
) -> None:
    """Log a one-line training progress message.

    Example output::

        I 2026-02-15 14:30:22.123 [vibe_rl] step 5000/100000 (5.0%) | loss=0.42 reward=195.0

    Parameters
    ----------
    step:
        Current training step.
    total_steps:
        Total planned training steps.
    metrics:
        Optional dict of scalar metrics to append.
    logger_name:
        Logger name (default ``"vibe_rl"``).
    """
    pct = 100.0 * step / total_steps if total_steps > 0 else 0.0
    parts = [f"step {step}/{total_steps} ({pct:.1f}%)"]
    if metrics:
        kv = " ".join(
            f"{k}={_to_python(v):.4g}" if isinstance(v, float) else f"{k}={_to_python(v)}"
            for k, v in metrics.items()
            if k not in ("step", "wall_time")
        )
        if kv:
            parts.append(kv)
    logging.getLogger(logger_name).info(" | ".join(parts))


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
