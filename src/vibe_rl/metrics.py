"""Structured JSONL metrics logger.

Writes one JSON object per line to ``logs/metrics.jsonl`` inside a
:class:`~vibe_rl.run_dir.RunDir`. Each line is self-describing — fields
can vary between entries.

Usage::

    from vibe_rl.run_dir import RunDir
    from vibe_rl.metrics import MetricsLogger

    run = RunDir("dqn_cartpole")
    logger = MetricsLogger(run)
    logger.write({"step": 1000, "loss": 0.42, "reward": 195.0})
    logger.close()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import IO, Any

import jax.numpy as jnp
import numpy as np


class MetricsLogger:
    """Append-only JSONL logger.

    Parameters
    ----------
    path:
        Path to the JSONL file.  Parent directories are created
        automatically.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = open(self._path, "a")  # noqa: SIM115
        self._start_time = time.monotonic()

    def write(self, record: dict[str, Any]) -> None:
        """Write a single metrics record as one JSON line.

        Automatically adds ``wall_time`` (seconds since logger creation)
        if not already present.  JAX/numpy scalars are converted to
        Python floats.
        """
        row = {k: _to_python(v) for k, v in record.items()}
        if "wall_time" not in row:
            row["wall_time"] = round(time.monotonic() - self._start_time, 3)
        self._file.write(json.dumps(row, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

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
