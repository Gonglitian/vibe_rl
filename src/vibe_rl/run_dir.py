"""Unified output directory management for training runs.

Provides ``RunDir``, a lightweight handle that creates and exposes the
standard directory layout for a single experiment run::

    runs/
    └── dqn_cartpole_20260215_143022/
        ├── config.json
        ├── checkpoints/
        │   ├── step_10000/
        │   ├── step_20000/
        │   └── best -> step_20000
        ├── logs/
        │   ├── metrics.jsonl
        │   └── events.out.*        (TensorBoard, optional)
        ├── videos/
        │   ├── eval_step_10000.mp4
        │   └── eval_step_20000.mp4
        └── artifacts/
            └── final_report.json

Usage::

    run = RunDir("dqn_cartpole", base_dir="runs")
    run.save_config(config)          # snapshot frozen dataclass / dict
    run.checkpoints / "step_10000"   # Path for orbax checkpoint
    run.log_path("metrics.jsonl")    # Path inside logs/
    run.video_path(step=10000)       # videos/eval_step_10000.mp4
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


class RunDir:
    """A lightweight handle for a single experiment's output directory.

    Parameters
    ----------
    experiment_name:
        Human-readable name (e.g. ``"dqn_cartpole"``). Combined with a
        UTC timestamp to form the run directory name.
    base_dir:
        Parent directory for all runs. Defaults to ``"runs"``.
    run_id:
        Explicit run directory name, bypassing auto-generation. Useful
        for resuming or testing with deterministic paths.
    """

    def __init__(
        self,
        experiment_name: str = "default",
        base_dir: str | Path = "runs",
        *,
        run_id: str | None = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        dirname = run_id if run_id is not None else f"{experiment_name}_{_timestamp()}"
        self._root = self._base_dir / dirname

        # Create the standard subdirectories eagerly so downstream code
        # can rely on them existing.
        for subdir in ("checkpoints", "logs", "videos", "artifacts"):
            (self._root / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core path properties
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Top-level run directory."""
        return self._root

    @property
    def checkpoints(self) -> Path:
        """Directory for orbax / torch checkpoints."""
        return self._root / "checkpoints"

    @property
    def logs(self) -> Path:
        """Directory for metric logs (JSONL, TensorBoard, etc.)."""
        return self._root / "logs"

    @property
    def videos(self) -> Path:
        """Directory for evaluation video recordings."""
        return self._root / "videos"

    @property
    def artifacts(self) -> Path:
        """Directory for miscellaneous outputs (reports, plots, etc.)."""
        return self._root / "artifacts"

    # ------------------------------------------------------------------
    # Convenience path builders
    # ------------------------------------------------------------------

    def checkpoint_dir(self, step: int) -> Path:
        """Return ``checkpoints/step_{step}/``, creating it."""
        p = self.checkpoints / f"step_{step}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def log_path(self, filename: str = "metrics.jsonl") -> Path:
        """Return a path inside ``logs/``."""
        return self.logs / filename

    def video_path(self, step: int, prefix: str = "eval") -> Path:
        """Return ``videos/{prefix}_step_{step}.mp4``."""
        return self.videos / f"{prefix}_step_{step}.mp4"

    def artifact_path(self, filename: str) -> Path:
        """Return a path inside ``artifacts/``."""
        return self.artifacts / filename

    # ------------------------------------------------------------------
    # Config snapshot
    # ------------------------------------------------------------------

    def save_config(self, config: Any, filename: str = "config.json") -> Path:
        """Serialize *config* to JSON in the run root.

        Accepts frozen dataclasses, plain dicts, or any object with a
        ``to_dict()`` method.

        Returns the path to the written file.
        """
        path = self._root / filename
        data = _config_to_dict(config)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
        return path

    def load_config(self, filename: str = "config.json") -> dict[str, Any]:
        """Load a previously saved config snapshot."""
        path = self._root / filename
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # Best-checkpoint symlink
    # ------------------------------------------------------------------

    def mark_best(self, step: int) -> Path:
        """Point ``checkpoints/best`` symlink at ``step_{step}/``.

        Returns the symlink path.
        """
        link = self.checkpoints / "best"
        target = f"step_{step}"
        # Atomically replace: write to a temp name then rename.
        tmp_link = link.with_suffix(".tmp")
        tmp_link.unlink(missing_ok=True)
        tmp_link.symlink_to(target)
        tmp_link.rename(link)
        return link

    @property
    def best_checkpoint(self) -> Path | None:
        """Resolve the ``best`` symlink, or *None* if it doesn't exist."""
        link = self.checkpoints / "best"
        if link.is_symlink():
            return link.resolve()
        return None

    # ------------------------------------------------------------------
    # Checkpoint cleanup
    # ------------------------------------------------------------------

    def cleanup_checkpoints(self, keep: int = 5) -> list[Path]:
        """Keep only the *keep* most recent step checkpoints.

        The ``best`` symlink target is always preserved regardless of
        *keep*.  Returns the list of removed directories.
        """
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")

        # Collect step_* directories, sorted ascending by step number.
        step_dirs: list[tuple[int, Path]] = []
        for entry in self.checkpoints.iterdir():
            if entry.is_dir() and entry.name.startswith("step_"):
                try:
                    step_num = int(entry.name.split("_", 1)[1])
                    step_dirs.append((step_num, entry))
                except ValueError:
                    continue
        step_dirs.sort()

        # Resolve best target so we never delete it.
        best_target: Path | None = None
        best_link = self.checkpoints / "best"
        if best_link.is_symlink():
            best_target = best_link.resolve()

        # Keep the last *keep* plus the best target.
        to_keep = {p for _, p in step_dirs[-keep:]}
        if best_target is not None and best_target.is_dir():
            to_keep.add(best_target)

        removed: list[Path] = []
        for _, p in step_dirs:
            if p not in to_keep:
                shutil.rmtree(p)
                removed.append(p)

        return removed

    # ------------------------------------------------------------------
    # Listing / discovery
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """Return sorted ``(step, path)`` pairs for all step checkpoints."""
        result: list[tuple[int, Path]] = []
        for entry in self.checkpoints.iterdir():
            if entry.is_dir() and entry.name.startswith("step_"):
                try:
                    step_num = int(entry.name.split("_", 1)[1])
                    result.append((step_num, entry))
                except ValueError:
                    continue
        result.sort()
        return result

    @property
    def latest_checkpoint(self) -> Path | None:
        """Path to the highest-step checkpoint, or *None*."""
        ckpts = self.list_checkpoints()
        return ckpts[-1][1] if ckpts else None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"RunDir({self._root})"

    def __str__(self) -> str:
        return str(self._root)

    def __fspath__(self) -> str:
        return str(self._root)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _config_to_dict(obj: Any) -> dict[str, Any]:
    """Recursively convert a config object to a plain dict."""
    if isinstance(obj, dict):
        return {k: _config_to_dict(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _config_to_dict(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_config_to_dict(v) for v in obj)
    return obj


def find_runs(
    base_dir: str | Path = "runs",
    experiment_name: str | None = None,
) -> list[RunDir]:
    """Discover existing run directories under *base_dir*.

    Returns a list of ``RunDir`` handles sorted by directory name
    (i.e. chronological if using default timestamp naming).

    If *experiment_name* is given, only runs whose directory name starts
    with ``{experiment_name}_`` are returned.
    """
    base = Path(base_dir)
    if not base.is_dir():
        return []

    dirs: list[str] = sorted(
        d.name
        for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if experiment_name is not None:
        prefix = f"{experiment_name}_"
        dirs = [d for d in dirs if d.startswith(prefix)]

    return [RunDir(base_dir=base_dir, run_id=d) for d in dirs]
