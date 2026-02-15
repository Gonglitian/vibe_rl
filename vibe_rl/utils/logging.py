from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any


class Logger:
    """
    Dual console + CSV logger. Prints metrics to stdout and writes
    every row to a CSV file for later plotting.
    """

    def __init__(self, log_dir: Path | str, filename: str = "progress.csv") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.log_dir / filename
        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer: csv.DictWriter | None = None
        self._start_time = time.time()
        self._episode = 0

    def log_episode(self, metrics: dict[str, Any]) -> None:
        self._episode += 1
        metrics["episode"] = self._episode
        metrics["elapsed_s"] = round(time.time() - self._start_time, 1)

        if self._writer is None:
            self._writer = csv.DictWriter(
                self._csv_file, fieldnames=list(metrics.keys()), extrasaction="ignore"
            )
            self._writer.writeheader()

        # If new keys appear, restart the writer with updated fieldnames
        new_keys = set(metrics.keys()) - set(self._writer.fieldnames)
        if new_keys:
            self._csv_file.close()
            # Re-read existing rows, rewrite with expanded header
            rows = []
            if self._csv_path.exists():
                with open(self._csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            fieldnames = list(self._writer.fieldnames) + sorted(new_keys)
            self._csv_file = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(
                self._csv_file, fieldnames=fieldnames, extrasaction="ignore"
            )
            self._writer.writeheader()
            for row in rows:
                self._writer.writerow(row)

        self._writer.writerow(metrics)
        self._csv_file.flush()

        parts = [
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ]
        print(" | ".join(parts), file=sys.stdout)

    def close(self) -> None:
        self._csv_file.close()
