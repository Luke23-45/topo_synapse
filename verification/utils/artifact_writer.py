"""
Artifact writing utilities for verification experiments.

Provides helpers to persist experiment data as JSONL, CSV, and numpy arrays,
so that each verification script can write its own output for later analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class ArtifactWriter:
    """Manages the output directory for a single experiment and writes data files."""

    def __init__(self, output_dir: str | Path, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        self.root = Path(output_dir) / experiment_id
        self.artifacts = self.root / "artifacts"
        self.metrics = self.root / "metrics"
        self.graphs = self.root / "graphs"
        for d in (self.artifacts, self.metrics, self.graphs):
            d.mkdir(parents=True, exist_ok=True)

    # ── JSONL ──────────────────────────────────────────────────────────

    def write_jsonl(self, filename: str, records: Sequence[dict[str, Any]]) -> Path:
        """Append or create a JSONL file with the given record dicts."""
        path = self.metrics / filename
        with path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, default=_json_default) + "\n")
        return path

    def log_row(self, filename: str, **kwargs: Any) -> Path:
        """Append a single JSONL row (convenience wrapper)."""
        return self.write_jsonl(filename, [kwargs])

    # ── CSV ────────────────────────────────────────────────────────────

    def write_csv(self, filename: str, rows: Sequence[dict[str, Any]]) -> Path:
        """Write rows to a CSV file. Columns are the union of all row keys."""
        if not rows:
            return self.metrics / filename
        path = self.metrics / filename
        fieldnames = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if path.stat().st_size == 0:
                writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_safe(v) for k, v in row.items()})
        return path

    # ── Numpy / graph data ────────────────────────────────────────────

    def save_numpy(self, filename: str, array: np.ndarray) -> Path:
        """Save a numpy array to the graphs directory."""
        path = self.graphs / filename
        np.save(str(path), array)
        return path

    def save_numpy_text(self, filename: str, array: np.ndarray, **kwargs) -> Path:
        """Save a numpy array as a human-readable text file."""
        path = self.graphs / filename
        np.savetxt(str(path), array, **kwargs)
        return path

    # ── JSON (full dump) ──────────────────────────────────────────────

    def save_json(self, filename: str, payload: dict[str, Any]) -> Path:
        """Save a complete JSON payload (e.g. summary/metadata)."""
        path = self.artifacts / filename
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path


# ── Helpers ────────────────────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _csv_safe(value: Any) -> str:
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating,)):
        return str(float(value))
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return str(value)
