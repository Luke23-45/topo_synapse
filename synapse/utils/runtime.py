from __future__ import annotations

from pathlib import Path


def build_run_artifacts(output_dir: str | Path):
    root = Path(output_dir)
    return {
        "root": root,
        "checkpoint": root / "checkpoints" / "best.pt",
        "history": root / "training_history.json",
        "final_metrics": root / "final_metrics.json",
        "analysis": root / "analysis",
    }
