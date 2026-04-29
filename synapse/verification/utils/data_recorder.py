from __future__ import annotations

from pathlib import Path
from typing import Any

from synapse.verification.utils.artifact_writer import ArtifactWriter


class VerificationRecorder:
    """Thin convenience wrapper around ArtifactWriter for scalar logging."""

    def __init__(self, root: str | Path, experiment_id: str = "") -> None:
        if experiment_id:
            self._writer = ArtifactWriter(root, experiment_id)
        else:
            root_path = Path(root)
            experiment_id = root_path.name
            self._writer = ArtifactWriter(root_path.parent, experiment_id)

    @property
    def writer(self) -> ArtifactWriter:
        return self._writer

    def log_scalar(self, **kwargs: Any) -> None:
        self._writer.log_row("scalars.jsonl", **kwargs)

    def log_csv_row(self, **kwargs: Any) -> None:
        self._writer.write_csv("scalars.csv", [kwargs])
