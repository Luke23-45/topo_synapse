from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopologyClassificationTask:
    name: str = "synthetic_topology_classification"
    objective: str = "Predict coarse topological class from structured sequences."
