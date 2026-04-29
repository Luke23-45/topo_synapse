"""
Exact Topology — Z3 Reference: §9 of 01_main_definition.md

Wraps the exact persistent homology computation (Vietoris-Rips filtration)
and diagram summarization. This is the deployment-time exact topological
object T^exact_Θ, NOT the differentiable proxy.

Delegates to synapse_core.topological_summary for the actual Gudhi/Ripser
computation.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


PersistenceDiagram = List[Tuple[float, float]]


def compute_persistence_diagrams(
    cloud: np.ndarray,
    Q: int,
    max_edge_length: float | None = None,
) -> List[PersistenceDiagram]:
    """Compute exact persistence diagrams via Vietoris-Rips filtration.

    Delegates to synapse_core.topological_summary which uses Gudhi/Ripser.
    """
    from .topological_summary import compute_persistence_diagrams as _compute
    return _compute(cloud, Q, max_edge_length=max_edge_length)


def hausdorff_distance(P: np.ndarray, Q_cloud: np.ndarray) -> float:
    """Compute Hausdorff distance between two point clouds."""
    from .topological_summary import hausdorff_distance as _hausdorff
    return _hausdorff(P, Q_cloud)


def summarize_diagrams(diagrams: List[Any]) -> np.ndarray:
    """Summarize persistence diagrams into a fixed-size feature vector.

    For each degree q, extracts:
        [count, mean_persistence, max_persistence, total_persistence]
    """
    summary: list[float] = []
    for dgm in diagrams:
        points = np.asarray(dgm, dtype=np.float64)
        if points.size == 0:
            summary.extend([0.0, 0.0, 0.0, 0.0])
            continue
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        finite = points[np.isfinite(points[:, 1])]
        if finite.size == 0:
            summary.extend([0.0, 0.0, 0.0, 0.0])
            continue
        persistence = finite[:, 1] - finite[:, 0]
        summary.extend([
            float(len(finite)),
            float(np.mean(persistence)),
            float(np.max(persistence)),
            float(np.sum(persistence)),
        ])
    return np.asarray(summary, dtype=np.float32)


__all__ = [
    "compute_persistence_diagrams",
    "hausdorff_distance",
    "summarize_diagrams",
]
