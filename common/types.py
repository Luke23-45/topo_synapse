from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class SynapseSample:
    sequence: np.ndarray
    label: int | float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SequenceBatch:
    sequences: torch.Tensor
    lengths: torch.Tensor
    targets: torch.Tensor


@dataclass
class ExactTopologyAudit:
    anchor_indices: list[int]
    anchor_vectors: np.ndarray
    normalized_anchor_vectors: np.ndarray
    point_cloud: np.ndarray
    persistence_diagrams: list[list[tuple[float, float]]]
    topology_summary: np.ndarray
    event_scores: np.ndarray
    saliency_scores: np.ndarray
    y_star: np.ndarray


@dataclass
class ProxyComputation:
    dense_vectors: torch.Tensor
    dense_lifted_cloud: torch.Tensor
    y_star: torch.Tensor
    proxy_features: torch.Tensor
    event_scores: torch.Tensor
    saliency_scores: torch.Tensor


@dataclass
class TrainingArtifacts:
    output_dir: str
    best_checkpoint: str
    metrics_path: str
