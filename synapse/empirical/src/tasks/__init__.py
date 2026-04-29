"""Z3 Evaluation Task Modules.

Task-specific evaluation utilities for the three Z3 tracks:

- **classification** — Standard multi-class accuracy and loss.
- **anomaly_detection** — AUROC/AUPRC for temporal anomaly detection.
- **retrieval** — Nearest-neighbor recall in proxy feature space.
"""

from __future__ import annotations

from .anomaly_detection import AnomalyDetectionResult, evaluate_anomaly_detection
from .topology_classification import TopologyClassificationTask
from .retrieval import RetrievalResult, evaluate_retrieval

__all__ = [
    "AnomalyDetectionResult",
    "RetrievalResult",
    "TopologyClassificationTask",
    "evaluate_anomaly_detection",
    "evaluate_retrieval",
]
