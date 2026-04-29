"""Anomaly Detection Task Module.

Provides evaluation utilities for anomaly / out-of-distribution detection
on temporal datasets (e.g. TelecomTS).  The Z3 model's proxy spectral
signature shift is used to detect distributional anomalies.

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §3, Track 1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AnomalyDetectionResult:
    """Result of an anomaly detection evaluation.

    Attributes
    ----------
    auroc : float
        Area under the ROC curve.
    auprc : float
        Area under the precision-recall curve.
    threshold : float
        Optimal decision threshold (maximizing F1).
    f1 : float
        F1 score at the optimal threshold.
    """
    auroc: float
    auprc: float
    threshold: float
    f1: float


def compute_proxy_anomaly_scores(
    model: torch.nn.Module,
    sequences: np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Compute per-sample anomaly scores from proxy spectral deviation.

    The anomaly score is the L2 distance between a sample's proxy
    features and the mean proxy features of the training distribution.
    Larger distances indicate greater deviation (potential anomaly).

    Parameters
    ----------
    model : torch.nn.Module
        A trained Z3TopologyFirstModel.
    sequences : np.ndarray
        Shape ``(N, T, d)`` — sequences to score.
    device : torch.device or str
        Computation device.

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` — anomaly scores (higher = more anomalous).
    """
    model.eval()
    device = torch.device(device)
    with torch.no_grad():
        x = torch.from_numpy(sequences).float().to(device)
        out = model.compute_proxy(x)
        features = out.proxy_features.cpu().numpy()  # (N, F)

    # Score = L2 distance from centroid.
    centroid = features.mean(axis=0, keepdims=True)
    scores = np.linalg.norm(features - centroid, axis=1)
    return scores


def evaluate_anomaly_detection(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> AnomalyDetectionResult:
    """Evaluate anomaly detection from pre-computed scores.

    Parameters
    ----------
    normal_scores : np.ndarray
        Shape ``(N_normal,)`` — scores for normal samples.
    anomaly_scores : np.ndarray
        Shape ``(N_anomaly,)`` — scores for anomalous samples.

    Returns
    -------
    AnomalyDetectionResult
    """
    labels = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(anomaly_scores)),
    ])
    scores = np.concatenate([normal_scores, anomaly_scores])

    # AUROC (trapezoidal approximation).
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    tpr = np.cumsum(sorted_labels) / max(sorted_labels.sum(), 1)
    fpr = np.cumsum(1 - sorted_labels) / max((1 - sorted_labels).sum(), 1)
    auroc = float(np.trapz(tpr, fpr))

    # AUPRC.
    precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
    recall = np.cumsum(sorted_labels) / max(sorted_labels.sum(), 1)
    auprc = float(np.trapz(precision, recall))

    # Optimal threshold (maximize F1).
    best_f1 = 0.0
    best_threshold = float(scores.mean())
    for threshold in np.linspace(scores.min(), scores.max(), 100):
        preds = (scores >= threshold).astype(float)
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return AnomalyDetectionResult(
        auroc=auroc,
        auprc=auprc,
        threshold=best_threshold,
        f1=best_f1,
    )


__all__ = [
    "AnomalyDetectionResult",
    "compute_proxy_anomaly_scores",
    "evaluate_anomaly_detection",
]
