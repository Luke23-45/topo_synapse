"""Classification metrics for Z3 SYNAPSE models.

Provides task-level evaluation metrics for classification tasks,
including per-class breakdowns, top-k accuracy, and confusion
matrix summaries.
"""

from __future__ import annotations

import numpy as np
import torch


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy from logits and integer targets.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, C)
        Raw class logits.
    targets : torch.Tensor, shape (B,)
        Integer class labels.

    Returns
    -------
    float
        Fraction of correct predictions.
    """
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


def per_class_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute per-class accuracy.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, C)
    targets : torch.Tensor, shape (B,)
    num_classes : int or None
        If provided, forces the class range; otherwise inferred from data.

    Returns
    -------
    dict mapping "class_{i}" → accuracy (float)
    """
    preds = logits.argmax(dim=-1)
    n_classes = num_classes or int(max(targets.max().item(), preds.max().item()) + 1)
    result: dict[str, float] = {}
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() > 0:
            acc = float((preds[mask] == targets[mask]).float().mean().item())
        else:
            acc = 0.0
        result[f"class_{c}"] = acc
    return result


def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 3,
) -> float:
    """Compute top-k accuracy.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, C)
    targets : torch.Tensor, shape (B,)
    k : int

    Returns
    -------
    float
    """
    _, top_k_preds = logits.topk(k, dim=-1)
    correct = top_k_preds.eq(targets.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


def confusion_matrix_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute confusion matrix and derived metrics.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, C)
    targets : torch.Tensor, shape (B,)
    num_classes : int or None

    Returns
    -------
    dict with keys:
        "confusion_matrix" : np.ndarray, shape (C, C)
        "macro_precision" : float
        "macro_recall" : float
        "macro_f1" : float
    """
    preds = logits.argmax(dim=-1).cpu().numpy()
    tgt = targets.cpu().numpy()
    n_classes = num_classes or int(max(tgt.max(), preds.max()) + 1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(tgt, preds):
        cm[int(t), int(p)] += 1

    precisions = np.zeros(n_classes, dtype=np.float64)
    recalls = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        tp = cm[c, c]
        precisions[c] = tp / max(cm[:, c].sum(), 1)
        recalls[c] = tp / max(cm[c, :].sum(), 1)

    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)

    return {
        "confusion_matrix": cm,
        "macro_precision": float(precisions.mean()),
        "macro_recall": float(recalls.mean()),
        "macro_f1": float(f1s.mean()),
    }
