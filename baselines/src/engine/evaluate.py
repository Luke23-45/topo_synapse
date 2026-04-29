"""Evaluation Engine — Z3 Baseline Study.

Evaluates a trained backbone on a held-out test set, computing
classification metrics (accuracy, F1, per-class precision/recall).

Reuses
------
- ``synapse.synapse.losses.combined_loss.compute_loss`` for loss computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


@dataclass
class BackboneEvaluation:
    """Evaluation results for a single backbone condition.

    Attributes
    ----------
    backbone : str
        Backbone name (e.g. "mlp", "tcn").
    accuracy : float
        Top-1 accuracy.
    f1_macro : float
        Macro-averaged F1 score.
    per_class_accuracy : np.ndarray
        Per-class accuracy.
    confusion_matrix : np.ndarray
        Confusion matrix.
    mean_loss : float
        Mean cross-entropy loss on test set.
    per_seed_accuracy : np.ndarray
        Accuracy from each seed run (for statistical testing).
    """
    backbone: str
    accuracy: float = 0.0
    f1_macro: float = 0.0
    per_class_accuracy: np.ndarray = field(default_factory=lambda: np.array([]))
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_loss: float = 0.0
    per_seed_accuracy: np.ndarray = field(default_factory=lambda: np.array([]))


@torch.no_grad()
def evaluate_backbone(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> BackboneEvaluation:
    """Evaluate a Z3UnifiedModel on a test set.

    Parameters
    ----------
    model : nn.Module
        A ``Z3UnifiedModel`` instance.
    test_loader : DataLoader
        Test dataloader yielding ``SequenceBatch`` or similar.
    device : str

    Returns
    -------
    BackboneEvaluation
    """
    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in test_loader:
        # Batch format from synapse.synapse.data: {"sequences": ..., "targets": ..., "lengths": ...}
        sequences = batch["sequences"].to(device)
        labels = batch["targets"].to(device)

        output = model(sequences)
        logits = output.logits

        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds = all_logits.argmax(dim=1)
    num_classes = all_logits.shape[1]

    # Accuracy
    accuracy = (preds == all_labels).float().mean().item()

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels.numpy(), preds.numpy()):
        cm[t, p] += 1

    # Per-class accuracy
    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    # Macro F1
    f1_scores = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1_scores.append(2 * prec * rec / max(prec + rec, 1e-8))
    f1_macro = float(np.mean(f1_scores))

    return BackboneEvaluation(
        backbone=getattr(model, "backbone_type", "unknown"),
        accuracy=accuracy,
        f1_macro=f1_macro,
        per_class_accuracy=per_class_acc,
        confusion_matrix=cm,
        mean_loss=total_loss / max(n_batches, 1),
    )


__all__ = ["BackboneEvaluation", "evaluate_backbone"]
