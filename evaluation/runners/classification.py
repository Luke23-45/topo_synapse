"""Classification task evaluator for Z3 SYNAPSE models.

Works across all modalities (temporal, geometric, scientific) as long
as the task is classification.  Computes:
    - Loss and accuracy
    - Per-class accuracy, top-k accuracy, confusion matrix
    - Proxy-exact topology alignment
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import Z3TopologyFirstModel

from .base import BaseEvaluator, EvalResult

log = logging.getLogger(__name__)


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks across all modalities.

    Parameters
    ----------
    config : SynapseConfig
    test_loader : DataLoader
    output_dir : Path
    max_topology_samples : int
        Maximum samples for proxy-exact alignment (compute budget).
    top_k : int
        K for top-k accuracy computation.
    """

    def __init__(
        self,
        config: SynapseConfig,
        test_loader: DataLoader,
        output_dir: Path,
        max_topology_samples: int = 64,
        top_k: int = 3,
    ) -> None:
        super().__init__(config, test_loader, output_dir)
        self.max_topology_samples = max_topology_samples
        self.top_k = top_k

    @torch.no_grad()
    def evaluate(self, model: Z3TopologyFirstModel) -> EvalResult:
        """Run classification evaluation protocol.

        Parameters
        ----------
        model : Z3TopologyFirstModel
            Trained model.

        Returns
        -------
        EvalResult
        """
        model.eval()
        model.to(self.device)

        all_losses: list[float] = []
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_confidences: list[np.ndarray] = []

        for batch in self.test_loader:
            batch_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            sequences = batch_device["sequences"]
            targets = batch_device["targets"]

            out = model(sequences)
            loss = F.cross_entropy(out.logits, targets)

            probs = F.softmax(out.logits, dim=-1)
            confidences, preds = probs.max(dim=-1)

            all_losses.append(float(loss.item()))
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        confidences = np.concatenate(all_confidences)

        # Primary metrics
        accuracy = float((preds == targets).mean())
        mean_loss = float(np.mean(all_losses))
        mean_confidence = float(confidences.mean())

        primary_metrics: dict[str, float] = {
            "accuracy": accuracy,
            "loss": mean_loss,
            "mean_confidence": mean_confidence,
        }

        # Per-class accuracy
        n_classes = self.config.output_dim
        per_class_acc = {}
        for c in range(n_classes):
            mask = targets == c
            if mask.sum() > 0:
                per_class_acc[f"class_{c}_accuracy"] = float((preds[mask] == c).mean())
            else:
                per_class_acc[f"class_{c}_accuracy"] = 0.0
        primary_metrics.update(per_class_acc)

        # Top-k accuracy
        if self.top_k > 1 and self.top_k < n_classes:
            top_k_correct = 0
            total = 0
            for batch in self.test_loader:
                batch_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = model(batch_device["sequences"])
                _, top_k_preds = out.logits.topk(self.top_k, dim=-1)
                correct = top_k_preds.eq(batch_device["targets"].unsqueeze(1)).any(dim=1)
                top_k_correct += int(correct.sum().item())
                total += batch_device["targets"].shape[0]
            primary_metrics[f"top_{self.top_k}_accuracy"] = top_k_correct / max(total, 1)

        # Per-sample metrics
        per_sample_metrics: dict[str, np.ndarray] = {
            "accuracy": (preds == targets).astype(np.float64),
            "confidence": confidences.astype(np.float64),
        }

        # Topology metrics
        topology_metrics: dict[str, float] = {}
        try:
            subset = self._extract_test_samples(max_samples=self.max_topology_samples)
            topology_metrics = self._compute_proxy_exact_alignment(model, subset)
        except Exception as e:
            log.warning("Proxy-exact alignment computation failed: %s", e)

        return EvalResult(
            dataset_name=self.config.task,
            modality="classification",
            task="classification",
            primary_metrics=primary_metrics,
            per_sample_metrics=per_sample_metrics,
            topology_metrics=topology_metrics,
            metadata={
                "num_test_samples": len(targets),
                "num_classes": n_classes,
                "top_k": self.top_k,
            },
        )
