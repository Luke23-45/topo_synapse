"""Scientific-data specific evaluation extensions.

Adds feature-importance analysis and regime-detection accuracy
on top of the base classification evaluator.  Used for scientific
modalities (e.g. photonic topology).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import Z3TopologyFirstModel

from .base import BaseEvaluator, EvalResult
from .classification import ClassificationEvaluator

log = logging.getLogger(__name__)


class ScientificEvaluator(ClassificationEvaluator):
    """Evaluator for scientific-data classification tasks.

    Extends ``ClassificationEvaluator`` with:
        - Feature importance via gradient-based sensitivity analysis
        - Per-class topology alignment breakdown

    Parameters
    ----------
    config : SynapseConfig
    test_loader : DataLoader
    output_dir : Path
    max_topology_samples : int
    """

    def __init__(
        self,
        config: SynapseConfig,
        test_loader: DataLoader,
        output_dir: Path,
        max_topology_samples: int = 64,
    ) -> None:
        super().__init__(config, test_loader, output_dir, max_topology_samples)

    @torch.no_grad()
    def evaluate(self, model: Z3TopologyFirstModel) -> EvalResult:
        """Run scientific evaluation: classification + feature importance."""
        result = super().evaluate(model)
        result.modality = "scientific"

        # Feature importance analysis
        feature_importance = self._feature_importance(model)
        result.robustness_metrics["feature_importance"] = feature_importance

        return result

    def _feature_importance(
        self,
        model: Z3TopologyFirstModel,
    ) -> dict[str, Any]:
        """Gradient-based feature importance via input perturbation.

        For each input dimension, measures the drop in confidence when
        that dimension is zeroed out.
        """
        model.eval()
        importance_scores: dict[str, Any] = {}

        # Collect a batch
        batch = next(iter(self.test_loader))
        batch_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        sequences = batch_device["sequences"]
        targets = batch_device["targets"]

        # Baseline confidence
        with torch.no_grad():
            out = model(sequences)
            baseline_conf = F.softmax(out.logits, dim=-1)
            baseline_correct = (out.logits.argmax(dim=-1) == targets).float().mean().item()

        input_dim = sequences.shape[-1]
        dim_drops = []

        for d in range(input_dim):
            perturbed = sequences.clone()
            perturbed[:, :, d] = 0.0

            with torch.no_grad():
                out_pert = model(perturbed)
                pert_conf = F.softmax(out_pert.logits, dim=-1)
                pert_correct = (out_pert.logits.argmax(dim=-1) == targets).float().mean().item()

                # Mean confidence drop for the true class
                true_conf_drop = float(
                    (baseline_conf[range(len(targets)), targets] -
                     pert_conf[range(len(targets)), targets]).mean().item()
                )
                dim_drops.append(true_conf_drop)

        importance_scores["per_dimension_confidence_drop"] = dim_drops
        importance_scores["baseline_accuracy"] = baseline_correct
        importance_scores["input_dim"] = input_dim

        return importance_scores
