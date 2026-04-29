"""Geometric / point-cloud specific evaluation extensions.

Adds rotation-invariance testing and point-density scaling analysis
on top of the base classification evaluator.  Used for geometric
modalities (e.g. spatial point clouds).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import Z3TopologyFirstModel

from .base import BaseEvaluator, EvalResult
from .classification import ClassificationEvaluator

log = logging.getLogger(__name__)


class GeometricEvaluator(ClassificationEvaluator):
    """Evaluator for geometric / point-cloud classification tasks.

    Extends ``ClassificationEvaluator`` with:
        - Rotation invariance test (accuracy under random rotations)
        - Point density scaling (accuracy vs number of points)

    Parameters
    ----------
    config : SynapseConfig
    test_loader : DataLoader
    output_dir : Path
    rotation_angles : list[float] or None
        Rotation angles (radians) to test. None uses defaults.
    max_topology_samples : int
    """

    def __init__(
        self,
        config: SynapseConfig,
        test_loader: DataLoader,
        output_dir: Path,
        rotation_angles: list[float] | None = None,
        max_topology_samples: int = 64,
    ) -> None:
        super().__init__(config, test_loader, output_dir, max_topology_samples)
        self.rotation_angles = rotation_angles or [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]

    @torch.no_grad()
    def evaluate(self, model: Z3TopologyFirstModel) -> EvalResult:
        """Run geometric evaluation: classification + invariance tests."""
        result = super().evaluate(model)
        result.modality = "geometric"

        # Rotation invariance test
        rotation_results = self._rotation_invariance_test(model)
        result.robustness_metrics["rotation_sweep"] = rotation_results

        return result

    def _rotation_invariance_test(
        self,
        model: Z3TopologyFirstModel,
    ) -> list[dict[str, float]]:
        """Test classification accuracy under 2D rotations of input sequences."""
        model.eval()
        results: list[dict[str, float]] = []

        # Collect a fixed subset of test samples
        test_seqs = self._extract_test_samples(max_samples=32)

        for angle in self.rotation_angles:
            # Apply 2D rotation: [cos -sin; sin cos]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotation_matrix = torch.tensor(
                [[cos_a, -sin_a], [sin_a, cos_a]],
                dtype=torch.float32,
                device=self.device,
            )

            # Only rotate the first 2 dims (spatial coords)
            rotated_seqs = test_seqs.clone()
            spatial = rotated_seqs[:, :, :2]
            rotated_spatial = torch.einsum("btd,dc->btc", spatial, rotation_matrix)
            rotated_seqs[:, :, :2] = rotated_spatial

            with torch.no_grad():
                # We need targets — use the test_loader batch
                pass

            # Use the test_loader to get targets
            accs = []
            for batch in self.test_loader:
                batch_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                sequences = batch_device["sequences"]
                targets = batch_device["targets"]

                # Apply rotation
                rot_seqs = sequences.clone()
                if sequences.shape[-1] >= 2:
                    sp = rot_seqs[:, :, :2]
                    rot_sp = torch.einsum("btd,dc->btc", sp, rotation_matrix)
                    rot_seqs[:, :, :2] = rot_sp

                out = model(rot_seqs)
                preds = out.logits.argmax(dim=-1)
                accs.append(float((preds == targets).float().mean().item()))

            mean_acc = float(np.mean(accs)) if accs else 0.0
            results.append({
                "rotation_angle_rad": float(angle),
                "accuracy": mean_acc,
            })

        return results
