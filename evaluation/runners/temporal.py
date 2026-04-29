"""Temporal-sequence specific evaluation extensions.

Adds noise-robustness sweeps and sequence-length scaling analysis
on top of the base classification evaluator.  Used for temporal
modalities (e.g. synthetic topology, telecom).
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
from synapse.empirical.datasets.synthetic_topology import generate_topology_dataset

from .base import BaseEvaluator, EvalResult
from .classification import ClassificationEvaluator

log = logging.getLogger(__name__)


class TemporalEvaluator(ClassificationEvaluator):
    """Evaluator for temporal-sequence classification tasks.

    Extends ``ClassificationEvaluator`` with:
        - Noise robustness sweep (accuracy vs noise level)
        - Sequence length scaling (accuracy vs sequence length)

    Parameters
    ----------
    config : SynapseConfig
    test_loader : DataLoader
    output_dir : Path
    noise_levels : list[float]
        Noise standard deviations to sweep.
    length_scales : list[int] or None
        Sequence lengths to sweep (None to skip).
    max_topology_samples : int
    """

    def __init__(
        self,
        config: SynapseConfig,
        test_loader: DataLoader,
        output_dir: Path,
        noise_levels: list[float] | None = None,
        length_scales: list[int] | None = None,
        max_topology_samples: int = 64,
    ) -> None:
        super().__init__(config, test_loader, output_dir, max_topology_samples)
        self.noise_levels = noise_levels or [0.0, 0.01, 0.03, 0.05, 0.1, 0.2]
        self.length_scales = length_scales

    @torch.no_grad()
    def evaluate(self, model: Z3TopologyFirstModel) -> EvalResult:
        """Run temporal evaluation: classification + robustness sweeps."""
        result = super().evaluate(model)
        result.modality = "temporal"

        # Noise robustness sweep
        noise_results = self._noise_robustness_sweep(model)
        result.robustness_metrics["noise_sweep"] = noise_results

        # Length scaling sweep
        if self.length_scales is not None:
            length_results = self._length_scaling_sweep(model)
            result.robustness_metrics["length_sweep"] = length_results

        return result

    def _noise_robustness_sweep(
        self,
        model: Z3TopologyFirstModel,
    ) -> list[dict[str, float]]:
        """Sweep noise levels and measure accuracy at each level."""
        model.eval()
        results: list[dict[str, float]] = []

        # Extract a stable subset for the sweep
        try:
            subset_seqs = self._extract_test_samples(max_samples=64).cpu()
            # Find labels for these samples
            subset_labels = []
            count = 0
            for batch in self.test_loader:
                targets = batch["targets"]
                subset_labels.append(targets)
                count += targets.shape[0]
                if count >= 64:
                    break
            subset_labels = torch.cat(subset_labels, dim=0)[:subset_seqs.shape[0]].to(self.device)
        except Exception as e:
            log.warning("Could not extract subset for noise sweep: %s", e)
            return []

        for noise_std in self.noise_levels:
            # Apply additive Gaussian noise to the base sequences
            noise = torch.randn_like(subset_seqs) * noise_std
            seq_tensor = (subset_seqs + noise).float().to(self.device)
            label_tensor = subset_labels

            with torch.no_grad():
                out = model(seq_tensor)
                preds = out.logits.argmax(dim=-1)
                acc = float((preds == label_tensor).float().mean().item())
                confidence = float(
                    out.logits.softmax(dim=-1).max(dim=-1).values.mean().item()
                )

            results.append({
                "noise_std": float(noise_std),
                "accuracy": acc,
                "mean_confidence": confidence,
            })

        return results

    def _length_scaling_sweep(
        self,
        model: Z3TopologyFirstModel,
    ) -> list[dict[str, float]]:
        """Sweep sequence lengths and measure accuracy at each length."""
        # Only valid for synthetic data where we can generate arbitrary lengths
        # For real data, sequence lengths are fixed by the adapter.
        if self.config.get("dataset", "") != "synthetic":
            log.info("Skipping length scaling sweep for non-synthetic dataset.")
            return []

        model.eval()
        results: list[dict[str, float]] = []

        for length in self.length_scales:
            sequences, labels, _ = generate_topology_dataset(
                64,
                length=length,
                noise_std=self.config.noise_std,
                seed=self.config.seed + length,
            )
            seq_tensor = torch.from_numpy(sequences).float().to(self.device)
            label_tensor = torch.from_numpy(labels).long().to(self.device)

            with torch.no_grad():
                out = model(seq_tensor)
                preds = out.logits.argmax(dim=-1)
                acc = float((preds == label_tensor).float().mean().item())

            results.append({
                "sequence_length": length,
                "accuracy": acc,
            })

        return results
