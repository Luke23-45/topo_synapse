"""Autoregressive rollout evaluation for Z3 SYNAPSE models.

Adapted from the baselines rollout logic for the Z3 architecture.
Tests the model's robustness under compounding prediction errors by
feeding predictions back as input in an autoregressive loop.

Rollout metrics:
    - rollout_accuracy_curve: Accuracy at each rollout step
    - rollout_confidence_curve: Mean confidence at each step
    - rollout_divergence_step: First step where confidence drops below threshold
    - rollout_slope: Linear regression slope of confidence curve
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import Z3TopologyFirstModel

log = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Results from autoregressive rollout evaluation.

    Attributes
    ----------
    accuracy_per_step : np.ndarray, shape (N,)
        Classification accuracy at each rollout step.
    confidence_per_step : np.ndarray, shape (N,)
        Mean confidence at each rollout step.
    divergence_step : int or None
        First step where confidence drops below 50% of baseline.
    slope : float
        Linear regression slope of the confidence curve.
    baseline_confidence : float
        Confidence from clean (non-corrupted) input.
    """

    accuracy_per_step: np.ndarray
    confidence_per_step: np.ndarray
    divergence_step: Optional[int]
    slope: float
    baseline_confidence: float


def rollout_evaluate(
    model: Z3TopologyFirstModel,
    sequences: torch.Tensor,
    targets: torch.Tensor,
    n_steps: int = 10,
    noise_scale: float = 0.05,
    device: torch.device | None = None,
) -> RolloutResult:
    """Autoregressive rollout: inject noise, measure confidence decay.

    At each step:
        1. Add cumulative noise to the input sequence
        2. Predict class from the corrupted sequence
        3. Measure accuracy and confidence against ground-truth targets
        4. Record metrics

    Parameters
    ----------
    model : Z3TopologyFirstModel
        Trained model.
    sequences : torch.Tensor, shape (B, T, d)
        Clean input sequences.
    targets : torch.Tensor, shape (B,)
        Ground-truth class labels.
    n_steps : int
        Number of rollout steps.
    noise_scale : float
        Standard deviation of Gaussian noise added per step.
    device : torch.device, optional

    Returns
    -------
    RolloutResult
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    sequences = sequences.to(device)
    targets = targets.to(device)

    # Baseline confidence (clean input)
    with torch.no_grad():
        out = model(sequences)
        probs = F.softmax(out.logits, dim=-1)
        baseline_conf = float(probs.max(dim=-1).values.mean().item())
        baseline_acc = float((out.logits.argmax(dim=-1) == targets).float().mean().item())

    corrupted = sequences.clone()
    acc_per_step: List[float] = []
    conf_per_step: List[float] = []

    for step in range(n_steps):
        with torch.no_grad():
            # Add noise to simulate compounding corruption
            noise = torch.randn_like(corrupted) * noise_scale
            corrupted = corrupted + noise

            out = model(corrupted)
            preds = out.logits.argmax(dim=-1)
            probs = F.softmax(out.logits, dim=-1)
            confidences = probs.max(dim=-1).values

            acc = float((preds == targets).float().mean().item())
            conf = float(confidences.mean().item())

            acc_per_step.append(acc)
            conf_per_step.append(conf)

    acc_curve = np.array(acc_per_step, dtype=np.float64)
    conf_curve = np.array(conf_per_step, dtype=np.float64)

    # Divergence step: first step where confidence drops below 50% of baseline
    divergence_step = None
    if baseline_conf > 0:
        threshold = 0.5 * baseline_conf
        for i, c in enumerate(conf_per_step):
            if c < threshold:
                divergence_step = i
                break

    # Slope of confidence curve
    slope = 0.0
    if len(conf_curve) >= 2:
        x = np.arange(len(conf_curve), dtype=np.float64)
        slope = float(np.polyfit(x, conf_curve, 1)[0])

    return RolloutResult(
        accuracy_per_step=acc_curve,
        confidence_per_step=conf_curve,
        divergence_step=divergence_step,
        slope=slope,
        baseline_confidence=baseline_conf,
    )


def rollout_evaluate_dataset(
    model: Z3TopologyFirstModel,
    dataloader: torch.utils.data.DataLoader,
    config: SynapseConfig,
    n_steps: int = 10,
    noise_scale: float = 0.05,
    max_batches: int = 10,
) -> Dict[str, RolloutResult]:
    """Run rollout evaluation across a dataset.

    Parameters
    ----------
    model : Z3TopologyFirstModel
    dataloader : DataLoader
    config : SynapseConfig
    n_steps : int
    noise_scale : float
    max_batches : int
        Maximum batches to evaluate (compute budget).

    Returns
    -------
    dict mapping batch_id → RolloutResult
    """
    device = next(model.parameters()).device
    model.eval()
    results: Dict[str, RolloutResult] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            sequences = batch_device["sequences"]
            targets = batch_device["targets"]

            result = rollout_evaluate(
                model, sequences, targets,
                n_steps=n_steps,
                noise_scale=noise_scale,
                device=device,
            )
            results[f"batch_{batch_idx:04d}"] = result

    log.info(
        "Rollout evaluation: %d batches, %d steps",
        len(results), n_steps,
    )
    return results


def aggregate_rollout_results(
    results: Dict[str, RolloutResult],
) -> Dict[str, float]:
    """Aggregate rollout results across batches.

    Returns
    -------
    dict with keys:
        mean_baseline_confidence, std_baseline_confidence,
        mean_divergence_step, std_divergence_step,
        mean_slope, std_slope,
        mean_confidence_per_step (list), std_confidence_per_step (list),
        mean_accuracy_per_step (list), std_accuracy_per_step (list),
    """
    if not results:
        return {}

    baselines = [r.baseline_confidence for r in results.values()]
    div_steps = [r.divergence_step for r in results.values()
                 if r.divergence_step is not None]
    slopes = [r.slope for r in results.values()]

    # Average curves (may have different lengths)
    conf_curves = [r.confidence_per_step for r in results.values()]
    acc_curves = [r.accuracy_per_step for r in results.values()]

    def _pad_and_stack(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        max_len = max(len(c) for c in curves)
        padded = np.zeros((len(curves), max_len))
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        return padded.mean(axis=0), padded.std(axis=0)

    mean_conf, std_conf = _pad_and_stack(conf_curves)
    mean_acc, std_acc = _pad_and_stack(acc_curves)

    return {
        "mean_baseline_confidence": float(np.mean(baselines)),
        "std_baseline_confidence": float(np.std(baselines)),
        "mean_divergence_step": float(np.mean(div_steps)) if div_steps else float("inf"),
        "std_divergence_step": float(np.std(div_steps)) if len(div_steps) > 1 else 0.0,
        "mean_slope": float(np.mean(slopes)),
        "std_slope": float(np.std(slopes)),
        "mean_confidence_per_step": mean_conf.tolist(),
        "std_confidence_per_step": std_conf.tolist(),
        "mean_accuracy_per_step": mean_acc.tolist(),
        "std_accuracy_per_step": std_acc.tolist(),
    }
