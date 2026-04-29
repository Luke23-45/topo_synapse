"""Z3 Baseline Study — Autoregressive Rollout Evaluation
======================================================

Adapts the legacy rollout concept for classification tasks.

In the legacy robotics study, rollout measures compounding error when
predicted actions are fed back as proprioception.  For classification,
we simulate compounding degradation by progressively corrupting the
input sequence and measuring how accuracy degrades — a "robustness
rollout" that tests how well each backbone handles accumulating input
noise (analogous to compounding prediction errors in robotics).

Rollout protocol:
    1. Start with a clean test sequence
    2. At each step, add Gaussian noise to the input (simulating
       sensor drift / compounding error)
    3. Classify the corrupted sequence
    4. Record accuracy at each noise level → degradation curve

Rollout metrics:
    - accuracy_per_step: Accuracy at each corruption step
    - area_under_curve: Integral of accuracy curve (robustness summary)
    - divergence_step: First step where accuracy drops below 50% of clean
    - degradation_slope: Linear regression slope of accuracy curve
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Results from rollout (robustness) evaluation.

    Attributes
    ----------
    accuracy_per_step : np.ndarray, shape (N,)
        Classification accuracy at each corruption step.
    area_under_curve : float
        Integral of the accuracy curve (higher = more robust).
    divergence_step : int or None
        First step where accuracy drops below 50% of clean accuracy.
        None if divergence never occurs.
    degradation_slope : float
        Linear regression slope of the accuracy curve (negative = degrading).
    clean_accuracy : float
        Accuracy with zero corruption (baseline reference).
    """

    accuracy_per_step: np.ndarray
    area_under_curve: float
    divergence_step: Optional[int]
    degradation_slope: float
    clean_accuracy: float


def rollout_evaluate(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    labels: torch.Tensor,
    n_steps: int = 10,
    noise_scale: float = 0.1,
    device: str = "cpu",
) -> RolloutResult:
    """Autoregressive rollout: progressively corrupt input and measure accuracy decay.

    At each step:
        1. Add cumulative Gaussian noise to the input sequence
        2. Classify the corrupted sequence
        3. Record accuracy

    Parameters
    ----------
    model : nn.Module
        Trained Z3UnifiedModel.
    sequences : torch.Tensor, shape (B, T, D)
        Clean input sequences.
    labels : torch.Tensor, shape (B,)
        Ground-truth class labels.
    n_steps : int
        Number of rollout (corruption) steps.
    noise_scale : float
        Standard deviation of Gaussian noise added per step.
        Total noise at step k = noise_scale * sqrt(k) (Brownian motion).
    device : str

    Returns
    -------
    RolloutResult
    """
    model.eval()
    model.to(device)
    sequences = sequences.to(device)
    labels = labels.to(device)

    # Compute clean accuracy first
    with torch.no_grad():
        clean_output = model(sequences)
        clean_preds = clean_output.logits.argmax(dim=1)
        clean_accuracy = (clean_preds == labels).float().mean().item()

    accuracy_per_step = [clean_accuracy]

    for step in range(1, n_steps + 1):
        with torch.no_grad():
            # Cumulative noise: Brownian-motion-style corruption
            # noise at step k ~ N(0, noise_scale * sqrt(k))
            noise_std = noise_scale * np.sqrt(step)
            noise = torch.randn_like(sequences) * noise_std
            corrupted = sequences + noise

            output = model(corrupted)
            preds = output.logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            accuracy_per_step.append(acc)

    acc_curve = np.array(accuracy_per_step, dtype=np.float64)

    # Area under curve (trapezoidal)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    area_under_curve = float(_trapz(acc_curve))

    # Divergence step: first step where accuracy < 50% of clean
    divergence_step = None
    if clean_accuracy > 0:
        threshold = 0.5 * clean_accuracy
        for i, acc_val in enumerate(accuracy_per_step):
            if acc_val < threshold:
                divergence_step = i
                break

    # Degradation slope: linear regression on accuracy curve
    degradation_slope = 0.0
    if len(acc_curve) >= 2:
        x = np.arange(len(acc_curve), dtype=np.float64)
        degradation_slope = float(np.polyfit(x, acc_curve, 1)[0])

    return RolloutResult(
        accuracy_per_step=acc_curve,
        area_under_curve=area_under_curve,
        divergence_step=divergence_step,
        degradation_slope=degradation_slope,
        clean_accuracy=clean_accuracy,
    )


def rollout_evaluate_dataset(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    n_steps: int = 10,
    noise_scale: float = 0.1,
    max_samples: int = 50,
    device: str = "cpu",
) -> Dict[str, RolloutResult]:
    """Run rollout evaluation across a dataset, aggregating per-sample results.

    Parameters
    ----------
    model : nn.Module
    test_loader : DataLoader
        Yields batches with "sequences" and "targets".
    n_steps : int
        Number of rollout steps.
    noise_scale : float
        Noise standard deviation per step.
    max_samples : int
        Maximum samples to evaluate (for compute budget).
    device : str

    Returns
    -------
    dict mapping sample_id → RolloutResult
    """
    model.eval()
    model.to(device)

    results = {}
    sample_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            if sample_idx >= max_samples:
                break

            sequences = batch["sequences"].to(device)
            labels = batch["targets"].to(device)

            B = sequences.shape[0]
            for b in range(B):
                if sample_idx >= max_samples:
                    break

                seq = sequences[b:b+1]
                lbl = labels[b:b+1]

                result = rollout_evaluate(
                    model, seq, lbl,
                    n_steps=n_steps,
                    noise_scale=noise_scale,
                    device=device,
                )

                results[f"sample_{sample_idx:04d}"] = result
                sample_idx += 1

    log.info(
        "Rollout evaluation: %d samples, %d steps, noise_scale=%.3f",
        len(results), n_steps, noise_scale,
    )

    return results


def aggregate_rollout_results(
    results: Dict[str, RolloutResult],
) -> Dict[str, float]:
    """Aggregate rollout results across samples.

    Returns
    -------
    dict with keys:
        mean_auc, std_auc,
        mean_slope, std_slope,
        mean_divergence_step, std_divergence_step,
        mean_clean_accuracy, std_clean_accuracy,
        mean_accuracy_per_step (list), std_accuracy_per_step (list)
    """
    if not results:
        return {}

    aucs = [r.area_under_curve for r in results.values()]
    slopes = [r.degradation_slope for r in results.values()]
    div_steps = [r.divergence_step for r in results.values()
                 if r.divergence_step is not None]
    clean_accs = [r.clean_accuracy for r in results.values()]

    # Average accuracy curves (may have different lengths)
    curves = [r.accuracy_per_step for r in results.values()]
    max_len = max(len(c) for c in curves)
    padded = np.zeros((len(curves), max_len))
    for i, c in enumerate(curves):
        padded[i, :len(c)] = c
    mean_curve = padded.mean(axis=0)
    std_curve = padded.std(axis=0)

    return {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_slope": float(np.mean(slopes)),
        "std_slope": float(np.std(slopes)),
        "mean_divergence_step": float(np.mean(div_steps)) if div_steps else float("inf"),
        "std_divergence_step": float(np.std(div_steps)) if len(div_steps) > 1 else 0.0,
        "mean_clean_accuracy": float(np.mean(clean_accs)),
        "std_clean_accuracy": float(np.std(clean_accs)),
        "mean_accuracy_per_step": mean_curve.tolist(),
        "std_accuracy_per_step": std_curve.tolist(),
    }


__all__ = [
    "RolloutResult",
    "rollout_evaluate",
    "rollout_evaluate_dataset",
    "aggregate_rollout_results",
]
