"""Rollout-style robustness evaluation for classification baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Dataset-level robustness curve summary."""

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
    """Progressively corrupt inputs and measure dataset accuracy decay.

    The returned AUC is normalized to ``[0, 1]`` so a perfect rollout stays at
    ``1.0`` regardless of how many steps were evaluated.
    """
    model.eval()
    model.to(device)
    sequences = sequences.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        clean_output = model(sequences)
        clean_preds = clean_output.logits.argmax(dim=1)
        clean_accuracy = float((clean_preds == labels).float().mean().item())

    accuracy_per_step = [clean_accuracy]

    for step in range(1, n_steps + 1):
        with torch.no_grad():
            noise_std = noise_scale * np.sqrt(step)
            corrupted = sequences + torch.randn_like(sequences) * noise_std
            output = model(corrupted)
            preds = output.logits.argmax(dim=1)
            accuracy_per_step.append(float((preds == labels).float().mean().item()))

    acc_curve = np.asarray(accuracy_per_step, dtype=np.float64)
    x_axis = np.linspace(0.0, 1.0, num=len(acc_curve), dtype=np.float64)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    area_under_curve = float(trapz(acc_curve, x=x_axis))

    divergence_step = None
    if clean_accuracy > 0:
        threshold = 0.5 * clean_accuracy
        for i, acc_val in enumerate(accuracy_per_step):
            if acc_val < threshold:
                divergence_step = i
                break

    degradation_slope = 0.0
    if len(acc_curve) >= 2:
        x = np.linspace(0.0, 1.0, num=len(acc_curve), dtype=np.float64)
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
    """Evaluate rollout robustness on one dataset-level subset.

    The old implementation averaged many per-sample 0/1 curves, which made AUC
    difficult to interpret and excessively noisy. This version evaluates a
    single dataset-level accuracy curve over up to ``max_samples`` examples.
    """
    collected_sequences: list[torch.Tensor] = []
    collected_labels: list[torch.Tensor] = []
    collected = 0

    for batch in test_loader:
        if collected >= max_samples:
            break
        take = min(batch["sequences"].shape[0], max_samples - collected)
        collected_sequences.append(batch["sequences"][:take].cpu())
        collected_labels.append(batch["targets"][:take].cpu())
        collected += take

    if not collected_sequences:
        return {}

    sequences = torch.cat(collected_sequences, dim=0)
    labels = torch.cat(collected_labels, dim=0)
    result = rollout_evaluate(
        model,
        sequences,
        labels,
        n_steps=n_steps,
        noise_scale=noise_scale,
        device=device,
    )

    log.info(
        "Rollout evaluation: %d samples, %d steps, noise_scale=%.3f",
        int(labels.shape[0]), n_steps, noise_scale,
    )
    return {"dataset": result}


def aggregate_rollout_results(
    results: Dict[str, RolloutResult],
) -> Dict[str, float]:
    """Aggregate rollout results across seeds or multiple evaluation subsets."""
    if not results:
        return {}

    aucs = [r.area_under_curve for r in results.values()]
    slopes = [r.degradation_slope for r in results.values()]
    div_steps = [r.divergence_step for r in results.values() if r.divergence_step is not None]
    clean_accs = [r.clean_accuracy for r in results.values()]

    curves = [r.accuracy_per_step for r in results.values()]
    max_len = max(len(c) for c in curves)
    padded = np.zeros((len(curves), max_len))
    for i, curve in enumerate(curves):
        padded[i, :len(curve)] = curve
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
