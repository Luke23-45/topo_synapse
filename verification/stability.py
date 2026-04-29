"""
Topology Stability Verification — Z3 Reference: §9 of 02_rigorous_architecture.md

Theorem 9.1: For matched cloud perturbations with max||p_j - p'_j|| ≤ ε,
    d_B(Dgm_q(P), Dgm_q(P')) ≤ 2ε.

This module empirically validates that the exact topology audit is stable
under small noise perturbations.
"""

from __future__ import annotations

import numpy as np
import torch

from synapse.synapse_arch.model import Z3TopologyFirstModel
from synapse.synapse_core.topology import hausdorff_distance


def topology_stability_trial(
    model: Z3TopologyFirstModel,
    sequence: np.ndarray,
    *,
    noise_std: float,
) -> dict[str, float]:
    """Run a single stability trial: compare exact audit of clean vs noisy sequence."""
    clean = model.exact_audit(torch.from_numpy(sequence).float().unsqueeze(0))[0]
    noisy_sequence = sequence + np.random.normal(scale=noise_std, size=sequence.shape)
    noisy = model.exact_audit(torch.from_numpy(noisy_sequence).float().unsqueeze(0))[0]
    return {
        "noise_std": noise_std,
        "hausdorff_distance": float(hausdorff_distance(clean.point_cloud, noisy.point_cloud)),
        "anchor_count_clean": float(len(clean.anchor_indices)),
        "anchor_count_noisy": float(len(noisy.anchor_indices)),
    }
