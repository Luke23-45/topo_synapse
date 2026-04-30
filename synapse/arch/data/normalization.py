"""Normalization statistics for the structure-aware lift layer."""

from __future__ import annotations

import numpy as np

from synapse.synapse_core.topology_features import compute_structural_normalization_stats


def compute_normalization_stats(sequences: np.ndarray) -> dict[str, np.ndarray]:
    """Compute stats for the universal topology feature contract."""
    return compute_structural_normalization_stats(sequences)
