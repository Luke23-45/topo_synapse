from __future__ import annotations

import numpy as np
import torch

from synapse.arch.data.normalization import compute_normalization_stats
from synapse.synapse_core.topology_features import (
    build_structural_feature_tensor,
    structural_feature_dim,
)


def test_structural_feature_dim_and_stats_match() -> None:
    sequences = np.random.randn(4, 6, 3).astype(np.float32)
    stats = compute_normalization_stats(sequences)

    expected_dim = structural_feature_dim(3, include_selection=False)
    assert stats["mu"].shape == (expected_dim,)
    assert stats["sigma"].shape == (expected_dim,)
    assert np.all(stats["sigma"] > 0.0)


def test_structural_channels_are_translation_invariant() -> None:
    base = torch.tensor(
        [[[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]]],
        dtype=torch.float32,
    )
    shifted = base + torch.tensor([10.0, -7.0], dtype=torch.float32)

    base_features = build_structural_feature_tensor(base)
    shifted_features = build_structural_feature_tensor(shifted)

    d = base.shape[-1]
    invariant_slice = slice(d, None)
    assert torch.allclose(
        base_features[:, :, invariant_slice],
        shifted_features[:, :, invariant_slice],
        atol=1e-5,
    )
