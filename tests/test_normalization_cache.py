from __future__ import annotations

import numpy as np

from synapse.arch.training.builders.builder import resolve_normalization
from synapse.dataset.adapters.base import DatasetBundle, DatasetSpec


def test_resolve_normalization_uses_cached_bundle_metadata() -> None:
    bundle = DatasetBundle(
        train_sequences=np.zeros((2, 3, 4), dtype=np.float32),
        train_labels=np.zeros(2, dtype=np.int64),
        val_sequences=np.zeros((1, 3, 4), dtype=np.float32),
        val_labels=np.zeros(1, dtype=np.int64),
        test_sequences=np.zeros((1, 3, 4), dtype=np.float32),
        test_labels=np.zeros(1, dtype=np.int64),
        spec=DatasetSpec(input_dim=4, sequence_length=3, num_classes=2),
        metadata={
            "normalization_mu": np.arange(15, dtype=np.float64),
            "normalization_sigma": np.arange(15, dtype=np.float64) + 1.0,
        },
    )

    stats = resolve_normalization(bundle)

    np.testing.assert_allclose(stats["mu"], bundle.metadata["normalization_mu"])
    np.testing.assert_allclose(stats["sigma"], bundle.metadata["normalization_sigma"])
