"""Synthetic Topology Adapter.

Wraps the existing ``synapse.empirical.datasets.synthetic_topology``
module to produce a ``DatasetBundle`` compatible with the Z3 multi-dataset
pipeline.  This is the only adapter that requires no external data source
and is always available.

The adapter preserves the exact behaviour of ``build_synthetic_bundle()``
so that existing code continues to work identically.

Reference
---------
- Generator: ``synapse/empirical/datasets/synthetic_topology.py``
- Z3 plan:   ``docs/implementions/plan.md`` §4.2
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..registry import register_adapter
from .base import DatasetBundle, DatasetSpec, Z3Adapter
from .persistence import get_prepared_cache_dir, load_prepared_bundle, save_prepared_bundle

log = logging.getLogger(__name__)


class SyntheticAdapter(Z3Adapter):
    """Adapter for in-memory synthetic topology sequences.

    Generates four topological classes (line, circle, figure-eight,
    branch) as 2D trajectories with configurable noise and length.

    Parameters
    ----------
    train_size : int
        Number of training samples.
    val_size : int
        Number of validation samples.
    test_size : int
        Number of test samples.
    length : int
        Sequence length *T* per sample.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        train_size: int = 512,
        val_size: int = 128,
        test_size: int = 128,
        length: int = 128,
        noise_std: float = 0.03,
        seed: int = 7,
        data_root: str | None = None,
    ) -> None:
        self._train_size = train_size
        self._val_size = val_size
        self._test_size = test_size
        self._length = length
        self._noise_std = noise_std
        self._seed = seed
        self._data_root = data_root

        self._spec = DatasetSpec(
            name="synthetic",
            modality="temporal",
            source="synthetic",
            input_dim=2,
            sequence_length=length,
            num_classes=4,
            task="classification",
        )

    # ---- Z3Adapter interface -----------------------------------------------

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    @property
    def input_dim(self) -> int:
        return self._spec.input_dim

    @property
    def num_classes(self) -> int:
        return self._spec.num_classes

    def load_splits(self) -> DatasetBundle:
        """Generate synthetic topology data and return pre-split arrays (with disk caching)."""
        # 1. Check for prepared cache
        cache_root = Path(self._data_root or "data/datasets").resolve()
        
        # Calculate ratio for the cache key
        total = self._train_size + self._val_size + self._test_size
        cache_dir = get_prepared_cache_dir(
            cache_root, "synthetic", self._spec, self._seed,
            self._train_size / total if total > 0 else 0.8,
            self._val_size / total if total > 0 else 0.1,
            extra_key=(
                f"N{self._train_size}_{self._val_size}_{self._test_size}"
                f"_noise{int(round(self._noise_std * 10000))}"
            ),
        )
        
        bundle = load_prepared_bundle(cache_dir, self._spec)
        if bundle is not None:
            return bundle

        # 2. Generate on the fly
        log.info("Generating synthetic topology data...")
        from synapse.empirical.datasets.synthetic_topology import (
            build_synthetic_bundle,
        )

        raw_bundle = build_synthetic_bundle(
            self._train_size,
            self._val_size,
            self._test_size,
            length=self._length,
            noise_std=self._noise_std,
            seed=self._seed,
        )

        res_bundle = DatasetBundle(
            train_sequences=raw_bundle.train_sequences.astype(np.float32),
            train_labels=raw_bundle.train_labels.astype(np.int64),
            val_sequences=raw_bundle.val_sequences.astype(np.float32),
            val_labels=raw_bundle.val_labels.astype(np.int64),
            test_sequences=raw_bundle.test_sequences.astype(np.float32),
            test_labels=raw_bundle.test_labels.astype(np.int64),
            spec=self._spec,
            metadata={"topology_names": ["line", "circle", "figure_eight", "branch"]},
        )
        
        # 3. Save to disk
        save_prepared_bundle(cache_dir, res_bundle)
        
        return res_bundle


# Self-register with the global adapter registry.
register_adapter("synthetic", SyntheticAdapter)


__all__ = ["SyntheticAdapter"]
