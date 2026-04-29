"""Spatial Geometry Adapter.

Loads the SpatialLM-Dataset from HuggingFace and produces a
``DatasetBundle`` for the Z3 geometric evaluation track.

The dataset contains 12,328 indoor scenes as 3D point clouds.
Z3 uses the ``DifferentiableHodgeProxy`` (L1) to identify structural
"holes" (doorways, windows, furniture voids) that distinguish room types.

Source: https://huggingface.co/datasets/manycore-research/SpatialLM-Dataset

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §3, Track 2
- Data source doc: ``docs/dev/data_source.md`` §3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ..preprocess.geometric import GeometricPreprocessor
from ..registry import register_adapter
from .base import DatasetBundle, DatasetSpec, Z3Adapter
from .persistence import get_prepared_cache_dir, load_prepared_bundle, save_prepared_bundle
from .split_utils import apply_split, try_load_from_local

log = logging.getLogger(__name__)


class SpatialAdapter(Z3Adapter):
    """Adapter for the SpatialLM 3D point-cloud dataset.

    Parameters
    ----------
    target_length : int
        Fixed number of points per sample after sub-sampling.
    train_ratio : float
        Fraction of data used for training (default 0.8).
    val_ratio : float
        Fraction of data used for validation (default 0.1).
    seed : int
        Random seed for reproducible splitting.
    max_samples : int or None
        Cap on total samples loaded (for smoke tests).
    data_root : str or None
        Root directory for downloaded datasets.  If set, the adapter
        will try loading from local disk before HuggingFace.
    """

    def __init__(
        self,
        *,
        target_length: int = 512,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: int | None = None,
        data_root: str | None = None,
    ) -> None:
        self._target_length = target_length
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._seed = seed
        self._max_samples = max_samples
        self._data_root = data_root

        self._spec = DatasetSpec(
            name="spatial",
            modality="geometric",
            source="huggingface",
            hf_repo="manycore-research/SpatialLM-Dataset",
            input_dim=3,
            sequence_length=target_length,
            num_classes=10,
            task="classification",
            max_samples=max_samples,
            data_root=data_root,
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
        """Load SpatialLM and return splits (with disk caching).
        
        This is the primary entry point. It checks for prepared split
        files on disk before falling back to raw extraction.

        Returns
        -------
        DatasetBundle
        """
        # 1. Check for final prepared splits first
        cache_dir = Path(self._data_root or "data/datasets")
        cache_path = get_prepared_cache_dir(
            cache_dir, "spatial", self._spec, self._seed, self._train_ratio, self._val_ratio
        )
        
        bundle = load_prepared_bundle(cache_path, self._spec)
        if bundle is not None:
            return bundle

        # 2. If no prepared cache, load raw data (local or HF)
        log.info("No prepared cache found for SpatialLM. Starting extraction...")
        raw_clouds, raw_labels = self._load_data()

        # 3. Apply splits
        clouds, labels = apply_split(
            raw_clouds,
            raw_labels,
            train_ratio=self._train_ratio,
            val_ratio=self._val_ratio,
            seed=self._seed,
        )

        # 4. Wrap and Preprocess
        preprocessor = GeometricPreprocessor(target_length=self._target_length)
        bundle = DatasetBundle(
            train_sequences=clouds["train"],
            train_labels=labels["train"],
            val_sequences=clouds["val"],
            val_labels=labels["val"],
            test_sequences=clouds["test"],
            test_labels=labels["test"],
            spec=self._spec,
        )
        bundle = preprocessor(bundle)

        # 5. Save the FINAL PREPARED data to disk for next time
        save_prepared_bundle(cache_path, bundle)
        
        return bundle

    # ------------------------------------------------------------------ #

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load point clouds + labels, preferring local data over HuggingFace.

        Returns
        -------
        clouds : np.ndarray
            Shape ``(N_total, N_pts, 3)``.
        labels : np.ndarray
            Shape ``(N_total,)``.
        """
        # --- Try local disk first ---
        local_ds = try_load_from_local(self._data_root, "spatial")
        if local_ds is not None:
            log.info("SpatialAdapter: loading from local data_root=%s", self._data_root)
            return self._extract_from_dataset(local_ds)

        # --- Fall back to HuggingFace ---
        return self._load_from_huggingface()

    def _load_from_huggingface(self) -> tuple[np.ndarray, np.ndarray]:
        """Download and extract point clouds + labels from HuggingFace.

        Returns
        -------
        clouds : np.ndarray
            Shape ``(N_total, N_pts, 3)``.
        labels : np.ndarray
            Shape ``(N_total,)``.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load SpatialLM. "
                "Install it with: pip install datasets"
            ) from exc

        log.info("Loading SpatialLM from HuggingFace: %s", self._spec.hf_repo)
        ds = load_dataset(self._spec.hf_repo, split="train", trust_remote_code=True)

        return self._extract_from_dataset(ds)

    def _extract_from_dataset(self, ds) -> tuple[np.ndarray, np.ndarray]:
        """Extract point clouds + labels from a HuggingFace Dataset object.

        Parameters
        ----------
        ds : datasets.Dataset or datasets.DatasetDict
            Loaded dataset (from local disk or HuggingFace).

        Returns
        -------
        clouds : np.ndarray
            Shape ``(N_total, N_pts, 3)``.
        labels : np.ndarray
            Shape ``(N_total,)``.
        """
        if hasattr(ds, "values"):
            from datasets import concatenate_datasets
            ds = concatenate_datasets(list(ds.values()))

        clouds_list: list[np.ndarray] = []
        labels_list: list[int] = []

        total = min(len(ds), self._max_samples) if self._max_samples else len(ds)
        for record in tqdm(ds, desc="Extracting SpatialLM", total=total):
            if self._max_samples is not None and len(clouds_list) >= self._max_samples:
                break

            cloud = self._extract_cloud(record)
            label = self._extract_label(record)

            if cloud is not None:
                clouds_list.append(cloud)
                labels_list.append(label)

        if not clouds_list:
            raise RuntimeError(
                f"No valid point clouds extracted from SpatialLM. "
                f"Dataset columns: {ds.column_names}"
            )

        clouds = np.stack(clouds_list, axis=0).astype(np.float32)
        labels = np.asarray(labels_list, dtype=np.int64)

        log.info(
            "SpatialAdapter: loaded %d clouds, shape=%s, classes=%d",
            clouds.shape[0],
            clouds.shape,
            len(np.unique(labels)),
        )
        return clouds, labels

    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_cloud(record: dict) -> np.ndarray | None:
        """Extract a (N, 3) point cloud from a HuggingFace record."""
        for key in ("points", "cloud", "xyz", "vertices", "coordinates"):
            if key in record:
                val = record[key]
                if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] >= 3:
                    return val[:, :3]
                if isinstance(val, list):
                    arr = np.asarray(val, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                        return arr[:, :3]
        return None

    @staticmethod
    def _extract_label(record: dict) -> int:
        """Extract an integer label from a HuggingFace record."""
        for key in ("label", "room_type", "category", "class"):
            if key in record:
                val = record[key]
                if isinstance(val, int):
                    return val
                if isinstance(val, (np.integer,)):
                    return int(val)
                try:
                    return int(val)
                except (ValueError, TypeError):
                    continue
        log.warning(
            "No label column found in record (keys: %s), defaulting to 0",
            list(record.keys()),
        )
        return 0


register_adapter("spatial", SpatialAdapter)


__all__ = ["SpatialAdapter"]
