"""2D Photonic Topology Adapter.

Loads the 2D photonic topology dataset from HuggingFace and produces
a ``DatasetBundle`` for the Z3 scientific evaluation track.

The dataset contains 10,000 photonic crystal unit cells with labels
corresponding to topological symmetry settings and dielectric contrasts.
This is a direct test of the **Hodge Laplacian Proxy** — in photonics,
topology determines the existence of edge states.

Source: https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §3, Track 3
- Data source doc: ``docs/dev/data_source.md`` §4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ..preprocess.scientific import ScientificPreprocessor
from ..registry import register_adapter
from .base import DatasetBundle, DatasetSpec, Z3Adapter
from .persistence import get_prepared_cache_dir, load_prepared_bundle, save_prepared_bundle
from .split_utils import apply_split, extract_predefined_splits, try_load_from_local

log = logging.getLogger(__name__)


class PhotonicAdapter(Z3Adapter):
    """Adapter for the 2D photonic topology dataset.

    Parameters
    ----------
    inject_coordinates : bool
        If ``True``, prepend (x, y) grid coordinates to each feature
        vector, increasing ``input_dim`` by 2.
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
        inject_coordinates: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: int | None = None,
        data_root: str | None = None,
    ) -> None:
        self._inject_coordinates = inject_coordinates
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._seed = seed
        self._max_samples = max_samples
        self._data_root = data_root

        # input_dim is 8 (features) + 2 (coordinates) if injected.
        effective_dim = 8 + (2 if inject_coordinates else 0)

        self._spec = DatasetSpec(
            name="photonic",
            modality="scientific",
            source="huggingface",
            hf_repo="cgeorgiaw/2d-photonic-topology",
            input_dim=effective_dim,
            sequence_length=64,
            num_classes=4,
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
        """Load 2D photonic topology and return splits (with disk caching).
        
        This is the primary entry point. It checks for prepared split
        files on disk before falling back to raw extraction.

        Returns
        -------
        DatasetBundle
        """
        # 1. Check for final prepared splits first
        cache_dir = Path(self._data_root or "data/datasets")
        cache_path = get_prepared_cache_dir(
            cache_dir, "photonic", self._spec, self._seed, self._train_ratio, self._val_ratio
        )
        
        bundle = load_prepared_bundle(cache_path, self._spec)
        if bundle is not None:
            return bundle

        # 2. If no prepared cache, load raw data (local or HF)
        log.info("No prepared cache found for PhotonicTopology. Starting extraction...")
        ds = self._load_data()
        split_result = extract_predefined_splits(
            ds,
            extract_array=self._extract_grid,
            extract_label=self._extract_label,
            max_samples=self._max_samples,
            train_ratio=self._train_ratio,
            val_ratio=self._val_ratio,
        )
        if split_result is not None:
            grids, labels = split_result
        else:
            raw_grids, raw_labels = self._extract_from_dataset(ds)
            grids, labels = apply_split(
                raw_grids,
                raw_labels,
                train_ratio=self._train_ratio,
                val_ratio=self._val_ratio,
                seed=self._seed,
            )

        # 4. Wrap and Preprocess
        preprocessor = ScientificPreprocessor(
            inject_coordinates=self._inject_coordinates,
        )
        bundle = DatasetBundle(
            train_sequences=grids["train"],
            train_labels=labels["train"],
            val_sequences=grids["val"],
            val_labels=labels["val"],
            test_sequences=grids["test"],
            test_labels=labels["test"],
            spec=self._spec,
        )
        bundle = preprocessor(bundle)

        # 5. Save the FINAL PREPARED data to disk for next time
        save_prepared_bundle(cache_path, bundle)
        
        return bundle

    # ------------------------------------------------------------------ #

    def _load_data(self):
        """Load the raw photonic dataset object."""
        # --- Try local disk first ---
        local_ds = try_load_from_local(self._data_root, "photonic")
        if local_ds is not None:
            log.info("PhotonicAdapter: loading from local data_root=%s", self._data_root)
            return local_ds

        # --- Fall back to HuggingFace ---
        return self._load_from_huggingface()

    def _load_from_huggingface(self):
        """Download the raw photonic dataset object from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load the photonic "
                "topology dataset. Install it with: pip install datasets"
            ) from exc

        log.info(
            "Loading 2D photonic topology from HuggingFace: %s",
            self._spec.hf_repo,
        )
        ds = load_dataset(self._spec.hf_repo, trust_remote_code=True)

        return ds

    def _extract_from_dataset(self, ds) -> tuple[np.ndarray, np.ndarray]:
        """Extract grids + labels from a HuggingFace Dataset object.

        Parameters
        ----------
        ds : datasets.Dataset or datasets.DatasetDict
            Loaded dataset (from local disk or HuggingFace).

        Returns
        -------
        grids : np.ndarray
            Shape ``(N_total, H, W, F)``.
        labels : np.ndarray
            Shape ``(N_total,)``.
        """
        if hasattr(ds, "values"):
            from datasets import concatenate_datasets
            ds = concatenate_datasets(list(ds.values()))

        grids_list: list[np.ndarray] = []
        labels_list: list[int] = []

        total = min(len(ds), self._max_samples) if self._max_samples else len(ds)
        for record in tqdm(ds, desc="Extracting PhotonicTopology", total=total):
            if self._max_samples is not None and len(grids_list) >= self._max_samples:
                break

            grid = self._extract_grid(record)
            label = self._extract_label(record)

            if grid is not None:
                grids_list.append(grid)
                labels_list.append(label)

        if not grids_list:
            raise RuntimeError(
                f"No valid grids extracted from photonic topology. "
                f"Dataset columns: {ds.column_names}"
            )

        grids = np.stack(grids_list, axis=0).astype(np.float32)
        labels = np.asarray(labels_list, dtype=np.int64)

        log.info(
            "PhotonicAdapter: loaded %d grids, shape=%s, classes=%d",
            grids.shape[0],
            grids.shape,
            len(np.unique(labels)),
        )
        return grids, labels


    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_grid(record: dict) -> np.ndarray | None:
        """Extract a (H, W, F) grid from a HuggingFace record."""
        for key in ("grid", "unit_cell", "image", "features", "data"):
            if key in record:
                val = record[key]
                if isinstance(val, np.ndarray) and val.ndim in (2, 3):
                    if val.ndim == 2:
                        val = val[:, :, np.newaxis]
                    return val
                if isinstance(val, list):
                    arr = np.asarray(val, dtype=np.float32)
                    if arr.ndim in (2, 3):
                        if arr.ndim == 2:
                            arr = arr[:, :, np.newaxis]
                        return arr
        return None

    @staticmethod
    def _extract_label(record: dict) -> int:
        """Extract an integer label from a HuggingFace record."""
        for key in ("label", "topology", "symmetry", "class", "category"):
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


register_adapter("photonic", PhotonicAdapter)


__all__ = ["PhotonicAdapter"]
