"""TelecomTS Adapter.

Loads the TelecomTS dataset from HuggingFace and produces a
``DatasetBundle`` for the Z3 temporal evaluation track.

The dataset contains 32,000 samples from a live 5G network with
multivariate time-series observations (signal strength, latency,
packet loss, + derived features).  Z3 uses the ``CausalEventEncoder``
to detect topological invariants in telemetry state-space trajectories.

Source: https://huggingface.co/datasets/AliMaatouk/TelecomTS

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §3, Track 1
- Data source doc: ``docs/dev/data_source.md`` §2
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ..preprocess.temporal import TemporalPreprocessor
from ..registry import register_adapter
from .base import DatasetBundle, DatasetSpec, Z3Adapter
from .persistence import (
    get_prepared_cache_dir,
    get_raw_cache_dir,
    load_prepared_bundle,
    load_raw_data,
    save_prepared_bundle,
    save_raw_data,
)
from .split_utils import apply_split, try_load_from_local

log = logging.getLogger(__name__)


class TelecomAdapter(Z3Adapter):
    """Adapter for the TelecomTS 5G telemetry dataset.

    Parameters
    ----------
    target_length : int
        Fixed sequence length after padding/truncation.
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
        target_length: int = 256,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: int | None = None,
        data_root: str | None = None,
        input_dim: int = 16,
    ) -> None:
        self._target_length = target_length
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._seed = seed
        self._max_samples = max_samples
        self._data_root = data_root

        self._spec = DatasetSpec(
            name="telecom",
            modality="temporal",
            source="huggingface",
            hf_repo="AliMaatouk/TelecomTS",
            input_dim=input_dim,
            sequence_length=target_length,
            num_classes=3,
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
        """Load TelecomTS and return pre-split arrays (with two-tier caching).

        Cache resolution order:
        1. **Prepared bundle** (seed-specific) — instant if found.
        2. **Raw data cache** (seed-independent) — skip HF download,
           only re-split and preprocess.
        3. **HuggingFace download** — first run only; saves to both
           raw cache and prepared bundle cache.

        Returns
        -------
        DatasetBundle
        """
        data_root = Path(self._data_root or "data/datasets").resolve()

        # 1. Check for final prepared splits (fastest path)
        prepared_path = get_prepared_cache_dir(
            data_root, "telecom", self._spec, self._seed,
            self._train_ratio, self._val_ratio,
        )
        bundle = load_prepared_bundle(prepared_path, self._spec)
        if bundle is not None:
            return bundle

        # 2. Check raw data cache (skip HF download)
        raw_path = get_raw_cache_dir(data_root, "telecom", self._spec)
        raw_data = load_raw_data(raw_path, self._spec)

        if raw_data is not None:
            log.info("Raw data cache hit — skipping HuggingFace download.")
            raw_sequences, raw_labels = raw_data
        else:
            # 3. Download / extract from HuggingFace (first run only)
            log.info("No raw cache found for TelecomTS. Downloading...")
            raw_sequences, raw_labels = self._load_data()
            # Save raw data for future seeds
            save_raw_data(raw_path, raw_sequences, raw_labels, self._spec)

        # 4. Apply seed-dependent split
        sequences, labels = apply_split(
            raw_sequences,
            raw_labels,
            train_ratio=self._train_ratio,
            val_ratio=self._val_ratio,
            seed=self._seed,
        )

        # 5. Preprocess
        preprocessor = TemporalPreprocessor(target_length=self._target_length)
        bundle = DatasetBundle(
            train_sequences=sequences["train"],
            train_labels=labels["train"],
            val_sequences=sequences["val"],
            val_labels=labels["val"],
            test_sequences=sequences["test"],
            test_labels=labels["test"],
            spec=self._spec,
        )
        bundle = preprocessor(bundle)

        # 6. Save prepared bundle for this seed
        save_prepared_bundle(prepared_path, bundle)

        return bundle

    # ------------------------------------------------------------------ #

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load sequences + labels, preferring local data over HuggingFace.

        Returns
        -------
        sequences : np.ndarray
            Shape ``(N_total, T_raw, d)``.
        labels : np.ndarray
            Shape ``(N_total,)``.
        """
        # --- Try local disk first ---
        local_ds = try_load_from_local(self._data_root, "telecom")
        if local_ds is not None:
            log.info("TelecomAdapter: loading from local data_root=%s", self._data_root)
            return self._extract_from_dataset(local_ds)

        # --- Fall back to HuggingFace ---
        return self._load_from_huggingface()

    def _load_from_huggingface(self) -> tuple[np.ndarray, np.ndarray]:
        """Download and extract sequences + labels from HuggingFace.

        Returns
        -------
        sequences : np.ndarray
            Shape ``(N_total, T_raw, d)``.
        labels : np.ndarray
            Shape ``(N_total,)``.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load TelecomTS. "
                "Install it with: pip install datasets"
            ) from exc

        log.info("Loading TelecomTS from HuggingFace: %s", self._spec.hf_repo)
        ds = load_dataset(self._spec.hf_repo, split="train").with_format("numpy")

        return self._extract_from_dataset(ds)

    def _extract_from_dataset(self, ds) -> tuple[np.ndarray, np.ndarray]:
        """Extract sequences + labels from a HuggingFace Dataset object.

        Parameters
        ----------
        ds : datasets.Dataset or datasets.DatasetDict
            Loaded dataset (from local disk or HuggingFace).

        Returns
        -------
        sequences : np.ndarray
            Shape ``(N_total, T_raw, d)``.
        labels : np.ndarray
            Shape ``(N_total,)``.
        """
        if hasattr(ds, "values"):
            # DatasetDict — concatenate all splits.
            from datasets import concatenate_datasets
            ds = concatenate_datasets(list(ds.values()))

        sequences_list: list[np.ndarray] = []
        labels_list: list[int] = []

        total = min(len(ds), self._max_samples) if self._max_samples else len(ds)
        for record in tqdm(ds, desc="Extracting TelecomTS", total=total):
            if self._max_samples is not None and len(sequences_list) >= self._max_samples:
                break

            seq = self._extract_sequence(record)
            label = self._extract_label(record)

            if seq is not None:
                sequences_list.append(seq)
                labels_list.append(label)

        if not sequences_list:
            raise RuntimeError(
                f"No valid sequences extracted from TelecomTS. "
                f"Dataset columns: {ds.column_names}"
            )

        sequences = np.stack(sequences_list, axis=0).astype(np.float32)
        labels = np.asarray(labels_list, dtype=np.int64)

        # Self-healing: update spec if dimension mismatch
        if sequences.shape[-1] != self._spec.input_dim:
            log.warning(
                "Dimension mismatch: spec.input_dim=%d, actual=%d. Updating spec.",
                self._spec.input_dim, sequences.shape[-1]
            )
            self._spec.input_dim = sequences.shape[-1]

        log.info(
            "TelecomAdapter: loaded %d samples, shape=%s, classes=%d",
            sequences.shape[0],
            sequences.shape,
            len(np.unique(labels)),
        )
        return sequences, labels

    # ------------------------------------------------------------------ #
    # Column extraction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_sequence(record: dict) -> np.ndarray | None:
        """Extract a (T, d) sequence from a HuggingFace record.

        TelecomTS structure: record['KPIs'] is a dict of (T,) arrays.
        """
        kpis = record.get("KPIs")
        if isinstance(kpis, dict):
            # Sort keys for consistent feature ordering
            feature_keys = sorted(kpis.keys())
            arrays = []
            for k in feature_keys:
                val = kpis[k]
                if isinstance(val, (np.ndarray, list)):
                    # Check if numerical
                    try:
                        arr = np.asarray(val, dtype=np.float32)
                        arrays.append(arr)
                    except (ValueError, TypeError):
                        # Skip categorical/string KPIs
                        continue
            
            if arrays:
                # Stack to (T, d)
                stacked = np.stack(arrays, axis=-1)
                return stacked
        
        # Fallback for other formats
        for key in ("sequence", "timeseries", "trajectory", "features", "data"):
            if key in record:
                val = record[key]
                if isinstance(val, np.ndarray) and val.ndim >= 2:
                    return val
                if isinstance(val, list):
                    arr = np.asarray(val, dtype=np.float32)
                    if arr.ndim >= 2:
                        return arr
        return None

    @staticmethod
    def _extract_label(record: dict) -> int:
        """Extract an integer label from a HuggingFace record.

        TelecomTS structure: record['labels'] is a dict.
        Target: 'application' (mapped to {0..4}).
        """
        labels_dict = record.get("labels")
        if isinstance(labels_dict, dict):
            # Map application names to IDs if they are strings, or use directly if int
            val = labels_dict.get("application", labels_dict.get("Application"))
            if val is not None:
                if isinstance(val, (int, np.integer)):
                    return int(val)
                # Actual classes in AliMaatouk/TelecomTS are: Youtube, Twitch, File
                # Mapping: Youtube=0, Twitch=1, File=2
                app_map = {"youtube": 0, "twitch": 1, "file": 2}
                if isinstance(val, str):
                    val_lower = val.lower()
                    for k, v in app_map.items():
                        if k in val_lower:
                            return v
                    return 2 # Default to File/Other if unknown

        # Generic fallback
        keys = {k.lower(): k for k in record.keys()}
        for target in ("labels", "label", "target", "class", "category"):
            if target in keys:
                val = record[keys[target]]
                if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                    if len(val) > 0:
                        try:
                            val = val[0]
                        except (KeyError, TypeError, IndexError):
                            pass
                try:
                    return int(val)
                except (ValueError, TypeError):
                    continue
        return 0


register_adapter("telecom", TelecomAdapter)


__all__ = ["TelecomAdapter"]
