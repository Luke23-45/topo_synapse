"""Dataset Bundle Persistence.

Two-tier caching strategy:

1. **Raw data cache** — stores the extracted (sequences, labels) arrays
   *before* any seed-dependent splitting.  Keyed by dataset name,
   input_dim, and max_samples (no seed).  This avoids re-downloading
   from HuggingFace when only the seed changes.

2. **Prepared bundle cache** — stores the final split + preprocessed
   arrays, keyed by seed + ratios.  Allows instant resumption of
   training for a specific seed without re-splitting or
   re-preprocessing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import DatasetBundle, DatasetSpec

log = logging.getLogger(__name__)


def get_prepared_cache_dir(
    data_root: str | Path,
    dataset_name: str,
    spec: DatasetSpec,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    *,
    extra_key: str | None = None,
) -> Path:
    """Generate a unique directory path for a prepared dataset split."""
    root = Path(data_root) / dataset_name / "prepared"
    # Construct a unique configuration ID
    params = f"T{spec.sequence_length}_D{spec.input_dim}_S{seed}_R{int(train_ratio*100)}_{int(val_ratio*100)}"
    if spec.max_samples is not None:
        params += f"_M{spec.max_samples}"
    if extra_key:
        safe_extra = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in extra_key)
        params += f"_{safe_extra}"
    
    return root / params


def save_prepared_bundle(cache_dir: Path, bundle: DatasetBundle) -> None:
    """Save a DatasetBundle to a modular directory structure.

    Numpy arrays in metadata are saved as separate ``.npz`` files under
    a ``metadata_arrays/`` sub-directory so that the JSON spec file only
    contains JSON-serializable scalars and strings.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save individual splits
        for split in ["train", "val", "test"]:
            seqs = getattr(bundle, f"{split}_sequences")
            labels = getattr(bundle, f"{split}_labels")
            np.savez_compressed(
                cache_dir / f"{split}.npz",
                sequences=seqs,
                labels=labels,
            )

        # 2. Separate ndarray values from JSON-safe metadata
        json_metadata: dict[str, Any] = {}
        array_keys: list[str] = []

        for key, value in bundle.metadata.items():
            if isinstance(value, np.ndarray):
                array_keys.append(key)
            else:
                json_metadata[key] = value

        # Save ndarray metadata values as individual .npz files
        if array_keys:
            arr_dir = cache_dir / "metadata_arrays"
            arr_dir.mkdir(exist_ok=True)
            for key in array_keys:
                np.save(arr_dir / f"{key}.npy", bundle.metadata[key])

        # 3. Save spec and JSON-safe metadata
        spec_dict = {
            "name": bundle.spec.name,
            "modality": bundle.spec.modality,
            "input_dim": bundle.spec.input_dim,
            "sequence_length": bundle.spec.sequence_length,
            "num_classes": bundle.spec.num_classes,
            "task": bundle.spec.task,
        }

        with open(cache_dir / "bundle_spec.json", "w") as f:
            json.dump({
                "spec": spec_dict,
                "metadata": json_metadata,
                "metadata_array_keys": array_keys,
            }, f, indent=2)

        log.info("Saved modular prepared bundle to: %s", cache_dir)
    except Exception as e:
        log.warning("Failed to save modular prepared bundle: %s", e)


def load_prepared_bundle(cache_dir: Path, expected_spec: DatasetSpec) -> DatasetBundle | None:
    """Load a DatasetBundle from a modular directory structure.

    Reconstructs ndarray metadata values from the ``metadata_arrays/``
    sub-directory if present.
    """
    if not cache_dir.is_dir():
        return None

    try:
        # 1. Verify spec
        spec_path = cache_dir / "bundle_spec.json"
        metadata: dict[str, Any] = {}
        if spec_path.exists():
            with open(spec_path, "r") as f:
                cached_data = json.load(f)
                cached_spec = cached_data["spec"]
                if (cached_spec["input_dim"] != expected_spec.input_dim or
                    cached_spec["sequence_length"] != expected_spec.sequence_length):
                    log.warning("Spec mismatch in %s. Re-extracting.", cache_dir.name)
                    return None
                metadata = cached_data.get("metadata", {})

            # Reconstruct ndarray metadata values
            array_keys = cached_data.get("metadata_array_keys", [])
            if array_keys:
                arr_dir = cache_dir / "metadata_arrays"
                for key in array_keys:
                    arr_path = arr_dir / f"{key}.npy"
                    if arr_path.exists():
                        metadata[key] = np.load(arr_path)

        # 2. Load splits
        splits = {}
        for split in ["train", "val", "test"]:
            path = cache_dir / f"{split}.npz"
            if not path.exists():
                return None
            data = np.load(path)
            splits[f"{split}_sequences"] = data["sequences"]
            splits[f"{split}_labels"] = data["labels"]

        bundle = DatasetBundle(
            train_sequences=splits["train_sequences"],
            train_labels=splits["train_labels"],
            val_sequences=splits["val_sequences"],
            val_labels=splits["val_labels"],
            test_sequences=splits["test_sequences"],
            test_labels=splits["test_labels"],
            spec=expected_spec,
            metadata=metadata,
        )
        log.info("Loaded modular prepared bundle from: %s", cache_dir.name)
        return bundle
    except Exception as e:
        log.warning("Failed to load modular bundle from %s: %s", cache_dir, e)
        return None


# ---------------------------------------------------------------------------
# Raw data cache (seed-independent)
# ---------------------------------------------------------------------------

def get_raw_cache_dir(
    data_root: str | Path,
    dataset_name: str,
    spec: DatasetSpec,
) -> Path:
    """Generate a unique directory path for raw (un-split) data.

    The key does **not** include the seed, so the same raw cache is
    reused across all seeds for a given dataset configuration.
    """
    root = Path(data_root) / dataset_name / "raw"
    params = f"D{spec.input_dim}"
    if spec.max_samples is not None:
        params += f"_M{spec.max_samples}"
    return root / params


def save_raw_data(
    cache_dir: Path,
    sequences: np.ndarray,
    labels: np.ndarray,
    spec: DatasetSpec,
) -> None:
    """Save raw (un-split) sequences + labels to disk.

    This is called once after the initial download/extraction from
    HuggingFace.  Subsequent seeds load from this cache instead of
    re-downloading.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_dir / "raw.npz",
            sequences=sequences,
            labels=labels,
        )
        spec_dict = {
            "name": spec.name,
            "input_dim": spec.input_dim,
            "num_classes": spec.num_classes,
        }
        with open(cache_dir / "raw_spec.json", "w") as f:
            json.dump(spec_dict, f, indent=2)
        log.info("Saved raw data cache to: %s", cache_dir)
    except Exception as e:
        log.warning("Failed to save raw data cache: %s", e)


def load_raw_data(
    cache_dir: Path,
    expected_spec: DatasetSpec,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load raw (un-split) sequences + labels from disk.

    Returns ``(sequences, labels)`` if a valid cache exists, or ``None``
    if the cache is missing or has mismatched dimensions.
    """
    npz_path = cache_dir / "raw.npz"
    if not npz_path.exists():
        return None

    try:
        # Verify spec
        spec_path = cache_dir / "raw_spec.json"
        if spec_path.exists():
            with open(spec_path, "r") as f:
                cached = json.load(f)
            if cached.get("input_dim") != expected_spec.input_dim:
                log.warning("Raw cache spec mismatch in %s. Re-extracting.", cache_dir.name)
                return None

        data = np.load(npz_path)
        sequences = data["sequences"]
        labels = data["labels"]
        log.info("Loaded raw data cache from: %s", cache_dir.name)
        return sequences, labels
    except Exception as e:
        log.warning("Failed to load raw data cache from %s: %s", cache_dir, e)
        return None
