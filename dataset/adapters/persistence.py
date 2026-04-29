"""Dataset Bundle Persistence.

Provides functions to save and load prepared ``DatasetBundle`` objects
to/from disk as compressed numpy archives, allowing for instant
resumption of training without re-extracting from raw sources.
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
) -> Path:
    """Generate a unique directory path for a prepared dataset split."""
    root = Path(data_root) / dataset_name / "prepared"
    # Construct a unique configuration ID
    params = f"T{spec.sequence_length}_D{spec.input_dim}_S{seed}_R{int(train_ratio*100)}_{int(val_ratio*100)}"
    if spec.max_samples is not None:
        params += f"_M{spec.max_samples}"
    
    return root / params


def save_prepared_bundle(cache_dir: Path, bundle: DatasetBundle) -> None:
    """Save a DatasetBundle to a modular directory structure."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save individual splits
        for split in ["train", "val", "test"]:
            seqs = getattr(bundle, f"{split}_sequences")
            labels = getattr(bundle, f"{split}_labels")
            np.savez_compressed(
                cache_dir / f"{split}.npz",
                sequences=seqs,
                labels=labels
            )
        
        # 2. Save spec and metadata
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
                "metadata": bundle.metadata
            }, f, indent=2)
            
        log.info("Saved modular prepared bundle to: %s", cache_dir)
    except Exception as e:
        log.warning("Failed to save modular prepared bundle: %s", e)


def load_prepared_bundle(cache_dir: Path, expected_spec: DatasetSpec) -> DatasetBundle | None:
    """Load a DatasetBundle from a modular directory structure."""
    if not cache_dir.is_dir():
        return None

    try:
        # 1. Verify spec
        spec_path = cache_dir / "bundle_spec.json"
        if spec_path.exists():
            with open(spec_path, "r") as f:
                cached_data = json.load(f)
                cached_spec = cached_data["spec"]
                if (cached_spec["input_dim"] != expected_spec.input_dim or 
                    cached_spec["sequence_length"] != expected_spec.sequence_length):
                    log.warning("Spec mismatch in %s. Re-extracting.", cache_dir.name)
                    return None
                metadata = cached_data.get("metadata", {})
        else:
            metadata = {}

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
            metadata=metadata
        )
        log.info("Loaded modular prepared bundle from: %s", cache_dir.name)
        return bundle
    except Exception as e:
        log.warning("Failed to load modular bundle from %s: %s", cache_dir, e)
        return None
