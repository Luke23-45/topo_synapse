"""Shared dataset splitting and official-split extraction utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def apply_split(
    arrays: np.ndarray,
    labels: np.ndarray,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split arrays into train/val/test using stratified class sampling."""
    if arrays.shape[0] != labels.shape[0]:
        raise ValueError("arrays and labels must have the same first dimension")
    if arrays.shape[0] == 0:
        raise ValueError("cannot split an empty dataset")

    rng = np.random.default_rng(seed)
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)

        n_cls = int(cls_idx.shape[0])
        n_train = int(round(n_cls * train_ratio))
        n_val = int(round(n_cls * val_ratio))
        n_train = min(n_train, n_cls)
        n_val = min(n_val, max(n_cls - n_train, 0))

        split_indices["train"].extend(cls_idx[:n_train].tolist())
        split_indices["val"].extend(cls_idx[n_train:n_train + n_val].tolist())
        split_indices["test"].extend(cls_idx[n_train + n_val:].tolist())

    materialized: dict[str, np.ndarray] = {}
    materialized_labels: dict[str, np.ndarray] = {}
    for split_name, idx_list in split_indices.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        rng.shuffle(idx)
        materialized[split_name] = arrays[idx]
        materialized_labels[split_name] = labels[idx]

    return materialized, materialized_labels


def extract_predefined_splits(
    ds: Any,
    *,
    extract_array: Callable[[dict[str, Any]], np.ndarray | None],
    extract_label: Callable[[dict[str, Any]], int],
    max_samples: int | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Use official dataset splits when available, else return ``None``.

    Supports HuggingFace ``DatasetDict``-style objects with canonical or
    near-canonical split names. If only train/test are present, a stratified
    validation split is carved out of train. Datasets that expose a single
    physical split with an embedded row-level ``split`` field are also
    supported.
    """
    embedded = _extract_embedded_splits(
        ds,
        extract_array=extract_array,
        extract_label=extract_label,
        max_samples=max_samples,
    )
    if embedded is not None:
        return embedded

    if not hasattr(ds, "keys"):
        return None

    split_map: dict[str, Any] = {}
    for raw_name in ds.keys():
        canonical = _canonical_split_name(str(raw_name))
        if canonical is not None and canonical not in split_map:
            split_map[canonical] = ds[raw_name]

    if "train" not in split_map:
        return None

    if len(split_map) == 1 and "train" in split_map:
        embedded = _extract_embedded_splits(
            split_map["train"],
            extract_array=extract_array,
            extract_label=extract_label,
            max_samples=max_samples,
        )
        if embedded is not None:
            return embedded

    if "test" not in split_map and "val" not in split_map:
        return None

    extracted: dict[str, np.ndarray] = {}
    extracted_labels: dict[str, np.ndarray] = {}

    budgets = _allocate_split_budgets(split_map, max_samples)
    for split_name, split_ds in split_map.items():
        limit = budgets.get(split_name)
        arr, lbl = _extract_records(split_ds, extract_array, extract_label, limit)
        if arr.size == 0:
            continue
        extracted[split_name] = arr
        extracted_labels[split_name] = lbl

    if "train" not in extracted:
        return None

    if "val" not in extracted and "test" in extracted:
        train_arrays, train_labels = apply_split(
            extracted["train"],
            extracted_labels["train"],
            train_ratio=train_ratio / max(train_ratio + val_ratio, 1e-8),
            val_ratio=val_ratio / max(train_ratio + val_ratio, 1e-8),
            seed=42,
        )
        extracted["train"] = train_arrays["train"]
        extracted_labels["train"] = train_labels["train"]
        extracted["val"] = train_arrays["val"]
        extracted_labels["val"] = train_labels["val"]

    if "test" not in extracted and "val" in extracted:
        extracted["test"] = extracted["val"]
        extracted_labels["test"] = extracted_labels["val"]

    if {"train", "val", "test"} <= extracted.keys():
        return (
            {
                "train": extracted["train"],
                "val": extracted["val"],
                "test": extracted["test"],
            },
            {
                "train": extracted_labels["train"],
                "val": extracted_labels["val"],
                "test": extracted_labels["test"],
            },
        )

    return None


def try_load_from_local(
    data_root: str | None,
    dataset_name: str,
) -> Any:
    """Try to load pre-saved raw data from the local data directory."""
    if data_root is None:
        return None

    from pathlib import Path

    from ..download_sources import is_downloaded

    if not is_downloaded(dataset_name, data_root):
        return None

    raw_dir = Path(data_root) / dataset_name / "raw"
    if not raw_dir.is_dir():
        return None

    try:
        from datasets import load_from_disk
    except ImportError:
        return None

    try:
        ds = load_from_disk(str(raw_dir))
    except Exception:
        return None

    return ds


def _canonical_split_name(name: str) -> str | None:
    lowered = name.lower()
    if lowered in {"train", "training"}:
        return "train"
    if lowered in {"validation", "val", "valid", "dev"}:
        return "val"
    if lowered in {"test", "testing", "eval", "evaluation"}:
        return "test"
    return None


def _allocate_split_budgets(split_map: dict[str, Any], max_samples: int | None) -> dict[str, int | None]:
    if max_samples is None:
        return {name: None for name in split_map}

    sizes = {name: len(split_ds) for name, split_ds in split_map.items()}
    total = sum(sizes.values())
    if total <= 0:
        return {name: 0 for name in split_map}

    budgets: dict[str, int] = {}
    remaining = max_samples
    ordered = ["train", "val", "test"]
    for idx, split_name in enumerate(ordered):
        if split_name not in sizes:
            continue
        if idx == len([s for s in ordered if s in sizes]) - 1:
            budgets[split_name] = min(remaining, sizes[split_name])
            continue
        portion = int(round(max_samples * (sizes[split_name] / total)))
        portion = max(1, min(portion, sizes[split_name], remaining))
        budgets[split_name] = portion
        remaining -= portion
    for split_name in split_map:
        budgets.setdefault(split_name, min(remaining, sizes[split_name]))
    return budgets


def _extract_records(
    split_ds: Any,
    extract_array: Callable[[dict[str, Any]], np.ndarray | None],
    extract_label: Callable[[dict[str, Any]], int],
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    arrays_list: list[np.ndarray] = []
    labels_list: list[int] = []

    for record in split_ds:
        if limit is not None and len(arrays_list) >= limit:
            break
        arr = extract_array(record)
        if arr is None:
            continue
        label = int(extract_label(record))
        if label < 0:
            continue
        arrays_list.append(arr.astype(np.float32, copy=False))
        labels_list.append(label)

    if not arrays_list:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return (
        np.stack(arrays_list, axis=0).astype(np.float32),
        np.asarray(labels_list, dtype=np.int64),
    )


def _extract_embedded_splits(
    ds: Any,
    *,
    extract_array: Callable[[dict[str, Any]], np.ndarray | None],
    extract_label: Callable[[dict[str, Any]], int],
    max_samples: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    if hasattr(ds, "keys"):
        return None

    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    saw_embedded_split = False

    for record in ds:
        split_name = _record_split_name(record)
        if split_name is None:
            continue
        saw_embedded_split = True
        split_records[split_name].append(record)

    if not saw_embedded_split or not split_records["train"]:
        return None
    if not split_records["val"] and not split_records["test"]:
        return None

    budgets = _allocate_record_budgets(split_records, max_samples)
    extracted: dict[str, np.ndarray] = {}
    extracted_labels: dict[str, np.ndarray] = {}

    for split_name in ("train", "val", "test"):
        arr, lbl = _extract_records(
            split_records[split_name],
            extract_array,
            extract_label,
            budgets.get(split_name),
        )
        if arr.size == 0:
            continue
        extracted[split_name] = arr
        extracted_labels[split_name] = lbl

    if {"train", "val", "test"} <= extracted.keys():
        return (
            {
                "train": extracted["train"],
                "val": extracted["val"],
                "test": extracted["test"],
            },
            {
                "train": extracted_labels["train"],
                "val": extracted_labels["val"],
                "test": extracted_labels["test"],
            },
        )
    return None


def _record_split_name(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    for key in ("split", "partition", "fold"):
        raw_name = record.get(key)
        if raw_name is None:
            continue
        canonical = _canonical_split_name(str(raw_name).strip())
        if canonical is not None:
            return canonical
    return None


def _allocate_record_budgets(
    split_records: dict[str, list[dict[str, Any]]],
    max_samples: int | None,
) -> dict[str, int | None]:
    if max_samples is None:
        return {name: None for name in split_records}

    sizes = {name: len(records) for name, records in split_records.items()}
    total = sum(sizes.values())
    if total <= 0:
        return {name: 0 for name in split_records}

    budgets: dict[str, int] = {}
    remaining = max_samples
    present = [name for name in ("train", "val", "test") if sizes[name] > 0]
    for idx, split_name in enumerate(present):
        if idx == len(present) - 1:
            budgets[split_name] = min(remaining, sizes[split_name])
            continue
        portion = int(round(max_samples * (sizes[split_name] / total)))
        portion = max(1, min(portion, sizes[split_name], remaining))
        budgets[split_name] = portion
        remaining -= portion
    for split_name in split_records:
        budgets.setdefault(split_name, 0)
    return budgets


__all__ = ["apply_split", "extract_predefined_splits", "try_load_from_local"]
