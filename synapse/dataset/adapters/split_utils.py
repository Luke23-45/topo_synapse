"""Shared dataset splitting utilities.

Provides a single ``apply_split()`` function used by all HuggingFace
adapters, eliminating the duplicated ratio-based split logic that was
previously copy-pasted across TelecomAdapter, SpatialAdapter, and
PhotonicAdapter.

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §4.3
"""

from __future__ import annotations

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
    """Split data into train/val/test by ratio.

    Parameters
    ----------
    arrays : np.ndarray
        Shape ``(N, ...)`` — data arrays to split along the first axis.
    labels : np.ndarray
        Shape ``(N,)`` — label array to split along the first axis.
    train_ratio : float
        Fraction of data used for training (default 0.8).
    val_ratio : float
        Fraction of data used for validation (default 0.1).
        The remainder goes to test.
    seed : int
        Random seed for reproducible splitting.

    Returns
    -------
    arrays_split : dict
        Keys ``"train"``, ``"val"``, ``"test"``.
    labels_split : dict
        Same keys.
    """
    rng = np.random.default_rng(seed)
    n = arrays.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        {
            "train": arrays[train_idx],
            "val": arrays[val_idx],
            "test": arrays[test_idx],
        },
        {
            "train": labels[train_idx],
            "val": labels[val_idx],
            "test": labels[test_idx],
        },
    )


def try_load_from_local(
    data_root: str | None,
    dataset_name: str,
) -> Any:
    """Try to load pre-saved raw data from the local data directory.

    Checks ``<data_root>/<dataset_name>/raw/`` for an Arrow dataset
    saved by ``download_sources.py``.  Returns a HuggingFace
    ``Dataset`` (or ``DatasetDict``) if found, or ``None`` if no
    local data is available.

    Parameters
    ----------
    data_root : str or None
        Root directory for downloaded datasets.
    dataset_name : str
        Canonical dataset name (e.g. ``"telecom"``).

    Returns
    -------
    datasets.Dataset or datasets.DatasetDict or None
        Loaded HuggingFace dataset object, or ``None``.
    """
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


__all__ = ["apply_split", "try_load_from_local"]
