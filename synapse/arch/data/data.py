"""Z3 Training Data Pipeline.

Builds DataLoaders from the adapter registry so that the training
engine can consume any registered dataset by name.  The legacy
synthetic-only path is preserved as the default fallback.

The primary entry point is ``build_dataloaders()``, which:
1. Resolves the dataset name from the config.
2. Instantiates the adapter via the registry.
3. Loads and preprocesses the data into a ``DatasetBundle``.
4. Wraps the arrays in ``TrajectoryDataset`` and ``DataLoader``.
5. Computes normalization statistics for the lift layer.

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §5, Data Flow
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from synapse.dataset.adapters.base import DatasetBundle
from synapse.dataset.registry import create_adapter

from .collate import trajectory_collate_fn
from .normalization import compute_normalization_stats
from .trajectory_dataset import TrajectoryDataset

log = logging.getLogger(__name__)


def build_dataloaders(
    cfg: Any,
    *,
    dataset_name: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, DatasetBundle]:
    """Build train/val/test DataLoaders from the adapter registry.

    Parameters
    ----------
    cfg : Any
        Configuration object.  Expected to have ``model``, ``data``,
        ``training``, and ``execution`` attributes (OmegaConf or
        dataclass).  If ``dataset_name`` is not provided, the config
        must have ``cfg.data.dataset`` or defaults to ``"synthetic"``.
    dataset_name : str or None
        Override the dataset name.  If ``None``, reads from
        ``cfg.data.dataset`` or falls back to ``"synthetic"``.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    bundle : DatasetBundle
        The pre-split dataset arrays and spec.
    """
    name = dataset_name or getattr(getattr(cfg, "data", None), "dataset", None) or "synthetic"

    log.info("Building dataloaders for dataset: %s", name)

    # Resolve adapter kwargs from config.
    adapter_kwargs = _adapter_kwargs_from_cfg(cfg, name)

    adapter = create_adapter(name, **adapter_kwargs)
    bundle = adapter.load_splits()

    log.info(
        "Dataset '%s': train=%d, val=%d, test=%d, input_dim=%d, seq_len=%d",
        name,
        bundle.train_size,
        bundle.val_size,
        bundle.test_size,
        bundle.input_dim,
        bundle.sequence_length,
    )

    batch_size = getattr(getattr(cfg, "training", None), "batch_size", 32)
    if bundle.spec.batch_size_override is not None:
        batch_size = bundle.spec.batch_size_override

    train_ds = TrajectoryDataset(bundle.train_sequences, bundle.train_labels)
    val_ds = TrajectoryDataset(bundle.val_sequences, bundle.val_labels)
    test_ds = TrajectoryDataset(bundle.test_sequences, bundle.test_labels)

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": trajectory_collate_fn,
    }

    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
        bundle,
    )


def _adapter_kwargs_from_cfg(cfg: Any, name: str) -> dict[str, Any]:
    """Extract adapter constructor kwargs from the config.

    Parameters
    ----------
    cfg : Any
        Configuration object.
    name : str
        Dataset name (used for default values).

    Returns
    -------
    dict
        Keyword arguments for the adapter constructor.
    """
    data_cfg = getattr(cfg, "data", None)
    model_cfg = getattr(cfg, "model", None)
    exec_cfg = getattr(cfg, "execution", None)

    kwargs: dict[str, Any] = {}

    if name == "synthetic":
        kwargs["train_size"] = getattr(data_cfg, "train_size", 512)
        kwargs["val_size"] = getattr(data_cfg, "val_size", 128)
        kwargs["test_size"] = getattr(data_cfg, "test_size", 128)
        kwargs["length"] = getattr(model_cfg, "max_history_tokens", 128)
        kwargs["noise_std"] = getattr(data_cfg, "noise_std", 0.03)
        kwargs["seed"] = getattr(exec_cfg, "seed", 7)
    else:
        # External adapters use target_length from model config.
        kwargs["target_length"] = getattr(model_cfg, "max_history_tokens", 128)
        kwargs["seed"] = getattr(exec_cfg, "seed", 42)
        max_samples = getattr(data_cfg, "max_samples", None)
        if max_samples is not None:
            kwargs["max_samples"] = max_samples
        data_root = getattr(data_cfg, "data_root", None)
        if data_root is not None:
            kwargs["data_root"] = data_root

    return kwargs


__all__ = [
    "build_dataloaders",
]
