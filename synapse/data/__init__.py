"""Z3 Training Data Package.

Provides DataLoader construction via the adapter registry, along with
supporting utilities for normalization, collation, and dataset wrapping.
"""

from __future__ import annotations

from .collate import trajectory_collate_fn
from .data import build_dataloaders
from .normalization import compute_normalization_stats
from .trajectory_dataset import TrajectoryDataset

__all__ = [
    "TrajectoryDataset",
    "build_dataloaders",
    "compute_normalization_stats",
    "trajectory_collate_fn",
]
