"""Collate Function for Trajectory Batches.

Stacks per-sample dicts into a single batch dict with
``"sequences"``, ``"targets"``, and ``"lengths"`` tensors.
"""

from __future__ import annotations

import torch


def trajectory_collate_fn(batch):
    """Collate a list of sample dicts into a batch dict.

    Parameters
    ----------
    batch : list of dict
        Each dict has keys ``"sequence"``, ``"target"``, ``"length"``.

    Returns
    -------
    dict
        Keys ``"sequences"`` (B, T, d), ``"targets"`` (B,),
        ``"lengths"`` (B,).
    """
    sequences = torch.stack([item["sequence"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    lengths = torch.as_tensor([item["length"] for item in batch], dtype=torch.long)
    return {"sequences": sequences, "targets": targets, "lengths": lengths}
