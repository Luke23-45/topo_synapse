"""Trajectory Dataset.

Wraps pre-split numpy arrays as a PyTorch ``Dataset`` that yields
per-sample dicts suitable for the Z3 training collate function.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Fixed-length trajectory dataset for Z3 models.

    Parameters
    ----------
    sequences : np.ndarray
        Shape ``(N, T, d)`` — observation sequences.
    labels : np.ndarray
        Shape ``(N,)`` — integer class labels.
    """

    def __init__(self, sequences, labels) -> None:
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int):
        return {
            "sequence": self.sequences[index],
            "target": self.labels[index],
            "length": self.sequences[index].shape[0],
        }
