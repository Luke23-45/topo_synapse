from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


TOPOLOGY_LABELS = {
    "line": 0,
    "circle": 1,
    "figure_eight": 2,
    "branch": 3,
}


def _line_sequence(length: int, noise_std: float) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, length, dtype=np.float64)
    seq = np.stack([t, 0.25 * t], axis=1)
    return seq + np.random.normal(scale=noise_std, size=seq.shape)


def _circle_sequence(length: int, noise_std: float) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, length, dtype=np.float64)
    seq = np.stack([np.cos(t), np.sin(t)], axis=1)
    return seq + np.random.normal(scale=noise_std, size=seq.shape)


def _figure_eight_sequence(length: int, noise_std: float) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, length, dtype=np.float64)
    seq = np.stack([np.sin(t), np.sin(t) * np.cos(t)], axis=1)
    return seq + np.random.normal(scale=noise_std, size=seq.shape)


def _branch_sequence(length: int, noise_std: float) -> np.ndarray:
    half = length // 2
    left = np.stack(
        [
            np.linspace(-1.0, 0.0, half, dtype=np.float64),
            np.zeros(half, dtype=np.float64),
        ],
        axis=1,
    )
    right = np.stack(
        [
            np.linspace(0.0, 1.0, length - half, dtype=np.float64),
            np.linspace(0.0, 1.0, length - half, dtype=np.float64),
        ],
        axis=1,
    )
    seq = np.concatenate([left, right], axis=0)
    return seq + np.random.normal(scale=noise_std, size=seq.shape)


GENERATORS = {
    "line": _line_sequence,
    "circle": _circle_sequence,
    "figure_eight": _figure_eight_sequence,
    "branch": _branch_sequence,
}


def generate_topology_dataset(
    size: int,
    *,
    length: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    classes = list(TOPOLOGY_LABELS.keys())
    sequences = []
    labels = []
    names = []
    for _ in range(size):
        name = classes[int(rng.integers(0, len(classes)))]
        state = np.random.get_state()
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        seq = GENERATORS[name](length=length, noise_std=noise_std)
        np.random.set_state(state)
        sequences.append(seq.astype(np.float32))
        labels.append(TOPOLOGY_LABELS[name])
        names.append(name)
    return np.stack(sequences), np.asarray(labels, dtype=np.int64), names


@dataclass
class SyntheticTopologyBundle:
    train_sequences: np.ndarray
    train_labels: np.ndarray
    val_sequences: np.ndarray
    val_labels: np.ndarray
    test_sequences: np.ndarray
    test_labels: np.ndarray


class SequenceClassificationDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


def build_synthetic_bundle(
    train_size: int,
    val_size: int,
    test_size: int,
    *,
    length: int,
    noise_std: float,
    seed: int,
) -> SyntheticTopologyBundle:
    train_sequences, train_labels, _ = generate_topology_dataset(
        train_size, length=length, noise_std=noise_std, seed=seed
    )
    val_sequences, val_labels, _ = generate_topology_dataset(
        val_size, length=length, noise_std=noise_std, seed=seed + 1
    )
    test_sequences, test_labels, _ = generate_topology_dataset(
        test_size, length=length, noise_std=noise_std, seed=seed + 2
    )
    return SyntheticTopologyBundle(
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        test_sequences=test_sequences,
        test_labels=test_labels,
    )
