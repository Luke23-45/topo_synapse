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
    slope = 0.25 * np.random.uniform(0.5, 2.0)
    seq = np.stack([t, slope * t], axis=1)
    seq = _apply_augmentations(seq, noise_std)
    return seq


def _circle_sequence(length: int, noise_std: float) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, length, dtype=np.float64)
    r = np.random.uniform(0.7, 1.3)
    seq = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    seq = _apply_augmentations(seq, noise_std)
    return seq


def _figure_eight_sequence(length: int, noise_std: float) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, length, dtype=np.float64)
    scale_x = np.random.uniform(0.8, 1.2)
    scale_y = np.random.uniform(0.8, 1.2)
    seq = np.stack([scale_x * np.sin(t), scale_y * np.sin(t) * np.cos(t)], axis=1)
    seq = _apply_augmentations(seq, noise_std)
    return seq


def _branch_sequence(length: int, noise_std: float) -> np.ndarray:
    half = length // 2
    branch_angle = np.random.uniform(0.3, 1.2)
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
            np.linspace(0.0, branch_angle, length - half, dtype=np.float64),
        ],
        axis=1,
    )
    seq = np.concatenate([left, right], axis=0)
    seq = _apply_augmentations(seq, noise_std)
    return seq


def _apply_augmentations(seq: np.ndarray, noise_std: float) -> np.ndarray:
    """Apply a bundle of noise and complexity augmentations.

    1. Per-sample noise scale jitter (heteroscedastic across samples)
    2. Per-dimension noise scale variation
    3. Random 2D rotation (preserves topology, breaks axis alignment)
    4. Sinusoidal drift along the sequence (non-stationary perturbation)
    5. Sparse outlier spikes (simulates sensor glitches)
    """
    length, dim = seq.shape

    # 1–2. Heteroscedastic Gaussian noise: each sample and dimension gets
    # a different effective scale, drawn around the base noise_std.
    scale_per_dim = noise_std * np.random.uniform(0.5, 2.0, size=(1, dim))
    seq = seq + np.random.normal(scale=1.0, size=seq.shape) * scale_per_dim

    # 3. Random rotation in 2D — preserves topological invariants but
    # breaks the privileged axis alignment that makes classification trivial.
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    seq = seq @ rotation.T

    # 4. Sinusoidal drift — a low-frequency perturbation that varies
    # across the sequence, making the noise non-stationary.
    freq = np.random.uniform(0.5, 3.0)
    phase = np.random.uniform(0.0, 2.0 * np.pi)
    amplitude = noise_std * np.random.uniform(1.0, 4.0)
    drift = amplitude * np.sin(freq * np.linspace(0, 2 * np.pi, length) + phase)
    seq = seq + drift[:, None]

    # 5. Sparse outlier spikes — random large deviations on a few points,
    # simulating sensor glitches or transient artefacts.
    spike_prob = 0.05
    spike_mask = np.random.random(length) < spike_prob
    spike_amplitude = noise_std * np.random.uniform(5.0, 15.0)
    seq[spike_mask] += np.random.normal(scale=spike_amplitude, size=seq[spike_mask].shape)

    return seq


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
