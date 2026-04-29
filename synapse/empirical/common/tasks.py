from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synapse.empirical.datasets.synthetic_topology import generate_topology_dataset as _generate_topology_dataset


@dataclass
class SequenceSample:
    sequence: np.ndarray
    target: int | float


def generate_topology_dataset(size: int, *, length: int, noise_std: float, seed: int):
    return _generate_topology_dataset(size=size, length=length, noise_std=noise_std, seed=seed)


def generate_memory_task_dataset(size: int, *, length: int, seed: int) -> list[SequenceSample]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(size):
        seq = rng.normal(size=(length, 2)).astype(np.float32)
        marker_idx = int(rng.integers(length // 4, length - 1))
        seq[marker_idx] += np.array([3.0, -2.0], dtype=np.float32)
        samples.append(SequenceSample(sequence=seq, target=marker_idx / max(length - 1, 1)))
    return samples


def generate_control_dataset(size: int, *, length: int, seed: int) -> list[SequenceSample]:
    rng = np.random.default_rng(seed)
    return [SequenceSample(sequence=rng.normal(size=(length, 2)).astype(np.float32), target=float(rng.normal())) for _ in range(size)]
