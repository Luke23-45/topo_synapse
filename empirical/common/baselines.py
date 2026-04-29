from __future__ import annotations

import numpy as np
import torch

from synapse.synapse_arch.model import Z3TopologyFirstModel


def summarize_diagrams(diagrams: list[list[tuple[float, float]]]) -> np.ndarray:
    summary = []
    for dgm in diagrams:
        if not dgm:
            summary.extend([0.0, 0.0, 0.0, 0.0])
            continue
        arr = np.asarray(dgm, dtype=np.float64)
        finite = arr[np.isfinite(arr[:, 1])]
        if finite.size == 0:
            summary.extend([0.0, 0.0, 0.0, 0.0])
            continue
        life = finite[:, 1] - finite[:, 0]
        summary.extend([float(len(finite)), float(life.mean()), float(life.max()), float(life.sum())])
    return np.asarray(summary, dtype=np.float32)


def proxy_topology_features(model: Z3TopologyFirstModel, sequences: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        proxy = model.compute_proxy(torch.from_numpy(sequences).float())
    return proxy.proxy_features.detach().cpu().numpy()


def uniform_feature(sequences: np.ndarray) -> np.ndarray:
    return sequences.mean(axis=1)
