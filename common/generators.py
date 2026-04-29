from __future__ import annotations

import numpy as np


def random_walk(d: int, T: int, step_std: float = 0.5, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=step_std, size=(T, d))
    return np.cumsum(steps, axis=0).astype(np.float64)


def piecewise_constant(d: int, T: int, num_segments: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_segments = max(1, min(T, num_segments))
    change_points = sorted(rng.choice(np.arange(1, T), size=max(0, num_segments - 1), replace=False).tolist())
    points = [0, *change_points, T]
    out = np.zeros((T, d), dtype=np.float64)
    current = rng.normal(size=d)
    for start, end in zip(points[:-1], points[1:]):
        current = current + rng.normal(scale=0.5, size=d)
        out[start:end] = current
    return out


def piecewise_constant_auto(d: int, T: int, num_segments: int, seed: int | None = None) -> np.ndarray:
    return piecewise_constant(d=d, T=T, num_segments=num_segments, seed=seed)
