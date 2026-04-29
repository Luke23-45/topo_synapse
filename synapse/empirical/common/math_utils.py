from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error


def make_orthogonal_W(k: int, D: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    return q[:k].astype(np.float64)


def ridge_probe_accuracy(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=500)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    return float(accuracy_score(test_y, pred))


def ridge_regression_mse(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
    reg = Ridge(alpha=1.0)
    reg.fit(train_x, train_y)
    pred = reg.predict(test_x)
    return float(mean_squared_error(test_y, pred))


def pad_rows(x: np.ndarray, width: int) -> np.ndarray:
    if x.shape[1] >= width:
        return x[:, :width]
    pad = np.zeros((x.shape[0], width - x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=1)


def timed_call(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start
