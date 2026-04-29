from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def match_f1(true_idx: list[int], pred_idx: list[int], tolerance: int = 1) -> float:
    if not true_idx and not pred_idx:
        return 1.0
    matched = 0
    used = set()
    for t in true_idx:
        for i, p in enumerate(pred_idx):
            if i in used:
                continue
            if abs(t - p) <= tolerance:
                matched += 1
                used.add(i)
                break
    precision = matched / max(len(pred_idx), 1)
    recall = matched / max(len(true_idx), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
