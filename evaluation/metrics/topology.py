"""Topology-specific evaluation metrics for Z3 SYNAPSE models.

Measures alignment between the differentiable Hodge-spectral proxy
and the exact Vietoris-Rips persistence audit, as well as Betti
number prediction accuracy and persistence diagram distances.
"""

from __future__ import annotations

import numpy as np


def proxy_exact_alignment(
    proxy_features: np.ndarray,
    exact_summaries: np.ndarray,
) -> dict[str, float]:
    """Cosine alignment between proxy features and exact topology summaries.

    Parameters
    ----------
    proxy_features : np.ndarray, shape (B, D_proxy)
        Differentiable Hodge-spectral features from the proxy branch.
    exact_summaries : np.ndarray, shape (B, D_exact)
        Topology summary vectors from the exact Vietoris-Rips audit.

    Returns
    -------
    dict with key "proxy_exact_cosine_alignment" → float in [-1, 1].
    """
    proxy = proxy_features.reshape(proxy_features.shape[0], -1)
    exact = exact_summaries.reshape(exact_summaries.shape[0], -1)
    width = min(proxy.shape[1], exact.shape[1])
    proxy = proxy[:, :width]
    exact = exact[:, :width]
    proxy_centered = proxy - proxy.mean(axis=0, keepdims=True)
    exact_centered = exact - exact.mean(axis=0, keepdims=True)
    numerator = float(np.sum(proxy_centered * exact_centered, dtype=np.float64))
    denom = float(
        np.sqrt(np.sum(proxy_centered**2, dtype=np.float64) * np.sum(exact_centered**2, dtype=np.float64))
        + 1e-8
    )
    return {"proxy_exact_cosine_alignment": numerator / denom}


def betti_number_accuracy(
    predicted_betti: np.ndarray,
    true_betti: np.ndarray,
) -> dict[str, float]:
    """Accuracy of predicted Betti numbers against ground truth.

    Parameters
    ----------
    predicted_betti : np.ndarray, shape (B, Q+1)
        Predicted Betti numbers per homology degree.
    true_betti : np.ndarray, shape (B, Q+1)
        Ground-truth Betti numbers.

    Returns
    -------
    dict with keys:
        "betti_exact_match" : fraction of samples where all Betti numbers match
        "betti_per_degree_accuracy" : mean accuracy per homology degree
    """
    pred = predicted_betti.astype(np.int64)
    true = true_betti.astype(np.int64)

    exact_match = float(np.all(pred == true, axis=1).mean())

    per_degree_acc = float((pred == true).mean())

    return {
        "betti_exact_match": exact_match,
        "betti_per_degree_accuracy": per_degree_acc,
    }


def persistence_diagram_distance(
    diagram_a: list[list[tuple[float, float]]],
    diagram_b: list[list[tuple[float, float]]],
    order: int = 2,
) -> dict[str, float]:
    """Approximate bottleneck / Wasserstein distance between persistence diagrams.

    Uses a simple matching heuristic: for each homology degree, sort both
    diagrams by persistence and compute the Lp distance between matched pairs,
    padding the shorter diagram with diagonal points (birth=death).

    Parameters
    ----------
    diagram_a, diagram_b : list of list of (birth, death)
        Persistence diagrams per homology degree.
    order : int
        Lp norm order (2 for Wasserstein-2, ∞ approximated by max for bottleneck).

    Returns
    -------
    dict with key "persistence_wasserstein_p" → float
    """
    total_dist = 0.0
    total_points = 0

    for dgm_a, dgm_b in zip(diagram_a, diagram_b):
        pts_a = np.array(dgm_a, dtype=np.float64) if len(dgm_a) > 0 else np.empty((0, 2))
        pts_b = np.array(dgm_b, dtype=np.float64) if len(dgm_b) > 0 else np.empty((0, 2))

        if len(pts_a) == 0 and len(pts_b) == 0:
            continue

        # Sort by persistence (death - birth) descending
        if len(pts_a) > 0:
            persist_a = pts_a[:, 1] - pts_a[:, 0]
            order_a = np.argsort(-persist_a)
            pts_a = pts_a[order_a]
        if len(pts_b) > 0:
            persist_b = pts_b[:, 1] - pts_b[:, 0]
            order_b = np.argsort(-persist_b)
            pts_b = pts_b[order_b]

        max_len = max(len(pts_a), len(pts_b))

        # Pad shorter diagram with diagonal points
        if len(pts_a) < max_len:
            diag_pts = np.zeros((max_len - len(pts_a), 2), dtype=np.float64)
            diag_pts[:, 0] = pts_b[len(pts_a):, 0] if len(pts_b) >= max_len else 0.0
            diag_pts[:, 1] = diag_pts[:, 0]
            pts_a = np.concatenate([pts_a, diag_pts], axis=0)
        if len(pts_b) < max_len:
            diag_pts = np.zeros((max_len - len(pts_b), 2), dtype=np.float64)
            diag_pts[:, 0] = pts_a[len(pts_b):, 0] if len(pts_a) >= max_len else 0.0
            diag_pts[:, 1] = diag_pts[:, 0]
            pts_b = np.concatenate([pts_b, diag_pts], axis=0)

        if order == 0:  # bottleneck (approximate)
            dists = np.max(np.abs(pts_a - pts_b), axis=1)
            total_dist += float(dists.max())
        else:
            dists = np.linalg.norm(pts_a - pts_b, ord=order, axis=1)
            total_dist += float(np.sum(dists ** order))
        total_points += max_len

    if total_points == 0:
        return {f"persistence_wasserstein_{order}": 0.0}

    if order == 0:
        result = total_dist
    else:
        result = (total_dist / max(total_points, 1)) ** (1.0 / order)

    return {f"persistence_wasserstein_{order}": result}
