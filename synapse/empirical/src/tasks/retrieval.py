"""Representation Retrieval Task Module.

Evaluates the quality of Z3 learned representations by measuring
nearest-neighbor retrieval accuracy in the proxy feature space.
A good topological representation should place samples with the same
topological class close together in feature space.

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §3, Track 2–3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RetrievalResult:
    """Result of a representation retrieval evaluation.

    Attributes
    ----------
    recall_at_1 : float
        Fraction of queries where the nearest neighbor has the same label.
    recall_at_5 : float
        Fraction of queries where a same-label sample is in the top-5.
    recall_at_10 : float
        Fraction of queries where a same-label sample is in the top-10.
    mean_rank : float
        Mean rank of the first same-label neighbor.
    """
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mean_rank: float


def compute_proxy_features(
    model: torch.nn.Module,
    sequences: np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Extract proxy features from a trained Z3 model.

    Parameters
    ----------
    model : torch.nn.Module
        A trained Z3TopologyFirstModel.
    sequences : np.ndarray
        Shape ``(N, T, d)`` — sequences to embed.
    device : torch.device or str
        Computation device.

    Returns
    -------
    np.ndarray
        Shape ``(N, F)`` — proxy feature vectors.
    """
    model.eval()
    device = torch.device(device)
    with torch.no_grad():
        x = torch.from_numpy(sequences).float().to(device)
        out = model.compute_proxy(x)
        features = out.proxy_features.cpu().numpy()
    return features


def evaluate_retrieval(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_features: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> RetrievalResult:
    """Evaluate nearest-neighbor retrieval in feature space.

    For each query, finds the closest gallery samples by L2 distance
    and checks whether same-label samples appear in the top-*k*.

    Parameters
    ----------
    query_features : np.ndarray
        Shape ``(N_q, F)`` — query feature vectors.
    query_labels : np.ndarray
        Shape ``(N_q,)`` — query labels.
    gallery_features : np.ndarray
        Shape ``(N_g, F)`` — gallery feature vectors.
    gallery_labels : np.ndarray
        Shape ``(N_g,)`` — gallery labels.
    ks : tuple of int
        Top-*k* values to evaluate.

    Returns
    -------
    RetrievalResult
    """
    # Pairwise L2 distances: (N_q, N_g).
    dists = (
        np.sum(query_features ** 2, axis=1, keepdims=True)
        - 2 * query_features @ gallery_features.T
        + np.sum(gallery_features ** 2, axis=1, keepdims=True).T
    )

    # Sort gallery by distance for each query.
    sorted_indices = np.argsort(dists, axis=1)
    sorted_labels = gallery_labels[sorted_indices]

    n_queries = query_labels.shape[0]
    recall_at: dict[int, float] = {}
    for k in ks:
        hits = 0
        for i in range(n_queries):
            top_k_labels = sorted_labels[i, :k]
            if query_labels[i] in top_k_labels:
                hits += 1
        recall_at[k] = hits / n_queries

    # Mean rank of first same-label neighbor.
    ranks = []
    for i in range(n_queries):
        same_label_mask = sorted_labels[i] == query_labels[i]
        if same_label_mask.any():
            ranks.append(float(np.argmax(same_label_mask)) + 1)
        else:
            ranks.append(float(len(gallery_labels)))
    mean_rank = float(np.mean(ranks))

    return RetrievalResult(
        recall_at_1=recall_at.get(1, 0.0),
        recall_at_5=recall_at.get(5, 0.0),
        recall_at_10=recall_at.get(10, 0.0),
        mean_rank=mean_rank,
    )


__all__ = [
    "RetrievalResult",
    "compute_proxy_features",
    "evaluate_retrieval",
]
