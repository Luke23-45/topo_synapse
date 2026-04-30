"""Structure-aware feature builders for topological encoders."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor


def _as_mask(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
    return mask.to(device=x.device, dtype=x.dtype)


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)


def _masked_var(x: Tensor, mask: Tensor, mean: Tensor) -> Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    sq = ((x - mean) ** 2) * mask.unsqueeze(-1)
    return sq.sum(dim=1, keepdim=True) / denom.unsqueeze(-1)


def structural_feature_dim(input_dim: int, include_selection: bool = True) -> int:
    """Feature dimension for the universal topology vectors."""
    return 3 * input_dim + 3 + int(include_selection)


def router_context_dim(input_dim: int) -> int:
    return 4 * input_dim


def precompute_structural_geometry(
    sequence: Tensor,
    mask: Optional[Tensor] = None,
    knn_k: int = 4,
    landmark_budget: int = 128,
) -> dict[str, Tensor]:
    """Precompute geometry terms reused across routing stages."""
    if sequence.ndim != 3:
        raise ValueError(f"Expected [B, T, D] input, got shape {tuple(sequence.shape)}")

    B, T, _ = sequence.shape
    valid = _as_mask(sequence, mask)
    mean = _masked_mean(sequence, valid)
    centered = sequence - mean
    std = _masked_var(sequence, valid, mean).clamp_min(1e-6).sqrt()
    standardized = centered / std
    radius = standardized.norm(dim=-1, keepdim=True)

    if T <= 1:
        local_scale = torch.zeros(B, T, 1, device=sequence.device, dtype=sequence.dtype)
        centrality = torch.zeros_like(local_scale)
    else:
        if T <= landmark_budget:
            pairwise = torch.cdist(standardized, standardized)
            valid_pairs = valid.unsqueeze(1) * valid.unsqueeze(2)
            eye = torch.eye(T, device=sequence.device, dtype=sequence.dtype).unsqueeze(0)
            pairwise = pairwise.masked_fill((eye > 0) | (valid_pairs == 0), float("inf"))

            k_eff = max(1, min(knn_k, T - 1))
            knn_dist = pairwise.topk(k_eff, dim=-1, largest=False).values
            local_scale = knn_dist.masked_fill(torch.isinf(knn_dist), 0.0).mean(dim=-1, keepdim=True)

            pairwise_mean = pairwise.masked_fill(torch.isinf(pairwise), 0.0).sum(dim=-1, keepdim=True)
            neighbor_count = valid_pairs.sum(dim=-1, keepdim=True).sub(1.0).clamp_min(1.0)
            centrality = pairwise_mean / neighbor_count
        else:
            landmark_idx = torch.linspace(
                0,
                T - 1,
                steps=landmark_budget,
                device=sequence.device,
                dtype=sequence.dtype,
            ).round().long()
            landmark_features = standardized.index_select(1, landmark_idx)
            landmark_valid = valid.index_select(1, landmark_idx)
            landmark_dist = torch.cdist(standardized, landmark_features)
            valid_landmarks = valid.unsqueeze(-1) * landmark_valid.unsqueeze(1)
            landmark_dist = landmark_dist.masked_fill(valid_landmarks == 0, float("inf"))

            token_idx = torch.arange(T, device=sequence.device).view(1, T, 1)
            self_hits = token_idx.eq(landmark_idx.view(1, 1, -1))
            landmark_dist = landmark_dist.masked_fill(self_hits, float("inf"))

            k_eff = max(1, min(knn_k, landmark_budget))
            knn_dist = landmark_dist.topk(k_eff, dim=-1, largest=False).values
            local_scale = knn_dist.masked_fill(torch.isinf(knn_dist), 0.0).mean(dim=-1, keepdim=True)

            centrality_sum = landmark_dist.masked_fill(torch.isinf(landmark_dist), 0.0).sum(dim=-1, keepdim=True)
            centrality_count = valid_landmarks.sum(dim=-1, keepdim=True).clamp_min(1.0)
            centrality = centrality_sum / centrality_count

    return {
        "valid": valid,
        "mean": mean,
        "centered": centered,
        "std": std,
        "standardized": standardized,
        "radius": radius,
        "local_scale": local_scale,
        "centrality": centrality,
    }


def build_structural_feature_tensor(
    sequence: Tensor,
    selection_weights: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    knn_k: int = 4,
    geometry_cache: Optional[dict[str, Tensor]] = None,
    include_selection: bool = True,
) -> Tensor:
    """Build translation-robust, structure-aware vectors for the lift layer.

    Output channels are:
    - raw observation
    - centered observation
    - standardized observation
    - radius to sample centroid
    - local scale via k-NN distance
    - global centrality via mean pairwise distance
    - selector weight / anchor prior
    """
    if geometry_cache is None:
        geometry_cache = precompute_structural_geometry(
            sequence,
            mask=mask,
            knn_k=knn_k,
        )
    valid = geometry_cache["valid"]
    features = [
        sequence,
        geometry_cache["centered"],
        geometry_cache["standardized"],
        geometry_cache["radius"],
        geometry_cache["local_scale"],
        geometry_cache["centrality"],
    ]
    if include_selection:
        B, T, _ = sequence.shape
        if selection_weights is None:
            selection = torch.zeros(B, T, 1, device=sequence.device, dtype=sequence.dtype)
        else:
            selection = selection_weights.unsqueeze(-1).to(dtype=sequence.dtype)
        features.append(selection)

    features = torch.cat(features, dim=-1)
    return features * valid.unsqueeze(-1)


def build_router_context(
    sequence: Tensor,
    mask: Optional[Tensor] = None,
    geometry_cache: Optional[dict[str, Tensor]] = None,
) -> Tensor:
    """Global structure summary broadcast into the candidate encoder."""
    if geometry_cache is None:
        geometry_cache = precompute_structural_geometry(sequence, mask=mask)
    valid = geometry_cache["valid"]
    mean = geometry_cache["mean"].squeeze(1)
    std = geometry_cache["std"].squeeze(1)

    max_vals = sequence.masked_fill(valid.unsqueeze(-1) == 0, float("-inf")).max(dim=1).values
    max_vals = torch.where(torch.isinf(max_vals), torch.zeros_like(max_vals), max_vals)
    min_vals = sequence.masked_fill(valid.unsqueeze(-1) == 0, float("inf")).min(dim=1).values
    min_vals = torch.where(torch.isinf(min_vals), torch.zeros_like(min_vals), min_vals)
    return torch.cat([mean, std, max_vals, min_vals], dim=-1)


def build_feature_similarity(features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Cosine similarity matrix used for diversity-aware routing."""
    if features.ndim != 3:
        raise ValueError(f"Expected [B, T, D] features, got shape {tuple(features.shape)}")

    valid = _as_mask(features, mask)
    normed = torch.nn.functional.normalize(features, dim=-1, eps=1e-6)
    similarity = torch.bmm(normed, normed.transpose(1, 2))
    similarity = (similarity + 1.0) * 0.5

    T = features.shape[1]
    valid_pairs = valid.unsqueeze(1) * valid.unsqueeze(2)
    eye = torch.eye(T, device=features.device, dtype=features.dtype).unsqueeze(0)
    similarity = similarity * valid_pairs
    similarity = similarity * (1.0 - eye)
    return similarity.clamp_min(0.0)


def compute_structural_normalization_stats(
    sequences: np.ndarray,
    knn_k: int = 4,
) -> dict[str, np.ndarray]:
    """Compute normalization stats for structure-aware topology vectors."""
    tensor = torch.from_numpy(sequences.astype(np.float32))
    with torch.no_grad():
        geometry_cache = precompute_structural_geometry(tensor, knn_k=knn_k)
        features = build_structural_feature_tensor(
            tensor,
            knn_k=knn_k,
            geometry_cache=geometry_cache,
            include_selection=False,
        )
    flat = features.reshape(-1, features.shape[-1]).cpu().numpy().astype(np.float64)
    sigma = flat.std(axis=0)
    sigma[sigma <= 0] = 1.0
    return {"mu": flat.mean(axis=0), "sigma": sigma}
