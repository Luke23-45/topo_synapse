"""Legacy Z3 topology-first model preserved under the legacy namespace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from synapse.common.types import ProxyComputation
from synapse.synapse_core.audit import compute_exact_topology_audit
from synapse.synapse_core.event import CausalEventModel
from synapse.synapse_core.lift import normalize_anchors
from synapse.synapse_core.proxy import DifferentiableHodgeProxy
from synapse.synapse_core.selection import solve_relaxed_selector
from synapse.synapse_core.topology_features import build_structural_feature_tensor, structural_feature_dim

from ..config import SynapseConfig
from ..deep_hodge import DeepHodgeTransformer
from ..normalized_lift import NormalizedLift


@dataclass
class ModelForwardOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor
    event_scores: torch.Tensor
    saliency_scores: torch.Tensor
    y_star: torch.Tensor
    proxy_features: torch.Tensor
    dense_lifted_cloud: torch.Tensor


class Z3TopologyFirstModel(nn.Module):
    """Legacy Z3 topology-first model."""

    def __init__(self, config: SynapseConfig) -> None:
        super().__init__()
        self.config = config
        anchor_dim = structural_feature_dim(config.input_dim, include_selection=False)
        self.event_model = CausalEventModel(config.input_dim, config.hidden_dim)
        self.lift = NormalizedLift(anchor_dim, config.k)

        if config.topology_mode == "baseline_proxy":
            self.proxy = DifferentiableHodgeProxy(
                lift_dim=config.k,
                hidden_dim=config.d_model,
                num_scales=config.num_scales,
                num_eigs=config.num_eigs,
                max_points=config.max_proxy_points,
                tau=config.tau,
            )
        elif config.topology_mode == "deep_hodge":
            self.proxy = DeepHodgeTransformer(
                num_layers=config.num_layers,
                d_model=config.d_model,
                k_dim=config.k,
                num_scales=config.num_scales,
                max_points=config.max_proxy_points,
            )
            self.topology_proj = nn.Linear(config.k, config.d_model)
        else:
            raise ValueError(f"Unsupported topology_mode: {config.topology_mode}")

        self.readout = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.output_dim),
        )

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _solve_batch_selector(self, saliency_scores: torch.Tensor) -> torch.Tensor:
        y_values = []
        for scores in saliency_scores.detach().cpu().numpy():
            y_values.append(
                solve_relaxed_selector(
                    scores.astype(np.float64),
                    K=self.config.K,
                    r=self.config.r,
                    lam=self.config.lam,
                )
            )
        return torch.from_numpy(np.stack(y_values, axis=0)).to(
            device=saliency_scores.device,
            dtype=saliency_scores.dtype,
        )

    def compute_proxy(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> ProxyComputation:
        event_scores, saliency_scores = self.event_model(sequence, mask)
        y_star = self._solve_batch_selector(saliency_scores)

        dense_vectors = build_structural_feature_tensor(
            sequence,
            knn_k=max(1, self.config.r),
            include_selection=False,
        )
        _, dense_lifted_cloud = self.lift(dense_vectors)

        with torch.autocast(device_type=sequence.device.type, enabled=False):
            if self.config.topology_mode == "baseline_proxy":
                proxy_features = self.proxy(dense_lifted_cloud, y_star)
            elif self.config.topology_mode == "deep_hodge":
                _, N, k_dim = dense_lifted_cloud.shape
                K_eff = min(N, self.config.max_proxy_points)
                _, top_idx = torch.topk(y_star, K_eff, dim=1)
                top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, k_dim)
                cloud = torch.gather(dense_lifted_cloud, 1, top_idx_exp)
                feat = self.topology_proj(cloud)
                out = self.proxy(feat)
                proxy_features = out.mean(dim=1)
            else:
                raise RuntimeError(f"Architecture logic not implemented for {self.config.topology_mode}")

        return ProxyComputation(
            dense_vectors=dense_vectors,
            dense_lifted_cloud=dense_lifted_cloud,
            y_star=y_star,
            proxy_features=proxy_features,
            event_scores=event_scores,
            saliency_scores=saliency_scores,
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> ModelForwardOutput:
        proxy = self.compute_proxy(sequence, mask)
        logits = self.readout(proxy.proxy_features)
        embeddings = proxy.proxy_features
        return ModelForwardOutput(
            logits=logits,
            embeddings=embeddings,
            event_scores=proxy.event_scores,
            saliency_scores=proxy.saliency_scores,
            y_star=proxy.y_star,
            proxy_features=proxy.proxy_features,
            dense_lifted_cloud=proxy.dense_lifted_cloud,
        )

    def exact_audit(self, sequence: torch.Tensor) -> list:
        event_scores, saliency_scores = self.event_model(sequence)
        y_star = self._solve_batch_selector(saliency_scores)
        audits = []
        for batch_idx in range(sequence.shape[0]):
            audits.append(
                compute_exact_topology_audit(
                    trajectory=sequence[batch_idx].detach().cpu().numpy().astype(np.float64),
                    event_scores=event_scores[batch_idx].detach().cpu().numpy().astype(np.float64),
                    saliency_scores=saliency_scores[batch_idx].detach().cpu().numpy().astype(np.float64),
                    y_star=y_star[batch_idx].detach().cpu().numpy().astype(np.float64),
                    W_theta=self.lift.W_theta.detach().cpu().numpy().astype(np.float64),
                    K=self.config.K,
                    r=self.config.r,
                    Q=self.config.Q,
                    mu=self.lift.mu.detach().cpu().numpy().astype(np.float64),
                    sigma=self.lift.sigma.detach().cpu().numpy().astype(np.float64),
                )
            )
        return audits

    def refresh_normalization(self, sequences: np.ndarray) -> None:
        dense = build_structural_feature_tensor(
            torch.from_numpy(sequences).float(),
            knn_k=max(1, self.config.r),
            include_selection=False,
        )
        stacked = dense.reshape(-1, dense.shape[-1]).cpu().numpy().astype(np.float64)
        _, mu, sigma = normalize_anchors(stacked)
        self.lift.set_normalization(
            torch.from_numpy(mu).float(),
            torch.from_numpy(sigma).float(),
        )
