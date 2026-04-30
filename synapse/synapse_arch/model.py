"""
Z3 Topology-First Model — Z3 Reference: §10–12 of 01_main_definition.md

Task-agnostic representation model with separated exact audit and proxy paths.

Architecture:
    1. CausalEventModel → event scores + saliency
    2. Relaxed selector → continuous anchor weights y*
    3. Dense anchor vectors → topology projection Π_top → NormalizedLift
    4. DifferentiableHodgeProxy → spectral features (L0 + L1)
    5. TaskTransformer → readout head → logits/embeddings
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from synapse.common.types import ProxyComputation
from synapse.synapse_core.audit import compute_exact_topology_audit
from synapse.synapse_core.event import CausalEventModel
from synapse.synapse_core.lift import (
    normalize_anchors,
)
from synapse.synapse_core.proxy import DifferentiableHodgeProxy
from synapse.synapse_core.selection import solve_relaxed_selector
from synapse.synapse_core.topology_features import build_structural_feature_tensor

from .config import SynapseConfig
from .deep_hodge import DeepHodgeTransformer
from .normalized_lift import NormalizedLift


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
    """General structured-data model with separated exact audit and proxy paths.

    The model consumes arbitrary structured sequences and produces:
        - logits: task predictions (classification or regression)
        - embeddings: learned representations
        - proxy_features: differentiable Hodge-spectral features
        - exact_audit: deployment-time Vietoris-Rips persistence (offline)
    """

    def __init__(self, config: SynapseConfig) -> None:
        super().__init__()
        self.config = config
        anchor_dim = 3 * config.input_dim + 3

        # Stage 1: Causal event detection and saliency normalization
        self.event_model = CausalEventModel(config.input_dim, config.hidden_dim)

        # Stage 2: Geometric lift (§8–9)
        self.lift = NormalizedLift(anchor_dim, config.k)

        # Stage 3: Topological Branch (§12)
        # Supports multiple architectures for robust ablation studies
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
            # Project k-dim lift to d_model for the transformer input
            self.topology_proj = nn.Linear(config.k, config.d_model)
        else:
            raise ValueError(f"Unsupported topology_mode: {config.topology_mode}")

        # Stage 4: Minimal Downstream Learner (Z3 §13)
        # Acting strictly on the Hodge-spectral proxy features
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
        """Solve the relaxed selector QP for each sample in the batch."""
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
        """Compute differentiable Hodge-spectral proxy features.

        Pipeline:
            sequence → event_model → saliency → relaxed_selector → y*
            sequence + saliency → dense_vectors → Π_top → NormalizedLift → cloud
            cloud + y* → HodgeProxy → spectral features (L0 + L1 eigenvalues)
        """
        # Stage 1: Event detection + saliency
        event_scores, saliency_scores = self.event_model(sequence, mask)

        # Stage 2: Anchor selection
        y_star = self._solve_batch_selector(saliency_scores)

        # Stage 3: Topology-projected lift
        dense_vectors = build_structural_feature_tensor(
            sequence,
            knn_k=max(1, self.config.r),
            include_selection=False,
        )
        _, dense_lifted_cloud = self.lift(dense_vectors)

        # Stage 4: Topological Branch Processing (fully differentiable)
        with torch.autocast(device_type=sequence.device.type, enabled=False):
            if self.config.topology_mode == "baseline_proxy":
                proxy_features = self.proxy(dense_lifted_cloud, y_star)
            elif self.config.topology_mode == "deep_hodge":
                # Select top-K anchors based on y_star saliency
                B, N, k_dim = dense_lifted_cloud.shape
                K_eff = min(N, self.config.max_proxy_points)
                _, top_idx = torch.topk(y_star, K_eff, dim=1)
                top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, k_dim)
                
                # Extract the geometry of selected anchors: (B, K_eff, k)
                cloud = torch.gather(dense_lifted_cloud, 1, top_idx_exp)
                
                # Project geometry to model dimension and route through topological layers
                feat = self.topology_proj(cloud)
                out = self.proxy(feat) # (B, K_eff, d_model)
                
                # Global representation via mean pooling (compatible with current readout)
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
        """Full forward pass: proxy computation → minimal readout.

        This is strictly Topology-First: the task prediction is a function
        of the differentiable topological proxy features.
        """
        # Compute proxy
        proxy = self.compute_proxy(sequence, mask)

        # Readout strictly from proxy features
        logits = self.readout(proxy.proxy_features)
        
        # Embeddings are the penultimate layer of the readout
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
        """Run deployment-time exact topology audit (Vietoris-Rips persistence).

        This is NOT differentiable. It uses Gudhi/Ripser for exact computation.
        Used only for validation and visualization.
        """
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
        """Recompute lift normalization statistics from a batch of sequences.

        Must be called before training to set μ and σ for the affine normalization.
        """
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
