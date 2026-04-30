"""Unified model orchestration for active encoder and backbone paths."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from synapse.baselines.base import BaselineBackbone
from synapse.baselines.registry import create_backbone
from synapse.common.encoders import create_encoder

from .deep_hodge import DeepHodgeTransformer

log = logging.getLogger(__name__)


@dataclass
class UnifiedModelOutput:
    """Output compatible with the active training loss pipeline."""

    logits: Tensor
    embeddings: Tensor
    event_scores: Tensor
    saliency_scores: Tensor
    y_star: Tensor
    proxy_features: Tensor
    dense_lifted_cloud: Tensor


class DeepHodgeBackbone(BaselineBackbone):
    """Wrapper that adapts DeepHodgeTransformer to the backbone interface."""

    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 2,
        k_dim: int = 16,
        num_scales: int = 3,
        max_points: int = 16,
        num_classes: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.transformer = DeepHodgeTransformer(
            num_layers=num_layers,
            d_model=d_model,
            k_dim=k_dim,
            num_scales=num_scales,
            max_points=max_points,
        )
        self.head = nn.Linear(d_model, num_classes)

    @property
    def backbone_name(self) -> str:
        return "deep_hodge"

    def forward(
        self,
        x: Tensor,
        *,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        h = self.transformer(x)
        features = h.mean(dim=1)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


class UnifiedModel(nn.Module):
    """Unified model: encoder -> backbone -> shared output contract."""

    def __init__(
        self,
        backbone_type: str = "mlp",
        modality: str = "temporal",
        input_dim: int = 2,
        d_model: int = 64,
        num_classes: int = 4,
        num_tokens: int = 128,
        dropout: float = 0.1,
        grid_h: int = 32,
        grid_w: int = 32,
        patch_size: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        k_dim: int = 16,
        num_scales: int = 3,
        max_proxy_points: int = 16,
        k_neighbors: int = 16,
        K: int = 8,
        r: int = 1,
        lam: float = 0.5,
        hidden_dim: int = 64,
        drop_path: float = 0.0,
        use_coords: bool = True,
        kernel_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone_type = backbone_type
        self.d_model = d_model
        self.num_classes = num_classes

        if backbone_type == "deep_hodge":
            modality = "topological"
        self.modality = modality

        if modality == "topological":
            self.encoder = create_encoder(
                "topological",
                input_dim=input_dim,
                d_model=d_model,
                hidden_dim=hidden_dim,
                k=k_dim,
                K=K,
                r=r,
                lam=lam,
                L=max_proxy_points,
                max_proxy_points=max_proxy_points,
            )
        elif modality == "temporal":
            self.encoder = create_encoder(
                "temporal",
                input_dim=input_dim,
                d_model=d_model,
                max_len=num_tokens,
                dropout=dropout,
            )
        elif modality == "geometric":
            self.encoder = create_encoder(
                "geometric",
                input_dim=input_dim,
                d_model=d_model,
                dropout=dropout,
            )
        elif modality == "scientific":
            self.encoder = create_encoder(
                "scientific",
                input_dim=input_dim,
                grid_h=grid_h,
                grid_w=grid_w,
                patch_size=patch_size,
                d_model=d_model,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown modality: '{modality}'")

        if backbone_type == "deep_hodge":
            self.backbone = DeepHodgeBackbone(
                d_model=d_model,
                num_layers=num_layers,
                k_dim=k_dim,
                num_scales=num_scales,
                max_points=max_proxy_points,
                num_classes=num_classes,
            )
        else:
            backbone_kwargs = {
                "d_model": d_model,
                "num_classes": num_classes,
                "dropout": dropout,
            }
            if backbone_type == "mlp":
                backbone_kwargs["num_tokens"] = num_tokens
            elif backbone_type == "ptv3":
                backbone_kwargs.update(
                    num_layers=num_layers,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    drop_path=drop_path,
                    use_coords=use_coords,
                )
            elif backbone_type == "snn":
                backbone_kwargs.update(
                    num_layers=num_layers,
                    ffn_ratio=ffn_ratio,
                    k_neighbors=k_neighbors,
                    drop_path=drop_path,
                    use_coords=use_coords,
                )
            elif backbone_type == "tcn":
                backbone_kwargs["kernel_size"] = kernel_size
            self.backbone = create_backbone(backbone_type, **backbone_kwargs)

        self._has_topological_encoder = modality == "topological"

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_normalization(self, mu: Tensor, sigma: Tensor) -> None:
        if self._has_topological_encoder:
            self.encoder.set_normalization(mu, sigma)

    def forward(self, sequence: Tensor, mask: Tensor | None = None) -> UnifiedModelOutput:
        if self._has_topological_encoder:
            tokens, y_star, _all_y, _all_memory = self.encoder(sequence, mask=mask)
        else:
            tokens = self.encoder(sequence)
            y_star = torch.zeros(sequence.shape[0], sequence.shape[1], device=sequence.device)

        logits, features = self.backbone(tokens, return_features=True)

        B = sequence.shape[0]
        device = sequence.device
        d = self.d_model
        zeros = torch.zeros(B, sequence.shape[1], device=device)

        return UnifiedModelOutput(
            logits=logits,
            embeddings=features,
            event_scores=zeros,
            saliency_scores=zeros,
            y_star=y_star,
            proxy_features=features,
            dense_lifted_cloud=torch.zeros(B, 1, d, device=device),
        )


Z3UnifiedModel = UnifiedModel


__all__ = ["DeepHodgeBackbone", "UnifiedModel", "UnifiedModelOutput", "Z3UnifiedModel"]
