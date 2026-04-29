"""Z3 Unified Model — encoder + backbone + head orchestration.

Provides a single entry point that composes any modality encoder with
any backbone (including the proposed Deep Hodge) and a shared
classification head.  This enables fair comparison across all 5 models
by keeping the encoder and task head constant while varying only the
backbone.

Usage
-----
    from synapse.synapse_arch.unified import Z3UnifiedModel

    model = Z3UnifiedModel(
        backbone_type="mlp",        # or "tcn", "ptv3", "snn", "deep_hodge"
        modality="temporal",        # or "geometric", "scientific", "topological"
        input_dim=2,
        d_model=64,
        num_classes=4,
    )
    logits = model(sequence)

Design
------
- **Encoder**: Selected by ``modality`` — normalizes raw data to tokens.
- **Backbone**: Selected by ``backbone_type`` — processes tokens.
- **Task head**: Shared ``ClassificationHead`` for all models.
- For ``backbone_type="deep_hodge"``, the encoder is forced to
  ``"topological"`` and the backbone wraps ``DeepHodgeTransformer``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from synapse.baselines.base import BaselineBackbone
from synapse.baselines.registry import create_backbone, registered_names
from synapse.common.encoders import create_encoder
from synapse.common.layers import ClassificationHead

from .deep_hodge import DeepHodgeTransformer

log = logging.getLogger(__name__)


@dataclass
class UnifiedModelOutput:
    """Output of Z3UnifiedModel — compatible with existing loss functions.

    Fields match ``ModelForwardOutput`` from ``synapse_arch/model.py``
    so that ``compute_loss()`` works without modification.  Baseline
    models fill auxiliary fields with zeros.
    """
    logits: Tensor
    embeddings: Tensor
    event_scores: Tensor
    saliency_scores: Tensor
    y_star: Tensor
    proxy_features: Tensor
    dense_lifted_cloud: Tensor


class DeepHodgeBackbone(BaselineBackbone):
    """Wrapper that adapts DeepHodgeTransformer to the BaselineBackbone interface.

    Parameters
    ----------
    d_model : int
    num_layers : int
    k_dim : int
    num_scales : int
    max_points : int
    num_classes : int
    """

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
        h = self.transformer(x)  # [B, K, d_model]
        features = h.mean(dim=1)  # [B, d_model]
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


class Z3UnifiedModel(nn.Module):
    """Unified model: encoder → backbone → task head.

    Parameters
    ----------
    backbone_type : str
        One of ``"mlp"``, ``"tcn"``, ``"ptv3"``, ``"snn"``,
        ``"deep_hodge"``.
    modality : str
        One of ``"temporal"``, ``"geometric"``, ``"scientific"``,
        ``"topological"``.  Auto-forced to ``"topological"`` when
        ``backbone_type="deep_hodge"``.
    input_dim : int
        Raw input dimension per timestep/point.
    d_model : int
        Shared hidden dimension.
    num_classes : int
        Number of output classes.
    num_tokens : int
        Expected number of input tokens (sequence length / point count).
        Used by MLP for flattening.
    dropout : float
        Dropout probability.
    grid_h, grid_w : int
        Grid dimensions (only used for ``modality="scientific"``).
    patch_size : int
        Patch size for scientific encoder.
    num_layers : int
        Number of backbone layers (for PTv3, SNN, Deep Hodge).
    num_heads : int
        Number of attention heads (for PTv3).
    ffn_ratio : int
        FFN expansion ratio.
    k_dim : int
        Geometric lift dimension (for Deep Hodge).
    num_scales : int
        Spectral scales (for Deep Hodge).
    max_proxy_points : int
        Maximum simplicial complex size (for Deep Hodge, SNN).
    k_neighbors : int
        KNN neighbors (for SNN).
    K : int
        Max anchors (for topological encoder).
    r : int
        Refractory separation (for topological encoder).
    lam : float
        Selector regularization (for topological encoder).
    hidden_dim : int
        Event model hidden dim (for topological encoder).
    """

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

        # Force topological encoder for Deep Hodge
        if backbone_type == "deep_hodge":
            modality = "topological"
        self.modality = modality

        # --- Encoder ---
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

        # --- Backbone ---
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
            backbone_kwargs = dict(
                d_model=d_model,
                num_classes=num_classes,
                dropout=dropout,
            )
            if backbone_type == "mlp":
                backbone_kwargs["num_tokens"] = num_tokens
            elif backbone_type == "ptv3":
                backbone_kwargs.update(
                    num_layers=num_layers, num_heads=num_heads,
                    ffn_ratio=ffn_ratio, drop_path=drop_path,
                    use_coords=use_coords,
                )
            elif backbone_type == "snn":
                backbone_kwargs.update(
                    num_layers=num_layers, ffn_ratio=ffn_ratio,
                    k_neighbors=k_neighbors, drop_path=drop_path,
                    use_coords=use_coords,
                )
            elif backbone_type == "tcn":
                backbone_kwargs["kernel_size"] = kernel_size
            self.backbone = create_backbone(backbone_type, **backbone_kwargs)

        # Track whether encoder produces y_star (topological only)
        self._has_topological_encoder = (modality == "topological")

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_normalization(self, mu: Tensor, sigma: Tensor) -> None:
        """Set lift normalization (only for topological encoder)."""
        if self._has_topological_encoder:
            self.encoder.set_normalization(mu, sigma)

    def forward(self, sequence: Tensor, mask: Tensor | None = None) -> UnifiedModelOutput:
        """Full forward pass: encoder → backbone → output.

        Parameters
        ----------
        sequence : Tensor
            Raw input ``[B, T, input_dim]`` or ``[B, N, input_dim]``.
        mask : Tensor or None
            Optional mask (used by Deep Hodge backbone).

        Returns
        -------
        UnifiedModelOutput
            Compatible with ``compute_loss()`` from the training pipeline.
        """
        # Encode
        if self._has_topological_encoder:
            tokens, y_star = self.encoder(sequence)
        else:
            tokens = self.encoder(sequence)
            y_star = torch.zeros(sequence.shape[0], sequence.shape[1], device=sequence.device)

        # Backbone
        result = self.backbone(tokens, return_features=True)
        logits, features = result

        # Build output compatible with existing loss pipeline
        B = sequence.shape[0]
        device = sequence.device
        d = self.d_model

        return UnifiedModelOutput(
            logits=logits,
            embeddings=features,
            event_scores=torch.zeros(B, sequence.shape[1], device=device),
            saliency_scores=torch.zeros(B, sequence.shape[1], device=device),
            y_star=y_star,
            proxy_features=features,
            dense_lifted_cloud=torch.zeros(B, 1, d, device=device),
        )


__all__ = ["DeepHodgeBackbone", "UnifiedModelOutput", "Z3UnifiedModel"]
