"""Baseline 1: Multi-Layer Perceptron (Sanity Check).

A simple 3-layer MLP that flattens the input tokens and maps directly
to class logits.  Establishes the performance floor — tests whether
geometry, temporal order, or topology actually matter for the task.

Architecture:
    [B, N, d_model] → flatten → Linear(N*d_model, hidden) → GELU
                    → Linear(hidden, hidden) → GELU
                    → Linear(hidden, num_classes)
"""

from __future__ import annotations

import torch
from torch import nn, Tensor

from ..base import BaselineBackbone
from ..registry import register_backbone


class MLPBackbone(BaselineBackbone):
    """3-layer MLP baseline (sanity check).

    Parameters
    ----------
    d_model : int
        Input token dimension (from encoder).
    num_tokens : int
        Number of input tokens (sequence length or point count).
    num_classes : int
        Number of output classes.
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_tokens: int = 128,
        num_classes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.num_classes = num_classes

        flat_dim = d_model * num_tokens

        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @property
    def backbone_name(self) -> str:
        return "mlp"

    def forward(
        self,
        x: Tensor,
        *,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        B, N, D = x.shape
        flat = x.reshape(B, N * D)
        logits = self.net(flat)

        if return_features:
            features = x.mean(dim=1)
            return logits, features
        return logits


register_backbone("mlp", MLPBackbone)


__all__ = ["MLPBackbone"]
