"""Baseline 2: Temporal Convolutional Network (Temporal Baseline).

Implements a TCN with dilated causal convolutions and residual connections.
Tests whether standard 1-D convolutional processing captures temporal
patterns as effectively as topological attention.

Architecture:
    [B, T, d_model] → stack of TemporalBlocks(d=1,2,4,8)
                    → global average pool → Linear → logits

Reference
---------
- Bai et al., "An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling", 2018
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn.utils import weight_norm

from ..base import BaselineBackbone
from ..registry import register_backbone


class CausalConv1d(nn.Module):
    """Dilated causal 1-D convolution with padding to preserve length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        ))
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self._crop = self.padding

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.act(out)
        out = self.drop(out)
        if self._crop > 0:
            out = out[:, :, :-self._crop]
        return out


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop(out) + residual
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        return out


class TCNBackbone(BaselineBackbone):
    """Temporal Convolutional Network baseline.

    Parameters
    ----------
    d_model : int
        Input token dimension (from encoder).
    num_classes : int
        Number of output classes.
    num_channels : list[int]
        Channel sizes for each TemporalBlock.
    kernel_size : int
        Convolution kernel size.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_classes: int = 4,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_channels is None:
            num_channels = [d_model, d_model, d_model, d_model]

        layers = []
        in_ch = d_model
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels[-1], num_classes)

    @property
    def backbone_name(self) -> str:
        return "tcn"

    def forward(
        self,
        x: Tensor,
        *,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        h = x.transpose(1, 2)  # [B, d_model, T]
        h = self.network(h)     # [B, C_last, T]
        features = h.mean(dim=2)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


register_backbone("tcn", TCNBackbone)


__all__ = ["TCNBackbone"]
