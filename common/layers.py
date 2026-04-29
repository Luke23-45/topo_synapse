"""Shared neural network building blocks for Z3 baseline models.

Provides reusable layers (MLP, DropPath, positional encodings, etc.)
so that each baseline implementation stays concise and consistent.
"""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# MLP block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer MLP with GELU activation and optional dropout.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    hidden_dim : int
        Hidden dimension (typically ``in_dim * ffn_ratio``).
    out_dim : int
        Output dimension.
    dropout : float
        Dropout probability applied after the activation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ---------------------------------------------------------------------------
# DropPath (stochastic depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth layer (drop entire residual path).

    Used in deep residual networks to regularize by randomly dropping
    entire residual branches during training.

    Parameters
    ----------
    drop_prob : float
        Probability of dropping the path (0.0 = no drop).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob


# ---------------------------------------------------------------------------
# Positional encoding (1-D, learnable)
# ---------------------------------------------------------------------------

class PositionalEncoding1D(nn.Module):
    """Learnable 1-D positional embedding for sequence data.

    Parameters
    ----------
    max_len : int
        Maximum sequence length.
    d_model : int
        Embedding dimension.
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional embedding to input.

        Parameters
        ----------
        x : Tensor
            Shape ``[B, T, d_model]`` where *T* ≤ ``max_len``.

        Returns
        -------
        Tensor
            Shape ``[B, T, d_model]``.
        """
        return x + self.pos[:, :x.size(1), :]


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (fixed, no learnable params)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Parameters
    ----------
    max_len : int
        Maximum sequence length.
    d_model : int
        Embedding dimension (must be even).
    """

    def __init__(self, max_len: int = 512, d_model: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1), :]


# ---------------------------------------------------------------------------
# Layer scale
# ---------------------------------------------------------------------------

class LayerScale(nn.Module):
    """Per-channel learnable scale initialized near zero.

    Used in deep transformers to stabilize training (Touvron et al., 2021).

    Parameters
    ----------
    dim : int
        Number of channels.
    init_value : float
        Initial value for the scale parameter.
    """

    def __init__(self, dim: int, init_value: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


# ---------------------------------------------------------------------------
# Task head (shared classification head)
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Lightweight classification head used by all backbones.

    Two-layer MLP: d_model → d_model → num_classes with GELU and
    LayerNorm in between.

    Parameters
    ----------
    d_model : int
        Input (pooled) feature dimension.
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout before the final linear layer.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.norm(x))


__all__ = [
    "ClassificationHead",
    "DropPath",
    "LayerScale",
    "MLP",
    "PositionalEncoding1D",
    "SinusoidalPositionalEncoding",
]
