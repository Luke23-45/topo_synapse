"""Temporal Encoder — 1-D time-series → latent tokens.

Projects each timestep of a 1-D sequence into ``d_model``-dimensional
tokens and adds learnable positional embeddings.  Suitable for
TelecomTS and other temporal datasets.

Input:  [B, T, input_dim] raw time-series
Output: [B, T, d_model]   encoded tokens
"""

from __future__ import annotations

import torch
from torch import nn, Tensor

from synapse.common.layers import PositionalEncoding1D


class TemporalEncoder(nn.Module):
    """1-D temporal sequence encoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each observation vector.
    d_model : int
        Output token dimension.
    max_len : int
        Maximum sequence length for positional encoding.
    dropout : float
        Dropout after projection + positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding1D(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Encode temporal sequence.

        Parameters
        ----------
        x : Tensor
            Raw time-series ``[B, T, input_dim]``.

        Returns
        -------
        Tensor
            Encoded tokens ``[B, T, d_model]``.
        """
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.drop(x)
        x = self.norm(x)
        return x


__all__ = ["TemporalEncoder"]
