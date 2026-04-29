"""Geometric Encoder — 3-D point clouds → latent tokens.

Projects point coordinates and per-point features into ``d_model``-
dimensional tokens with 3-D sinusoidal positional encoding derived
from the normalized coordinates.  Suitable for SpatialLM and other
point-cloud datasets.

Input:  [B, N, 3+feat_dim] raw point cloud (xyz + features)
Output: [B, N, d_model]    encoded tokens
"""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor


class GeometricEncoder(nn.Module):
    """3-D point cloud encoder with sinusoidal coordinate encoding.

    Parameters
    ----------
    input_dim : int
        Total dimension per point (3 for xyz + feature_dim).
    d_model : int
        Output token dimension.
    dropout : float
        Dropout after projection.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Point-wise feature projection
        self.proj = nn.Linear(input_dim, d_model)

        # 3-D sinusoidal positional encoding for xyz coordinates
        # Each axis gets d_model // 6 sin + cos pairs → 3 * (d_model//3) = d_model
        self.coord_enc_dim = d_model // 3
        # Precompute frequency bands
        freqs = torch.exp(
            torch.arange(0, self.coord_enc_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.coord_enc_dim)
        )
        self.register_buffer("freqs", freqs)

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _encode_coords(self, xyz: Tensor) -> Tensor:
        """Sinusoidal encoding of 3-D coordinates.

        Parameters
        ----------
        xyz : Tensor
            Shape ``[B, N, 3]`` — normalized xyz coordinates.

        Returns
        -------
        Tensor
            Shape ``[B, N, d_model]`` — positional encoding.
        """
        B, N, _ = xyz.shape
        half = self.coord_enc_dim // 2
        full = half * 2

        parts = []
        for axis in range(3):
            coord = xyz[:, :, axis : axis + 1]  # [B, N, 1]
            angles = coord * self.freqs[:half].unsqueeze(0).unsqueeze(0)  # [B, N, half]
            parts.append(torch.sin(angles))
            parts.append(torch.cos(angles))

        enc = torch.cat(parts, dim=-1)  # [B, N, 3 * 2 * half]
        # Pad or truncate to d_model
        if enc.size(-1) < self.d_model:
            pad = torch.zeros(
                B, N, self.d_model - enc.size(-1),
                device=enc.device, dtype=enc.dtype,
            )
            enc = torch.cat([enc, pad], dim=-1)
        elif enc.size(-1) > self.d_model:
            enc = enc[:, :, : self.d_model]
        return enc

    def forward(self, x: Tensor) -> Tensor:
        """Encode point cloud.

        Parameters
        ----------
        x : Tensor
            Raw point cloud ``[B, N, input_dim]`` where the first 3
            channels are xyz coordinates (zero-padded if input_dim < 3).

        Returns
        -------
        Tensor
            Encoded tokens ``[B, N, d_model]``.
        """
        # Ensure at least 3 channels for xyz coordinates
        if x.size(-1) < 3:
            pad = torch.zeros(
                *x.shape[:-1], 3 - x.size(-1),
                device=x.device, dtype=x.dtype,
            )
            xyz = torch.cat([x, pad], dim=-1)  # [B, N, 3]
        else:
            xyz = x[:, :, :3]  # [B, N, 3]
        pos_enc = self._encode_coords(xyz)

        feat = self.proj(x)  # [B, N, d_model]
        feat = feat + pos_enc
        feat = self.drop(feat)
        feat = self.norm(feat)
        return feat


__all__ = ["GeometricEncoder"]
