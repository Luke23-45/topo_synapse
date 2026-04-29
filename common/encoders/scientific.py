"""Scientific Encoder — 2-D grid/field data → latent tokens.

Splits a 2-D field into non-overlapping patches (like ViT), linearly
projects each patch into ``d_model`` dimensions, and adds 2-D learnable
positional embeddings.  Suitable for Photonic and other grid-based
scientific datasets.

Input:  [B, H, W, C] or [B, C, H, W] raw 2-D field
Output: [B, N_patches, d_model] encoded tokens
"""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor


class ScientificEncoder(nn.Module):
    """2-D patch-based encoder for scientific grid data.

    Parameters
    ----------
    input_dim : int
        Number of channels in the input field (C).
    grid_h : int
        Height of the input grid.
    grid_w : int
        Width of the input grid.
    patch_size : int
        Spatial patch size (patches are patch_size × patch_size).
    d_model : int
        Output token dimension.
    dropout : float
        Dropout after projection + positional encoding.
    channels_first : bool
        If ``True``, input is ``[B, C, H, W]``; else ``[B, H, W, C]``.
    """

    def __init__(
        self,
        input_dim: int,
        grid_h: int,
        grid_w: int,
        patch_size: int = 4,
        d_model: int = 64,
        dropout: float = 0.1,
        channels_first: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.channels_first = channels_first
        self.grid_h = grid_h
        self.grid_w = grid_w

        assert grid_h % patch_size == 0 and grid_w % patch_size == 0, (
            f"Grid dimensions ({grid_h}, {grid_w}) must be divisible by "
            f"patch_size ({patch_size})"
        )

        n_patches_h = grid_h // patch_size
        n_patches_w = grid_w // patch_size
        self.n_patches = n_patches_h * n_patches_w
        patch_dim = input_dim * patch_size * patch_size

        self.proj = nn.Linear(patch_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Encode 2-D field into patch tokens.

        Parameters
        ----------
        x : Tensor
            Raw field data.  Shape ``[B, H, W, C]`` if
            ``channels_first=False``, else ``[B, C, H, W]``.

        Returns
        -------
        Tensor
            Encoded tokens ``[B, n_patches, d_model]``.
        """
        if self.channels_first:
            # [B, C, H, W] → [B, H, W, C]
            x = x.permute(0, 2, 3, 1)

        B, H, W, C = x.shape
        p = self.patch_size

        # Extract patches: [B, H/p, p, W/p, p, C]
        patches = x.reshape(B, H // p, p, W // p, p, C)
        # Rearrange to [B, n_patches, p*p*C]
        patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(B, self.n_patches, p * p * C)

        tokens = self.proj(patches)  # [B, n_patches, d_model]
        tokens = tokens + self.pos
        tokens = self.drop(tokens)
        tokens = self.norm(tokens)
        return tokens


__all__ = ["ScientificEncoder"]
