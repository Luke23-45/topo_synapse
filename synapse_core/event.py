"""
Causal Event Model — Z3 Reference: §4 of 01_main_definition.md

Implements the causal event-score functional E_t and saliency normalizer S_t.
For each t, E_t : X_t → [0,∞) is computed from finite differences.
"""

from __future__ import annotations

import torch
from torch import nn

from .saliency import SaliencyNormalizer


class CausalEventEncoder(nn.Module):
    """Computes per-frame event scores from finite differences.

    Input: sequence (B, T, d)
    Output: event_scores (B, T) — raw causal event logits
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        prev = torch.cat([sequence[:, :1, :], sequence[:, :-1, :]], dim=1)
        diff = sequence - prev
        pair = torch.cat([sequence, prev, diff], dim=-1)
        hidden = self.net(pair)
        return self.score_head(hidden).squeeze(-1)


class CausalEventModel(nn.Module):
    """Full event pipeline: encoder + normalizer.

    Returns (event_scores, saliency_scores) where saliency is non-negative
    and suitable for the relaxed selector.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = CausalEventEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.normalizer = SaliencyNormalizer()

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        event_scores = self.encoder(sequence)
        saliency = self.normalizer(event_scores, mask)
        return event_scores, saliency
