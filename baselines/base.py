"""Abstract interface for all Z3 baseline backbones.

Every baseline model (MLP, TCN, PTv3, SNN) and the proposed Deep Hodge
model must implement ``BaselineBackbone``.  The interface guarantees a
uniform call signature so the unified training pipeline can swap
backbones without code changes.

Design contract
---------------
- Input:  [B, N, d_model] encoded tokens (from any modality encoder)
- Output: [B, num_classes] logits (or features when ``return_features=True``)
- All backbones expose ``num_parameters`` for fair comparison reporting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaselineBackbone(nn.Module, ABC):
    """Abstract base class for Z3 baseline backbones.

    Subclasses must implement:
        - ``forward(x, return_features)`` → logits or (logits, features)
        - ``backbone_name`` property → str identifier

    The ``num_parameters`` property is provided by default but can be
    overridden for custom parameter counting (e.g. excluding frozen layers).
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Process encoded tokens and produce task output.

        Parameters
        ----------
        x : Tensor
            Encoded tokens of shape ``[B, N, d_model]`` where *N* is
            the number of tokens (sequence length or point count) and
            *d_model* is the shared hidden dimension.
        return_features : bool
            If ``True``, return a tuple ``(logits, features)`` where
            *features* is a ``[B, d_model]`` pooled representation
            suitable for downstream metric computation or visualization.
            If ``False``, return only logits.

        Returns
        -------
        Tensor or tuple[Tensor, Tensor]
            ``[B, num_classes]`` logits, or ``(logits, features)`` when
            ``return_features=True``.
        """
        raise NotImplementedError

    @property
    def backbone_name(self) -> str:
        """Canonical name used in configs, logging, and reports."""
        return self.__class__.__name__

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count (for fair comparison)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["BaselineBackbone"]
