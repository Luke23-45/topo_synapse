"""Temporal Preprocessor.

Transforms variable-length time-series data into fixed-shape
``(N, T, d)`` arrays suitable for the Z3 CausalEventEncoder.

Operations
----------
1. **Pad / truncate** each sequence to the target length *T*.
   - Shorter sequences are zero-padded at the end.
   - Longer sequences are truncated from the end.
2. **Z-score normalization** computed from the training split only
   (to prevent data leakage).  Statistics are stored in the bundle
   metadata so they can be reused at inference time.

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §4.4
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..adapters.base import DatasetBundle, DatasetSpec
from .base import Preprocessor


class TemporalPreprocessor(Preprocessor):
    """Preprocessor for temporal / time-series modalities.

    Parameters
    ----------
    target_length : int
        Fixed sequence length *T* after padding/truncation.
    normalize : bool
        Whether to apply z-score normalization (default ``True``).
    eps : float
        Small constant to prevent division by zero in normalization.
    """

    def __init__(
        self,
        *,
        target_length: int = 128,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        self._target_length = target_length
        self._normalize = normalize
        self._eps = eps

    @property
    def name(self) -> str:
        return "temporal"

    def __call__(self, bundle: DatasetBundle) -> DatasetBundle:
        train_seqs = self._reshape_sequences(bundle.train_sequences)
        val_seqs = self._reshape_sequences(bundle.val_sequences)
        test_seqs = self._reshape_sequences(bundle.test_sequences)

        mu: np.ndarray | None = None
        sigma: np.ndarray | None = None

        if self._normalize:
            # Compute normalization from training split only.
            flat = train_seqs.reshape(-1, train_seqs.shape[-1]).astype(np.float64)
            mu = flat.mean(axis=0).astype(np.float32)
            sigma = flat.std(axis=0).astype(np.float32)
            sigma[sigma < self._eps] = 1.0

            train_seqs = (train_seqs - mu) / sigma
            val_seqs = (val_seqs - mu) / sigma
            test_seqs = (test_seqs - mu) / sigma

        updated_spec = replace(
            bundle.spec,
            sequence_length=self._target_length,
        )

        metadata = {**bundle.metadata}
        if mu is not None:
            metadata["normalization_mean"] = mu
            metadata["normalization_std"] = sigma

        return DatasetBundle(
            train_sequences=train_seqs.astype(np.float32),
            train_labels=bundle.train_labels,
            val_sequences=val_seqs.astype(np.float32),
            val_labels=bundle.val_labels,
            test_sequences=test_seqs.astype(np.float32),
            test_labels=bundle.test_labels,
            spec=updated_spec,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #

    def _reshape_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Pad or truncate sequences to the target length.

        Parameters
        ----------
        sequences : np.ndarray
            Shape ``(N, T_raw, d)`` or ``(N, d)`` (single-timestep).

        Returns
        -------
        np.ndarray
            Shape ``(N, target_length, d)``.
        """
        if sequences.ndim == 2:
            # (N, d) → (N, 1, d)
            sequences = sequences[:, np.newaxis, :]

        N, T_raw, d = sequences.shape
        T = self._target_length

        if T_raw == T:
            return sequences

        if T_raw < T:
            # Zero-pad at the end.
            pad_width = T - T_raw
            padding = np.zeros((N, pad_width, d), dtype=sequences.dtype)
            return np.concatenate([sequences, padding], axis=1)

        # Truncate from the end.
        return sequences[:, :T, :]


__all__ = ["TemporalPreprocessor"]
