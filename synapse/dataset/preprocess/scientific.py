"""Scientific Preprocessor.

Transforms 2D grid data (e.g. photonic crystal unit cells) into
fixed-shape ``(N, T, d)`` sequences suitable for the Z3 Hodge
Laplacian Proxy.

Operations
----------
1. **Grid flattening** — reshape ``(H, W, F)`` grids into
   ``(H*W, F)`` sequences by raster scan (row-major).  The spatial
   adjacency is implicitly captured by the Z3 simplicial complex
   construction (Vietoris-Rips on the flattened coordinates).
2. **Per-field z-score normalization** — each feature channel is
   normalized independently using statistics from the training split.
   This prevents a single high-variance channel from dominating the
   proxy's spectral decomposition.
3. **Spatial coordinate injection** — optionally prepend (x, y)
   grid coordinates to each feature vector so the proxy can recover
   spatial structure.  When enabled, ``input_dim`` becomes ``F + 2``.

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §4.4
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..adapters.base import DatasetBundle, DatasetSpec
from .base import Preprocessor


class ScientificPreprocessor(Preprocessor):
    """Preprocessor for scientific / grid-based modalities.

    Parameters
    ----------
    inject_coordinates : bool
        If ``True``, prepend (x, y) normalized grid coordinates to
        each feature vector, increasing ``input_dim`` by 2.
    normalize : bool
        Whether to apply per-field z-score normalization.
    eps : float
        Small constant to prevent division by zero.
    """

    def __init__(
        self,
        *,
        inject_coordinates: bool = True,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        self._inject_coordinates = inject_coordinates
        self._normalize = normalize
        self._eps = eps

    @property
    def name(self) -> str:
        return "scientific"

    def __call__(self, bundle: DatasetBundle) -> DatasetBundle:
        train_seqs = self._flatten_grids(bundle.train_sequences)
        val_seqs = self._flatten_grids(bundle.val_sequences)
        test_seqs = self._flatten_grids(bundle.test_sequences)

        mu: np.ndarray | None = None
        sigma: np.ndarray | None = None

        if self._normalize:
            flat = train_seqs.reshape(-1, train_seqs.shape[-1]).astype(np.float64)
            mu = flat.mean(axis=0).astype(np.float32)
            sigma = flat.std(axis=0).astype(np.float32)
            sigma[sigma < self._eps] = 1.0

            train_seqs = (train_seqs - mu) / sigma
            val_seqs = (val_seqs - mu) / sigma
            test_seqs = (test_seqs - mu) / sigma

        new_input_dim = train_seqs.shape[2]
        new_seq_len = train_seqs.shape[1]
        updated_spec = replace(
            bundle.spec,
            input_dim=new_input_dim,
            sequence_length=new_seq_len,
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

    def _flatten_grids(self, grids: np.ndarray) -> np.ndarray:
        """Flatten grid samples into sequences with optional coordinates.

        Parameters
        ----------
        grids : np.ndarray
            Shape ``(N, H, W, F)`` — grid-structured data, or
            ``(N, T, d)`` if already flattened.

        Returns
        -------
        np.ndarray
            Shape ``(N, H*W, d_out)`` where ``d_out`` is ``F`` or
            ``F + 2`` depending on ``inject_coordinates``.
        """
        if grids.ndim == 3:
            # Already in sequence form — nothing to flatten.
            return grids

        if grids.ndim != 4:
            raise ValueError(
                f"Expected grids with ndim=4 (N, H, W, F), got ndim={grids.ndim}"
            )

        N, H, W, F = grids.shape
        # Row-major flatten: (N, H, W, F) → (N, H*W, F)
        sequences = grids.reshape(N, H * W, F)

        if not self._inject_coordinates:
            return sequences

        # Build normalized (x, y) coordinates.
        x_coords = np.linspace(0.0, 1.0, W, dtype=np.float32)
        y_coords = np.linspace(0.0, 1.0, H, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)  # (H, W)
        coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # (H*W, 2)
        coords = np.broadcast_to(coords, (N, H * W, 2)).copy()

        return np.concatenate([coords, sequences], axis=2)


__all__ = ["ScientificPreprocessor"]
