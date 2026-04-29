"""Geometric Preprocessor.

Transforms variable-size 3D point clouds into fixed-shape
``(N, T, d)`` arrays suitable for the Z3 DifferentiableHodgeProxy.

Operations
----------
1. **Anchor sub-sampling** — reduce large point clouds to *T* salient
   anchors using farthest-point sampling (deterministic, no gradient
   required at preprocessing time).  This mirrors the Z3 selector's
   role but operates as a coarse pre-filter before the model's own
   anchor selection.
2. **Unit-sphere normalization** — center each cloud at the origin
   and scale so that the maximum radius is 1.0.  This ensures metric
   stability across datasets with vastly different spatial extents.
3. **Padding** — for clouds with fewer than *T* points, zero-pad.

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §4.4
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..adapters.base import DatasetBundle, DatasetSpec
from .base import Preprocessor


class GeometricPreprocessor(Preprocessor):
    """Preprocessor for geometric / point-cloud modalities.

    Parameters
    ----------
    target_length : int
        Fixed number of points *T* per sample after sub-sampling.
    normalize : bool
        Whether to apply unit-sphere normalization (default ``True``).
    """

    def __init__(
        self,
        *,
        target_length: int = 512,
        normalize: bool = True,
    ) -> None:
        self._target_length = target_length
        self._normalize = normalize

    @property
    def name(self) -> str:
        return "geometric"

    def __call__(self, bundle: DatasetBundle) -> DatasetBundle:
        train_seqs = self._process_clouds(bundle.train_sequences)
        val_seqs = self._process_clouds(bundle.val_sequences)
        test_seqs = self._process_clouds(bundle.test_sequences)

        updated_spec = replace(
            bundle.spec,
            sequence_length=self._target_length,
        )

        return DatasetBundle(
            train_sequences=train_seqs.astype(np.float32),
            train_labels=bundle.train_labels,
            val_sequences=val_seqs.astype(np.float32),
            val_labels=bundle.val_labels,
            test_sequences=test_seqs.astype(np.float32),
            test_labels=bundle.test_labels,
            spec=updated_spec,
            metadata=bundle.metadata,
        )

    # ------------------------------------------------------------------ #

    def _process_clouds(self, clouds: np.ndarray) -> np.ndarray:
        """Sub-sample, normalize, and pad/truncate point clouds.

        Parameters
        ----------
        clouds : np.ndarray
            Shape ``(N, T_raw, d)`` — raw point clouds.

        Returns
        -------
        np.ndarray
            Shape ``(N, target_length, d)`` — processed clouds.
        """
        N = clouds.shape[0]
        d = clouds.shape[2]
        T = self._target_length
        result = np.zeros((N, T, d), dtype=np.float32)

        for i in range(N):
            cloud = clouds[i].astype(np.float64)

            # Unit-sphere normalization: center + scale.
            if self._normalize:
                centroid = cloud.mean(axis=0)
                cloud = cloud - centroid
                max_radius = np.linalg.norm(cloud, axis=1).max()
                if max_radius > 1e-8:
                    cloud = cloud / max_radius

            # Sub-sample or pad to target length.
            n_pts = cloud.shape[0]
            if n_pts >= T:
                indices = self._farthest_point_sample(cloud, T)
                result[i] = cloud[indices].astype(np.float32)
            else:
                result[i, :n_pts, :] = cloud.astype(np.float32)
                # Remaining rows stay zero-padded.

        return result

    @staticmethod
    def _farthest_point_sample(
        cloud: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Deterministic farthest-point sampling.

        Iteratively selects the point farthest from the current set
        of chosen points.  The first point is always the centroid-
        nearest point for determinism.

        Parameters
        ----------
        cloud : np.ndarray
            Shape ``(N, d)`` — point cloud (already centered).
        n_samples : int
            Number of points to select.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` — indices into the input cloud.
        """
        n_pts = cloud.shape[0]
        if n_samples >= n_pts:
            return np.arange(n_pts)

        # Start from the point closest to the origin (centroid).
        distances = np.linalg.norm(cloud, axis=1)
        selected_indices = [int(np.argmin(distances))]

        # Initialize min-distances from each point to the selected set.
        min_distances = np.full(n_pts, np.inf, dtype=np.float64)
        min_distances = np.minimum(
            min_distances,
            np.linalg.norm(cloud - cloud[selected_indices[0]], axis=1),
        )

        for _ in range(1, n_samples):
            next_idx = int(np.argmax(min_distances))
            selected_indices.append(next_idx)
            new_dists = np.linalg.norm(cloud - cloud[next_idx], axis=1)
            min_distances = np.minimum(min_distances, new_dists)

        return np.array(selected_indices, dtype=np.intp)


__all__ = ["GeometricPreprocessor"]
