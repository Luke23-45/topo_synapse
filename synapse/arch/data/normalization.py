"""Normalization Statistics for the Z3 Lift Layer.

Computes per-feature mean and standard deviation from training
sequences, augmenting each vector with time, delta, and saliency
channels to match the Z3 anchor vector structure.
"""

from __future__ import annotations

import numpy as np


def compute_normalization_stats(sequences: np.ndarray) -> dict[str, np.ndarray]:
    """Compute z-score normalization statistics for the lift layer.

    Augments each observation with a time coordinate (0), a delta
    indicator (1 except at t=0), and a saliency channel (0), then
    computes mean and std over the flattened training set.

    .. note::
       The returned stats have shape ``(d + 3,)`` because of the
       augmentation channels.  The lift layer must also add these
       three channels to each input vector **before** applying the
       normalization.  If the lift layer normalizes raw ``(d,)``
       vectors directly, the shapes will mismatch.

    Parameters
    ----------
    sequences : np.ndarray
        Shape ``(N, T, d)`` — training sequences.

    Returns
    -------
    dict
        ``"mu"`` — shape ``(d + 3,)`` mean vector.
        ``"sigma"`` — shape ``(d + 3,)`` std vector (zeros replaced
        with 1.0 to avoid division by zero).
    """
    time = np.zeros((sequences.shape[0], sequences.shape[1], 1), dtype=np.float64)
    delta = np.ones((sequences.shape[0], sequences.shape[1], 1), dtype=np.float64)
    delta[:, 0, 0] = 0.0
    saliency = np.zeros((sequences.shape[0], sequences.shape[1], 1), dtype=np.float64)
    vectors = np.concatenate([time, sequences.astype(np.float64), delta, saliency], axis=-1)
    flat = vectors.reshape(-1, vectors.shape[-1])
    sigma = flat.std(axis=0)
    sigma[sigma <= 0] = 1.0
    return {"mu": flat.mean(axis=0), "sigma": sigma}
