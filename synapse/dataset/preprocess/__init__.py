"""Z3 Modality Preprocessors.

Each preprocessor transforms raw samples from a specific modality into
the uniform ``(N, T, d)`` sequence format that the Z3 model consumes.

Available preprocessors
----------------------
- ``TemporalPreprocessor``  — pad/truncate/z-score for time-series.
- ``GeometricPreprocessor`` — farthest-point sub-sample + unit-sphere
  normalization for point clouds.
- ``ScientificPreprocessor`` — grid flattening + per-field z-score
  for structured scientific data.
"""

from __future__ import annotations

from .base import Preprocessor
from .geometric import GeometricPreprocessor
from .scientific import ScientificPreprocessor
from .temporal import TemporalPreprocessor

__all__ = [
    "GeometricPreprocessor",
    "Preprocessor",
    "ScientificPreprocessor",
    "TemporalPreprocessor",
]
