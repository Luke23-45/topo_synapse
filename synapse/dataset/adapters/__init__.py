"""Z3 Dataset Adapters.

Each adapter implements ``Z3Adapter`` and produces a ``DatasetBundle``
with pre-split numpy arrays suitable for the Z3 training pipeline.

Adapters self-register with the global registry at import time so that
``create_adapter(name)`` can instantiate them by canonical name.

Available adapters
------------------
- ``synthetic`` — In-memory synthetic topology sequences.
- ``telecom``   — TelecomTS time-series (HuggingFace).
- ``spatial``   — SpatialLM 3D point clouds (HuggingFace).
- ``photonic``  — 2D photonic topology grids (HuggingFace).
"""

from __future__ import annotations

from .base import DatasetBundle, DatasetSpec, Z3Adapter
from .split_utils import apply_split, try_load_from_local
from .synthetic_adapter import SyntheticAdapter

# External adapters — import triggers self-registration.
# These require the 'datasets' package but are imported lazily
# by the registry when actually needed.
from .telecom_adapter import TelecomAdapter  # noqa: F401
from .spatial_adapter import SpatialAdapter  # noqa: F401
from .photonic_adapter import PhotonicAdapter  # noqa: F401

__all__ = [
    "DatasetBundle",
    "DatasetSpec",
    "PhotonicAdapter",
    "SpatialAdapter",
    "SyntheticAdapter",
    "TelecomAdapter",
    "Z3Adapter",
    "apply_split",
    "try_load_from_local",
]
