"""Active topological encoder entry point.

The active codebase standardizes on the Z4 encoder. The legacy Z3
implementation now lives under ``synapse.common.encoders.legacy``.
"""

from __future__ import annotations

from .z4_topological_encoder import Z4TopologicalEncoder


class TopologicalEncoder(Z4TopologicalEncoder):
    """Compatibility alias for the active Z4 topological encoder."""


__all__ = ["TopologicalEncoder"]
