"""Modality-specific encoders for the Z3 unified model.

Each encoder normalizes a specific data modality into a common
``[B, N, d_model]`` token format that any backbone can consume.

Encoders
--------
- ``TemporalEncoder``    : 1-D time-series → tokens
- ``GeometricEncoder``   : 3-D point clouds → tokens
- ``ScientificEncoder``  : 2-D grid/field data → tokens
- ``TopologicalEncoder`` : Deep Hodge preprocessing (event + lift) → tokens
"""

from .temporal import TemporalEncoder
from .geometric import GeometricEncoder
from .scientific import ScientificEncoder
from .topological_encoder import TopologicalEncoder

_ENCODER_REGISTRY = {
    "temporal": TemporalEncoder,
    "geometric": GeometricEncoder,
    "scientific": ScientificEncoder,
    "topological": TopologicalEncoder,
}


def create_encoder(modality: str, **kwargs):
    """Factory: instantiate an encoder by modality name.

    Parameters
    ----------
    modality : str
        One of ``"temporal"``, ``"geometric"``, ``"scientific"``,
        ``"topological"``.
    **kwargs
        Forwarded to the encoder constructor.

    Returns
    -------
    nn.Module
    """
    if modality not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown modality: '{modality}'. "
            f"Available: {sorted(_ENCODER_REGISTRY.keys())}"
        )
    return _ENCODER_REGISTRY[modality](**kwargs)


__all__ = [
    "TemporalEncoder",
    "GeometricEncoder",
    "ScientificEncoder",
    "TopologicalEncoder",
    "create_encoder",
]
