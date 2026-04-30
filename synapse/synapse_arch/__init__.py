"""Task-agnostic topology-first architecture package."""

from .config import SynapseConfig
from .normalized_lift import NormalizedLift

__all__ = ["ModelForwardOutput", "NormalizedLift", "SynapseConfig", "TopologyFirstModel", "Z3TopologyFirstModel"]


def __getattr__(name: str):
    if name in {"ModelForwardOutput", "TopologyFirstModel", "Z3TopologyFirstModel"}:
        from .model import ModelForwardOutput, TopologyFirstModel, Z3TopologyFirstModel

        exports = {
            "ModelForwardOutput": ModelForwardOutput,
            "TopologyFirstModel": TopologyFirstModel,
            "Z3TopologyFirstModel": Z3TopologyFirstModel,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
