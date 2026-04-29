"""Backbone Registry — factory for all Z3 baseline and proposed models.

Mirrors the pattern used by ``synapse.dataset.registry`` so that
backbone construction is centralized and any model can be instantiated
by name from a config string.

Usage
-----
    from synapse.baselines.registry import create_backbone
    backbone = create_backbone("mlp", d_model=64, num_classes=4)
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaselineBackbone

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal registry state
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[BaselineBackbone]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_backbone(name: str, cls: type[BaselineBackbone]) -> None:
    """Register a backbone class under a canonical name.

    Called by each backbone module at import time.  If a duplicate name
    is registered, the later registration wins and a warning is emitted.

    Parameters
    ----------
    name : str
        Canonical backbone name (e.g. ``"mlp"``, ``"tcn"``, ``"ptv3"``).
    cls : type[BaselineBackbone]
        Concrete backbone class (not an instance).
    """
    if name in _BACKENDS:
        log.warning(
            "Overwriting existing backbone registration for '%s': "
            "%s → %s",
            name,
            _BACKENDS[name].__name__,
            cls.__name__,
        )
    _BACKENDS[name] = cls


def create_backbone(name: str, **kwargs: Any) -> BaselineBackbone:
    """Instantiate a registered backbone by name.

    Parameters
    ----------
    name : str
        Canonical backbone name.
    **kwargs
        Keyword arguments forwarded to the backbone constructor.

    Returns
    -------
    BaselineBackbone
        A ready-to-use backbone instance.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backbone: '{name}'. "
            f"Available: {sorted(_BACKENDS.keys())}"
        )
    return _BACKENDS[name](**kwargs)


def list_available_backbones() -> dict[str, type[BaselineBackbone]]:
    """Return a copy of the current backbone registry."""
    return dict(_BACKENDS)


def registered_names() -> list[str]:
    """Return the sorted list of currently registered backbone names."""
    return sorted(_BACKENDS.keys())


__all__ = [
    "create_backbone",
    "list_available_backbones",
    "register_backbone",
    "registered_names",
]
