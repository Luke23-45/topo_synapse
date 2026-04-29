"""Z3 Dataset Registry — Adapter Factory.

Provides a single entry point for loading any supported dataset by name,
abstracting away format differences.  Adapters register themselves by
calling ``register_adapter()`` from their own module scope.

**Important:** This module does *not* auto-import adapters at load time
to avoid circular imports.  The caller is responsible for ensuring that
the desired adapter modules have been imported before calling
``create_adapter()``.  The ``adapters/__init__.py`` package handles
this by importing all built-in adapters.

Usage
-----
    from synapse.dataset.adapters import SyntheticAdapter  # triggers registration
    from synapse.dataset.registry import create_adapter

    adapter = create_adapter("synthetic", length=128, noise_std=0.03, seed=7)
    bundle = adapter.load_splits()

Reference
---------
- Baselines: ``baselines/src/data/registry.py``
- Z3 plan:   ``docs/implementions/plan.md`` §4.3
"""

from __future__ import annotations

import logging
from typing import Any

from .adapters.base import DatasetSpec, Z3Adapter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal registry state
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, type[Z3Adapter]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_adapter(name: str, cls: type[Z3Adapter]) -> None:
    """Register an adapter class under a canonical name.

    Called by each adapter module at import time.  If a duplicate name
    is registered, the later registration wins and a warning is emitted.

    Parameters
    ----------
    name : str
        Canonical dataset name (e.g. ``"synthetic"``, ``"telecom"``).
    cls : type[Z3Adapter]
        Concrete adapter class (not an instance).
    """
    if name in _ADAPTERS:
        log.warning(
            "Overwriting existing adapter registration for '%s': "
            "%s → %s",
            name,
            _ADAPTERS[name].__name__,
            cls.__name__,
        )
    _ADAPTERS[name] = cls


def create_adapter(name: str, **kwargs: Any) -> Z3Adapter:
    """Instantiate a registered adapter by name.

    Parameters
    ----------
    name : str
        Canonical dataset name.
    **kwargs
        Keyword arguments forwarded to the adapter constructor.

    Returns
    -------
    Z3Adapter
        A ready-to-use adapter instance.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    if name not in _ADAPTERS:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            f"Available: {sorted(_ADAPTERS.keys())}"
        )
    return _ADAPTERS[name](**kwargs)


def list_available_datasets() -> dict[str, DatasetSpec]:
    """Return metadata about all registered datasets.

    Instantiates each adapter with default arguments to read its spec.
    Adapters that require constructor arguments (e.g. ``local_path``)
    may raise; such entries are silently skipped.

    Returns
    -------
    dict
        Mapping ``dataset_name → DatasetSpec``.
    """
    datasets: dict[str, DatasetSpec] = {}
    for name, cls in sorted(_ADAPTERS.items()):
        try:
            adapter = cls()
            datasets[name] = adapter.spec
        except TypeError:
            log.debug(
                "Skipping adapter '%s' in list_available_datasets: "
                "constructor requires arguments.",
                name,
            )
    return datasets


def registered_names() -> list[str]:
    """Return the sorted list of currently registered adapter names."""
    return sorted(_ADAPTERS.keys())


__all__ = [
    "create_adapter",
    "list_available_datasets",
    "register_adapter",
    "registered_names",
]
