"""Empirical experiment configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synapse.utils.io import load_yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "empirical.yaml"


def load_empirical_config() -> dict[str, Any]:
    """Load the full empirical YAML configuration."""
    return load_yaml(_CONFIG_PATH)


def get_experiment_config(experiment_id: str) -> dict[str, Any]:
    """Return the config block for a single experiment (e.g. 'EZ3-01').

    The returned dict includes an ``experiment_id`` key so that scripts
    never need to hardcode their own ID.
    """
    full = load_empirical_config()
    if experiment_id not in full:
        raise KeyError(f"Experiment '{experiment_id}' not found in empirical.yaml")
    cfg = dict(full[experiment_id])
    cfg["experiment_id"] = experiment_id
    return cfg


__all__ = ["get_experiment_config", "load_empirical_config"]
