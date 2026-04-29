"""Evaluation configuration files and loader.

Provides YAML-based evaluation configs that specify which metrics,
robustness sweeps, and report formats to use for each dataset modality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


EVAL_CONFIG_DIR = Path(__file__).resolve().parent


def load_eval_config(dataset_name: str) -> dict[str, Any]:
    """Load evaluation configuration for a dataset.

    Falls back to ``default.yaml`` if no dataset-specific config exists.

    Parameters
    ----------
    dataset_name : str
        Canonical dataset name (e.g. ``"synthetic"``, ``"telecom"``).

    Returns
    -------
    dict
    """
    specific = EVAL_CONFIG_DIR / f"{dataset_name}.yaml"
    default = EVAL_CONFIG_DIR / "default.yaml"

    path = specific if specific.exists() else default
    return yaml.safe_load(path.read_text(encoding="utf-8"))


__all__ = ["load_eval_config", "EVAL_CONFIG_DIR"]
