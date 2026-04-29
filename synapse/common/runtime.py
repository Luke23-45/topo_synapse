from __future__ import annotations

from pathlib import Path

from synapse.synapse_arch.config import SynapseConfig
from synapse.utils.io import load_yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT
CONFIG_ROOT = PACKAGE_ROOT / "config"


def load_config(path: str | None) -> SynapseConfig:
    payload = load_yaml(path or CONFIG_ROOT / "default.yaml")
    return SynapseConfig(**payload)
