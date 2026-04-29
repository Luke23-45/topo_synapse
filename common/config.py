from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class SimpleNamespaceConfig:
    payload: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:
        try:
            value = self.payload[item]
            if isinstance(value, dict):
                return SimpleNamespaceConfig(value)
            return value
        except KeyError:
            raise AttributeError(f"'SimpleNamespaceConfig' object has no attribute '{item}'")

    def get(self, item: str, default: Any = None) -> Any:
        return self.payload.get(item, default)


def load_config(path: str | Path) -> SimpleNamespaceConfig:
    return SimpleNamespaceConfig(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
