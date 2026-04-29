from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synapse.common.config import load_config


@dataclass
class EmpiricalConfig:
    output_dir: str = "synapse_outputs/empirical"
    seed: int = 7
    length: int = 64
    train_size: int = 128
    val_size: int = 64
    test_size: int = 64
    noise_std: float = 0.03
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-3
    K_values: tuple[int, ...] = (4, 8)
    r_values: tuple[int, ...] = (1, 2)


def load_emp_config(path: str | Path | None = None) -> EmpiricalConfig:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    cfg = load_config(path)
    return EmpiricalConfig(**cfg.payload)
