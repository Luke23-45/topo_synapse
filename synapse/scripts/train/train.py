"""Training CLI script for Z3 SYNAPSE models.

Uses Hydra + OmegaConf for production-grade experiment management.

Usage:
    python -m synapse.synapse.scripts.train
    python -m synapse.synapse.scripts.train dataset=telecom
    python -m synapse.synapse.scripts.train experiment=full
    python -m synapse.synapse.scripts.train --multirun dataset=synthetic,telecom,spatial,photonic
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ...training import train
from ...utils.runtime import build_run_artifacts


@hydra.main(
    config_path="../../config/experiment",
    config_name="smoke",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:
    """Train a Z3 SYNAPSE model using Hydra configuration.

    The cfg is composed from:
    - config/training/default.yaml (base)
    - config/training/{dataset}.yaml (dataset override)
    - Command line overrides (e.g., training.max_epochs=50)
    """
    # Resolve config to ensure all interpolations are evaluated
    OmegaConf.resolve(cfg)

    # Extract Lightning trainer parameters from config
    trainer_cfg = cfg.get("trainer", {})
    lightning_params = {
        "max_epochs": cfg.training.get("epochs", cfg.training.get("max_epochs", 10)),
        "accelerator": trainer_cfg.get("accelerator", "auto"),
        "devices": trainer_cfg.get("devices", "auto"),
        "precision": trainer_cfg.get("precision", "32-true"),
        "warmup_steps": cfg.training.get("warmup_steps", 0),
        "ema_decay": cfg.get("ema", {}).get("decay", 0.999),
        "gradient_clip_norm": cfg.training.get("gradient_clip_norm", 1.0),
        "early_stopping_patience": cfg.training.get("early_stopping_patience", 10),
        "logger_name": cfg.get("logging", {}).get("logger", "csv"),
    }

    # Determine output directory
    output_dir = Path(cfg.get("logging", {}).get("output_dir", "synapse_outputs"))
    if "hydra" in cfg:
        output_dir = Path(cfg.hydra.run.dir)

    # Build a temporary config dict for the train function (backward compat)
    # The train function expects a config with model, data, training, execution sections
    train_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(f"DEBUG: train_cfg_dict keys: {list(train_cfg_dict.keys())}")

    # Run training
    result = train(
        train_cfg_dict,
        str(output_dir),
        dataset_name=cfg.get("data", {}).get("dataset", "synthetic"),
        **lightning_params,
    )

    # Save artifacts
    artifacts = build_run_artifacts(str(output_dir))
    artifacts["root"].mkdir(parents=True, exist_ok=True)
    artifacts["history"].write_text(
        json.dumps(result["history"], indent=2), encoding="utf-8"
    )
    artifacts["final_metrics"].write_text(
        json.dumps(result["test_metrics"], indent=2), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
