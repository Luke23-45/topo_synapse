"""Z3 SYNAPSE Experiment Runner.

Orchestrates multi-dataset, multi-seed experiments by programmatically
composing Hydra configurations and executing the training pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf

from synapse.arch.training import train
from synapse.arch.runtime import resolve_project_path, save_json

log = logging.getLogger(__name__)


def run_experiment(config_name: str, overrides: list[str] | None = None) -> None:
    """Run a full Z3 experiment based on a configuration file.

    Parameters
    ----------
    config_name : str
        Name of the experiment config (e.g., "experiment/full").
    overrides : list of str, optional
        Additional Hydra overrides.
    """
    from hydra import initialize_config_dir
    import os
    
    # 1. Initialize Hydra and compose the experiment config
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config"))
    
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    # 2. Extract experiment metadata
    exp_name = cfg.get("experiment", {}).get("name", "synapse_exp")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_project_path(f"synapse_outputs/{timestamp}_{exp_name}")
    output_root.mkdir(parents=True, exist_ok=True)

    log.info("Starting Z3 Experiment: %s", exp_name)
    log.info("Output directory: %s", output_root)

    # 3. Identify datasets and seeds to sweep over
    datasets = cfg.get("dataset", ["synthetic"])
    if isinstance(datasets, str):
        datasets = [datasets]

    seeds = cfg.get("execution", {}).get("seed", [42])
    if isinstance(seeds, int):
        seeds = [seeds]

    # 4. Multi-dataset / Multi-seed loop
    all_results = {}

    for ds_name in datasets:
        all_results[ds_name] = {}
        log.info("\n" + "="*60)
        log.info("DATASET: %s", ds_name.upper())
        log.info("="*60)

        for seed in seeds:
            run_name = f"{ds_name}_seed{seed}"
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            log.info("--- Running Seed %d ---", seed)

            # Re-compose config for this specific run
            run_overrides = (overrides or []) + [
                f"+execution.seed={seed}",
            ]
            # Only add dataset override if we are actually sweeping
            if len(datasets) > 1 or ds_name != datasets[0]:
                run_overrides.append(f"+dataset={ds_name}")
                
            with initialize_config_dir(version_base="1.3", config_dir=config_dir):
                run_cfg = compose(config_name=config_name, overrides=run_overrides)

            full_cfg = OmegaConf.to_container(run_cfg, resolve=True)
            log.info("Available config keys: %s", list(full_cfg.keys()))

            # Extract lightning params
            training_cfg = full_cfg.get("training", {})
            trainer_cfg = full_cfg.get("trainer", {})
            ema_cfg = full_cfg.get("ema", {})

            lightning_params = {
                "max_epochs": training_cfg.get("epochs", 10),
                "accelerator": trainer_cfg.get("accelerator") if trainer_cfg.get("accelerator") != "auto" else None,
                "devices": trainer_cfg.get("devices") if trainer_cfg.get("devices") != "auto" else None,
                "precision": trainer_cfg.get("precision") if trainer_cfg.get("precision") != "auto" else None,
                "warmup_steps": training_cfg.get("warmup_steps", 0),
                "ema_decay": ema_cfg.get("decay", 0.999),
                "gradient_clip_norm": training_cfg.get("gradient_clip_norm", 1.0),
                "early_stopping_patience": training_cfg.get("early_stopping_patience", 10),
                "logger_name": "csv",
            }

            try:
                # Run training
                result = train(
                    full_cfg,
                    str(run_dir),
                    dataset_name=ds_name,
                    **lightning_params,
                )
                
                all_results[ds_name][seed] = {
                    "test_metrics": result.get("test_metrics"),
                    "run_dir": str(run_dir),
                }
                
                log.info("Seed %d complete: Accuracy=%.4f", seed, result.get("test_metrics", {}).get("accuracy", 0.0))

            except Exception as e:
                log.error("Run failed for dataset %s, seed %d: %s", ds_name, seed, e, exc_info=True)
                all_results[ds_name][seed] = {"error": str(e)}

    # 5. Save global experiment summary
    save_json(output_root / "experiment_summary.json", all_results)
    log.info("\nExperiment complete. Summary saved to %s", output_root / "experiment_summary.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="Z3 Experiment Runner")
    parser.add_argument("--config", default="experiment/smoke", help="Experiment config name")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    
    args = parser.parse_args()
    run_experiment(args.config, args.overrides)
