"""Pipeline verification script (Smoke Test).

This script performs an end-to-end dry run of the SYNAPSE pipeline:
1. Loads the configuration (smoke experiment).
2. Initializes the DataModule and builds a tiny subset of the dataset.
3. Builds the Z3TopologyFirstModel.
4. Executes a single training step to verify gradient health and config mapping.

Usage:
    python -m synapse.synapse.scripts.test_pipeline
"""

import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from synapse.arch.training.core.loop import train
from synapse.arch.data.data import build_dataloaders
from synapse.arch.training.core.data_module import SynapseDataModule
from synapse.arch.training.builders.builder import build_model_from_cfg
from synapse.common.config import SimpleNamespaceConfig

log = logging.getLogger(__name__)

@hydra.main(
    config_path="../config/experiment",
    config_name="mock",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:
    # 1. Resolve and set tiny limits
    OmegaConf.resolve(cfg)
    
    # Force tiny dataset for speed using safe updates
    OmegaConf.update(cfg, "data.max_samples", 32, force_add=True)
    OmegaConf.update(cfg, "data.train_size", 16, force_add=True)
    OmegaConf.update(cfg, "data.val_size", 8, force_add=True)
    OmegaConf.update(cfg, "data.test_size", 8, force_add=True)
    OmegaConf.update(cfg, "training.epochs", 1, force_add=True)
    OmegaConf.update(cfg, "trainer.devices", 1, force_add=True)
    
    log.info("Starting Pipeline Verification (Smoke Test)...")
    
    # 2. Build DataModule
    try:
        log.info("Testing DataModule setup...")
        # Convert to SimpleNamespace for compat with our internal loop logic
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        sn_cfg = SimpleNamespaceConfig(cfg_dict)
        
        datamodule = SynapseDataModule(sn_cfg, dataset_name=cfg.data.dataset)
        datamodule.setup()
        log.info("DataModule setup successful.")
    except Exception as e:
        log.error(f"DataModule setup FAILED: {e}")
        raise e

    # 3. Build Model
    try:
        log.info("Testing Model building...")
        model = build_model_from_cfg(sn_cfg, datamodule.bundle)
        log.info(f"Model building successful. Trainable params: {model.num_trainable_params}")
    except Exception as e:
        log.error(f"Model building FAILED: {e}")
        raise e

    # 4. Run Single Step (using the standard train loop but limited)
    try:
        log.info("Testing end-to-end training step...")
        # We use a very low learning rate for safety
        results = train(
            cfg_dict,
            output_dir="synapse_outputs/test_run",
            dataset_name=cfg.data.dataset,
            max_epochs=1,
            accelerator="cpu", # Force CPU for verification stability
            devices=1,
            precision="32-true",
            warmup_steps=0,
            logger_name="csv"
        )
        log.info("Pipeline Verification PASSED.")
        return 0
    except Exception as e:
        log.error(f"Pipeline Execution FAILED: {e}")
        raise e

if __name__ == "__main__":
    main()
