"""Evaluation CLI script for Z3 SYNAPSE models.

Uses Hydra + OmegaConf for production-grade experiment management.

Usage:
    python -m synapse.synapse.scripts.eval
    python -m synapse.synapse.scripts.eval dataset=telecom
    python -m synapse.synapse.scripts.eval checkpoint.path=path/to/best.pt
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from synapse.evaluation.configs import load_eval_config
from synapse.evaluation.reporting import generate_json_report, generate_markdown_report
from synapse.arch.data.data import build_dataloaders
from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import Z3TopologyFirstModel

log = logging.getLogger(__name__)


def _build_evaluator(config, test_loader, output_dir, dataset_name):
    """Instantiate the correct evaluator based on dataset modality."""
    from synapse.dataset.registry import create_adapter

    eval_cfg = load_eval_config(dataset_name)
    adapter = create_adapter(dataset_name)
    modality = adapter.spec.modality

    from synapse.evaluation.runners.classification import ClassificationEvaluator
    from synapse.evaluation.runners.geometric import GeometricEvaluator
    from synapse.evaluation.runners.scientific import ScientificEvaluator
    from synapse.evaluation.runners.temporal import TemporalEvaluator

    if modality == "temporal":
        noise_levels = eval_cfg.get("robustness", {}).get("noise_sweep", {}).get("levels")
        length_scales = eval_cfg.get("robustness", {}).get("length_sweep", {}).get("lengths")
        return TemporalEvaluator(
            config, test_loader, output_dir,
            noise_levels=noise_levels, length_scales=length_scales,
        )
    elif modality == "geometric":
        rotation_angles = eval_cfg.get("robustness", {}).get("rotation_sweep", {}).get("angles_rad")
        return GeometricEvaluator(
            config, test_loader, output_dir, rotation_angles=rotation_angles,
        )
    elif modality == "scientific":
        return ScientificEvaluator(config, test_loader, output_dir)
    else:
        return ClassificationEvaluator(config, test_loader, output_dir)


@hydra.main(
    config_path="../../../config",
    config_name="eval/default",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:
    """Evaluate a trained Z3 SYNAPSE model using Hydra configuration."""
    OmegaConf.resolve(cfg)

    # Build model config from Hydra config
    model_cfg = SynapseConfig(
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        hidden_dim=cfg.model.hidden_dim,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        ffn_ratio=cfg.model.ffn_ratio,
        dropout=cfg.model.dropout,
        K=cfg.model.K,
        r=cfg.model.r,
        lam=cfg.model.lam,
        Q=cfg.model.Q,
        k=cfg.model.k,
        max_history_tokens=cfg.model.max_history_tokens,
    )

    # Build model and load checkpoint
    model = Z3TopologyFirstModel(model_cfg)
    checkpoint_path = Path(cfg.checkpoint.path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    # Build dataloaders
    train_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    _, _, test_loader, bundle = build_dataloaders(
        train_cfg_dict, dataset_name=cfg.data.dataset
    )

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Output directory
    output_dir = Path(cfg.output.dir)
    if "hydra" in cfg:
        output_dir = Path(cfg.hydra.run.dir) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    evaluator = _build_evaluator(model_cfg, test_loader, output_dir, cfg.data.dataset)
    result = evaluator.evaluate(model)

    # Save results
    if cfg.output.save_json:
        result.save_json(output_dir / "eval_result.json")
        generate_json_report(
            {cfg.data.dataset: result},
            output_path=output_dir / "evaluation_report.json",
        )
    if cfg.output.save_markdown:
        generate_markdown_report(
            {cfg.data.dataset: result},
            output_path=output_dir / "evaluation_report.md",
        )

    log.info("Evaluation complete. Results saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
