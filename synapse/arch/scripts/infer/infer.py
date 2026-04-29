"""Inference CLI script for Z3 SYNAPSE models.

Uses Hydra + OmegaConf for production-grade experiment management.

Usage:
    python -m synapse.synapse.scripts.infer
    python -m synapse.synapse.scripts.infer checkpoint.path=path/to/best.pt
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from synapse.empirical.datasets.synthetic_topology import TOPOLOGY_LABELS, generate_topology_dataset
from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import Z3TopologyFirstModel
from synapse.utils.io import save_json


@hydra.main(
    config_path="../../../config",
    config_name="infer/default",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:
    """Run inference with a trained Z3 model using Hydra configuration."""
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

    # Generate synthetic samples
    sequences, _, names = generate_topology_dataset(
        cfg.data.num_samples,
        length=model_cfg.max_history_tokens,
        noise_std=cfg.data.noise_std,
        seed=cfg.execution.seed + 10,
    )

    # Run inference
    with torch.no_grad():
        out = model(torch.from_numpy(sequences).float())

    inv_labels = {v: k for k, v in TOPOLOGY_LABELS.items()}
    preds = out.logits.argmax(dim=-1).tolist()

    # Output path
    output_path = Path(cfg.output.path)
    if "hydra" in cfg:
        output_path = Path(cfg.hydra.run.dir) / "infer.json"

    # Save results
    if cfg.output.save_predictions:
        save_json(
            output_path,
            {
                "predictions": [inv_labels[p] for p in preds],
                "ground_truth": names,
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
