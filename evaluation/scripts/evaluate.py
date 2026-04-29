"""CLI script for evaluating a trained Z3 SYNAPSE model.

Runs the full evaluation protocol (classification metrics, topology
alignment, robustness sweeps) based on the dataset modality and
evaluation config, then generates JSON and Markdown reports.

Usage:
    python -m synapse.evaluation.scripts.evaluate \\
        --checkpoint path/to/best.pt \\
        --dataset synthetic \\
        --output-dir synapse_outputs/eval
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from synapse.common.runtime import load_config
from synapse.dataset.registry import create_adapter
from synapse.evaluation.configs import load_eval_config
from synapse.evaluation.runners.classification import ClassificationEvaluator
from synapse.evaluation.runners.temporal import TemporalEvaluator
from synapse.evaluation.runners.geometric import GeometricEvaluator
from synapse.evaluation.runners.scientific import ScientificEvaluator
from synapse.evaluation.reporting import generate_json_report, generate_markdown_report
from synapse.synapse.data.data import build_dataloaders
from synapse.synapse_arch.model import Z3TopologyFirstModel

log = logging.getLogger(__name__)


def _build_evaluator(
    config,
    test_loader: DataLoader,
    output_dir: Path,
    eval_cfg: dict,
    dataset_name: str,
):
    """Instantiate the correct evaluator based on dataset modality."""
    adapter = create_adapter(dataset_name)
    modality = adapter.spec.modality

    if modality == "temporal":
        noise_levels = eval_cfg.get("robustness", {}).get("noise_sweep", {}).get("levels")
        length_scales = eval_cfg.get("robustness", {}).get("length_sweep", {}).get("lengths")
        return TemporalEvaluator(
            config, test_loader, output_dir,
            noise_levels=noise_levels,
            length_scales=length_scales,
        )
    elif modality == "geometric":
        rotation_angles = eval_cfg.get("robustness", {}).get("rotation_sweep", {}).get("angles_rad")
        return GeometricEvaluator(
            config, test_loader, output_dir,
            rotation_angles=rotation_angles,
        )
    elif modality == "scientific":
        return ScientificEvaluator(config, test_loader, output_dir)
    else:
        return ClassificationEvaluator(config, test_loader, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Z3 SYNAPSE model.")
    parser.add_argument("--config", default=None, help="Path to model config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name")
    parser.add_argument("--output-dir", default="synapse_outputs/eval", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = Z3TopologyFirstModel(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    _, _, test_loader, bundle = build_dataloaders(config, dataset_name=args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = load_eval_config(args.dataset)
    evaluator = _build_evaluator(config, test_loader, output_dir, eval_cfg, args.dataset)
    result = evaluator.evaluate(model)

    # Save results
    result.save_json(output_dir / "eval_result.json")
    generate_json_report({"default": result}, output_path=output_dir / "evaluation_report.json")
    generate_markdown_report({"default": result}, output_path=output_dir / "evaluation_report.md")

    log.info("Evaluation complete. Results saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
