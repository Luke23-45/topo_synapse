"""Post-training evaluation hooks for Z3 SYNAPSE training.

Provides the evaluator factory and the post-training evaluation
protocol that runs automatically after training completes, mirroring
the baselines rollout pattern.

Works with both plain ``nn.Module`` and ``LightningModule`` models.
When a ``LightningModule`` is passed, the underlying ``.model``
attribute is extracted automatically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from synapse.dataset.adapters.base import DatasetBundle
from synapse.evaluation.configs import load_eval_config
from synapse.evaluation.reporting import generate_json_report, generate_markdown_report
from synapse.evaluation.rollout import aggregate_rollout_results, rollout_evaluate_dataset
from synapse.evaluation.runners.classification import ClassificationEvaluator
from synapse.evaluation.runners.geometric import GeometricEvaluator
from synapse.evaluation.runners.scientific import ScientificEvaluator
from synapse.evaluation.runners.temporal import TemporalEvaluator
from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import Z3TopologyFirstModel

log = logging.getLogger(__name__)


def _unwrap_model(model):
    """Extract the underlying Z3TopologyFirstModel from a LightningModule or return as-is."""
    return getattr(model, "model", model)


def build_evaluator(
    config: SynapseConfig,
    test_loader: DataLoader,
    output_dir: Path,
    dataset_name: str,
):
    """Instantiate the correct evaluator based on dataset modality.

    Parameters
    ----------
    config : SynapseConfig
    test_loader : DataLoader
    output_dir : Path
    dataset_name : str

    Returns
    -------
    BaseEvaluator subclass instance.
    """
    from synapse.dataset.registry import create_adapter

    eval_cfg = load_eval_config(dataset_name)
    adapter = create_adapter(dataset_name)
    modality = adapter.spec.modality

    if modality == "temporal":
        noise_levels = (
            eval_cfg.get("robustness", {})
            .get("noise_sweep", {})
            .get("levels")
        )
        length_scales = (
            eval_cfg.get("robustness", {})
            .get("length_sweep", {})
            .get("lengths")
        )
        return TemporalEvaluator(
            config, test_loader, output_dir,
            noise_levels=noise_levels,
            length_scales=length_scales,
        )
    elif modality == "geometric":
        rotation_angles = (
            eval_cfg.get("robustness", {})
            .get("rotation_sweep", {})
            .get("angles_rad")
        )
        return GeometricEvaluator(
            config, test_loader, output_dir,
            rotation_angles=rotation_angles,
        )
    elif modality == "scientific":
        return ScientificEvaluator(config, test_loader, output_dir)
    else:
        return ClassificationEvaluator(config, test_loader, output_dir)


def run_post_training_evaluation(
    model,
    config: SynapseConfig,
    test_loader: DataLoader,
    bundle: DatasetBundle,
    output_root: Path,
    dataset_name: str,
) -> dict:
    """Run the full evaluation protocol after training completes.

    Mirrors the baselines rollout pattern: after training, automatically
    runs evaluation (metrics, topology alignment, robustness sweeps,
    rollout) and generates reports.

    Parameters
    ----------
    model : Z3TopologyFirstModel or LightningModule
        The trained model.  If a LightningModule, the underlying
        ``.model`` attribute is extracted automatically.
    config : SynapseConfig
    test_loader : DataLoader
    bundle : DatasetBundle
    output_root : Path
    dataset_name : str

    Returns
    -------
    dict with evaluation results and report paths.
    """
    base_model = _unwrap_model(model)

    eval_dir = output_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running post-training evaluation for dataset: %s", dataset_name)

    # 1. Modality-specific evaluation
    evaluator = build_evaluator(config, test_loader, eval_dir, dataset_name)
    eval_result = evaluator.evaluate(base_model)
    eval_result.save_json(eval_dir / "eval_result.json")

    # 2. Rollout evaluation (robustness under compounding noise)
    rollout_results: dict = {}
    try:
        rollout_dict = rollout_evaluate_dataset(
            base_model, test_loader, config,
            n_steps=10, noise_scale=0.05, max_batches=10,
        )
        rollout_agg = aggregate_rollout_results(rollout_dict)
        rollout_results = rollout_agg
        (eval_dir / "rollout_results.json").write_text(
            json.dumps(rollout_agg, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning("Rollout evaluation failed: %s", e)

    # 3. Generate reports
    report_dir = eval_dir / "reports"
    report_dir.mkdir(exist_ok=True)

    generate_json_report(
        {dataset_name: eval_result},
        output_path=report_dir / "evaluation_report.json",
    )
    generate_markdown_report(
        {dataset_name: eval_result},
        output_path=report_dir / "evaluation_report.md",
    )

    log.info("Post-training evaluation complete. Reports saved to %s", report_dir)

    return {
        "eval_result": eval_result.to_dict(),
        "rollout": rollout_results,
        "report_dir": str(report_dir),
    }
