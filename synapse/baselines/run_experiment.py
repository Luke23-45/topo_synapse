#!/usr/bin/env python3
"""
Z3 Baseline Study — Multi-Backbone Experiment Runner
=====================================================

Trains all backbone conditions across configured datasets, evaluates,
runs statistical comparisons, and generates cross-backbone reports.

Usage:
    python -m synapse.baselines.run_experiment --config configs/experiment/smoke.yaml
    python -m synapse.baselines.run_experiment --config configs/experiment/full.yaml --backbones all
    python -m synapse.baselines.run_experiment --config configs/experiment/full.yaml --backbones mlp tcn
    python -m synapse.baselines.run_experiment --config configs/experiment/full.yaml --datasets synthetic telecom
    python -m synapse.baselines.run_experiment --config configs/experiment/smoke.yaml --seeds 1

This script orchestrates:
    1. Load experiment YAML config
    2. Build dataloaders via ``synapse.synapse.data.build_dataloaders``
    3. Train each backbone × dataset × seed combination
    4. Evaluate on held-out test sets
    5. Cross-backbone statistical comparisons (Welch's t-test)
    6. Publication-quality plots and reports

Reuses from ``synapse.synapse``:
    - ``synapse.synapse.data.data.build_dataloaders`` — data pipeline
    - ``synapse.synapse.losses.combined_loss.compute_loss`` — loss computation
    - ``synapse.synapse.training.builders.builder.resolve_normalization`` — normalization
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .src.core.config import (
    BackboneCondition,
    Z3DatasetSpec,
    Z3ExperimentConfig,
    load_experiment_config,
)
from .src.engine.train import train_backbone, TrainState
from .src.engine.evaluate import evaluate_backbone, BackboneEvaluation
from .src.engine.metrics import compare_conditions, ComparisonResult
from .src.engine.rollout import (
    rollout_evaluate_dataset,
    aggregate_rollout_results,
    RolloutResult,
)
from .src.reporting.report import generate_json_report, generate_markdown_report
from .src.reporting.visualize import plot_accuracy_comparison, plot_learning_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

ALL_BACKBONES = ["mlp", "tcn", "ptv3", "snn", "deep_hodge"]


# ---------------------------------------------------------------------------
# Data loading — delegates to synapse.synapse.data
# ---------------------------------------------------------------------------

def _build_dataloaders(
    config: Z3ExperimentConfig,
    dataset_spec: Z3DatasetSpec,
    seed: int,
):
    """Build dataloaders by delegating to ``synapse.synapse.data.build_dataloaders``."""
    from synapse.arch.data.data import build_dataloaders

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.model = _Cfg()
    cfg.model.input_dim = dataset_spec.input_dim
    cfg.model.output_dim = dataset_spec.num_classes
    cfg.model.max_history_tokens = dataset_spec.sequence_length
    cfg.model.hidden_dim = config.model.hidden_dim
    cfg.model.d_model = config.model.d_model
    cfg.model.num_heads = config.model.num_heads
    cfg.model.num_layers = config.model.num_layers
    cfg.model.ffn_ratio = config.model.ffn_ratio
    cfg.model.dropout = config.model.dropout
    cfg.model.K = config.model.K
    cfg.model.r = config.model.r
    cfg.model.lam = config.model.lam
    cfg.model.Q = 1
    cfg.model.k = config.model.k
    cfg.model.max_proxy_points = config.model.max_proxy_points
    cfg.model.num_scales = config.model.num_scales
    cfg.model.tau = 1e-4

    cfg.data = _Cfg()
    cfg.data.dataset = dataset_spec.name
    cfg.data.train_size = dataset_spec.train_size
    cfg.data.val_size = dataset_spec.val_size
    cfg.data.test_size = dataset_spec.test_size
    cfg.data.noise_std = dataset_spec.noise_std
    cfg.data.data_root = dataset_spec.data_root

    cfg.training = _Cfg()
    cfg.training.batch_size = dataset_spec.batch_size_override or config.training.batch_size

    cfg.execution = _Cfg()
    cfg.execution.seed = seed

    return build_dataloaders(cfg, dataset_name=dataset_spec.name)


# ---------------------------------------------------------------------------
# Per-Dataset × Per-Seed Experiment Pipeline
# ---------------------------------------------------------------------------

def _run_single(
    config: Z3ExperimentConfig,
    dataset_spec: Z3DatasetSpec,
    backbone: BackboneCondition,
    seed: int,
    output_base: Path,
    device: str,
) -> dict | None:
    """Run a single backbone × dataset × seed combination.

    Returns {"evaluation": ..., "train_state": ..., "rollout": dict or None}
    or None on failure.
    """
    # Derive configs: dataset override → backbone override → seed override
    ds_config = config.for_dataset(dataset_spec)
    bk_config = ds_config.for_backbone(backbone)
    seed_config = bk_config.for_seed(seed)

    cond_dir = output_base / f"dataset_{dataset_spec.name}" / f"backbone_{backbone.value}" / f"seed_{seed}"
    cond_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_loader, val_loader, test_loader, bundle = _build_dataloaders(
            seed_config, dataset_spec, seed,
        )

        train_state = train_backbone(
            seed_config, train_loader, val_loader, cond_dir, bundle=bundle, device=device,
        )

        evaluation = evaluate_backbone(
            train_state.model, test_loader, device=device,
        )

        # Rollout evaluation (robustness under input corruption)
        rollout_agg = None
        try:
            rollout_results = rollout_evaluate_dataset(
                train_state.model, test_loader,
                n_steps=seed_config.rollout.n_steps,
                noise_scale=seed_config.rollout.noise_scale,
                max_samples=seed_config.rollout.max_samples,
                device=device,
            )
            rollout_agg = aggregate_rollout_results(rollout_results)
            log.info(
                "  %s × %s × seed=%d: rollout_auc=%.4f, degradation_slope=%.4f",
                dataset_spec.name, backbone.value, seed,
                rollout_agg["mean_auc"], rollout_agg["mean_slope"],
            )
        except Exception as e:
            log.warning("  %s × %s × seed=%d: rollout FAILED: %s", dataset_spec.name, backbone.value, seed, e)

        log.info(
            "  %s × %s × seed=%d: accuracy=%.4f, f1=%.4f",
            dataset_spec.name, backbone.value, seed,
            evaluation.accuracy, evaluation.f1_macro,
        )

        return {"evaluation": evaluation, "train_state": train_state, "rollout": rollout_agg}

    except Exception as e:
        log.error("  %s × %s × seed=%d FAILED: %s", dataset_spec.name, backbone.value, seed, e, exc_info=True)
        return None


def _run_dataset_experiment(
    dataset_spec: Z3DatasetSpec,
    config: Z3ExperimentConfig,
    backbones: List[BackboneCondition],
    output_base: Path,
    device: str = "cpu",
) -> Dict[str, dict]:
    """Run the full experiment for a single dataset × all backbones × all seeds.

    Per-dataset num_seeds overrides global stats.num_seeds if specified.

    Returns dict mapping backbone_name -> {
        "evaluations": [BackboneEvaluation, ...],  # per seed
        "train_states": [TrainState, ...],          # per seed
        "rollouts": [dict, ...],                    # per seed (aggregate_rollout_results)
        "mean_accuracy": float,
        "std_accuracy": float,
    }
    """
    # Use per-dataset num_seeds if specified, otherwise fall back to global stats
    num_seeds = dataset_spec.num_seeds if dataset_spec.num_seeds is not None else config.stats.num_seeds
    seeds = [config.seed + i for i in range(num_seeds)]

    results: Dict[str, dict] = {}

    for backbone in backbones:
        backbone_name = backbone.value
        log.info("═══ %s × %s ═══", dataset_spec.name.upper(), backbone.label)

        seed_evals: List[BackboneEvaluation] = []
        seed_states: List[TrainState] = []
        seed_rollouts: List[dict] = []

        for seed in seeds:
            result = _run_single(config, dataset_spec, backbone, seed, output_base, device)
            if result is not None:
                seed_evals.append(result["evaluation"])
                seed_states.append(result["train_state"])
                if result.get("rollout") is not None:
                    seed_rollouts.append(result["rollout"])

        if seed_evals:
            accs = np.array([e.accuracy for e in seed_evals])
            results[backbone_name] = {
                "evaluations": seed_evals,
                "train_states": seed_states,
                "rollouts": seed_rollouts,
                "mean_accuracy": float(accs.mean()),
                "std_accuracy": float(accs.std()),
            }

            log.info(
                "  %s × %s: mean_acc=%.4f ± %.4f (%d seeds)",
                dataset_spec.name, backbone_name,
                results[backbone_name]["mean_accuracy"],
                results[backbone_name]["std_accuracy"],
                len(seed_evals),
            )

    return results


# ---------------------------------------------------------------------------
# Cross-Backbone Reporting
# ---------------------------------------------------------------------------

def _generate_cross_backbone_report(
    all_results: Dict[str, Dict[str, dict]],
    config: Z3ExperimentConfig,
    output_base: Path,
) -> None:
    """Generate cross-backbone comparison tables and plots."""
    report_dir = output_base / "cross_backbone"
    report_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, ds_results in all_results.items():
        # Aggregate evaluations (use first seed for per-class metrics)
        # Defensive: skip backbones with empty evaluation lists (Issue 5)
        evaluations = {
            k: v["evaluations"][0]
            for k, v in ds_results.items()
            if v.get("evaluations")
        }
        aggregate_stats = {
            k: {
                "std_accuracy": v["std_accuracy"],
                "std_f1_macro": float(np.std([e.f1_macro for e in v["evaluations"]])),
                "std_loss": float(np.std([e.mean_loss for e in v["evaluations"]])),
                "per_seed_accuracy": [float(e.accuracy) for e in v["evaluations"]],
                "per_seed_f1_macro": [float(e.f1_macro) for e in v["evaluations"]],
                "per_seed_mean_loss": [float(e.mean_loss) for e in v["evaluations"]],
            }
            for k, v in ds_results.items()
            if v.get("evaluations")
        }
        for k, v in ds_results.items():
            if not v.get("evaluations"):
                continue
            evals = v["evaluations"]
            representative = evaluations[k]
            representative.accuracy = float(np.mean([e.accuracy for e in evals]))
            representative.f1_macro = float(np.mean([e.f1_macro for e in evals]))
            representative.mean_loss = float(np.mean([e.mean_loss for e in evals]))

        train_losses = {
            k: v["train_states"][0].train_losses
            for k, v in ds_results.items()
            if v.get("train_states")
        }
        val_losses = {
            k: v["train_states"][0].val_losses
            for k, v in ds_results.items()
            if v.get("train_states")
        }

        # Multi-seed statistical comparisons (Issue 2: flexible, not Deep Hodge only)
        # Strategy: pick the best-performing backbone as reference, then compare
        # each other backbone against it. If only one backbone, no comparisons.
        comparisons: List[ComparisonResult] = []
        backbones_with_seeds = {
            k: v for k, v in ds_results.items()
            if v.get("evaluations") and len(v["evaluations"]) >= 1
        }

        if len(backbones_with_seeds) >= 2:
            # Determine reference backbone: prefer "deep_hodge" (proposed method)
            # if present; otherwise use the backbone with highest mean accuracy.
            if "deep_hodge" in backbones_with_seeds:
                reference_name = "deep_hodge"
            else:
                reference_name = max(
                    backbones_with_seeds,
                    key=lambda k: backbones_with_seeds[k]["mean_accuracy"],
                )
            ref_accs = np.array([
                e.accuracy for e in backbones_with_seeds[reference_name]["evaluations"]
            ])

            for bk_name, bk_data in backbones_with_seeds.items():
                if bk_name == reference_name:
                    continue
                bk_accs = np.array([e.accuracy for e in bk_data["evaluations"]])
                comp = compare_conditions(
                    bk_accs,
                    ref_accs,
                    metric_name="accuracy",
                    condition_a=bk_name,
                    condition_b=reference_name,
                    alpha=config.stats.significance_level,
                    n_bootstrap=config.stats.num_bootstrap_samples,
                )
                comparisons.append(comp)

        # Reports
        generate_json_report(
            evaluations,
            comparisons,
            report_dir / f"{ds_name}_results.json",
            aggregate_stats=aggregate_stats,
        )
        generate_markdown_report(
            evaluations,
            comparisons,
            report_dir / f"{ds_name}_summary.md",
            aggregate_stats=aggregate_stats,
        )

        # Plots — use mean ± std from multi-seed
        accuracies = {k: v["mean_accuracy"] for k, v in ds_results.items()}
        stds = {k: v["std_accuracy"] for k, v in ds_results.items()}
        plot_accuracy_comparison(accuracies, stds, output_path=report_dir / f"{ds_name}_accuracy.pdf")
        plot_learning_curves(train_losses, val_losses, output_path=report_dir / f"{ds_name}_learning.pdf")

        # Rollout degradation curves (if available)
        rollout_data = {k: v["rollouts"] for k, v in ds_results.items() if v.get("rollouts")}
        if rollout_data:
            _save_rollout_report(rollout_data, report_dir, ds_name)

    log.info("Cross-backbone reports saved to %s", report_dir)


def _save_rollout_report(
    rollout_data: Dict[str, list],
    report_dir: Path,
    ds_name: str,
) -> None:
    """Save rollout degradation curves as JSON and plot."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Aggregate across seeds: mean ± std of mean_accuracy_per_step
    summary = {}
    for bk_name, seed_rollouts in rollout_data.items():
        if not seed_rollouts:
            continue
        curves = [r["mean_accuracy_per_step"] for r in seed_rollouts if "mean_accuracy_per_step" in r]
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.zeros((len(curves), max_len))
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        summary[bk_name] = {
            "mean_curve": padded.mean(axis=0).tolist(),
            "std_curve": padded.std(axis=0).tolist(),
            "mean_auc": float(np.mean([r["mean_auc"] for r in seed_rollouts])),
            "mean_slope": float(np.mean([r["mean_slope"] for r in seed_rollouts])),
        }

    if not summary:
        return

    # JSON
    with open(report_dir / f"{ds_name}_rollout.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot degradation curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for bk_name, data in summary.items():
        curve = np.array(data["mean_curve"])
        std = np.array(data["std_curve"])
        steps = np.arange(len(curve))
        ax.plot(steps, curve, label=bk_name)
        ax.fill_between(steps, curve - std, curve + std, alpha=0.2)
    ax.set_xlabel("Corruption Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Rollout Robustness — {ds_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(report_dir / f"{ds_name}_rollout.pdf", bbox_inches="tight")
    plt.close(fig)

    log.info("Rollout report saved for %s", ds_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Z3 Baseline Study: Multi-backbone experiment"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--backbones", type=str, nargs="+", default=None,
        help="Override backbones to run (e.g., mlp tcn deep_hodge, or 'all')",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Override datasets to run (e.g., synthetic telecom)",
    )
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="Override number of seeds for multi-seed runs",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cpu, cuda, cuda:0). Auto-detected if not set.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    log.info("Loaded config: %s", args.config)

    # Override seeds if specified
    if args.seeds is not None:
        from .src.core.config import StatsParams
        config = Z3ExperimentConfig(
            backbone=config.backbone,
            seed=config.seed,
            model=config.model,
            training=config.training,
            loss=config.loss,
            stats=StatsParams(
                num_seeds=args.seeds,
                significance_level=config.stats.significance_level,
                confidence_level=config.stats.confidence_level,
                num_bootstrap_samples=config.stats.num_bootstrap_samples,
            ),
            rollout=config.rollout,
            datasets=config.datasets,
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(config.output_dir) / f"{timestamp}_{config.experiment_name}"
    output_base.mkdir(parents=True, exist_ok=True)
    config.save_yaml(output_base / "config_snapshot.yaml")

    # Determine backbones
    if args.backbones:
        if "all" in args.backbones:
            backbones = [BackboneCondition(b) for b in ALL_BACKBONES]
        else:
            valid_names = {b.value for b in BackboneCondition}
            invalid = [b for b in args.backbones if b not in valid_names]
            if invalid:
                raise ValueError(
                    f"Invalid backbone name(s): {invalid}. "
                    f"Valid options: {sorted(valid_names)}"
                )
            backbones = [BackboneCondition(b) for b in args.backbones]
    else:
        backbones = [config.backbone]

    # Determine datasets
    if args.datasets:
        ds_specs = [ds for ds in config.datasets if ds.name in args.datasets]
        if not ds_specs:
            log.error("None of the requested datasets found in config: %s", args.datasets)
            sys.exit(1)
    else:
        ds_specs = list(config.datasets)

    log.info("═══════════════════════════════════════════════════════════")
    log.info(
        "Z3 Baseline Study: %d datasets × %d backbones × %d seeds",
        len(ds_specs), len(backbones), config.stats.num_seeds,
    )
    log.info("  Datasets: %s", [ds.name for ds in ds_specs])
    log.info("  Backbones: %s", [b.value for b in backbones])
    log.info("  Seeds: %d (base=%d)", config.stats.num_seeds, config.seed)
    log.info("  Device: %s", device)
    log.info("═══════════════════════════════════════════════════════════")

    # Run experiment for each dataset
    all_results: Dict[str, Dict[str, dict]] = {}
    total_start = time.time()

    for ds_spec in ds_specs:
        log.info("\n▓▓▓ DATASET: %s ▓▓▓\n", ds_spec.name.upper())
        ds_start = time.time()

        results = _run_dataset_experiment(
            ds_spec, config, backbones, output_base, device=device,
        )

        ds_elapsed = time.time() - ds_start
        log.info("Dataset '%s' complete in %.1fs (%d backbones)", ds_spec.name, ds_elapsed, len(results))

        if results:
            all_results[ds_spec.name] = results

    total_elapsed = time.time() - total_start

    # Generate cross-backbone reports
    if all_results:
        _generate_cross_backbone_report(all_results, config, output_base)

    log.info("═══════════════════════════════════════════════════════════")
    log.info("EXPERIMENT COMPLETE: %d datasets, %.1fs total", len(all_results), total_elapsed)
    log.info("Output directory: %s", output_base)
    log.info("═══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
