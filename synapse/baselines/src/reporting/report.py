"""Report Generation — Z3 Baseline Study.

Generates JSON and Markdown reports from evaluation results.
Mirrors the legacy ``baselines/src/reporting/report.py`` pattern.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..engine.evaluate import BackboneEvaluation
from ..engine.metrics import ComparisonResult

log = logging.getLogger(__name__)


def generate_json_report(
    evaluations: Dict[str, BackboneEvaluation],
    comparisons: List[ComparisonResult],
    output_path: Path,
) -> Path:
    """Generate a JSON report with all metrics and statistical tests."""
    report = {
        "evaluations": {},
        "comparisons": [],
    }

    for name, ev in evaluations.items():
        report["evaluations"][name] = {
            "backbone": ev.backbone,
            "accuracy": ev.accuracy,
            "f1_macro": ev.f1_macro,
            "mean_loss": ev.mean_loss,
            "per_class_accuracy": ev.per_class_accuracy.tolist(),
            "confusion_matrix": ev.confusion_matrix.tolist(),
        }

    for comp in comparisons:
        report["comparisons"].append({
            "metric": comp.metric_name,
            "condition_a": comp.condition_a,
            "condition_b": comp.condition_b,
            "mean_a": comp.mean_a,
            "std_a": comp.std_a,
            "mean_b": comp.mean_b,
            "std_b": comp.std_b,
            "cohens_d": comp.cohens_d,
            "d_ci": [comp.d_ci_lower, comp.d_ci_upper],
            "p_value": comp.p_value,
            "significant": comp.welch_significant,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("JSON report saved to %s", output_path)
    return output_path


def generate_markdown_report(
    evaluations: Dict[str, BackboneEvaluation],
    comparisons: List[ComparisonResult],
    output_path: Path,
) -> Path:
    """Generate a Markdown summary table."""
    lines = [
        "# Z3 Baseline Study — Results\n",
        "",
        "## Classification Accuracy (mean ± std over seeds)\n",
        "",
    ]

    # Header
    backbones = sorted(evaluations.keys())
    header = "| Metric | " + " | ".join(backbones) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(backbones)) + "|"
    lines.extend([header, sep])

    # Accuracy row
    row = "| Accuracy |"
    for name in backbones:
        ev = evaluations[name]
        row += f" {ev.accuracy:.4f} |"
    lines.append(row)

    # F1 row
    row = "| F1 Macro |"
    for name in backbones:
        ev = evaluations[name]
        row += f" {ev.f1_macro:.4f} |"
    lines.append(row)

    # Loss row
    row = "| Mean Loss |"
    for name in backbones:
        ev = evaluations[name]
        row += f" {ev.mean_loss:.4f} |"
    lines.append(row)

    # Statistical comparisons
    if comparisons:
        lines.append("")
        lines.append("## Statistical Comparisons (Welch's t-test)\n")
        for comp in comparisons:
            sig = "Significant" if comp.welch_significant else "Not significant"
            lines.append(
                f"- **{comp.condition_a} vs {comp.condition_b}**: "
                f"p={comp.p_value:.4f}, d={comp.cohens_d:.3f} ({sig})"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    log.info("Markdown report saved to %s", output_path)
    return output_path


__all__ = ["generate_json_report", "generate_markdown_report"]
