"""Markdown report generation for Z3 SYNAPSE evaluation.

Produces a human-readable Markdown report suitable for publications
and review, containing:
    - Per-condition metric tables
    - Statistical comparison summaries
    - Topology alignment analysis
    - Robustness sweep results
    - Conclusion section
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from synapse.evaluation.runners.base import EvalResult
from synapse.evaluation.metrics.statistical import ComparisonResult

log = logging.getLogger(__name__)


def generate_markdown_report(
    evaluations: Dict[str, EvalResult],
    comparisons: List[ComparisonResult] | None = None,
    output_path: str | Path = "evaluation_report.md",
) -> Path:
    """Generate a Markdown report from evaluation results.

    Parameters
    ----------
    evaluations : dict mapping condition_name → EvalResult
    comparisons : list of ComparisonResult, optional
    output_path : str or Path

    Returns
    -------
    Path to saved Markdown file.
    """
    output_path = Path(output_path)
    lines: list[str] = []

    lines.append("# SYNAPSE Z3 Evaluation Report")
    lines.append("")

    # Configuration summary
    lines.append("## Configuration")
    lines.append("")
    for cond_key, eval_result in evaluations.items():
        lines.append(f"- **{cond_key}**: dataset={eval_result.dataset_name}, "
                     f"modality={eval_result.modality}, task={eval_result.task}")
    lines.append("")

    # Primary metrics table
    lines.append("## Primary Metrics")
    lines.append("")

    # Build table header from union of all metric keys
    all_metric_keys: list[str] = []
    for eval_result in evaluations.values():
        for k in eval_result.primary_metrics:
            if k not in all_metric_keys:
                all_metric_keys.append(k)

    header = "| Condition | " + " | ".join(all_metric_keys) + " |"
    sep = "|-----------|" + "|".join(["--------"] * len(all_metric_keys)) + "|"
    lines.extend([header, sep])

    for cond_key, eval_result in evaluations.items():
        row = f"| **{cond_key}** |"
        for k in all_metric_keys:
            val = eval_result.primary_metrics.get(k, "—")
            if isinstance(val, float):
                row += f" {val:.4f} |"
            else:
                row += f" {val} |"
        lines.append(row)
    lines.append("")

    # Topology alignment
    lines.append("## Topology Alignment")
    lines.append("")
    for cond_key, eval_result in evaluations.items():
        if eval_result.topology_metrics:
            lines.append(f"**{cond_key}**:")
            for metric_name, value in eval_result.topology_metrics.items():
                if isinstance(value, float):
                    lines.append(f"- {metric_name}: {value:.4f}")
                else:
                    lines.append(f"- {metric_name}: {value}")
            lines.append("")

    # Robustness analysis
    lines.append("## Robustness Analysis")
    lines.append("")
    for cond_key, eval_result in evaluations.items():
        if eval_result.robustness_metrics:
            lines.append(f"### {cond_key}")
            lines.append("")
            _write_robustness_section(eval_result.robustness_metrics, lines)
            lines.append("")

    # Statistical comparisons
    if comparisons:
        lines.append("## Statistical Comparisons")
        lines.append("")
        for comp in comparisons:
            sig = "✅ Significant" if comp.welch_significant else "❌ Not significant"
            lines.append(f"**{comp.condition_a} vs {comp.condition_b}** ({comp.metric_name}):")
            lines.append(f"- Welch's t: t={comp.t_statistic:.4f}, p={comp.p_value:.4f}")
            lines.append(f"- Cohen's d: {comp.cohens_d:.3f} "
                        f"(95% CI: [{comp.d_ci_lower:.3f}, {comp.d_ci_upper:.3f}])")
            lines.append(f"- Normality (Shapiro-Wilk): p={comp.shapiro_p:.4f} "
                        f"({'normal' if comp.is_normal else 'non-normal'})")
            if comp.wilcoxon_p is not None:
                lines.append(f"- Wilcoxon signed-rank: p={comp.wilcoxon_p:.4f}")
            lines.append(f"- Bonferroni-corrected α: {comp.bonferroni_alpha:.4f}")
            lines.append(f"- Significant after correction: {comp.significant_after_correction}")
            lines.append(f"- **{sig}**")
            lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    if len(evaluations) >= 2:
        keys = list(evaluations.keys())
        acc_a = evaluations[keys[0]].primary_metrics.get("accuracy", 0.0)
        acc_b = evaluations[keys[1]].primary_metrics.get("accuracy", 0.0)
        if isinstance(acc_a, (int, float)) and isinstance(acc_b, (int, float)):
            if acc_b > acc_a:
                lines.append(f"{keys[1]} outperforms {keys[0]} "
                            f"(accuracy: {acc_b:.4f} vs {acc_a:.4f}).")
            elif acc_a > acc_b:
                lines.append(f"{keys[0]} outperforms {keys[1]} "
                            f"(accuracy: {acc_a:.4f} vs {acc_b:.4f}).")
            else:
                lines.append(f"No accuracy difference between {keys[0]} and {keys[1]}.")
        else:
            lines.append("Insufficient data for conclusion.")
    else:
        lines.append("Single condition evaluated — no comparison available.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Saved Markdown report to %s", output_path)
    return output_path


def _write_robustness_section(
    robustness: Dict[str, Any],
    lines: list[str],
) -> None:
    """Write robustness sweep data to markdown lines."""
    for sweep_name, sweep_data in robustness.items():
        if isinstance(sweep_data, list) and len(sweep_data) > 0 and isinstance(sweep_data[0], dict):
            lines.append(f"**{sweep_name}**:")
            for entry in sweep_data:
                parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in entry.items()]
                lines.append(f"- {', '.join(parts)}")
            lines.append("")
        elif isinstance(sweep_data, dict):
            lines.append(f"**{sweep_name}**:")
            for k, v in sweep_data.items():
                if isinstance(v, (int, float)):
                    lines.append(f"- {k}: {v:.4f}")
                elif isinstance(v, list):
                    lines.append(f"- {k}: [{', '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in v)}]")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")
