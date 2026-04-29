"""JSON report generation for Z3 SYNAPSE evaluation.

Produces a machine-readable JSON report containing:
    - Per-condition metric summaries
    - Statistical comparison results
    - Topology alignment metrics
    - Robustness sweep data
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from synapse.evaluation.runners.base import EvalResult
from synapse.evaluation.metrics.statistical import ComparisonResult

log = logging.getLogger(__name__)


def generate_json_report(
    evaluations: Dict[str, EvalResult],
    comparisons: List[ComparisonResult] | None = None,
    output_path: str | Path = "evaluation_report.json",
) -> Path:
    """Generate a JSON report from evaluation results.

    Parameters
    ----------
    evaluations : dict mapping condition_name → EvalResult
    comparisons : list of ComparisonResult, optional
    output_path : str or Path

    Returns
    -------
    Path to saved JSON file.
    """
    output_path = Path(output_path)
    report: Dict[str, Any] = {
        "condition_summaries": {},
        "topology_analysis": {},
        "robustness_analysis": {},
        "statistical_comparisons": [],
    }

    # Per-condition summaries
    for cond_key, eval_result in evaluations.items():
        report["condition_summaries"][cond_key] = eval_result.to_dict()

    # Topology analysis
    for cond_key, eval_result in evaluations.items():
        if eval_result.topology_metrics:
            report["topology_analysis"][cond_key] = eval_result.topology_metrics

    # Robustness analysis
    for cond_key, eval_result in evaluations.items():
        if eval_result.robustness_metrics:
            report["robustness_analysis"][cond_key] = eval_result.robustness_metrics

    # Statistical comparisons
    if comparisons:
        for comp in comparisons:
            report["statistical_comparisons"].append({
                "metric": comp.metric_name,
                "condition_a": comp.condition_a,
                "condition_b": comp.condition_b,
                "mean_a": comp.mean_a,
                "std_a": comp.std_a,
                "mean_b": comp.mean_b,
                "std_b": comp.std_b,
                "mean_diff": comp.mean_diff,
                "cohens_d": comp.cohens_d,
                "d_ci": [comp.d_ci_lower, comp.d_ci_upper],
                "t_statistic": comp.t_statistic,
                "p_value": comp.p_value,
                "welch_significant": comp.welch_significant,
                "significant_after_bonferroni": comp.significant_after_correction,
                "is_normal": comp.is_normal,
                "shapiro_p": comp.shapiro_p,
                "wilcoxon_p": comp.wilcoxon_p,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Saved JSON report to %s", output_path)
    return output_path
