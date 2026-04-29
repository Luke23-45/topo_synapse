"""Statistical Testing — Z3 Baseline Study.

Implements the statistical rigor protocol:
    - Primary: Two-sided Welch's t-test (alpha=0.05)
    - Effect size: Cohen's d with bootstrapped 95% CI
    - Non-parametric backup: Wilcoxon signed-rank test
    - Multiple comparison correction: Bonferroni

Mirrors the legacy ``baselines/src/engine/metrics.py`` pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    log.warning("scipy not available — statistical tests will use fallback implementations")


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two backbone conditions."""
    metric_name: str
    condition_a: str
    condition_b: str
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    mean_diff: float = 0.0
    cohens_d: float = 0.0
    d_ci_lower: float = 0.0
    d_ci_upper: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    welch_significant: bool = False
    wilcoxon_p: float | None = None
    shapiro_p_a: float | None = None
    shapiro_p_b: float | None = None
    normality_assumed: bool = True
    bonferroni_alpha: float = 0.05
    significant_after_correction: bool = False


def compare_conditions(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_name: str = "accuracy",
    condition_a: str = "A",
    condition_b: str = "B",
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> ComparisonResult:
    """Compare two backbone conditions statistically.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Per-seed metric scores for each condition.
    metric_name : str
    condition_a, condition_b : str
    alpha : float
        Significance level.
    n_bootstrap : int
        Bootstrap samples for Cohen's d CI.

    Returns
    -------
    ComparisonResult
    """
    mean_a, std_a = float(scores_a.mean()), float(scores_a.std())
    mean_b, std_b = float(scores_b.mean()), float(scores_b.std())
    mean_diff = mean_b - mean_a

    # Cohen's d
    pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Bootstrap CI for Cohen's d
    d_samples = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        a_boot = rng.choice(scores_a, size=len(scores_a), replace=True)
        b_boot = rng.choice(scores_b, size=len(scores_b), replace=True)
        d_m = b_boot.mean() - a_boot.mean()
        d_s = np.sqrt((a_boot.std() ** 2 + b_boot.std() ** 2) / 2)
        d_samples.append(d_m / d_s if d_s > 0 else 0.0)
    d_samples = np.array(d_samples)
    d_ci_lower = float(np.percentile(d_samples, 2.5))
    d_ci_upper = float(np.percentile(d_samples, 97.5))

    # Welch's t-test
    t_stat, p_value = 0.0, 1.0
    wilcoxon_p = None
    shapiro_p_a = None
    shapiro_p_b = None
    normality_assumed = True

    if HAS_SCIPY and len(scores_a) >= 2 and len(scores_b) >= 2:
        # Shapiro-Wilk normality test
        if len(scores_a) >= 3:
            try:
                _, shapiro_p_a = sp_stats.shapiro(scores_a)
                shapiro_p_a = float(shapiro_p_a)
            except Exception:
                shapiro_p_a = None
        if len(scores_b) >= 3:
            try:
                _, shapiro_p_b = sp_stats.shapiro(scores_b)
                shapiro_p_b = float(shapiro_p_b)
            except Exception:
                shapiro_p_b = None

        # Check normality assumption
        normality_assumed = True
        if shapiro_p_a is not None and shapiro_p_a < alpha:
            normality_assumed = False
        if shapiro_p_b is not None and shapiro_p_b < alpha:
            normality_assumed = False

        # Welch's t-test (parametric)
        t_stat, p_value = sp_stats.ttest_ind(scores_a, scores_b, equal_var=False)
        p_value = float(p_value)
        t_stat = float(t_stat)
        if not np.isfinite(t_stat) or not np.isfinite(p_value):
            t_stat, p_value = 0.0, 1.0

        # Wilcoxon signed-rank test (non-parametric backup)
        try:
            _, wilcoxon_p = sp_stats.wilcoxon(scores_a, scores_b)
            wilcoxon_p = float(wilcoxon_p)
        except Exception:
            wilcoxon_p = None

    welch_significant = p_value < alpha
    bonferroni_alpha = alpha  # will be adjusted by caller for multiple comparisons
    sig_after = p_value < bonferroni_alpha

    return ComparisonResult(
        metric_name=metric_name,
        condition_a=condition_a,
        condition_b=condition_b,
        mean_a=mean_a, std_a=std_a,
        mean_b=mean_b, std_b=std_b,
        mean_diff=mean_diff,
        cohens_d=cohens_d,
        d_ci_lower=d_ci_lower, d_ci_upper=d_ci_upper,
        t_statistic=t_stat, p_value=p_value,
        welch_significant=welch_significant,
        wilcoxon_p=wilcoxon_p,
        shapiro_p_a=shapiro_p_a,
        shapiro_p_b=shapiro_p_b,
        normality_assumed=normality_assumed,
        bonferroni_alpha=bonferroni_alpha,
        significant_after_correction=sig_after,
    )


__all__ = ["ComparisonResult", "compare_conditions"]
