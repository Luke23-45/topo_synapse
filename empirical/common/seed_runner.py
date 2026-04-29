from __future__ import annotations

from statistics import mean, stdev


def run_multi_seed(seeds: list[int], fn):
    results = [fn(seed) for seed in seeds]
    return {
        "per_seed": results,
        "mean": mean(results),
        "std": stdev(results) if len(results) > 1 else 0.0,
    }
