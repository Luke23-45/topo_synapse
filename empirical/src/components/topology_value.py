from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from synapse.empirical.common.baselines import proxy_topology_features, uniform_feature
from synapse.empirical.common.math_utils import ridge_probe_accuracy
from synapse.empirical.common.shared import ArtifactWriter, build_empirical_model_from_config, exact_summary_matrix


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    p = config["params"]

    t0 = perf_counter()
    model, bundle = build_empirical_model_from_config(config)

    exact_x = exact_summary_matrix(model, bundle.train_sequences)
    exact_test = exact_summary_matrix(model, bundle.test_sequences)
    proxy_x = proxy_topology_features(model, bundle.train_sequences)
    proxy_test = proxy_topology_features(model, bundle.test_sequences)
    uniform_x = uniform_feature(bundle.train_sequences)
    uniform_test = uniform_feature(bundle.test_sequences)

    exact_acc = ridge_probe_accuracy(exact_x, bundle.train_labels, exact_test, bundle.test_labels)
    proxy_acc = ridge_probe_accuracy(proxy_x, bundle.train_labels, proxy_test, bundle.test_labels)
    uniform_acc = ridge_probe_accuracy(uniform_x, bundle.train_labels, uniform_test, bundle.test_labels)

    row = {
        "exact_acc": exact_acc,
        "proxy_acc": proxy_acc,
        "uniform_acc": uniform_acc,
        "exact_gt_uniform": exact_acc >= uniform_acc,
        "proxy_gt_uniform": proxy_acc >= uniform_acc,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("exact_features.npy", exact_x)
    writer.save_numpy("proxy_features.npy", proxy_x)
    writer.save_numpy("uniform_features.npy", uniform_x)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "exact_acc": exact_acc,
        "proxy_acc": proxy_acc,
        "uniform_acc": uniform_acc,
        "exact_gt_uniform": exact_acc >= uniform_acc,
        "proxy_gt_uniform": proxy_acc >= uniform_acc,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-01",
            run_experiment_fn=run_experiment,
        )
    )
