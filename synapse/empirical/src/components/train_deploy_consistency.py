from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import torch

from synapse.empirical.common.shared import ArtifactWriter, build_empirical_model_from_config, exact_summary_matrix


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    p = config["params"]
    n_samples = p["n_samples"]

    t0 = perf_counter()
    model, bundle = build_empirical_model_from_config(config)

    batch = torch.from_numpy(bundle.test_sequences[:n_samples]).float()
    proxy = model.compute_proxy(batch).proxy_features.detach().cpu().numpy()
    exact = exact_summary_matrix(model, bundle.test_sequences[:n_samples])

    aligned = proxy.shape[0] == exact.shape[0]
    proxy_finite = bool(np.isfinite(proxy).all())
    exact_finite = bool(np.isfinite(exact).all())
    passed = aligned and proxy_finite and exact_finite

    row = {
        "n_samples": int(proxy.shape[0]),
        "proxy_finite": proxy_finite,
        "exact_finite": exact_finite,
        "sample_aligned": aligned,
        "consistency_ok": passed,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("proxy_features.npy", proxy)
    writer.save_numpy("exact_features.npy", exact)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "consistency_ok": passed,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-13",
            run_experiment_fn=run_experiment,
        )
    )
