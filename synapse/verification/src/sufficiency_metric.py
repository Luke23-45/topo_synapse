from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from synapse.verification.utils._shared import ArtifactWriter, build_model, make_config_from_yaml


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    params_cfg = config["params"]
    batch_size = params_cfg["batch_size"]
    input_dim = params_cfg["input_dim"]
    max_history_tokens = params_cfg["max_history_tokens"]

    t0 = perf_counter()

    model_cfg = make_config_from_yaml(config["experiment_id"])
    model = build_model(model_cfg)
    batch = torch.randn(batch_size, max_history_tokens, input_dim)
    out = model(batch)

    proxy_dim = int(out.proxy_features.shape[-1])
    logit_dim = int(out.logits.shape[-1])
    separated = proxy_dim != logit_dim

    row = {
        "proxy_dim": proxy_dim,
        "logit_dim": logit_dim,
        "readout_separated": separated,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("proxy_features.npy", out.proxy_features.detach().numpy())
    writer.save_numpy("logits.npy", out.logits.detach().numpy())

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "proxy_dim": proxy_dim,
        "logit_dim": logit_dim,
        "readout_separated": separated,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.verification.utils._shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="VZ3-08",
            run_experiment_fn=run_experiment,
        )
    )
