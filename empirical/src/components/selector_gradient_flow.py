from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from synapse.empirical.common.shared import ArtifactWriter, build_empirical_model_from_config


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    p = config["params"]
    n_batch = p["n_batch"]

    t0 = perf_counter()
    model, bundle = build_empirical_model_from_config(config)
    model.train()

    batch = torch.from_numpy(bundle.train_sequences[:n_batch]).float()
    targets = torch.from_numpy(bundle.train_labels[:n_batch]).long()
    out = model(batch)
    loss = torch.nn.CrossEntropyLoss()(out.logits, targets)
    loss.backward()

    lift_grad = model.lift.W_theta.grad
    seq_grad = model.sequence_proj.weight.grad

    lift_finite = lift_grad is not None and bool(torch.isfinite(lift_grad).all())
    seq_finite = seq_grad is not None and bool(torch.isfinite(seq_grad).all())
    passed = lift_finite and seq_finite

    lift_norm = float(lift_grad.norm().item()) if lift_grad is not None else 0.0
    seq_norm = float(seq_grad.norm().item()) if seq_grad is not None else 0.0

    row = {
        "lift_grad_norm": lift_norm,
        "seq_grad_norm": seq_norm,
        "lift_finite": lift_finite,
        "seq_finite": seq_finite,
        "backprop_ok": passed,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    if lift_grad is not None:
        writer.save_numpy("lift_grad.npy", lift_grad.detach().cpu().numpy())
    if seq_grad is not None:
        writer.save_numpy("seq_grad.npy", seq_grad.detach().cpu().numpy())

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "backprop_ok": passed,
        "lift_grad_norm": lift_norm,
        "seq_grad_norm": seq_norm,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-03",
            run_experiment_fn=run_experiment,
        )
    )
