from __future__ import annotations

import argparse

import torch

from synapse.common.runtime import load_config
from synapse.empirical.datasets.synthetic_topology import generate_topology_dataset
from synapse.synapse_arch.model import Z3TopologyFirstModel
from synapse.utils.io import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute exact topology audits.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="synapse_outputs/topology_audit.json")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = Z3TopologyFirstModel(config)
    model.load_state_dict(payload["model_state"])
    model.eval()

    sequences, _, _ = generate_topology_dataset(2, length=config.max_history_tokens, noise_std=config.noise_std, seed=config.seed + 20)
    audits = model.exact_audit(torch.from_numpy(sequences).float())
    save_json(
        args.output,
        {
            "audits": [
                {
                    "anchor_indices": audit.anchor_indices,
                    "topology_summary": audit.topology_summary.tolist(),
                    "point_cloud_shape": list(audit.point_cloud.shape),
                }
                for audit in audits
            ]
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
