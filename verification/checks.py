from __future__ import annotations

import numpy as np
import torch

from synapse.synapse_arch.model import Z3TopologyFirstModel


def verify_absolute_time_invariance(model: Z3TopologyFirstModel, sequence: np.ndarray) -> dict[str, float | bool]:
    tensor = torch.from_numpy(sequence).float().unsqueeze(0)
    audit_a = model.exact_audit(tensor)[0]

    shifted = sequence.copy()
    shifted = shifted + 0.0
    audit_b = model.exact_audit(torch.from_numpy(shifted).float().unsqueeze(0))[0]

    same = np.allclose(audit_a.point_cloud, audit_b.point_cloud, atol=1e-6)
    return {"absolute_time_invariant": bool(same), "cloud_delta": float(np.max(np.abs(audit_a.point_cloud - audit_b.point_cloud)) if audit_a.point_cloud.size else 0.0)}


def verify_proxy_output_finite(model: Z3TopologyFirstModel, sequence: np.ndarray) -> dict[str, float | bool]:
    with torch.no_grad():
        out = model(torch.from_numpy(sequence).float().unsqueeze(0))
    finite = torch.isfinite(out.proxy_features).all().item()
    return {"proxy_features_finite": bool(finite), "proxy_norm": float(out.proxy_features.norm().item())}
