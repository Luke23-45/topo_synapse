from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synapse.common.encoders.history_aware_router import HistoryAwareAnchorRouter
from synapse.common.encoders.z4_topological_encoder import Z4TopologicalEncoder
from synapse.synapse_arch.deep_hodge import DeepHodgeTransformer
from synapse.synapse_core.topology_features import (
    build_feature_similarity,
    build_static_structural_features,
    precompute_structural_geometry,
)


def _bench(fn, *, warmup: int, iters: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / max(iters, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the Deep Hodge encoder pipeline on synthetic data.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    device = torch.device(args.device)
    x = torch.randn(args.batch_size, args.seq_len, args.input_dim, device=device)
    mask = torch.ones(args.batch_size, args.seq_len, device=device)

    router = HistoryAwareAnchorRouter(
        input_dim=args.input_dim,
        d_u=args.d_model,
        d_a=max(16, args.d_model // 2),
        d_m=args.d_model,
        K=args.K,
        r=args.r,
        L=args.L,
    ).to(device)
    encoder = Z4TopologicalEncoder(
        input_dim=args.input_dim,
        d_model=args.d_model,
        hidden_dim=args.d_model,
        d_u=args.d_model,
        d_a=max(16, args.d_model // 2),
        d_m=args.d_model,
        k=args.k,
        K=args.K,
        r=args.r,
        L=args.L,
    ).to(device)
    encoder.set_normalization(
        torch.zeros(3 * args.input_dim + 3, device=device),
        torch.ones(3 * args.input_dim + 3, device=device),
    )
    backbone = DeepHodgeTransformer(
        num_layers=2,
        d_model=args.d_model,
        k_dim=args.k,
        num_scales=3,
        max_points=args.L,
    ).to(device)

    geometry = precompute_structural_geometry(x, mask=mask, knn_k=max(1, args.r))
    static_features = build_static_structural_features(
        x,
        mask=mask,
        knn_k=max(1, args.r),
        geometry_cache=geometry,
    )

    timings = {
        "geometry_s": _bench(
            lambda: precompute_structural_geometry(x, mask=mask, knn_k=max(1, args.r)),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "static_features_s": _bench(
            lambda: build_static_structural_features(
                x,
                mask=mask,
                knn_k=max(1, args.r),
                geometry_cache=geometry,
            ),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "feature_similarity_s": _bench(
            lambda: build_feature_similarity(static_features, mask=mask),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "router_s": _bench(
            lambda: router(x, mask=mask),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "encoder_s": _bench(
            lambda: encoder(x, mask=mask),
            warmup=args.warmup,
            iters=args.iters,
        ),
    }

    tokens, *_ = encoder(x, mask=mask)
    timings["deep_hodge_backbone_s"] = _bench(
        lambda: backbone(tokens),
        warmup=args.warmup,
        iters=args.iters,
    )

    print(
        {
            "device": args.device,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "input_dim": args.input_dim,
            **{key: round(value, 6) for key, value in timings.items()},
        }
    )


if __name__ == "__main__":
    main()
