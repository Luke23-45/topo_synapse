from __future__ import annotations

import torch

from synapse.common.encoders.z4_topological_encoder import Z4TopologicalEncoder
from synapse.synapse_core.topology_features import build_static_structural_features


def test_z4_encoder_matches_manual_router_lift_pipeline() -> None:
    torch.manual_seed(2)
    encoder = Z4TopologicalEncoder(
        input_dim=4,
        d_model=12,
        hidden_dim=16,
        d_u=12,
        d_a=8,
        d_m=12,
        k=5,
        K=3,
        r=1,
        L=4,
    )
    encoder.set_normalization(torch.zeros(15), torch.ones(15))

    x = torch.randn(2, 6, 4)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        tokens, y_star, all_y, all_memory = encoder(x, mask=mask)

        ref_all_y, _all_z, ref_all_memory, _all_anchors, geometry_cache = encoder.router(
            x,
            feedback=None,
            mask=mask,
            hard=False,
        )
        dense_vectors = build_static_structural_features(
            x,
            mask=mask,
            knn_k=max(1, encoder.r),
            geometry_cache=geometry_cache,
        )
        _, dense_lifted_cloud = encoder.lift(dense_vectors)
        stage_mass = ref_all_y.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        cloud = torch.einsum("blt,btk->blk", ref_all_y, dense_lifted_cloud) / stage_mass
        ref_tokens = encoder.topology_proj(cloud)
        ref_y_star = ref_all_y.mean(dim=1)

    torch.testing.assert_close(tokens, ref_tokens)
    torch.testing.assert_close(y_star, ref_y_star)
    torch.testing.assert_close(all_y, ref_all_y)
    torch.testing.assert_close(all_memory, ref_all_memory)


def test_z4_encoder_respects_masked_positions() -> None:
    torch.manual_seed(3)
    encoder = Z4TopologicalEncoder(
        input_dim=3,
        d_model=10,
        hidden_dim=12,
        d_u=10,
        d_a=6,
        d_m=10,
        k=4,
        K=2,
        r=1,
        L=3,
    )
    encoder.set_normalization(torch.zeros(12), torch.ones(12))

    x = torch.randn(2, 5, 3)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        tokens, y_star, all_y, all_memory = encoder(x, mask=mask)

    assert tokens.shape == (2, 3, 10)
    assert all_memory.shape == (2, 4, 10)
    torch.testing.assert_close(all_y * (1.0 - mask.unsqueeze(1)), torch.zeros_like(all_y))
    torch.testing.assert_close(y_star * (1.0 - mask), torch.zeros_like(y_star))
