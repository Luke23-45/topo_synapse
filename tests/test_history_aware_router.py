from __future__ import annotations

import torch

from synapse.common.encoders.history_aware_router import HistoryAwareAnchorRouter
from synapse.synapse_core.topology_features import (
    build_feature_similarity,
    build_router_context,
    build_static_structural_features,
    precompute_structural_geometry,
)


def _naive_router_forward(
    router: HistoryAwareAnchorRouter,
    x: torch.Tensor,
    *,
    feedback: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    hard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = x.shape
    geometry_cache = precompute_structural_geometry(x, mask=mask, knn_k=max(1, router.r))
    context = build_router_context(x, mask=mask, geometry_cache=geometry_cache)
    static_features = build_static_structural_features(
        x,
        mask=mask,
        knn_k=max(1, router.r),
        geometry_cache=geometry_cache,
    )
    similarity = build_feature_similarity(static_features, mask=mask)

    memory = router.memory_init(context).to(dtype=x.dtype)
    cumulative_y = torch.zeros(batch_size, seq_len, device=x.device, dtype=x.dtype)
    context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)
    if feedback is None:
        feedback = torch.zeros(batch_size, router.feedback_dim, device=x.device, dtype=x.dtype)

    all_y = []
    all_z = []
    all_memory = [memory]
    all_anchors = []

    for _ in range(router.L):
        stage_u, stage_features, stage_similarity, _ = router.candidate_encoder(
            x,
            selection_weights=cumulative_y,
            mask=mask,
            geometry_cache=geometry_cache,
            context=context,
            context_expanded=context_expanded,
            similarity=similarity,
            static_features=static_features,
        )
        scores, values = router.scorer(
            stage_u,
            stage_features,
            stage_similarity,
            memory,
            prev_selected=cumulative_y,
        )
        y = router.selector(scores, similarity=stage_similarity, mask=mask, hard=hard)
        z = torch.bmm(y.unsqueeze(1), values).squeeze(1)
        stats = router.statistics(y, stage_u)
        memory = router.gru(torch.cat([z, stats, feedback], dim=-1), memory)
        cumulative_y = cumulative_y + y

        all_y.append(y)
        all_z.append(z)
        all_memory.append(memory)
        all_anchors.append(stage_u * y.unsqueeze(-1))

    return (
        torch.stack(all_y, dim=1),
        torch.stack(all_z, dim=1),
        torch.stack(all_memory, dim=1),
        torch.stack(all_anchors, dim=1),
    )


def test_candidate_encoder_projection_buffer_matches_concat_path() -> None:
    torch.manual_seed(0)
    router = HistoryAwareAnchorRouter(input_dim=5, d_u=16, d_a=8, d_m=16, K=4, r=1, L=3)

    x = torch.randn(2, 7, 5)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]],
        dtype=torch.float32,
    )
    selection = torch.rand(2, 7)
    geometry_cache = precompute_structural_geometry(x, mask=mask, knn_k=1)
    static_features = build_static_structural_features(x, mask=mask, knn_k=1, geometry_cache=geometry_cache)
    context = build_router_context(x, mask=mask, geometry_cache=geometry_cache)
    context_expanded = context.unsqueeze(1).expand(-1, x.shape[1], -1)

    structural_features = torch.cat([static_features, selection.unsqueeze(-1)], dim=-1)
    expected = router.candidate_encoder.project(
        structural_features,
        context,
        context_expanded=context_expanded,
    )

    projection_buffer = router.candidate_encoder.build_projection_buffer(
        static_features,
        context,
        context_expanded=context_expanded,
    )
    projection_buffer = router.candidate_encoder.update_projection_buffer(
        projection_buffer,
        selection_weights=selection,
        static_feature_dim=static_features.shape[-1],
    )
    actual = router.candidate_encoder.project_with_buffer(projection_buffer)

    torch.testing.assert_close(actual, expected)


def test_history_aware_router_matches_naive_reference() -> None:
    torch.manual_seed(1)
    router = HistoryAwareAnchorRouter(input_dim=6, d_u=24, d_a=12, d_m=24, K=4, r=1, L=5)

    x = torch.randn(3, 9, 6)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    feedback = torch.randn(3, 1)

    with torch.no_grad():
        all_y, all_z, all_memory, all_anchors, _ = router(x, feedback=feedback, mask=mask)
        ref_y, ref_z, ref_memory, ref_anchors = _naive_router_forward(
            router,
            x,
            feedback=feedback,
            mask=mask,
        )

    torch.testing.assert_close(all_y, ref_y)
    torch.testing.assert_close(all_z, ref_z)
    torch.testing.assert_close(all_memory, ref_memory)
    torch.testing.assert_close(all_anchors, ref_anchors)


def test_history_aware_router_backward_does_not_hit_inplace_autograd_error() -> None:
    torch.manual_seed(4)
    router = HistoryAwareAnchorRouter(input_dim=5, d_u=16, d_a=8, d_m=16, K=4, r=1, L=4)

    x = torch.randn(2, 7, 5, requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0, 0]],
        dtype=torch.float32,
    )

    all_y, all_z, all_memory, all_anchors, _ = router(x, mask=mask)
    loss = all_y.sum() + all_z.sum() + all_memory.sum() + all_anchors.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
