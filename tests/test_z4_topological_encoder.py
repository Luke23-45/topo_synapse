from __future__ import annotations

import torch

from synapse.common.encoders.history_aware_router import HistoryAwareAnchorRouter
from synapse.common.encoders.legacy import LegacyZ3TopologicalEncoder
from synapse.common.encoders.topological_encoder import TopologicalEncoder
from synapse.common.encoders.z4_topological_encoder import Z4TopologicalEncoder
from synapse.synapse_arch.deep_hodge import DeepHodgeLayer
from synapse.synapse_arch.model import TopologyFirstModel
from synapse.synapse_arch.unified import DeepHodgeStem
from synapse.synapse_arch.unified import UnifiedModel
from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_core.topology_features import structural_feature_dim


def test_router_is_permutation_equivariant() -> None:
    torch.manual_seed(0)
    router = HistoryAwareAnchorRouter(
        input_dim=3,
        d_u=12,
        d_a=8,
        d_m=10,
        K=2,
        r=2,
        L=2,
    )
    router.train()

    x = torch.randn(2, 5, 3)
    feedback = torch.zeros(2, 1)
    permutation = torch.tensor([2, 4, 0, 1, 3])
    inverse = torch.argsort(permutation)

    all_y, _, _, _, _ = router(x, feedback=feedback, hard=False)
    permuted_y, _, _, _, _ = router(x[:, permutation], feedback=feedback, hard=False)

    assert torch.allclose(all_y, permuted_y[:, :, inverse], atol=1e-4)


def test_router_initial_memory_depends_on_input_context() -> None:
    torch.manual_seed(7)
    router = HistoryAwareAnchorRouter(
        input_dim=2,
        d_u=8,
        d_a=4,
        d_m=6,
        K=2,
        r=1,
        L=1,
    )
    router.train()

    x = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[10.0, 0.0], [11.0, 0.0], [10.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    _, _, all_memory, _, _ = router(x, feedback=torch.zeros(2, 1), hard=False)

    initial_memory = all_memory[:, 0]
    assert torch.isfinite(initial_memory).all()
    assert not torch.allclose(initial_memory, torch.zeros_like(initial_memory))
    assert not torch.allclose(initial_memory[0], initial_memory[1])


def test_z4_encoder_outputs_finite_tokens() -> None:
    torch.manual_seed(1)
    encoder = Z4TopologicalEncoder(
        input_dim=4,
        d_model=16,
        d_u=12,
        d_a=8,
        d_m=10,
        k=6,
        K=3,
        r=2,
        L=2,
    )
    encoder.set_normalization(
        torch.zeros(structural_feature_dim(4, include_selection=False)),
        torch.ones(structural_feature_dim(4, include_selection=False)),
    )
    encoder.train()

    x = torch.randn(3, 7, 4)
    tokens, y_star, all_y, all_memory = encoder(x, feedback=torch.zeros(3, 1))

    assert tokens.shape == (3, 2, 16)
    assert y_star.shape == (3, 7)
    assert all_y.shape == (3, 2, 7)
    assert all_memory.shape == (3, 3, 10)
    assert torch.isfinite(tokens).all()


def test_legacy_topological_encoder_uses_structure_aware_lift() -> None:
    torch.manual_seed(2)
    encoder = LegacyZ3TopologicalEncoder(
        input_dim=3,
        d_model=8,
        hidden_dim=12,
        k=5,
        K=2,
        r=2,
        max_proxy_points=4,
    )
    encoder.set_normalization(
        torch.zeros(structural_feature_dim(3, include_selection=False)),
        torch.ones(structural_feature_dim(3, include_selection=False)),
    )

    tokens, y_star = encoder(torch.randn(2, 6, 3))
    assert tokens.shape == (2, 4, 8)
    assert y_star.shape == (2, 6)
    assert torch.isfinite(tokens).all()


def test_active_topological_encoder_matches_z4_shape() -> None:
    encoder = TopologicalEncoder(
        input_dim=3,
        d_model=8,
        hidden_dim=12,
        k=5,
        K=2,
        r=2,
        L=3,
    )
    encoder.set_normalization(
        torch.zeros(structural_feature_dim(3, include_selection=False)),
        torch.ones(structural_feature_dim(3, include_selection=False)),
    )
    tokens, y_star, all_y, all_memory = encoder(torch.randn(2, 6, 3))
    assert tokens.shape == (2, 3, 8)
    assert y_star.shape == (2, 6)
    assert all_y.shape == (2, 3, 6)


def test_unified_deep_hodge_uses_lightweight_proxy_stem() -> None:
    model = UnifiedModel(
        backbone_type="deep_hodge",
        input_dim=3,
        d_model=8,
        num_classes=2,
        num_tokens=6,
        k_dim=4,
        K=2,
        r=1,
        hidden_dim=8,
        max_proxy_points=4,
    )
    assert isinstance(model.encoder, DeepHodgeStem)


def _reference_deep_hodge_forward(
    layer: DeepHodgeLayer,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    B_batch, K_eff, d = x.shape
    pad_len = layer.max_points - K_eff
    if pad_len > 0:
        x_pad = torch.zeros(B_batch, pad_len, d, device=x.device, dtype=x.dtype)
        x = torch.cat([x, x_pad], dim=1)
        if mask is not None:
            m_pad = torch.zeros(B_batch, pad_len, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([mask, m_pad], dim=1)

    if mask is None:
        mask = torch.ones(B_batch, layer.max_points, device=x.device, dtype=x.dtype)
    else:
        mask = mask.to(device=x.device, dtype=x.dtype)

    residual = x
    x_norm = layer.norm1(x)
    P = layer.geom_proj(x_norm)
    mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
    D = torch.cdist(P, P) * mask_2d
    scales = torch.exp(layer.log_scales)

    V0 = layer.W_V0(x_norm).view(B_batch, layer.max_points, layer.num_scales, d)
    E_in_raw = torch.matmul(layer.abs_B1.t(), x_norm)
    E_in = layer.W_E_in(E_in_raw).view(B_batch, layer.abs_B1.shape[1], layer.num_scales, d)

    node_updates = []
    edge_updates = []
    for s_idx in range(layer.num_scales):
        sigma = scales[s_idx]
        affinity = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
        W1 = affinity[:, layer.e_idx_i, layer.e_idx_j]
        W2 = W1[:, layer.t_idx_ij] * W1[:, layer.t_idx_jk] * W1[:, layer.t_idx_ik]

        B1_scaled = layer.B1.unsqueeze(0) * W1.unsqueeze(1)
        L0 = torch.bmm(B1_scaled, layer.B1.unsqueeze(0).expand(B_batch, -1, -1).transpose(1, 2))
        L0 = L0 + layer.tau * layer._eye_K.unsqueeze(0)
        node_updates.append(torch.bmm(L0, V0[:, :, s_idx, :]))

        term_down = torch.bmm(
            layer.B1.t().unsqueeze(0) * mask.unsqueeze(1),
            layer.B1.unsqueeze(0).expand(B_batch, -1, -1),
        )
        B2_scaled = layer.B2.unsqueeze(0) * W2.unsqueeze(1)
        term_up = torch.bmm(
            B2_scaled,
            layer.B2.t().unsqueeze(0).expand(B_batch, -1, -1),
        )
        L1 = term_down + term_up + layer.tau * layer._eye_E.unsqueeze(0)
        M1 = torch.bmm(L1, E_in[:, :, s_idx, :])
        edge_updates.append(
            torch.bmm(layer.abs_B1.unsqueeze(0).expand(B_batch, -1, -1), M1)
        )

    all_updates = torch.cat(node_updates + edge_updates, dim=-1)
    x = residual + layer.dropout(layer.out_proj(all_updates))
    x = x + layer.ffn(layer.norm2(x))
    if pad_len > 0:
        x = x[:, :K_eff, :]
    return x


def test_vectorized_deep_hodge_layer_matches_reference() -> None:
    torch.manual_seed(5)
    layer = DeepHodgeLayer(
        d_model=8,
        k_dim=4,
        num_scales=2,
        max_points=5,
        dropout=0.0,
    )
    layer.eval()

    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.float32)

    with torch.no_grad():
        expected = _reference_deep_hodge_forward(layer, x, mask)
        actual = layer(x, mask)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_deep_hodge_stem_emits_proxy_tokens_and_attention_mass() -> None:
    torch.manual_seed(11)
    stem = DeepHodgeStem(input_dim=3, d_model=8, num_proxy_tokens=4, max_len=6, dropout=0.0)

    sequence = torch.randn(2, 6, 3)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    tokens, y_star = stem(sequence, mask=mask)

    assert tokens.shape == (2, 4, 8)
    assert y_star.shape == (2, 6)
    assert torch.allclose(y_star.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(tokens).all()


def test_active_topological_encoder_aliases_z4() -> None:
    assert issubclass(TopologicalEncoder, Z4TopologicalEncoder)
    assert LegacyZ3TopologicalEncoder is not TopologicalEncoder


def test_topology_first_model_uses_z4_encoder() -> None:
    config = SynapseConfig(
        input_dim=3,
        output_dim=2,
        hidden_dim=8,
        d_model=8,
        num_layers=1,
        num_scales=2,
        max_proxy_points=3,
        topology_mode="deep_hodge",
        k=4,
    )
    model = TopologyFirstModel(config)
    assert isinstance(model.encoder, Z4TopologicalEncoder)


def test_topology_first_model_uses_encoder_lift_for_dense_cloud() -> None:
    torch.manual_seed(3)
    config = SynapseConfig(
        input_dim=3,
        output_dim=2,
        hidden_dim=8,
        d_model=8,
        num_layers=1,
        num_scales=2,
        max_proxy_points=3,
        topology_mode="baseline_proxy",
        k=4,
        r=2,
    )
    model = TopologyFirstModel(config)
    model.set_normalization(
        torch.zeros(structural_feature_dim(3, include_selection=False)),
        torch.ones(structural_feature_dim(3, include_selection=False)),
    )

    sequence = torch.randn(2, 5, 3)
    dense_vectors, dense_lifted_cloud = model._compute_dense_lifted_cloud(sequence)
    _, encoder_lifted_cloud = model.encoder.lift(dense_vectors)

    assert not hasattr(model, "lift")
    assert torch.allclose(dense_lifted_cloud, encoder_lifted_cloud)
