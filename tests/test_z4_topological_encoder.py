from __future__ import annotations

import torch

from synapse.common.encoders.history_aware_router import HistoryAwareAnchorRouter
from synapse.common.encoders.legacy import LegacyZ3TopologicalEncoder
from synapse.common.encoders.topological_encoder import TopologicalEncoder
from synapse.common.encoders.z4_topological_encoder import Z4TopologicalEncoder
from synapse.synapse_arch.model import TopologyFirstModel
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


def test_unified_deep_hodge_defaults_to_z4_encoder() -> None:
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
    assert isinstance(model.encoder, Z4TopologicalEncoder)


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
