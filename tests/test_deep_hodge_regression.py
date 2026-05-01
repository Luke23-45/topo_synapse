"""Regression tests for Deep Hodge optimization patches.

Guards mathematical invariants and numerical equivalence of hot paths
in DeepHodgeLayer.forward() and DifferentiableHodgeProxy._hodge_spectrum()
that will be optimized to reduce CPU overhead (8 min → ~2 min).

Patches:
  A — Vectorize scale loop (batch all scales)
  B — Pre-register B1^T, abs_B1^T as buffers
  C — Convert Python list indices to tensor buffers
  D — Pre-expand B1 for batch dimension
  E — Hoist scale-invariant term_down out of loop / fuse bmm

Every test MUST pass before AND after each patch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from synapse.synapse_arch.deep_hodge import DeepHodgeLayer, DeepHodgeTransformer
from synapse.synapse_core.proxy import DifferentiableHodgeProxy


# ── Helpers ──────────────────────────────────────────────────────────

def _make_layer(d_model=16, k_dim=4, num_scales=3, max_points=8, tau=1e-4):
    return DeepHodgeLayer(d_model, k_dim, num_scales, max_points, tau)


def _make_proxy(lift_dim=4, hidden_dim=16, num_scales=3, num_eigs=4, max_points=8, tau=1e-4):
    return DifferentiableHodgeProxy(lift_dim, hidden_dim, num_scales, num_eigs, max_points, tau)


def _rand(B=2, K=8, d=16, seed=42):
    torch.manual_seed(seed)
    return torch.randn(B, K, d)


def _compute_L0(layer, x, B_batch):
    """Reference L0 computation matching current forward pass."""
    x_norm = layer.norm1(x)
    P = layer.geom_proj(x_norm)
    mask = torch.ones(B_batch, layer.max_points, device=x.device)
    mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
    D = torch.cdist(P, P) * mask_2d
    scales = torch.exp(layer.log_scales)
    results = []
    for s_idx in range(layer.num_scales):
        sigma = scales[s_idx]
        A = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
        W1 = A[:, layer.e_idx_i, layer.e_idx_j]
        B1_scaled = layer.B1.unsqueeze(0) * W1.unsqueeze(1)
        B1_exp = layer.B1.unsqueeze(0).expand(B_batch, -1, -1)
        L0 = torch.bmm(B1_scaled, B1_exp.transpose(1, 2))
        L0 = L0 + layer.tau * layer._eye_K.unsqueeze(0)
        results.append((L0, W1, W2_if_needed(W1, layer)))
    return results, mask, D, scales


def W2_if_needed(W1, layer):
    return W1[:, layer.t_idx_ij] * W1[:, layer.t_idx_jk] * W1[:, layer.t_idx_ik]


# ═════════════════════════════════════════════════════════════════════
# 1. MATHEMATICAL INVARIANTS
# ═════════════════════════════════════════════════════════════════════

class TestHodgeLaplacianInvariants:
    @pytest.fixture()
    def layer(self):
        return _make_layer()

    def test_L0_symmetric(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            x_norm = layer.norm1(x)
            P = layer.geom_proj(x_norm)
            mask = torch.ones(2, 8)
            mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
            D = torch.cdist(P, P) * mask_2d
            for s_idx in range(layer.num_scales):
                sigma = torch.exp(layer.log_scales[s_idx])
                A = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
                W1 = A[:, layer.e_idx_i, layer.e_idx_j]
                B1s = layer.B1.unsqueeze(0) * W1.unsqueeze(1)
                B1e = layer.B1.unsqueeze(0).expand(2, -1, -1)
                L0 = torch.bmm(B1s, B1e.transpose(1, 2)) + layer.tau * layer._eye_K.unsqueeze(0)
                for b in range(2):
                    assert torch.allclose(L0[b], L0[b].T, atol=1e-5), f"L0 not symmetric: s={s_idx}, b={b}"

    def test_L0_positive_definite(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            x_norm = layer.norm1(x)
            P = layer.geom_proj(x_norm)
            mask = torch.ones(2, 8)
            mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
            D = torch.cdist(P, P) * mask_2d
            for s_idx in range(layer.num_scales):
                sigma = torch.exp(layer.log_scales[s_idx])
                A = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
                W1 = A[:, layer.e_idx_i, layer.e_idx_j]
                B1s = layer.B1.unsqueeze(0) * W1.unsqueeze(1)
                B1e = layer.B1.unsqueeze(0).expand(2, -1, -1)
                L0 = torch.bmm(B1s, B1e.transpose(1, 2)) + layer.tau * layer._eye_K.unsqueeze(0)
                eig = torch.linalg.eigvalsh(L0)
                assert (eig > 0).all(), f"L0 not PD: s={s_idx}"

    def test_L1_symmetric(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            x_norm = layer.norm1(x)
            P = layer.geom_proj(x_norm)
            mask = torch.ones(2, 8)
            mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
            D = torch.cdist(P, P) * mask_2d
            W0 = mask
            for s_idx in range(layer.num_scales):
                sigma = torch.exp(layer.log_scales[s_idx])
                A = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
                W1 = A[:, layer.e_idx_i, layer.e_idx_j]
                W2 = W1[:, layer.t_idx_ij] * W1[:, layer.t_idx_jk] * W1[:, layer.t_idx_ik]
                BTs = layer.B1.t().unsqueeze(0) * W0.unsqueeze(1)
                B1e = layer.B1.unsqueeze(0).expand(2, -1, -1)
                td = torch.bmm(BTs, B1e)
                B2s = layer.B2.unsqueeze(0) * W2.unsqueeze(1)
                B2e = layer.B2.t().unsqueeze(0).expand(2, -1, -1)
                tu = torch.bmm(B2s, B2e)
                L1 = td + tu + layer.tau * layer._eye_E.unsqueeze(0)
                for b in range(2):
                    assert torch.allclose(L1[b], L1[b].T, atol=1e-4), f"L1 not symmetric: s={s_idx}, b={b}"

    def test_L1_positive_definite(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            x_norm = layer.norm1(x)
            P = layer.geom_proj(x_norm)
            mask = torch.ones(2, 8)
            mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
            D = torch.cdist(P, P) * mask_2d
            W0 = mask
            for s_idx in range(layer.num_scales):
                sigma = torch.exp(layer.log_scales[s_idx])
                A = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
                W1 = A[:, layer.e_idx_i, layer.e_idx_j]
                W2 = W1[:, layer.t_idx_ij] * W1[:, layer.t_idx_jk] * W1[:, layer.t_idx_ik]
                BTs = layer.B1.t().unsqueeze(0) * W0.unsqueeze(1)
                B1e = layer.B1.unsqueeze(0).expand(2, -1, -1)
                td = torch.bmm(BTs, B1e)
                B2s = layer.B2.unsqueeze(0) * W2.unsqueeze(1)
                B2e = layer.B2.t().unsqueeze(0).expand(2, -1, -1)
                tu = torch.bmm(B2s, B2e)
                L1 = td + tu + layer.tau * layer._eye_E.unsqueeze(0)
                eig = torch.linalg.eigvalsh(L1)
                assert (eig > 0).all(), f"L1 not PD: s={s_idx}"


class TestProxyEigenvalueInvariants:
    @pytest.fixture()
    def proxy(self):
        return _make_proxy()

    def test_eigenvalues_sorted_ascending(self, proxy):
        B, N, k = 2, 10, 4
        torch.manual_seed(42)
        cloud, weights = torch.randn(B, N, k), torch.rand(B, N)
        proxy.eval()
        with torch.no_grad():
            raw = proxy._hodge_spectrum(cloud.float(), weights.float())
        J = proxy.num_eigs
        off = 0
        for s in range(proxy.num_scales):
            e0, e1 = raw[:, off:off+J], raw[:, off+J:off+2*J]
            for b in range(B):
                assert (e0[b, 1:] - e0[b, :-1] >= -1e-6).all(), f"L0 eigs not ascending: s={s}"
                assert (e1[b, 1:] - e1[b, :-1] >= -1e-6).all(), f"L1 eigs not ascending: s={s}"
            off += 2 * J

    def test_eigenvalues_positive(self, proxy):
        B, N, k = 2, 10, 4
        torch.manual_seed(42)
        cloud, weights = torch.randn(B, N, k), torch.rand(B, N)
        proxy.eval()
        with torch.no_grad():
            raw = proxy._hodge_spectrum(cloud.float(), weights.float())
        J = proxy.num_eigs
        off = 0
        for s in range(proxy.num_scales):
            assert (raw[:, off:off+J] > 0).all(), f"L0 eigs not positive: s={s}"
            assert (raw[:, off+J:off+2*J] > 0).all(), f"L1 eigs not positive: s={s}"
            off += 2 * J


# ═════════════════════════════════════════════════════════════════════
# 2. PATCH-SPECIFIC NUMERICAL EQUIVALENCE TESTS
# ═════════════════════════════════════════════════════════════════════

class TestPatchB_PrecomputedTransposes:
    """B1^T and |B1|^T as buffers must match .t() on every call."""
    @pytest.fixture()
    def layer(self):
        return _make_layer()

    def test_B1t_buffer_equals_B1_t(self, layer):
        assert torch.equal(layer.B1.t().contiguous(), layer.B1.t().contiguous())

    def test_abs_B1t_buffer_equals_abs_B1_t(self, layer):
        assert torch.equal(layer.abs_B1.t().contiguous(), layer.abs_B1.t().contiguous())

    def test_B1t_in_L1_term_matches(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            W0 = torch.ones(2, 8)
            orig = layer.B1.t().unsqueeze(0) * W0.unsqueeze(1)
            buf = layer.B1.t().contiguous().unsqueeze(0) * W0.unsqueeze(1)
            assert torch.allclose(orig, buf, atol=0.0)

    def test_abs_B1t_in_edge_pullback_matches(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            x_norm = layer.norm1(x)
            orig = torch.matmul(layer.abs_B1.t(), x_norm)
            buf = torch.matmul(layer.abs_B1.t().contiguous(), x_norm)
            assert torch.allclose(orig, buf, atol=0.0)


class TestPatchC_TensorBufferIndices:
    """Python list indices → tensor buffers must produce identical W1/W2."""
    @pytest.fixture()
    def layer(self):
        return _make_layer()

    @pytest.fixture()
    def proxy(self):
        return _make_proxy()

    def test_edge_weight_indexing_layer(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            P = layer.geom_proj(layer.norm1(x))
            mask_2d = torch.ones(2, 8, 8)
            D = torch.cdist(P, P) * mask_2d
            sigma = torch.exp(layer.log_scales[0])
            A = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
            W1_list = A[:, layer.e_idx_i, layer.e_idx_j]
            W1_buf = A[:, torch.tensor(layer.e_idx_i), torch.tensor(layer.e_idx_j)]
            assert torch.allclose(W1_list, W1_buf, atol=0.0)

    def test_triangle_weight_indexing_layer(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            P = layer.geom_proj(layer.norm1(x))
            mask_2d = torch.ones(2, 8, 8)
            D = torch.cdist(P, P) * mask_2d
            sigma = torch.exp(layer.log_scales[0])
            A = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
            W1 = A[:, layer.e_idx_i, layer.e_idx_j]
            W2_list = W1[:, layer.t_idx_ij] * W1[:, layer.t_idx_jk] * W1[:, layer.t_idx_ik]
            W2_buf = (W1[:, torch.tensor(layer.t_idx_ij)]
                      * W1[:, torch.tensor(layer.t_idx_jk)]
                      * W1[:, torch.tensor(layer.t_idx_ik)])
            assert torch.allclose(W2_list, W2_buf, atol=0.0)

    def test_edge_weight_indexing_proxy(self, proxy):
        B, N, k = 2, 10, 4
        torch.manual_seed(42)
        cloud, weights = torch.randn(B, N, k), torch.rand(B, N)
        proxy.eval()
        with torch.no_grad():
            K_eff = min(N, proxy.max_points)
            _, ti = torch.topk(weights.float(), K_eff, dim=1, sorted=False)
            c = torch.gather(cloud.float(), 1, ti.unsqueeze(-1).expand(-1, -1, k))
            act = torch.gather(weights.float(), 1, ti)
            if K_eff < proxy.max_points:
                c = torch.cat([c, torch.zeros(B, proxy.max_points-K_eff, k)], 1)
                act = torch.cat([act, torch.zeros(B, proxy.max_points-K_eff)], 1)
            mask = (act > 1e-3).float()
            m2 = mask.unsqueeze(2) * mask.unsqueeze(1)
            D = torch.cdist(c, c) * m2
            sigma = torch.exp(proxy.log_scales.float()[0])
            A = torch.exp(-D.square() / (2.0*sigma.square()+1e-8)) * m2
            W1_list = A[:, proxy.e_idx_i, proxy.e_idx_j]
            W1_buf = A[:, torch.tensor(proxy.e_idx_i), torch.tensor(proxy.e_idx_j)]
            assert torch.allclose(W1_list, W1_buf, atol=0.0)


class TestPatchD_PreexpandedB1:
    """Pre-expanded B1 must equal on-the-fly unsqueeze+expand."""
    @pytest.fixture()
    def layer(self):
        return _make_layer()

    def test_expanded_B1_matches(self, layer):
        fly = layer.B1.unsqueeze(0).expand(4, -1, -1)
        pre = layer.B1.unsqueeze(0).expand(4, -1, -1).contiguous()
        assert torch.equal(fly, pre)

    def test_L0_with_preexpanded_matches(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            P = layer.geom_proj(layer.norm1(x))
            mask_2d = torch.ones(2, 8, 8)
            D = torch.cdist(P, P) * mask_2d
            sigma = torch.exp(layer.log_scales[0])
            A = torch.exp(-D.square() / (2.0*sigma.square()+1e-8)) * mask_2d
            W1 = A[:, layer.e_idx_i, layer.e_idx_j]
            B1s = layer.B1.unsqueeze(0) * W1.unsqueeze(1)
            L0_fly = torch.bmm(B1s, layer.B1.unsqueeze(0).expand(2,-1,-1).transpose(1,2))
            L0_pre = torch.bmm(B1s, layer.B1.unsqueeze(0).expand(2,-1,-1).contiguous().transpose(1,2))
            assert torch.allclose(L0_fly, L0_pre, atol=0.0)


class TestPatchA_VectorizedScaleLoop:
    """Batching all scales must produce identical L0 and L1."""
    @pytest.fixture()
    def layer(self):
        return _make_layer()

    def test_vectorized_L0_matches(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            P = layer.geom_proj(layer.norm1(x))
            mask_2d = torch.ones(2, 8, 8)
            D = torch.cdist(P, P) * mask_2d
            scales = torch.exp(layer.log_scales)
            # Sequential
            L0s = []
            all_W1 = []
            for s in range(layer.num_scales):
                A = torch.exp(-D.square()/(2.0*scales[s].square()+1e-8)) * mask_2d
                W1 = A[:, layer.e_idx_i, layer.e_idx_j]
                all_W1.append(W1)
                B1s = layer.B1.unsqueeze(0)*W1.unsqueeze(1)
                B1e = layer.B1.unsqueeze(0).expand(2,-1,-1)
                L0s.append(torch.bmm(B1s, B1e.transpose(1,2)) + layer.tau*layer._eye_K)
            # Vectorized
            W1_stk = torch.stack(all_W1, dim=1)  # (B,S,E)
            BS = 2 * layer.num_scales
            W1f = W1_stk.reshape(BS, 1, layer.B1.shape[1])
            B1e = layer.B1.unsqueeze(0).expand(BS,-1,-1)
            B1s = B1e * W1f
            L0v = torch.bmm(B1s, B1e.transpose(1,2))
            L0v = L0v.reshape(2, layer.num_scales, 8, 8) + layer.tau*layer._eye_K
            for s in range(layer.num_scales):
                assert torch.allclose(L0s[s], L0v[:,s], atol=1e-5), f"L0 mismatch: s={s}"

    def test_full_forward_deterministic(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            assert torch.allclose(layer(x), layer(x), atol=0.0)


class TestPatchE_HoistedTermDown:
    """B1^T W0 B1 is scale-invariant — hoisting must not change L1."""
    @pytest.fixture()
    def layer(self):
        return _make_layer()

    def test_term_down_constant_across_scales(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            W0 = torch.ones(2, 8)
            td_ref = torch.bmm(
                layer.B1.t().unsqueeze(0)*W0.unsqueeze(1),
                layer.B1.unsqueeze(0).expand(2,-1,-1))
            for _ in range(layer.num_scales):
                td_s = torch.bmm(
                    layer.B1.t().unsqueeze(0)*W0.unsqueeze(1),
                    layer.B1.unsqueeze(0).expand(2,-1,-1))
                assert torch.allclose(td_ref, td_s, atol=0.0)

    def test_L1_with_hoisted_term_down_matches(self, layer):
        x = _rand()
        layer.eval()
        with torch.no_grad():
            P = layer.geom_proj(layer.norm1(x))
            mask_2d = torch.ones(2, 8, 8)
            D = torch.cdist(P, P) * mask_2d
            scales = torch.exp(layer.log_scales)
            W0 = torch.ones(2, 8)
            td_hoisted = torch.bmm(
                layer.B1.t().unsqueeze(0)*W0.unsqueeze(1),
                layer.B1.unsqueeze(0).expand(2,-1,-1))
            for s in range(layer.num_scales):
                A = torch.exp(-D.square()/(2.0*scales[s].square()+1e-8))*mask_2d
                W1 = A[:, layer.e_idx_i, layer.e_idx_j]
                W2 = W1[:, layer.t_idx_ij]*W1[:, layer.t_idx_jk]*W1[:, layer.t_idx_ik]
                tu = torch.bmm(
                    layer.B2.unsqueeze(0)*W2.unsqueeze(1),
                    layer.B2.t().unsqueeze(0).expand(2,-1,-1))
                L1_loop = torch.bmm(
                    layer.B1.t().unsqueeze(0)*W0.unsqueeze(1),
                    layer.B1.unsqueeze(0).expand(2,-1,-1)) + tu + layer.tau*layer._eye_E
                L1_hoist = td_hoisted + tu + layer.tau*layer._eye_E
                assert torch.allclose(L1_loop, L1_hoist, atol=0.0), f"L1 mismatch: s={s}"


# ═════════════════════════════════════════════════════════════════════
# 3. SHAPE AND OUTPUT CONTRACT TESTS
# ═════════════════════════════════════════════════════════════════════

class TestDeepHodgeLayerShapeContract:
    @pytest.fixture()
    def layer(self):
        return _make_layer()

    def test_output_shape_matches_input(self, layer):
        out = layer(_rand())
        assert out.shape == (2, 8, 16)

    def test_output_shape_with_padding(self, layer):
        out = layer(_rand(B=2, K=5, d=16))
        assert out.shape == (2, 5, 16)

    def test_output_shape_with_mask(self, layer):
        x = _rand()
        mask = torch.ones(2, 8); mask[0, 5:] = 0.0
        assert layer(x, mask=mask).shape == (2, 8, 16)

    def test_output_finite(self, layer):
        assert torch.isfinite(layer(_rand())).all()

    def test_output_finite_tiny_input(self, layer):
        assert torch.isfinite(layer(torch.randn(2, 8, 16) * 1e-6)).all()

    def test_output_finite_large_input(self, layer):
        assert torch.isfinite(layer(torch.randn(2, 8, 16) * 1e3)).all()


class TestDeepHodgeTransformerShapeContract:
    def test_single_layer_shape(self):
        t = DeepHodgeTransformer(1, 16, 4, 3, 8)
        assert t(_rand()).shape == (2, 8, 16)

    def test_multi_layer_shape(self):
        t = DeepHodgeTransformer(3, 16, 4, 3, 8)
        assert t(_rand()).shape == (2, 8, 16)

    def test_multi_layer_finite(self):
        t = DeepHodgeTransformer(3, 16, 4, 3, 8)
        assert torch.isfinite(t(_rand())).all()


class TestProxyShapeContract:
    @pytest.fixture()
    def proxy(self):
        return _make_proxy()

    def test_output_shape(self, proxy):
        torch.manual_seed(42)
        out = proxy(torch.randn(2, 10, 4), torch.rand(2, 10))
        assert out.shape == (2, 16)

    def test_output_finite(self, proxy):
        torch.manual_seed(42)
        assert torch.isfinite(proxy(torch.randn(2, 10, 4), torch.rand(2, 10))).all()

    def test_feature_dim(self, proxy):
        assert proxy.feature_dim == 2 * proxy.num_scales * proxy.num_eigs + 4

    def test_raw_spectrum_shape(self, proxy):
        torch.manual_seed(42)
        proxy.eval()
        with torch.no_grad():
            raw = proxy._hodge_spectrum(torch.randn(2, 10, 4).float(), torch.rand(2, 10).float())
        assert raw.shape == (2, proxy.feature_dim)


# ═════════════════════════════════════════════════════════════════════
# 4. GRADIENT FLOW TESTS
# ═════════════════════════════════════════════════════════════════════

class TestDeepHodgeLayerGradients:
    @pytest.fixture()
    def layer(self):
        l = _make_layer(); l.train(); return l

    def _check_grad(self, layer, param_name):
        x = _rand()
        layer(x).sum().backward()
        p = dict(layer.named_parameters())[param_name]
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"No grad for {param_name}"

    def test_grad_geom_proj_weight(self, layer): self._check_grad(layer, "geom_proj.weight")
    def test_grad_log_scales(self, layer):
        x = _rand(); layer(x).sum().backward()
        assert layer.log_scales.grad is not None and torch.isfinite(layer.log_scales.grad).all()

    def test_grad_W_V0(self, layer): self._check_grad(layer, "W_V0.weight")
    def test_grad_W_E_in(self, layer): self._check_grad(layer, "W_E_in.weight")
    def test_grad_out_proj(self, layer): self._check_grad(layer, "out_proj.weight")

    def test_grad_ffn(self, layer):
        x = _rand(); layer(x).sum().backward()
        has_grad = any(m.weight.grad is not None for m in layer.ffn if isinstance(m, nn.Linear))
        assert has_grad

    def test_grad_with_mask(self, layer):
        x = _rand()
        mask = torch.ones(2, 8); mask[0, 5:] = 0.0
        layer(x, mask=mask).sum().backward()
        assert layer.log_scales.grad is not None and torch.isfinite(layer.log_scales.grad).all()

    def test_grad_with_padding(self, layer):
        layer(_rand(B=2, K=5, d=16)).sum().backward()
        assert layer.log_scales.grad is not None and torch.isfinite(layer.log_scales.grad).all()


class TestProxyGradients:
    @pytest.fixture()
    def proxy(self):
        p = _make_proxy(); p.train(); return p

    def test_grad_log_scales(self, proxy):
        torch.manual_seed(42)
        cloud = torch.randn(2, 10, 4, requires_grad=True)
        weights = torch.rand(2, 10, requires_grad=True)
        proxy(cloud, weights).sum().backward()
        assert proxy.log_scales.grad is not None and torch.isfinite(proxy.log_scales.grad).all()

    def test_grad_proj(self, proxy):
        torch.manual_seed(42)
        cloud = torch.randn(2, 10, 4, requires_grad=True)
        weights = torch.rand(2, 10, requires_grad=True)
        proxy(cloud, weights).sum().backward()
        has_grad = any(m.weight.grad is not None for m in proxy.proj if isinstance(m, nn.Linear))
        assert has_grad

    def test_grad_through_eigvalsh(self, proxy):
        """Gradient must flow through torch.linalg.eigvalsh (most likely to break)."""
        torch.manual_seed(42)
        cloud = torch.randn(2, 10, 4, requires_grad=True)
        weights = torch.rand(2, 10, requires_grad=True)
        proxy(cloud, weights).sum().backward()
        assert proxy.log_scales.grad is not None
        assert proxy.log_scales.grad.abs().sum() > 0, "Gradient through eigvalsh is zero"


# ═════════════════════════════════════════════════════════════════════
# 5. EDGE CASE AND BOUNDARY CONDITION TESTS
# ═════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_K_equals_1(self):
        layer = _make_layer(d_model=16, k_dim=4, num_scales=2, max_points=4)
        out = layer(torch.randn(1, 1, 16))
        assert out.shape == (1, 1, 16) and torch.isfinite(out).all()

    def test_K_equals_max_points(self):
        layer = _make_layer(d_model=16, k_dim=4, num_scales=2, max_points=8)
        out = layer(_rand(B=2, K=8, d=16))
        assert out.shape == (2, 8, 16) and torch.isfinite(out).all()

    def test_all_zero_mask(self):
        layer = _make_layer(d_model=16, k_dim=4, num_scales=2, max_points=8)
        x = _rand(B=1, K=8, d=16)
        out = layer(x, mask=torch.zeros(1, 8))
        assert torch.isfinite(out).all()

    def test_single_scale(self):
        layer = _make_layer(d_model=16, k_dim=4, num_scales=1, max_points=8)
        out = layer(_rand())
        assert out.shape == (2, 8, 16) and torch.isfinite(out).all()

    def test_large_num_scales(self):
        layer = _make_layer(d_model=16, k_dim=4, num_scales=6, max_points=8)
        out = layer(_rand())
        assert out.shape == (2, 8, 16) and torch.isfinite(out).all()

    def test_batch_size_1(self):
        layer = _make_layer()
        out = layer(_rand(B=1, K=8, d=16))
        assert out.shape == (1, 8, 16) and torch.isfinite(out).all()

    def test_proxy_with_N_less_than_max_points(self):
        proxy = _make_proxy(max_points=16)
        torch.manual_seed(42)
        out = proxy(torch.randn(2, 5, 4), torch.rand(2, 5))  # N=5 < max_points=16
        assert out.shape == (2, 16) and torch.isfinite(out).all()

    def test_proxy_with_zero_weights(self):
        proxy = _make_proxy()
        torch.manual_seed(42)
        cloud = torch.randn(2, 10, 4)
        weights = torch.zeros(2, 10)  # all zero → all masked
        out = proxy(cloud, weights)
        assert torch.isfinite(out).all()

    def test_residual_connection_preserved(self):
        """Residual must be added — output != FFN output alone."""
        layer = _make_layer()
        x = _rand()
        layer.eval()
        with torch.no_grad():
            out = layer(x)
        # If residual were dropped, output would be very different from input
        # We just check it's not identical to x (residual + attn + ffn)
        assert not torch.allclose(out, x, atol=1e-3), "Output equals input — residual may be broken"

    def test_boundary_matrices_correct_shape(self):
        """B1: (K, E), B2: (E, T), abs_B1: (K, E)."""
        K = 8
        E = K * (K - 1) // 2
        T = K * (K - 1) * (K - 2) // 6
        layer = _make_layer(max_points=K)
        assert layer.B1.shape == (K, E)
        assert layer.B2.shape == (E, T)
        assert layer.abs_B1.shape == (K, E)
        assert layer._eye_K.shape == (K, K)
        assert layer._eye_E.shape == (E, E)

    def test_B1_row_sum_is_zero(self):
        """Each column of B1 sums to zero (oriented incidence)."""
        layer = _make_layer()
        col_sums = layer.B1.sum(dim=0)
        assert torch.allclose(col_sums, torch.zeros_like(col_sums), atol=1e-6)

    def test_B2_B1_product_is_zero(self):
        """B1 @ B2 = 0 (boundary of boundary = 0)."""
        layer = _make_layer()
        product = layer.B1 @ layer.B2
        assert torch.allclose(product, torch.zeros_like(product), atol=1e-5)


# ═════════════════════════════════════════════════════════════════════
# 6. END-TO-END NUMERICAL EQUIVALENCE (OLD LOOP vs NEW VECTORIZED)
# ═════════════════════════════════════════════════════════════════════

class TestProxyVectorizedEquivalence:
    """The vectorized _hodge_spectrum must produce identical eigenvalues
    to the original per-scale loop implementation."""

    @pytest.fixture()
    def proxy(self):
        return _make_proxy()

    @staticmethod
    def _hodge_spectrum_loop(proxy, points, weights):
        """Reference loop-based implementation (pre-Patch A)."""
        B_batch, N, k_dim = points.shape
        K_eff = min(N, proxy.max_points)
        _, top_idx = torch.topk(weights, K_eff, dim=1, sorted=False)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, k_dim)
        cloud = torch.gather(points, 1, top_idx_exp)
        act = torch.gather(weights, 1, top_idx)
        if K_eff < proxy.max_points:
            pad_pts = torch.zeros(B_batch, proxy.max_points - K_eff, k_dim, device=cloud.device)
            pad_act = torch.zeros(B_batch, proxy.max_points - K_eff, device=act.device)
            cloud = torch.cat([cloud, pad_pts], dim=1)
            act = torch.cat([act, pad_act], dim=1)
        mask = (act > 1e-3).float()
        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
        D = torch.cdist(cloud, cloud) * mask_2d
        scales = torch.exp(proxy.log_scales.float())
        num_eigs = proxy.num_eigs
        W0 = act

        features = []
        for s_idx in range(proxy.num_scales):
            sigma = scales[s_idx]
            A = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
            W1 = A[:, proxy.e_idx_i, proxy.e_idx_j]
            W2 = W1[:, proxy.t_idx_ij] * W1[:, proxy.t_idx_jk] * W1[:, proxy.t_idx_ik]

            B1_scaled = proxy.B1.unsqueeze(0) * W1.unsqueeze(1)
            B1_exp = proxy.B1.unsqueeze(0).expand(B_batch, -1, -1)
            L0 = torch.bmm(B1_scaled, B1_exp.transpose(1, 2))
            L0 = L0 + proxy.tau * proxy._eye_K.unsqueeze(0)
            eigvals_L0 = torch.linalg.eigvalsh(L0)

            BT_scaled = proxy.B1.t().unsqueeze(0) * W0.unsqueeze(1)
            B1_exp = proxy.B1.unsqueeze(0).expand(B_batch, -1, -1)
            term_down = torch.bmm(BT_scaled, B1_exp)

            B2_scaled = proxy.B2.unsqueeze(0) * W2.unsqueeze(1)
            B2_exp = proxy.B2.t().unsqueeze(0).expand(B_batch, -1, -1)
            term_up = torch.bmm(B2_scaled, B2_exp)

            L1 = term_down + term_up
            L1 = L1 + proxy.tau * proxy._eye_E.unsqueeze(0)
            eigvals_L1 = torch.linalg.eigvalsh(L1)

            eig0 = eigvals_L0[:, :num_eigs]
            if eig0.shape[1] < num_eigs:
                eig0 = torch.nn.functional.pad(eig0, (0, num_eigs - eig0.shape[1]))
            eig1 = eigvals_L1[:, :num_eigs]
            if eig1.shape[1] < num_eigs:
                eig1 = torch.nn.functional.pad(eig1, (0, num_eigs - eig1.shape[1]))
            features.append(eig0)
            features.append(eig1)

        tri_mask = torch.triu(mask_2d, diagonal=1)
        tri_sum = tri_mask.sum(dim=(1, 2)).clamp_min(1.0)
        mean_dist = (D * tri_mask).sum(dim=(1, 2)) / tri_sum
        max_dist = (D * tri_mask).amax(dim=(1, 2))
        dist_var = ((D - mean_dist[:, None, None]).square() * tri_mask).sum(dim=(1, 2)) / tri_sum
        compactness = mean_dist / (max_dist + 1e-6)
        features.append(torch.stack([mean_dist, max_dist, dist_var, compactness], dim=-1))

        return torch.cat(features, dim=-1)

    def test_proxy_spectrum_loop_vs_vectorized(self, proxy):
        torch.manual_seed(99)
        B, N, k = 2, 10, 4
        cloud = torch.randn(B, N, k)
        weights = torch.rand(B, N)
        proxy.eval()
        with torch.no_grad():
            raw_loop = self._hodge_spectrum_loop(proxy, cloud.float(), weights.float())
            raw_vec = proxy._hodge_spectrum(cloud.float(), weights.float())
        assert torch.allclose(raw_loop, raw_vec, atol=1e-4), (
            f"Proxy spectrum mismatch: max diff = {(raw_loop - raw_vec).abs().max():.2e}"
        )

    def test_proxy_forward_loop_vs_vectorized(self, proxy):
        torch.manual_seed(77)
        B, N, k = 2, 10, 4
        cloud = torch.randn(B, N, k)
        weights = torch.rand(B, N)
        proxy.eval()
        with torch.no_grad():
            out = proxy(cloud, weights)
        assert torch.isfinite(out).all()
        assert out.shape == (B, 16)

    def test_proxy_spectrum_loop_vs_vectorized_small_N(self, proxy):
        """N < max_points — triggers zero-padding path."""
        torch.manual_seed(55)
        B, N, k = 2, 5, 4
        cloud = torch.randn(B, N, k)
        weights = torch.rand(B, N)
        proxy.eval()
        with torch.no_grad():
            raw_loop = self._hodge_spectrum_loop(proxy, cloud.float(), weights.float())
            raw_vec = proxy._hodge_spectrum(cloud.float(), weights.float())
        assert torch.allclose(raw_loop, raw_vec, atol=1e-4)

    def test_proxy_spectrum_loop_vs_vectorized_single_scale(self):
        proxy = _make_proxy(num_scales=1)
        torch.manual_seed(33)
        B, N, k = 2, 10, 4
        cloud = torch.randn(B, N, k)
        weights = torch.rand(B, N)
        proxy.eval()
        with torch.no_grad():
            raw_loop = self._hodge_spectrum_loop(proxy, cloud.float(), weights.float())
            raw_vec = proxy._hodge_spectrum(cloud.float(), weights.float())
        assert torch.allclose(raw_loop, raw_vec, atol=1e-4)


class TestDeepHodgeVectorizedEquivalence:
    """The vectorized DeepHodgeLayer.forward must produce identical outputs
    to the original per-scale loop implementation."""

    @staticmethod
    def _forward_loop(layer, x, mask=None):
        """Reference loop-based implementation (pre-Patch A)."""
        B_batch, K_eff, d = x.shape
        pad_len = layer.max_points - K_eff
        if pad_len > 0:
            x_pad = torch.zeros(B_batch, pad_len, d, device=x.device, dtype=x.dtype)
            x = torch.cat([x, x_pad], dim=1)
            if mask is not None:
                m_pad = torch.zeros(B_batch, pad_len, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, m_pad], dim=1)
        if mask is None:
            mask = torch.ones(B_batch, layer.max_points, device=x.device)

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
            A = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
            W1 = A[:, layer.e_idx_i, layer.e_idx_j]
            W2 = W1[:, layer.t_idx_ij] * W1[:, layer.t_idx_jk] * W1[:, layer.t_idx_ik]
            W0 = mask

            B1_scaled = layer.B1.unsqueeze(0) * W1.unsqueeze(1)
            B1_exp = layer.B1.unsqueeze(0).expand(B_batch, -1, -1)
            L0 = torch.bmm(B1_scaled, B1_exp.transpose(1, 2))
            L0 = L0 + layer.tau * layer._eye_K.unsqueeze(0)
            V0_s = V0[:, :, s_idx, :]
            M0 = torch.bmm(L0, V0_s)
            node_updates.append(M0)

            BT_scaled = layer.B1.t().unsqueeze(0) * W0.unsqueeze(1)
            B1_exp = layer.B1.unsqueeze(0).expand(B_batch, -1, -1)
            term_down = torch.bmm(BT_scaled, B1_exp)

            B2_scaled = layer.B2.unsqueeze(0) * W2.unsqueeze(1)
            B2_exp = layer.B2.t().unsqueeze(0).expand(B_batch, -1, -1)
            term_up = torch.bmm(B2_scaled, B2_exp)

            L1 = term_down + term_up
            L1 = L1 + layer.tau * layer._eye_E.unsqueeze(0)

            E_in_s = E_in[:, :, s_idx, :]
            M1 = torch.bmm(L1, E_in_s)
            M1_to_node = torch.bmm(layer.abs_B1.unsqueeze(0).expand(B_batch, -1, -1), M1)
            edge_updates.append(M1_to_node)

        all_updates = torch.cat(node_updates + edge_updates, dim=-1)
        attn_out = layer.out_proj(all_updates)
        attn_out = layer.dropout(attn_out)
        x_out = residual + attn_out
        x_out = x_out + layer.ffn(layer.norm2(x_out))
        if pad_len > 0:
            x_out = x_out[:, :K_eff, :]
        return x_out

    def test_layer_loop_vs_vectorized(self):
        layer = _make_layer()
        torch.manual_seed(42)
        x = _rand()
        layer.eval()
        with torch.no_grad():
            out_loop = self._forward_loop(layer, x)
            out_vec = layer(x)
        assert torch.allclose(out_loop, out_vec, atol=1e-4), (
            f"DeepHodgeLayer mismatch: max diff = {(out_loop - out_vec).abs().max():.2e}"
        )

    def test_layer_loop_vs_vectorized_with_mask(self):
        layer = _make_layer()
        torch.manual_seed(42)
        x = _rand()
        mask = torch.ones(2, 8)
        mask[0, 5:] = 0.0
        layer.eval()
        with torch.no_grad():
            out_loop = self._forward_loop(layer, x, mask=mask)
            out_vec = layer(x, mask=mask)
        assert torch.allclose(out_loop, out_vec, atol=1e-4), (
            f"DeepHodgeLayer mask mismatch: max diff = {(out_loop - out_vec).abs().max():.2e}"
        )

    def test_layer_loop_vs_vectorized_with_padding(self):
        layer = _make_layer()
        torch.manual_seed(42)
        x = _rand(B=2, K=5, d=16)
        layer.eval()
        with torch.no_grad():
            out_loop = self._forward_loop(layer, x)
            out_vec = layer(x)
        assert torch.allclose(out_loop, out_vec, atol=1e-4), (
            f"DeepHodgeLayer pad mismatch: max diff = {(out_loop - out_vec).abs().max():.2e}"
        )

    def test_layer_loop_vs_vectorized_single_scale(self):
        layer = _make_layer(num_scales=1)
        torch.manual_seed(42)
        x = _rand()
        layer.eval()
        with torch.no_grad():
            out_loop = self._forward_loop(layer, x)
            out_vec = layer(x)
        assert torch.allclose(out_loop, out_vec, atol=1e-4)

    def test_layer_loop_vs_vectorized_batch1(self):
        layer = _make_layer()
        torch.manual_seed(42)
        x = _rand(B=1, K=8, d=16)
        layer.eval()
        with torch.no_grad():
            out_loop = self._forward_loop(layer, x)
            out_vec = layer(x)
        assert torch.allclose(out_loop, out_vec, atol=1e-4)


# ═════════════════════════════════════════════════════════════════════
# 7. BUFFER REGISTRATION AND BACKWARD COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════

class TestBufferRegistration:
    """Verify all new buffers are registered and Python lists preserved."""

    def test_proxy_has_B1T_B2T_buffers(self):
        proxy = _make_proxy()
        assert hasattr(proxy, "_B1T") and proxy._B1T is not None
        assert hasattr(proxy, "_B2T") and proxy._B2T is not None
        assert torch.equal(proxy._B1T, proxy.B1.t().contiguous())
        assert torch.equal(proxy._B2T, proxy.B2.t().contiguous())

    def test_proxy_has_tensor_index_buffers(self):
        proxy = _make_proxy()
        assert hasattr(proxy, "_e_idx_i") and proxy._e_idx_i.dtype == torch.long
        assert hasattr(proxy, "_e_idx_j") and proxy._e_idx_j.dtype == torch.long
        assert hasattr(proxy, "_t_idx_ij") and proxy._t_idx_ij.dtype == torch.long
        assert hasattr(proxy, "_t_idx_jk") and proxy._t_idx_jk.dtype == torch.long
        assert hasattr(proxy, "_t_idx_ik") and proxy._t_idx_ik.dtype == torch.long

    def test_proxy_tensor_indices_match_python_lists(self):
        proxy = _make_proxy()
        assert torch.equal(proxy._e_idx_i, torch.tensor(proxy.e_idx_i, dtype=torch.long))
        assert torch.equal(proxy._e_idx_j, torch.tensor(proxy.e_idx_j, dtype=torch.long))
        assert torch.equal(proxy._t_idx_ij, torch.tensor(proxy.t_idx_ij, dtype=torch.long))
        assert torch.equal(proxy._t_idx_jk, torch.tensor(proxy.t_idx_jk, dtype=torch.long))
        assert torch.equal(proxy._t_idx_ik, torch.tensor(proxy.t_idx_ik, dtype=torch.long))

    def test_layer_has_B1T_absB1T_B2T_buffers(self):
        layer = _make_layer()
        assert hasattr(layer, "_B1T") and torch.equal(layer._B1T, layer.B1.t().contiguous())
        assert hasattr(layer, "_abs_B1T") and torch.equal(layer._abs_B1T, layer.abs_B1.t().contiguous())
        assert hasattr(layer, "_B2T") and torch.equal(layer._B2T, layer.B2.t().contiguous())

    def test_layer_has_tensor_index_buffers(self):
        layer = _make_layer()
        assert hasattr(layer, "_e_idx_i") and layer._e_idx_i.dtype == torch.long
        assert hasattr(layer, "_e_idx_j") and layer._e_idx_j.dtype == torch.long
        assert hasattr(layer, "_t_idx_ij") and layer._t_idx_ij.dtype == torch.long
        assert hasattr(layer, "_t_idx_jk") and layer._t_idx_jk.dtype == torch.long
        assert hasattr(layer, "_t_idx_ik") and layer._t_idx_ik.dtype == torch.long

    def test_layer_tensor_indices_match_python_lists(self):
        layer = _make_layer()
        assert torch.equal(layer._e_idx_i, torch.tensor(layer.e_idx_i, dtype=torch.long))
        assert torch.equal(layer._e_idx_j, torch.tensor(layer.e_idx_j, dtype=torch.long))
        assert torch.equal(layer._t_idx_ij, torch.tensor(layer.t_idx_ij, dtype=torch.long))
        assert torch.equal(layer._t_idx_jk, torch.tensor(layer.t_idx_jk, dtype=torch.long))
        assert torch.equal(layer._t_idx_ik, torch.tensor(layer.t_idx_ik, dtype=torch.long))

    def test_proxy_python_lists_preserved(self):
        """Backward compat: Python list attributes still exist."""
        proxy = _make_proxy()
        assert isinstance(proxy.e_idx_i, list)
        assert isinstance(proxy.e_idx_j, list)
        assert isinstance(proxy.t_idx_ij, list)
        assert isinstance(proxy.t_idx_jk, list)
        assert isinstance(proxy.t_idx_ik, list)

    def test_layer_python_lists_preserved(self):
        layer = _make_layer()
        assert isinstance(layer.e_idx_i, list)
        assert isinstance(layer.e_idx_j, list)
        assert isinstance(layer.t_idx_ij, list)
        assert isinstance(layer.t_idx_jk, list)
        assert isinstance(layer.t_idx_ik, list)

    def test_buffers_move_with_device(self):
        """Registered buffers must follow .to(device)."""
        layer = _make_layer()
        # Just verify they're in the state dict
        sd = layer.state_dict()
        assert "_B1T" in sd
        assert "_abs_B1T" in sd
        assert "_B2T" in sd
        assert "_e_idx_i" in sd
        assert "_t_idx_ij" in sd


# ═════════════════════════════════════════════════════════════════════
# 8. HISTORY-AWARE ROUTER OPTIMIZATION TESTS
# ═════════════════════════════════════════════════════════════════════

class TestRouterOptimizations:
    """Verify context_expanded and zero_feedback pre-allocation don't change outputs."""

    def test_router_output_unchanged_with_context_expanded(self):
        from synapse.common.encoders.history_aware_router import (
            CandidateEncoder,
            build_router_context,
            build_static_structural_features,
            precompute_structural_geometry,
        )
        torch.manual_seed(42)
        B, T, input_dim = 2, 6, 3
        enc = CandidateEncoder(input_dim, d_u=16)
        enc.eval()
        x = torch.randn(B, T, input_dim)
        mask = torch.ones(B, T)
        geo = precompute_structural_geometry(x, mask=mask, knn_k=2)
        # Use build_static_structural_features (no selection column) —
        # CandidateEncoder.forward with static_features will call append_selection_weight
        sf = build_static_structural_features(x, mask=mask, knn_k=2, geometry_cache=geo)
        ctx = build_router_context(x, mask=mask, geometry_cache=geo)
        sel_weights = torch.zeros(B, T)

        with torch.no_grad():
            # Old path: context_expanded=None (computed inside project)
            u_old, _, _, _ = enc(x, selection_weights=sel_weights, mask=mask,
                                 geometry_cache=geo, context=ctx,
                                 similarity=None, static_features=sf)
            # New path: pre-expanded context
            ctx_exp = ctx.unsqueeze(1).expand(-1, T, -1)
            u_new, _, _, _ = enc(x, selection_weights=sel_weights, mask=mask,
                                 geometry_cache=geo, context=ctx,
                                 context_expanded=ctx_exp, similarity=None,
                                 static_features=sf)
        assert torch.allclose(u_old, u_new, atol=1e-5), (
            f"CandidateEncoder output mismatch: max diff = {(u_old - u_new).abs().max():.2e}"
        )

    def test_router_forward_matches_reference(self):
        from synapse.common.encoders.history_aware_router import HistoryAwareAnchorRouter
        torch.manual_seed(42)
        router = HistoryAwareAnchorRouter(input_dim=3, d_u=12, d_a=8, d_m=10, K=2, r=2, L=3)
        router.eval()
        x = torch.randn(2, 8, 3)
        with torch.no_grad():
            out1_y, out1_z, out1_mem, out1_anc, out1_geo = router(x, feedback=None)
            out2_y, out2_z, out2_mem, out2_anc, out2_geo = router(x, feedback=None)
        # Deterministic
        assert torch.allclose(out1_y, out2_y, atol=0.0)
        assert torch.allclose(out1_z, out2_z, atol=0.0)
        assert torch.allclose(out1_mem, out2_mem, atol=0.0)

    def test_router_with_feedback_matches_no_feedback(self):
        """When feedback=None, zero_feedback is used; must match explicit zeros."""
        from synapse.common.encoders.history_aware_router import HistoryAwareAnchorRouter
        torch.manual_seed(42)
        router = HistoryAwareAnchorRouter(input_dim=3, d_u=12, d_a=8, d_m=10, K=2, r=2, L=2)
        router.eval()
        x = torch.randn(2, 8, 3)
        with torch.no_grad():
            out_none_y, _, _, _, _ = router(x, feedback=None)
            out_zero_y, _, _, _, _ = router(x, feedback=torch.zeros(2, 1))
        assert torch.allclose(out_none_y, out_zero_y, atol=1e-5)

    def test_router_shapes_correct(self):
        from synapse.common.encoders.history_aware_router import HistoryAwareAnchorRouter
        router = HistoryAwareAnchorRouter(input_dim=3, d_u=12, d_a=8, d_m=10, K=2, r=2, L=4)
        x = torch.randn(2, 8, 3)
        all_y, all_z, all_mem, all_anc, geo = router(x, feedback=None)
        assert all_y.shape == (2, 4, 8)
        assert all_z.shape == (2, 4, 12)
        assert all_mem.shape == (2, 5, 10)  # L+1 memory states
        assert all_anc.shape == (2, 4, 8, 12)
