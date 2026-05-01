"""
Differentiable Spectral Proxy — Z3 Reference: §12 of 01_main_definition.md

Implements the regularized weighted Hodge-spectral proxy:

    Φ^proxy_Θ(x_{1:T}) = Concat(
        {λ_j(Δ̂_{0,Θ}^(s))}_{s=1,j=1}^{S,J},
        {λ_j(Δ̂_{1,Θ}^(s))}_{s=1,j=1}^{S,J},
        mean_dist, max_dist, var_dist, compactness
    )

where:
    Δ̂_0^(s) = B1 W1^(s) B1^T + τ I_K        (vertex Laplacian, detects β_0)
    Δ̂_1^(s) = B1^T W0 B1 + B2 W2^(s) B2^T + τ I_E  (edge Laplacian, detects β_1)

The proxy feature dimension is fixed at 2*S*J + 4.
"""

from __future__ import annotations

import torch
from torch import nn


class DifferentiableHodgeProxy(nn.Module):
    """Full Hodge-spectral proxy for differentiable topology-aware training.

    Implements both L0 (connected components, β_0) and L1 (loops/holes, β_1)
    Hodge Laplacians on a fixed simplicial support, following the Z3 formulation.

    Parameters
    ----------
    lift_dim : int
        Dimensionality of the lifted point cloud (k in the formal definition).
    hidden_dim : int
        Output projection dimension (d_model).
    num_scales : int
        Number of learnable Gaussian kernel scales (S in formal definition).
    num_eigs : int
        Number of eigenvalues retained per operator per scale (J in formal definition).
    max_points : int
        Maximum number of vertices in the simplicial complex (MAX_K).
        Controls the size of precomputed boundary matrices.
    tau : float
        Ridge regularization parameter (τ > 0). Ensures positive definiteness
        but destroys exact Betti-number interpretation (Proposition 7.1 of
        02_rigorous_architecture.md).
    """

    def __init__(
        self,
        lift_dim: int,
        hidden_dim: int,
        num_scales: int = 3,
        num_eigs: int = 4,
        max_points: int = 16,
        tau: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.num_eigs = num_eigs
        self.max_points = max_points
        self.tau = tau

        # Learnable log-scales: σ_s = exp(log_scales[s])
        self.log_scales = nn.Parameter(torch.linspace(-1.5, 1.5, num_scales))

        # Feature dimension: S*J eigenvalues from L0 + S*J from L1 + 4 global stats
        feature_dim = 2 * num_scales * num_eigs + 4
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Precompute combinatorial boundary matrices for the complete 2-skeleton
        self._build_boundary_matrices(max_points)

        # Pre-register identity matrices as buffers (avoids per-forward allocation)
        num_edges = max_points * (max_points - 1) // 2
        self.register_buffer("_eye_K", torch.eye(max_points))
        self.register_buffer("_eye_E", torch.eye(num_edges))

    def _build_boundary_matrices(self, K: int) -> None:
        """Precompute B1 (vertex→edge) and B2 (edge→triangle) incidence matrices.

        These are fixed combinatorial operators for the complete 2-skeleton
        on K vertices. They do NOT depend on learned parameters.

        B1 ∈ ℝ^{K × |E|}  where |E| = K(K-1)/2
        B2 ∈ ℝ^{|E| × |T|}  where |T| = K(K-1)(K-2)/6
        """
        # Enumerate all edges (i, j) with i < j
        edges = []
        for i in range(K):
            for j in range(i + 1, K):
                edges.append((i, j))

        # Enumerate all triangles (i, j, k) with i < j < k
        triangles = []
        for i in range(K):
            for j in range(i + 1, K):
                for k in range(j + 1, K):
                    triangles.append((i, j, k))

        num_edges = len(edges)
        num_triangles = len(triangles)

        # B1: oriented vertex-edge incidence matrix
        # Convention: B1[v, e] = -1 if v is the source of edge e, +1 if target
        B1 = torch.zeros(K, num_edges)
        for e_idx, (i, j) in enumerate(edges):
            B1[i, e_idx] = -1.0
            B1[j, e_idx] = 1.0

        # B2: oriented edge-triangle incidence matrix
        # Convention: For triangle (i,j,k), the boundary is (j,k) - (i,k) + (i,j)
        edge_to_idx = {e: idx for idx, e in enumerate(edges)}
        B2 = torch.zeros(num_edges, num_triangles)
        for t_idx, (i, j, k) in enumerate(triangles):
            B2[edge_to_idx[(j, k)], t_idx] = 1.0
            B2[edge_to_idx[(i, k)], t_idx] = -1.0
            B2[edge_to_idx[(i, j)], t_idx] = 1.0

        self.register_buffer("B1", B1)
        self.register_buffer("B2", B2)

        # Patch B: Pre-register transposes as buffers (avoid repeated .t() calls)
        self.register_buffer("_B1T", B1.t().contiguous())
        self.register_buffer("_B2T", B2.t().contiguous())

        # Patch C: Convert Python list indices to tensor buffers for GPU-accelerated indexing
        self.register_buffer("_e_idx_i", torch.tensor([e[0] for e in edges], dtype=torch.long))
        self.register_buffer("_e_idx_j", torch.tensor([e[1] for e in edges], dtype=torch.long))
        self.register_buffer("_t_idx_ij", torch.tensor([edge_to_idx[(t[0], t[1])] for t in triangles], dtype=torch.long))
        self.register_buffer("_t_idx_jk", torch.tensor([edge_to_idx[(t[1], t[2])] for t in triangles], dtype=torch.long))
        self.register_buffer("_t_idx_ik", torch.tensor([edge_to_idx[(t[0], t[2])] for t in triangles], dtype=torch.long))

        # Keep Python lists for backward compatibility (tests reference them)
        self.e_idx_i = [e[0] for e in edges]
        self.e_idx_j = [e[1] for e in edges]
        self.t_idx_ij = [edge_to_idx[(t[0], t[1])] for t in triangles]
        self.t_idx_jk = [edge_to_idx[(t[1], t[2])] for t in triangles]
        self.t_idx_ik = [edge_to_idx[(t[0], t[2])] for t in triangles]

    @property
    def feature_dim(self) -> int:
        """Raw spectral feature dimension before projection: 2*S*J + 4."""
        return 2 * self.num_scales * self.num_eigs + 4

    def forward(self, dense_cloud: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
        """Compute the differentiable Hodge-spectral proxy features.

        Parameters
        ----------
        dense_cloud : (B, N, k) — lifted point cloud P̃_Θ
        y_star : (B, N) — continuous proxy weights ỹ_t

        Returns
        -------
        (B, hidden_dim) — projected spectral features
        """
        raw = self._hodge_spectrum(dense_cloud.float(), y_star.float())
        return self.proj(raw).to(dense_cloud.dtype)

    def _hodge_spectrum(self, points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Core Hodge-spectral computation following Z3 §12.

        1. Gather top-K activated points (implementation optimization)
        2. For each scale σ_s:
           a. Compute edge weights W1^(s) via Gaussian kernel
           b. Compute triangle weights W2^(s) as product of 3 edge weights
           c. Build Δ̂_0^(s) = B1 W1 B1^T + τI  (vertex Laplacian)
           d. Build Δ̂_1^(s) = B1^T W0 B1 + B2 W2 B2^T + τI  (edge Laplacian)
           e. Extract first J eigenvalues of each
        3. Append global distance statistics
        """
        B_batch, N, k_dim = points.shape
        K_eff = min(N, self.max_points)

        # ── Gather top-K_eff activated points ──
        _, top_idx = torch.topk(weights, K_eff, dim=1, sorted=False)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, k_dim)
        cloud = torch.gather(points, 1, top_idx_exp)      # (B, K_eff, k)
        act = torch.gather(weights, 1, top_idx)            # (B, K_eff)

        # Zero-pad to MAX_K so boundary matrices align
        if K_eff < self.max_points:
            pad_pts = torch.zeros(B_batch, self.max_points - K_eff, k_dim, device=cloud.device)
            pad_act = torch.zeros(B_batch, self.max_points - K_eff, device=act.device)
            cloud = torch.cat([cloud, pad_pts], dim=1)
            act = torch.cat([act, pad_act], dim=1)

        # Activation mask for valid vertices
        mask = (act > 1e-3).float()
        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)    # (B, K, K)

        # Pairwise distances (masked)
        D = torch.cdist(cloud, cloud) * mask_2d            # (B, K, K)

        scales = torch.exp(self.log_scales.float())
        num_eigs = self.num_eigs

        # W0: vertex weights = activations (§12: w_u^(0) = ỹ_u)
        W0 = act                                            # (B, K)

        # ── Patch A: Vectorized scale loop ──
        S = self.num_scales
        K = self.max_points
        E = self.B1.shape[1]
        T = self.B2.shape[1]

        # Compute all W1 and W2 for all scales at once
        sigma_sq = scales.square()  # (S,)
        D_sq = D.square().unsqueeze(1)  # (B, 1, K, K)
        denom = (2.0 * sigma_sq.view(1, S, 1, 1) + 1e-8)  # (1, S, 1, 1)
        A_all = torch.exp(-D_sq / denom) * mask_2d.unsqueeze(1)  # (B, S, K, K)

        # Patch C: use tensor buffer indices for GPU-accelerated fancy indexing
        W1_all = A_all[:, :, self._e_idx_i, self._e_idx_j]  # (B, S, E)
        W2_all = W1_all[:, :, self._t_idx_ij] * W1_all[:, :, self._t_idx_jk] * W1_all[:, :, self._t_idx_ik]  # (B, S, T)

        # ── L0 for all scales: (B1 * W1) @ B1^T + τI ──
        B1_u = self.B1.unsqueeze(0)  # (1, K, E)
        W1_flat = W1_all.reshape(B_batch * S, 1, E)  # (B*S, 1, E)
        B1_scaled = B1_u * W1_flat  # (B*S, K, E) — broadcast
        B1_exp = self.B1.unsqueeze(0).expand(B_batch * S, -1, -1)  # (B*S, K, E)
        L0_all = torch.bmm(B1_scaled, B1_exp.transpose(1, 2))  # (B*S, K, K)
        L0_all = L0_all + self.tau * self._eye_K.unsqueeze(0)
        L0_all = L0_all.reshape(B_batch, S, K, K)  # (B, S, K, K)

        # Batched eigvalsh over (B, S) — reshape to (B*S, K, K)
        eigvals_L0 = torch.linalg.eigvalsh(L0_all.reshape(B_batch * S, K, K))  # (B*S, K)
        eigvals_L0 = eigvals_L0.reshape(B_batch, S, K)  # (B, S, K)

        # ── Patch E: Hoist scale-invariant term_down out of loop ──
        # term_down = B1^T W0 B1 is the same for all scales (W0 doesn't depend on σ)
        # Patch B: use pre-registered _B1T buffer
        BT_scaled = self._B1T.unsqueeze(0) * W0.unsqueeze(1)  # (B, E, K)
        B1_exp_b = self.B1.unsqueeze(0).expand(B_batch, -1, -1)  # (B, K, E)
        term_down = torch.bmm(BT_scaled, B1_exp_b)  # (B, E, E)

        # term_up = B2 W2 B2^T varies per scale — batch over scales
        B2_u = self.B2.unsqueeze(0)  # (1, E, T)
        W2_flat = W2_all.reshape(B_batch * S, 1, T)  # (B*S, 1, T)
        B2_scaled = B2_u * W2_flat  # (B*S, E, T)
        # Patch B: use pre-registered _B2T buffer
        B2_exp = self._B2T.unsqueeze(0).expand(B_batch * S, -1, -1)  # (B*S, T, E)
        term_up = torch.bmm(B2_scaled, B2_exp)  # (B*S, E, E)

        # L1 = term_down + term_up + τI  (term_down is broadcast over S)
        L1_all = term_down.unsqueeze(1).expand(-1, S, -1, -1).reshape(B_batch * S, E, E) + term_up
        L1_all = L1_all + self.tau * self._eye_E.unsqueeze(0)
        L1_all = L1_all.reshape(B_batch, S, E, E)  # (B, S, E, E)

        # Batched eigvalsh over (B, S) — reshape to (B*S, E, E)
        eigvals_L1 = torch.linalg.eigvalsh(L1_all.reshape(B_batch * S, E, E))  # (B*S, E)
        eigvals_L1 = eigvals_L1.reshape(B_batch, S, E)  # (B, S, E)

        # Collect eigenvalue features
        features = []
        for s_idx in range(S):
            eig0 = eigvals_L0[:, s_idx, :num_eigs]
            if eig0.shape[1] < num_eigs:
                eig0 = torch.nn.functional.pad(eig0, (0, num_eigs - eig0.shape[1]))

            eig1 = eigvals_L1[:, s_idx, :num_eigs]
            if eig1.shape[1] < num_eigs:
                eig1 = torch.nn.functional.pad(eig1, (0, num_eigs - eig1.shape[1]))

            features.append(eig0)
            features.append(eig1)

        # ── Global distance statistics ──
        tri_mask = torch.triu(mask_2d, diagonal=1)
        tri_sum = tri_mask.sum(dim=(1, 2)).clamp_min(1.0)
        mean_dist = (D * tri_mask).sum(dim=(1, 2)) / tri_sum
        max_dist = (D * tri_mask).amax(dim=(1, 2))
        dist_var = ((D - mean_dist[:, None, None]).square() * tri_mask).sum(dim=(1, 2)) / tri_sum
        compactness = mean_dist / (max_dist + 1e-6)

        features.append(torch.stack([mean_dist, max_dist, dist_var, compactness], dim=-1))

        raw_features = torch.cat(features, dim=-1)          # (B, 2*S*J + 4)
        return raw_features
