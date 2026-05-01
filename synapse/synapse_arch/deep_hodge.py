"""
Deep Hodge Transformer Architecture — Stacking Z3 Topological Blocks

This module implements the "Topological Attention" mechanism discussed.
Instead of extracting eigenvalues as a global readout (which cannot be stacked),
this architecture uses the differentiable Hodge Laplacians (Δ̂_0 and Δ̂_1) as
dynamic diffusion (routing) operators. 

It passes messages across both nodes (0-simplices) and edges (1-simplices),
allowing the network to dynamically build and utilize the topological structure
of the latent space layer by layer.
"""

import torch
import torch.nn as nn
import math

class DeepHodgeLayer(nn.Module):
    """A single layer of the Deep Hodge-Transformer.
    
    Analogue to a Transformer Encoder layer, but instead of Self-Attention,
    it uses Hodge-Laplacian Message Passing across 0-simplices and 1-simplices.
    """
    def __init__(
        self, 
        d_model: int, 
        k_dim: int = 16, # geometric lifting dimension
        num_scales: int = 3, 
        max_points: int = 16, 
        tau: float = 1e-4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.k_dim = k_dim
        self.num_scales = num_scales
        self.max_points = max_points
        self.tau = tau
        
        # 1. Geometric Lift: Project features to a dynamic geometry to compute topology
        self.geom_proj = nn.Linear(d_model, k_dim)
        self.log_scales = nn.Parameter(torch.linspace(-1.5, 1.5, num_scales))
        
        # 2. Routing Projections (Analogue to Q, K, V in attention)
        # Node values
        self.W_V0 = nn.Linear(d_model, d_model * num_scales)
        # Edge input projection
        self.W_E_in = nn.Linear(d_model, d_model * num_scales)
        # Edge back-to-node projection
        self.W_E_out = nn.Linear(d_model * num_scales, d_model)
        
        # 3. Output aggregation
        self.out_proj = nn.Linear(d_model * num_scales * 2, d_model)
        
        # 4. Standard FFN block
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Precompute boundary matrices
        self._build_boundary_matrices(max_points)

        # Pre-register identity matrices as buffers (avoids per-forward allocation)
        num_edges = max_points * (max_points - 1) // 2
        self.register_buffer("_eye_K", torch.eye(max_points))
        self.register_buffer("_eye_E", torch.eye(num_edges))

    def _build_boundary_matrices(self, K: int) -> None:
        """Precomputes fixed boundary matrices for the complete 2-skeleton."""
        edges = []
        for i in range(K):
            for j in range(i + 1, K):
                edges.append((i, j))

        triangles = []
        for i in range(K):
            for j in range(i + 1, K):
                for k in range(j + 1, K):
                    triangles.append((i, j, k))

        num_edges = len(edges)
        num_triangles = len(triangles)

        # B1 (K x E)
        B1 = torch.zeros(K, num_edges)
        for e_idx, (i, j) in enumerate(edges):
            B1[i, e_idx] = -1.0
            B1[j, e_idx] = 1.0

        # B2 (E x T)
        edge_to_idx = {e: idx for idx, e in enumerate(edges)}
        B2 = torch.zeros(num_edges, num_triangles)
        for t_idx, (i, j, k) in enumerate(triangles):
            B2[edge_to_idx[(j, k)], t_idx] = 1.0
            B2[edge_to_idx[(i, k)], t_idx] = -1.0
            B2[edge_to_idx[(i, j)], t_idx] = 1.0

        self.register_buffer("B1", B1)
        self.register_buffer("B2", B2)

        # Unoriented B1 for pulling edge features back to nodes without canceling out
        abs_B1 = B1.abs()
        self.register_buffer("abs_B1", abs_B1)

        # Patch B: Pre-register transposes as buffers (avoid repeated .t() calls)
        self.register_buffer("_B1T", B1.t().contiguous())
        self.register_buffer("_abs_B1T", abs_B1.t().contiguous())
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Node features (B, K, d_model)
            mask: Optional boolean mask (B, K)
        """
        B_batch, K_eff, d = x.shape
        assert K_eff <= self.max_points, f"Expected K<={self.max_points}, got {K_eff}"
        
        # Pad to max_points if necessary
        pad_len = self.max_points - K_eff
        if pad_len > 0:
            x_pad = torch.zeros(B_batch, pad_len, d, device=x.device, dtype=x.dtype)
            x = torch.cat([x, x_pad], dim=1)
            if mask is not None:
                m_pad = torch.zeros(B_batch, pad_len, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, m_pad], dim=1)
        
        if mask is None:
            mask = torch.ones(B_batch, self.max_points, device=x.device)
            
        residual = x
        x_norm = self.norm1(x)
        
        # 1. Project to geometric space to build topology
        P = self.geom_proj(x_norm) # (B, K, k_dim)
        
        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
        D = torch.cdist(P, P) * mask_2d
        
        scales = torch.exp(self.log_scales)
        
        # Node Values (B, K, S, d_model)
        V0 = self.W_V0(x_norm).view(B_batch, self.max_points, self.num_scales, d)
        
        # Create Edge Features: E_in = |B1|^T X W_E_in
        # Using abs_B1 so features sum from both endpoints
        # Patch B: use pre-registered _abs_B1T buffer
        E_in_raw = torch.matmul(self._abs_B1T, x_norm)
        E_in = self.W_E_in(E_in_raw).view(B_batch, self._abs_B1T.shape[0], self.num_scales, d)

        # ── Patch A: Vectorized scale loop ──
        # Compute all W1 and W2 for all scales at once, then batch bmm
        S = self.num_scales
        K = self.max_points
        E = self.B1.shape[1]
        T = self.B2.shape[1]

        # (B, S, K, K) affinity: compute Gaussian kernel for all scales simultaneously
        # scales: (S,) -> (1, S, 1, 1) for broadcasting with (B, K, K)
        sigma_sq = scales.square()  # (S,)
        # D: (B, K, K) -> (B, 1, K, K) for broadcast over S
        D_sq = D.square().unsqueeze(1)  # (B, 1, K, K)
        denom = (2.0 * sigma_sq.view(1, S, 1, 1) + 1e-8)  # (1, S, 1, 1)
        A_all = torch.exp(-D_sq / denom) * mask_2d.unsqueeze(1)  # (B, S, K, K)

        # Patch C: use tensor buffer indices for GPU-accelerated fancy indexing
        # W1: (B, S, E) — extract edge weights from full affinity
        W1_all = A_all[:, :, self._e_idx_i, self._e_idx_j]  # (B, S, E)

        # W2: (B, S, T) — triangle weights as product of 3 edge weights
        W2_all = W1_all[:, :, self._t_idx_ij] * W1_all[:, :, self._t_idx_jk] * W1_all[:, :, self._t_idx_ik]  # (B, S, T)

        # ── L0 for all scales: (B1 * W1) @ B1^T + τI ──
        # Reshape to (B*S, K, E) for batched bmm
        B1_u = self.B1.unsqueeze(0)  # (1, K, E)
        W1_flat = W1_all.reshape(B_batch * S, 1, E)  # (B*S, 1, E)
        B1_scaled = B1_u * W1_flat  # (B*S, K, E) — broadcast
        B1_exp = self.B1.unsqueeze(0).expand(B_batch * S, -1, -1)  # (B*S, K, E)
        L0_all = torch.bmm(B1_scaled, B1_exp.transpose(1, 2))  # (B*S, K, K)
        L0_all = L0_all + self.tau * self._eye_K.unsqueeze(0)
        L0_all = L0_all.reshape(B_batch, S, K, K)  # (B, S, K, K)

        # Node message passing: M0 = L0 @ V0 for each scale
        # V0: (B, K, S, d) -> (B, S, K, d) for batched matmul with L0: (B, S, K, K)
        V0_t = V0.transpose(1, 2)  # (B, S, K, d)
        L0_flat = L0_all.reshape(B_batch * S, K, K)
        V0_flat = V0_t.reshape(B_batch * S, K, d)
        M0_flat = torch.bmm(L0_flat, V0_flat)  # (B*S, K, d)
        M0_all = M0_flat.reshape(B_batch, S, K, d)  # (B, S, K, d)

        # ── Patch E: Hoist scale-invariant term_down out of loop ──
        # term_down = B1^T W0 B1 is the same for all scales (W0 = mask doesn't depend on σ)
        # Patch B: use pre-registered _B1T buffer
        W0 = mask  # (B, K)
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

        # Edge message passing: M1 = L1 @ E_in for each scale
        # E_in: (B, E, S, d) -> (B, S, E, d)
        E_in_t = E_in.transpose(1, 2)  # (B, S, E, d)
        L1_flat = L1_all.reshape(B_batch * S, E, E)
        E_in_flat = E_in_t.reshape(B_batch * S, E, d)
        M1_flat = torch.bmm(L1_flat, E_in_flat)  # (B*S, E, d)
        M1_all = M1_flat.reshape(B_batch, S, E, d)  # (B, S, E, d)

        # Pull edge features back to nodes using |B1|
        # Patch B: use pre-registered abs_B1 buffer (no .t() needed, abs_B1 is K×E)
        # (B, S, K, E) @ (B, S, E, d) -> (B, S, K, d)
        abs_B1_u = self.abs_B1.unsqueeze(0).unsqueeze(0)  # (1, 1, K, E)
        abs_B1_exp = abs_B1_u.expand(B_batch * S, -1, -1, -1).reshape(B_batch * S, K, E)
        M1_flat2 = M1_all.reshape(B_batch * S, E, d)
        M1_to_node_flat = torch.bmm(abs_B1_exp, M1_flat2)  # (B*S, K, d)
        M1_to_node = M1_to_node_flat.reshape(B_batch, S, K, d)  # (B, S, K, d)

        # 3. Concatenate and Project
        # M0_all: (B, S, K, d), M1_to_node: (B, S, K, d)
        # -> (B, K, S*d) each, then cat -> (B, K, 2*S*d)
        node_cat = M0_all.transpose(1, 2).reshape(B_batch, K, S * d)  # (B, K, S*d)
        edge_cat = M1_to_node.transpose(1, 2).reshape(B_batch, K, S * d)  # (B, K, S*d)
        all_updates = torch.cat([node_cat, edge_cat], dim=-1)  # (B, K, 2*S*d)
        
        attn_out = self.out_proj(all_updates)
        attn_out = self.dropout(attn_out)
        
        # 4. Residual + FFN
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        
        # Truncate back to original K_eff if padded
        if pad_len > 0:
            x = x[:, :K_eff, :]
            
        return x


class DeepHodgeTransformer(nn.Module):
    """A full stack of DeepHodgeLayers forming a Topological Transformer."""
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        k_dim: int = 16,
        num_scales: int = 3,
        max_points: int = 16
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DeepHodgeLayer(d_model, k_dim, num_scales, max_points)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Passes the anchor sequence through a deep stack of topological routing layers.
        At each layer, the geometry is dynamically rebuilt and messages are passed
        across vertices and loops.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)
