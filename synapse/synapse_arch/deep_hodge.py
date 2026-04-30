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
        self.register_buffer("abs_B1", B1.abs())

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
        # (B, E, K) @ (B, K, d*S) -> (B, E, d*S)
        E_in_raw = torch.matmul(self.abs_B1.t(), x_norm) 
        E_in = self.W_E_in(E_in_raw).view(B_batch, self.abs_B1.shape[1], self.num_scales, d)
        
        node_updates = []
        edge_updates = []
        
        # 2. Build Hodge Laplacians and perform topological message passing
        for s_idx in range(self.num_scales):
            sigma = scales[s_idx]
            
            # Affinity based on distance
            A = torch.exp(-D.square() / (2.0 * sigma.square() + 1e-8)) * mask_2d
            W1 = A[:, self.e_idx_i, self.e_idx_j] # (B, E)
            W2 = W1[:, self.t_idx_ij] * W1[:, self.t_idx_jk] * W1[:, self.t_idx_ik] # (B, T)
            W0 = mask # (B, K)
            
            # --- Node Message Passing (0-Simplex) via Δ̂_0 ---
            # B2: (B1 * W1) @ B1.T avoids materializing the full diagonal matrix
            B1_scaled = self.B1.unsqueeze(0) * W1.unsqueeze(1)  # (B, K, E)
            B1_exp = self.B1.unsqueeze(0).expand(B_batch, -1, -1)  # (B, K, E)
            L0 = torch.bmm(B1_scaled, B1_exp.transpose(1, 2))  # (B, K, K)
            L0 = L0 + self.tau * self._eye_K.unsqueeze(0)  # B4: buffer
            
            # Diffusion: Apply L0 to V0. Note: Since L0 is a Laplacian, we actually want
            # (I - L0) or simply L0 directly to route features based on connectivity.
            # We use L0 directly as the routing matrix.
            # (B, K, K) @ (B, K, d_model) -> (B, K, d_model)
            V0_s = V0[:, :, s_idx, :]
            M0 = torch.bmm(L0, V0_s) 
            node_updates.append(M0)
            
            # --- Edge Message Passing (1-Simplex) via Δ̂_1 ---
            # B2: element-wise scaling for both terms
            BT_scaled = self.B1.t().unsqueeze(0) * W0.unsqueeze(1)  # (B, E, K)
            B1_exp = self.B1.unsqueeze(0).expand(B_batch, -1, -1)  # (B, K, E)
            term_down = torch.bmm(BT_scaled, B1_exp)  # (B, E, E)

            B2_scaled = self.B2.unsqueeze(0) * W2.unsqueeze(1)  # (B, E, T)
            B2_exp = self.B2.t().unsqueeze(0).expand(B_batch, -1, -1)  # (B, T, E)
            term_up = torch.bmm(B2_scaled, B2_exp)  # (B, E, E)

            L1 = term_down + term_up
            L1 = L1 + self.tau * self._eye_E.unsqueeze(0)  # B4: buffer
            
            # Route edge features through adjacent edges/triangles
            E_in_s = E_in[:, :, s_idx, :]
            M1 = torch.bmm(L1, E_in_s) # (B, E, d_model)
            
            # Pull edge features back to nodes using |B1|
            # (B, K, E) @ (B, E, d_model) -> (B, K, d_model)
            M1_to_node = torch.bmm(self.abs_B1.unsqueeze(0).expand(B_batch, -1, -1), M1)
            edge_updates.append(M1_to_node)
            
        # 3. Concatenate and Project
        # node_updates: S tensors of (B, K, d) -> (B, K, S*d)
        # edge_updates: S tensors of (B, K, d) -> (B, K, S*d)
        all_updates = torch.cat(node_updates + edge_updates, dim=-1) # (B, K, 2*S*d)
        
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
