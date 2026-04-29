"""
Option A: PS2-style learnable PPR with soft PPR adjacency within subgraphs.

Architecture:
  1. Extract subgraph S around (u,v) using score-threshold PPR  (extraction layer)
  2. Slice global PPR matrices at 3 alphas within S             (precomputed)
  3. PPRScaleSelector produces per-edge weights [w1,w2,w3]      (arch_net / theta)
  4. Soft adjacency: P_soft = sum_i w_i * P_alpha_i|_S          (differentiable)
  5. PPRDiffusionEncoder runs on S with P_soft                   (model / w)
  6. LinkPredictor on (h_u, h_v) -> score

Bi-level: theta (selector) on val loss, w (encoder+predictor) on train loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GNN encoder with dense PPR adjacency
# ---------------------------------------------------------------------------

class PPRDiffusionLayer(nn.Module):
    """h_out = ReLU(P_soft @ h @ W)  where P_soft is [N,N] row-normalized."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=True)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, P_soft):
        # x: [N, D_in], P_soft: [N, N]
        return self.W(P_soft @ x)


class PPRDiffusionEncoder(nn.Module):
    """Stack of PPR-diffusion layers sharing the same P_soft across all layers."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(PPRDiffusionLayer(in_channels, out_channels))
        else:
            self.layers.append(PPRDiffusionLayer(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(
                    PPRDiffusionLayer(hidden_channels, hidden_channels))
            self.layers.append(PPRDiffusionLayer(hidden_channels, out_channels))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, P_soft):
        for layer in self.layers[:-1]:
            x = layer(x, P_soft)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, P_soft)


# ---------------------------------------------------------------------------
# Selector (arch_net): produces per-edge alpha weights
# ---------------------------------------------------------------------------

class PPRScaleSelector(nn.Module):
    """
    Per-edge PPR scale selector (the architecture network / theta).

    Input:  cross-pair representations [B, num_scales, D]
            cross[b, k, :] = (PPR_alpha_k[u_b] @ x) * (PPR_alpha_k[v_b] @ x)
    Output: [B, num_scales] softmax weights

    During training: soft weights (temperature-annealed).
    During eval:     hard one-hot (argmax).
    """

    def __init__(self, in_channels, hidden_channels=256, num_layers=3,
                 num_scales=3, temperature=1.0, scale_emb_dim=32):
        super().__init__()
        self.temperature = temperature
        self.num_scales = num_scales
        self.scale_emb_dim = scale_emb_dim

        # Learnable per-scale embedding: gives the MLP a scale-identity signal
        # so it can express preferences like "always prefer alpha index 1" even
        # when cross-pair signals across scales are statistically similar.
        # Without this the alpha softmax tends to collapse onto whichever scale
        # has the largest cross-pair magnitude.
        self.scale_emb = nn.Parameter(
            torch.randn(num_scales, scale_emb_dim) * 0.02)

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Linear(in_channels + scale_emb_dim, hidden_channels, bias=False))
        for _ in range(num_layers - 2):
            self.layers.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.layers.append(nn.Linear(hidden_channels, 1, bias=False))

    def reset_parameters(self):
        nn.init.normal_(self.scale_emb, mean=0.0, std=0.02)
        for layer in self.layers:
            layer.reset_parameters()

    def set_temperature(self, temp):
        self.temperature = temp

    def _logits(self, cross_repr):
        """Forward through MLP → [B, num_scales] pre-softmax logits."""
        B, K, _ = cross_repr.shape
        scale_emb_b = self.scale_emb.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cross_repr, scale_emb_b], dim=-1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        return self.layers[-1](x).squeeze(-1)

    def forward(self, cross_repr):
        """
        Args:
            cross_repr: [B, num_scales, D]
        Returns:
            weights: [B, num_scales]
        """
        logits = self._logits(cross_repr)
        weights = torch.softmax(logits / self.temperature, dim=-1)

        if not self.training:
            indices = torch.argmax(weights, dim=-1)
            one_hot = torch.zeros_like(weights)
            one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
            weights = one_hot

        return weights

    def get_alpha_indices(self, cross_repr):
        """Hard argmax: which alpha index is selected per edge."""
        self.eval()
        with torch.no_grad():
            return torch.argmax(self._logits(cross_repr), dim=-1)


# ---------------------------------------------------------------------------
# Main GNN model (the w-parameters in bi-level)
# ---------------------------------------------------------------------------

class OptionAGNN(nn.Module):
    """
    PPR-diffusion GNN + link predictor for Option A.

    This is the 'model' (w) in bi-level optimization.
    PPRScaleSelector is the 'arch_net' (theta).

    compute_loss() takes both as inputs, mirroring AutoLinkPPR.compute_loss().
    """

    def __init__(self, feat_dim, hidden_channels, num_layers, dropout,
                 alphas, selector_hidden=256, selector_layers=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_channels = hidden_channels
        self.alphas = list(alphas)
        self.num_scales = len(alphas)
        self.dropout = dropout

        self.encoder = PPRDiffusionEncoder(
            feat_dim, hidden_channels, hidden_channels, num_layers, dropout)

        self.pre_mlp_norm = nn.LayerNorm(hidden_channels)

        self.predictor_lins = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Linear(hidden_channels, 1),
        ])

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.pre_mlp_norm.reset_parameters()
        for lin in self.predictor_lins:
            lin.reset_parameters()

    def predict(self, h_u, h_v):
        """Link predictor on paired embeddings — returns LOGITS.

        Sigmoid is applied at loss time (BCEWithLogitsLoss) for numerical
        stability. Eval ranking metrics (MRR/Hits/AUC/AP) are rank-based and
        unaffected by the monotonic sigmoid removal.
        """
        x = h_u * h_v
        x = self.pre_mlp_norm(x)
        for lin in self.predictor_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.predictor_lins[-1](x)

    # ------------------------------------------------------------------
    # Core PPR utilities
    # ------------------------------------------------------------------

    @staticmethod
    def compute_selector_input(u_indices, v_indices, x_full, ppr_dense, alphas):
        """
        Cross-pair PPR-diffused features for the selector.

        For each alpha k: cross[b,k,:] = (PPR_k[u_b] @ x) * (PPR_k[v_b] @ x)

        Args:
            u_indices: [B] global indices
            v_indices: [B] global indices
            x_full:    [N, D] raw node features
            ppr_dense: {alpha: [N, N]}
            alphas:    ordered list of alpha values

        Returns:
            [B, num_scales, D]
        """
        device = x_full.device
        scale_reprs = []
        for alpha in alphas:
            mat = ppr_dense[alpha]
            if mat.device != device:
                mat = mat.to(device, non_blocking=True)
            z_u = mat[u_indices] @ x_full   # [B, D]
            z_v = mat[v_indices] @ x_full   # [B, D]
            scale_reprs.append(z_u * z_v)
        return torch.stack(scale_reprs, dim=1)  # [B, num_scales, D]

    def _build_p_soft(self, nodes_S, weights, ppr_dense, device):
        """
        Soft PPR adjacency for one subgraph (sym-norm with explicit self-loops).

        Per-scale:  P̂_α = D_α^(-1/2) (I + P_α|_S) D_α^(-1/2)
        Mixture:    P_soft = Σ_k w_k · P̂_α_k

        Adding I before normalization fixes the all-zero-rows pathology that
        arose when score-threshold extraction at the widest alpha left
        local-alpha slices with empty rows for peripheral nodes — those rows
        would become uniform noise after row-normalization, drowning the
        selector signal. Sym-norm with self-loops is the GDC / APPNP / S²GC
        recipe (NeurIPS'19, ICLR'19, ICLR'21).

        Args:
            nodes_S: [|S|] global node indices (cpu or device)
            weights: [num_scales] selector output for this edge
            ppr_dense: {alpha: [N, N]}
            device: target device

        Returns:
            [|S|, |S|]
        """
        n = len(nodes_S)
        nodes_cpu = nodes_S.cpu()
        eye = torch.eye(n, device=device)
        P_soft = torch.zeros(n, n, device=device)
        for i, alpha in enumerate(self.alphas):
            mat = ppr_dense[alpha]
            P_slice = mat[nodes_cpu][:, nodes_cpu]
            if P_slice.device != device:
                P_slice = P_slice.to(device, non_blocking=True)
            A_tilde = eye + P_slice
            d_inv_sqrt = A_tilde.sum(dim=1).clamp(min=1e-6).pow(-0.5)
            P_hat = d_inv_sqrt.unsqueeze(1) * A_tilde * d_inv_sqrt.unsqueeze(0)
            P_soft = P_soft + weights[i] * P_hat
        return P_soft

    def forward_subgraph(self, nodes_S, u_local, v_local, x_full,
                         weights, ppr_dense):
        """
        GNN forward for one subgraph.

        Args:
            nodes_S: [|S|] global node indices
            u_local, v_local: int, local indices within nodes_S
            x_full: [N, D]
            weights: [num_scales] selector output for this edge
            ppr_dense: {alpha: [N, N]}

        Returns:
            h_u: [D],  h_v: [D]
        """
        device = x_full.device
        x_S = x_full[nodes_S.to(device)]
        P_soft = self._build_p_soft(nodes_S, weights, ppr_dense, device)
        h_S = self.encoder(x_S, P_soft)
        return h_S[u_local], h_S[v_local]

    # ------------------------------------------------------------------
    # Loss (called by bi-level searcher and fine-tuner)
    # ------------------------------------------------------------------

    def compute_loss(self, selector, edges, neg_edges, x_full, ppr_dense,
                     pos_subgraphs, neg_subgraphs):
        """
        Full loss for a batch using subgraph-level forward passes.

        Args:
            selector: PPRScaleSelector instance
            edges: [2, B]
            neg_edges: [2, B]
            x_full: [N, D]
            ppr_dense: {alpha: [N, N]}
            pos_subgraphs: list of (nodes_S, u_local, v_local) len=B
            neg_subgraphs: list of (nodes_S, u_local, v_local) len=B

        Returns:
            loss: scalar
        """
        B = edges.size(1)
        device = x_full.device

        cross_pos = self.compute_selector_input(
            edges[0], edges[1], x_full, ppr_dense, self.alphas)
        cross_neg = self.compute_selector_input(
            neg_edges[0], neg_edges[1], x_full, ppr_dense, self.alphas)

        w_pos = selector(cross_pos)  # [B, num_scales]
        w_neg = selector(cross_neg)

        pos_preds, neg_preds = [], []
        for i in range(B):
            nodes_S, u_loc, v_loc = pos_subgraphs[i]
            h_u, h_v = self.forward_subgraph(
                nodes_S, u_loc, v_loc, x_full, w_pos[i], ppr_dense)
            pos_preds.append(self.predict(h_u.unsqueeze(0), h_v.unsqueeze(0)))

        for i in range(B):
            nodes_S, u_loc, v_loc = neg_subgraphs[i]
            h_u, h_v = self.forward_subgraph(
                nodes_S, u_loc, v_loc, x_full, w_neg[i], ppr_dense)
            neg_preds.append(self.predict(h_u.unsqueeze(0), h_v.unsqueeze(0)))

        pos_logits = torch.cat(pos_preds)  # [B, 1] — logits from predict()
        neg_logits = torch.cat(neg_preds)

        # BCEWithLogitsLoss: numerically stable replacement for the previous
        # sigmoid + manual -log(p) - log(1-p) form.
        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_targets = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits),
        ], dim=0)
        return F.binary_cross_entropy_with_logits(all_logits, all_targets)
