"""
LPPR: PS2-style learnable PPR with soft PPR adjacency within subgraphs.

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
# Pluggable backbone encoders that take the LEARNED P_soft as adjacency.
# This is the apples-to-apples PS2-style framing: same backbone (GCN/SAGE/GAT)
# as the baselines, with the alpha mixture as the only changed variable.
# ---------------------------------------------------------------------------

class _DenseGCNLayer(nn.Module):
    """GCN-style: h_out = (P_soft @ h) W. P_soft already carries sym-norm + I."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=True)
    def reset_parameters(self):
        self.W.reset_parameters()
    def forward(self, x, P_soft):
        return self.W(P_soft @ x)


class _DenseSAGELayer(nn.Module):
    """SAGE-style: separate self and neighbor transforms, concatenated."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_self = nn.Linear(in_channels, out_channels, bias=True)
        self.W_neigh = nn.Linear(in_channels, out_channels, bias=False)
    def reset_parameters(self):
        self.W_self.reset_parameters()
        self.W_neigh.reset_parameters()
    def forward(self, x, P_soft):
        return self.W_self(x) + self.W_neigh(P_soft @ x)


class _DenseGATLayer(nn.Module):
    """
    GAT-style: P_soft acts as both an attention mask (where > 0 → eligible) and
    a soft edge weight that modulates attention magnitude. Single-head; cheap
    enough for small subgraphs.
    """
    def __init__(self, in_channels, out_channels, heads=4, leaky=0.2):
        super().__init__()
        assert out_channels % heads == 0, 'out_channels must be divisible by heads'
        self.heads = heads
        self.out_per_head = out_channels // heads
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.out_per_head))
        self.a_dst = nn.Parameter(torch.empty(heads, self.out_per_head))
        self.leaky = leaky
        self.reset_parameters()
    def reset_parameters(self):
        self.W.reset_parameters()
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    def forward(self, x, P_soft):
        n = x.size(0)
        h = self.W(x).view(n, self.heads, self.out_per_head)
        a_src = (h * self.a_src).sum(-1)               # [N, H]
        a_dst = (h * self.a_dst).sum(-1)               # [N, H]
        e = a_src.unsqueeze(1) + a_dst.unsqueeze(0)    # [N, N, H]
        e = F.leaky_relu(e, self.leaky)
        mask = (P_soft > 0)
        e = e.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        attn = F.softmax(e, dim=1)
        attn = attn * P_soft.unsqueeze(-1)             # weight by P_soft strength
        out = torch.einsum('ijh,jhf->ihf', attn, h)    # [N, H, F]
        return out.reshape(n, -1)                       # [N, H*F]


class _GenericDenseEncoder(nn.Module):
    """
    Stack of homogeneous layers. `layer_cls` is one of the dense layer classes
    above. P_soft is shared across all layers.
    """
    def __init__(self, layer_cls, in_channels, hidden_channels, out_channels,
                 num_layers, dropout, **layer_kwargs):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(layer_cls(in_channels, out_channels, **layer_kwargs))
        else:
            self.layers.append(layer_cls(in_channels, hidden_channels, **layer_kwargs))
            for _ in range(num_layers - 2):
                self.layers.append(layer_cls(hidden_channels, hidden_channels, **layer_kwargs))
            self.layers.append(layer_cls(hidden_channels, out_channels, **layer_kwargs))
    def reset_parameters(self):
        for l in self.layers: l.reset_parameters()
    def forward(self, x, P_soft):
        for layer in self.layers[:-1]:
            x = layer(x, P_soft)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, P_soft)


def make_lppr_encoder(kind, in_channels, hidden_channels, out_channels,
                      num_layers, dropout, gat_heads=4):
    """Factory: build the encoder backbone identified by `kind`."""
    kind = kind.upper()
    if kind == 'PPRDIFF':
        return PPRDiffusionEncoder(in_channels, hidden_channels, out_channels,
                                   num_layers, dropout)
    if kind == 'GCN':
        return _GenericDenseEncoder(_DenseGCNLayer, in_channels, hidden_channels,
                                    out_channels, num_layers, dropout)
    if kind == 'SAGE':
        return _GenericDenseEncoder(_DenseSAGELayer, in_channels, hidden_channels,
                                    out_channels, num_layers, dropout)
    if kind == 'GAT':
        return _GenericDenseEncoder(_DenseGATLayer, in_channels, hidden_channels,
                                    out_channels, num_layers, dropout, heads=gat_heads)
    raise ValueError(f"Unknown encoder kind: {kind!r}. "
                     "Use one of: GCN, SAGE, GAT, PPRDiff.")


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

class LPPRGNN(nn.Module):
    """
    PPR-diffusion GNN + link predictor for LPPR.

    This is the 'model' (w) in bi-level optimization.
    PPRScaleSelector is the 'arch_net' (theta).

    compute_loss() takes both as inputs, mirroring AutoLinkPPR.compute_loss().
    """

    def __init__(self, feat_dim, hidden_channels, num_layers, dropout,
                 alphas, selector_hidden=256, selector_layers=3,
                 encoder_type='GCN', gat_heads=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_channels = hidden_channels
        self.alphas = list(alphas)
        self.num_scales = len(alphas)
        self.dropout = dropout
        self.encoder_type = encoder_type

        # Pluggable backbone (PS2-style: same backbone as the baselines, only
        # the alpha mixture changes). Default to GCN to match the GCN baseline.
        self.encoder = make_lppr_encoder(
            encoder_type, feat_dim, hidden_channels, hidden_channels,
            num_layers, dropout, gat_heads=gat_heads)

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

        For each alpha k: cross[b,k,:] = (PPR_k[u_b] @ x) * (PPR_k[v_b] @ x),
        L2-normalized per (b, k).

        The L2-normalization kills the magnitude race between scales (α=0.9
        is near-one-hot ⇒ cross signal ≈ x[u]⊙x[v], large; α=0.15 averages
        ~50 nodes ⇒ cross is ~10× smaller per coord). Without normalization the
        selector picks the largest-magnitude scale at init and rich-get-richer
        locks it in (Liu et al., DARTS, ICLR'19). One-line fix; mandatory.

        Args:
            u_indices: [B] global indices
            v_indices: [B] global indices
            x_full:    [N, D] raw node features
            ppr_dense: {alpha: [N, N]}
            alphas:    ordered list of alpha values

        Returns:
            [B, num_scales, D] — each (b, k) row is unit L2 norm
        """
        device = x_full.device
        scale_reprs = []
        for alpha in alphas:
            mat = ppr_dense[alpha]
            if mat.device != device:
                mat = mat.to(device, non_blocking=True)
            z_u = mat[u_indices] @ x_full   # [B, D]
            z_v = mat[v_indices] @ x_full   # [B, D]
            cross = z_u * z_v
            cross = F.normalize(cross, p=2, dim=-1)  # unit L2 per (b, k)
            scale_reprs.append(cross)
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

    def _build_p_soft_full(self, weights, ppr_dense, num_nodes, device):
        """
        Full-graph soft PPR adjacency (sym-norm with self-loops, weighted mix).

        Per-scale:  P̂_α = D_α^(-1/2) (I + P_α) D_α^(-1/2)  (full N×N)
        Mixture:    P_full = Σ_k w_k · P̂_α_k

        Args:
            weights: [num_scales] single mixture (typically batch-mean)
            ppr_dense: {alpha: [N, N]}
            num_nodes: int N
            device: target device

        Returns:
            [N, N] full-graph propagator
        """
        N = num_nodes
        eye = torch.eye(N, device=device)
        P_full = torch.zeros(N, N, device=device)
        for i, alpha in enumerate(self.alphas):
            mat = ppr_dense[alpha]
            P_slice = mat.to(device) if mat.device != device else mat
            A_tilde = eye + P_slice
            d_inv_sqrt = A_tilde.sum(dim=1).clamp(min=1e-6).pow(-0.5)
            P_hat = d_inv_sqrt.unsqueeze(1) * A_tilde * d_inv_sqrt.unsqueeze(0)
            P_full = P_full + weights[i] * P_hat
        return P_full

    def forward_subgraph(self, nodes_S, u_local, v_local, x_full,
                         weights, ppr_dense):
        """
        GNN forward for one subgraph (legacy / ablation path). Kept so the
        old per-subgraph eval can still run if explicitly requested. The
        primary training path is forward_full_graph_batch.
        """
        device = x_full.device
        x_S = x_full[nodes_S.to(device)]
        P_soft = self._build_p_soft(nodes_S, weights, ppr_dense, device)
        h_S = self.encoder(x_S, P_soft)
        return h_S[u_local], h_S[v_local]

    def forward_full_graph_batch(self, edges, neg_edges, x_full,
                                 w_pos, w_neg, ppr_dense):
        """
        Full-graph forward for a batch of (pos, neg) edges.

        Uses batch-mean selector weights to build a single full-graph P_full,
        runs the encoder once over the entire graph, then indexes h_u / h_v
        for every edge in the batch.

        This is Sam's #2 fix: encoder now sees the full graph (not a tiny
        extracted subgraph), so it can't be starved of information by the
        score-threshold extractor. The selector still gets per-edge gradient
        via its contribution to the batch-mean mixture and via the entropy
        term — but the encoder's propagation is shared, which is what makes
        a single forward per batch tractable on Cora/CiteSeer/PubMed.

        Reference: Bojchevski et al., PPRGo (KDD'20) — full-graph PPR
        encoding with shared per-batch propagator.

        Returns:
            (pos_logits [B, 1], neg_logits [B, 1])
        """
        # Average weights across pos + neg for ONE shared propagator
        w_all = torch.cat([w_pos, w_neg], dim=0)  # [2B, num_scales]
        w_mean = w_all.mean(dim=0)                 # [num_scales]
        N = x_full.size(0)
        device = x_full.device
        P_full = self._build_p_soft_full(w_mean, ppr_dense, N, device)
        h = self.encoder(x_full, P_full)
        pos_logits = self.predict(h[edges[0]], h[edges[1]])
        neg_logits = self.predict(h[neg_edges[0]], h[neg_edges[1]])
        return pos_logits, neg_logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, selector, edges, neg_edges, x_full, ppr_dense,
                     entropy_coeff=0.0, return_aux=False):
        """
        Full-graph batch loss.

        Drops the subgraph-extraction code path entirely (Sam's #2). For each
        batch:
        1. Selector produces per-edge alpha weights w_pos, w_neg.
        2. Their batch mean defines a single full-graph P_full.
        3. Encoder runs ONCE over the entire graph.
        4. Predictor scores each (u,v) using the indexed h_u, h_v.
        5. Optional entropy bonus on the per-edge softmax keeps w from
           collapsing onto a single scale.
        """
        cross_pos = self.compute_selector_input(
            edges[0], edges[1], x_full, ppr_dense, self.alphas)
        cross_neg = self.compute_selector_input(
            neg_edges[0], neg_edges[1], x_full, ppr_dense, self.alphas)
        w_pos = selector(cross_pos)
        w_neg = selector(cross_neg)

        pos_logits, neg_logits = self.forward_full_graph_batch(
            edges, neg_edges, x_full, w_pos, w_neg, ppr_dense)

        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_targets = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits),
        ], dim=0)
        bce = F.binary_cross_entropy_with_logits(all_logits, all_targets)

        loss = bce
        ent = None
        if entropy_coeff != 0.0:
            w_all = torch.cat([w_pos, w_neg], dim=0)
            ent = -(w_all * (w_all + 1e-12).log()).sum(dim=-1).mean()
            loss = loss - entropy_coeff * ent

        if return_aux:
            return loss, {'bce': bce.detach(),
                          'entropy': ent.detach() if ent is not None else None,
                          'w_pos_mean': w_pos.detach().mean(dim=0),
                          'w_neg_mean': w_neg.detach().mean(dim=0)}
        return loss
