"""LCILP-style graph-level classifier for subgraph link prediction.

Scores a subgraph by pooling node representations and concatenating
head/tail embeddings — mirrors LCILP's GraphClassifier.

  score(u, v | subgraph) = MLP([mean_pool(h) || h_u || h_v])

Input features are DRNL structural labels concatenated with original node
features (when feature_dim > 0), matching the PS2 setup for citation networks.
Scores are raw unbounded logits — no sigmoid — for use with MarginRankingLoss.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from ..utils.models import GCN, SAGE, GAT


class SubgraphClassifier(nn.Module):
    """GNN encoder + graph-level MLP classifier.

    Parameters
    ----------
    drnl_dim : int
        DRNL feature dimension = 2*(max_dist+1).
    hidden : int
        Hidden and output dimension of the GNN layers.
    num_layers : int
        Number of GNN layers.
    dropout : float
    encoder_type : str
        'GCN', 'SAGE', or 'GAT'.
    feature_dim : int
        Original node feature dimension to concatenate with DRNL (0 = DRNL only).
    """

    def __init__(self, drnl_dim: int, hidden: int, num_layers: int,
                 dropout: float, encoder_type: str = "GCN", feature_dim: int = 0):
        super().__init__()

        in_channels = drnl_dim + feature_dim

        if encoder_type == "SAGE":
            self.encoder = SAGE(in_channels, hidden, hidden, num_layers, dropout)
        elif encoder_type == "GAT":
            self.encoder = GAT(in_channels, hidden, hidden, num_layers, dropout, heads=4)
        else:
            self.encoder = GCN(in_channels, hidden, hidden, num_layers, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for m in self.mlp:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch_vec: torch.Tensor, u_idx: torch.Tensor,
                v_idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [N, in_channels]  node features (DRNL or DRNL+content)
        edge_index : [2, E]            mega-batch edge index
        batch_vec  : [N]               batch assignment
        u_idx      : [B]               indices of u nodes in mega-batch
        v_idx      : [B]               indices of v nodes in mega-batch

        Returns
        -------
        scores : [B] unbounded logits (no sigmoid — use with MarginRankingLoss)
        """
        h = self.encoder(x, edge_index)               # [N, hidden]
        g = global_mean_pool(h, batch_vec)             # [B, hidden]
        hu = h[u_idx]                                  # [B, hidden]
        hv = h[v_idx]                                  # [B, hidden]
        logit = self.mlp(torch.cat([g, hu, hv], dim=1))  # [B, 1]
        return logit.squeeze(-1)                       # [B] unbounded
