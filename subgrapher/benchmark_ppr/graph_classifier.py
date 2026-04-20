"""LCILP-style graph-level classifier for subgraph link prediction.

Scores a subgraph by pooling node representations and concatenating
head/tail embeddings — mirrors LCILP's GraphClassifier.

  score(u, v | subgraph) = sigmoid(MLP([mean_pool(h) || h_u || h_v]))

The input features are DRNL labels (from subgrapher.utils.drnl), not
random node features, so the model learns purely from subgraph structure.
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
        Input feature dimension = 2*(max_dist+1).
    hidden : int
        Hidden and output dimension of the GNN layers.
    num_layers : int
        Number of GNN layers.
    dropout : float
    encoder_type : str
        'GCN' or 'SAGE'.
    """

    def __init__(self, drnl_dim: int, hidden: int, num_layers: int,
                 dropout: float, encoder_type: str = "GCN"):
        super().__init__()

        if encoder_type == "SAGE":
            self.encoder = SAGE(drnl_dim, hidden, hidden, num_layers, dropout)
        elif encoder_type == "GAT":
            self.encoder = GAT(drnl_dim, hidden, hidden, num_layers, dropout, heads=4)
        else:
            self.encoder = GCN(drnl_dim, hidden, hidden, num_layers, dropout)

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
        x          : [N, drnl_dim]  DRNL node features for the mega-batch
        edge_index : [2, E]         mega-batch edge index
        batch_vec  : [N]            batch assignment (node i belongs to graph batch_vec[i])
        u_idx      : [B]            global indices of u nodes in the mega-batch
        v_idx      : [B]            global indices of v nodes in the mega-batch

        Returns
        -------
        scores : [B] sigmoid scores in (0, 1)
        """
        h = self.encoder(x, edge_index)               # [N, hidden]
        g = global_mean_pool(h, batch_vec)             # [B, hidden]
        hu = h[u_idx]                                  # [B, hidden]
        hv = h[v_idx]                                  # [B, hidden]
        logit = self.mlp(torch.cat([g, hu, hv], dim=1))  # [B, 1]
        return torch.sigmoid(logit.squeeze(-1))        # [B]
