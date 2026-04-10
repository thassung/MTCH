"""
AutoLink model adapted for PPR-based architecture search.
Instead of selecting GNN layers, selects PPR teleport scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.models import GCN, SAGE, GAT


class AutoLinkPPR(nn.Module):
    """
    GNN encoder + PPR cross-pair link predictor for architecture search.

    The GNN runs on the full graph once. For each edge (u,v), cross-pair
    representations are formed from PPR-weighted aggregations at different
    teleport scales, and the arch_net selects among them.

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension for GNN and predictor
        num_layers: Number of GNN layers
        dropout: Dropout rate
        gnn_type: 'GCN', 'SAGE', or 'GAT'
        num_configs: Number of PPR configurations (num_scales^2)
        lin_layers: Number of layers in the link predictor MLP
        cat_type: 'multi' (element-wise product) or 'concat'
    """

    def __init__(self, in_channels, hidden_channels, num_layers, dropout,
                 gnn_type='SAGE', num_configs=9, lin_layers=3,
                 cat_type='multi'):
        super().__init__()

        self.gnn_type = gnn_type
        self.hidden_channels = hidden_channels
        self.num_configs = num_configs
        self.cat_type = cat_type

        if gnn_type == 'GCN':
            self.encoder = GCN(in_channels, hidden_channels, hidden_channels,
                               num_layers, dropout)
        elif gnn_type == 'SAGE':
            self.encoder = SAGE(in_channels, hidden_channels, hidden_channels,
                                num_layers, dropout)
        elif gnn_type == 'GAT':
            self.encoder = GAT(in_channels, hidden_channels, hidden_channels,
                               num_layers, dropout, heads=4)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        if cat_type == 'multi':
            pred_in = hidden_channels
        else:
            pred_in = hidden_channels * 2

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(pred_in, hidden_channels))
        for _ in range(lin_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, 1))

        self.dropout = dropout

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, edge_index):
        """Run GNN encoder on full graph -> node embeddings [N, D]."""
        return self.encoder(x, edge_index)

    def pred_pair(self, x):
        """MLP link predictor on combined representation."""
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def compute_loss(self, h, arch_net, multi_scale_ppr, edges, neg_edges):
        """
        Compute link prediction loss with architecture-selected PPR configs.

        Args:
            h: Node embeddings [N, D] from forward()
            arch_net: PPRSearchNet instance
            multi_scale_ppr: MultiScalePPR instance
            edges: [2, B] positive edges
            neg_edges: [2, B] negative edges

        Returns:
            loss: Scalar loss
        """
        pos_out = self._compute_pred_with_arch(h, arch_net, multi_scale_ppr,
                                               edges)
        neg_out = self._compute_pred_with_arch(h, arch_net, multi_scale_ppr,
                                               neg_edges)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        return pos_loss + neg_loss

    def compute_loss_with_arch(self, h, pos_atten, neg_atten, edges,
                               neg_edges, cross_pairs_pos, cross_pairs_neg):
        """
        Compute loss using pre-computed attention weights (for fine-tuning phase
        where arch_net is frozen and provides fixed attention).

        Args:
            h: Not used directly (cross-pairs already computed)
            pos_atten: [B, num_configs] attention for positive edges
            neg_atten: [B, num_configs] attention for negative edges
            edges: [2, B] positive edges
            neg_edges: [2, B] negative edges
            cross_pairs_pos: [B, num_configs, D] for positive edges
            cross_pairs_neg: [B, num_configs, D] for negative edges
        """
        pos_out = self._apply_attention_and_predict(cross_pairs_pos,
                                                    pos_atten)
        neg_out = self._apply_attention_and_predict(cross_pairs_neg,
                                                    neg_atten)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        return pos_loss + neg_loss

    def _compute_pred_with_arch(self, h, arch_net, multi_scale_ppr, edges):
        """Compute predictions using arch_net for configuration selection."""
        sources = edges[0]
        targets = edges[1]

        cross_pairs = multi_scale_ppr.get_ppr_cross_pair_batch(
            sources, targets, h)  # [B, C, D]

        atten = arch_net(cross_pairs)  # [B, C]

        return self._apply_attention_and_predict(cross_pairs, atten)

    def _apply_attention_and_predict(self, cross_pairs, atten):
        """Apply attention weights to cross-pairs and run predictor."""
        B, C, D = cross_pairs.shape
        weighted = cross_pairs * atten.view(B, C, 1)  # [B, C, D]
        combined = weighted.sum(dim=1)  # [B, D]
        return self.pred_pair(combined)

    def compute_pred(self, h, arch_net, multi_scale_ppr, edges):
        """Compute predictions (for evaluation)."""
        return self._compute_pred_with_arch(h, arch_net, multi_scale_ppr,
                                            edges)

    def compute_arch_attention(self, h, arch_net, multi_scale_ppr, edges):
        """
        Get attention weights from arch_net for given edges.
        Used to extract learned per-edge configurations.

        Returns:
            atten: [B, num_configs] attention weights
            cross_pairs: [B, num_configs, D]
        """
        sources = edges[0]
        targets = edges[1]
        cross_pairs = multi_scale_ppr.get_ppr_cross_pair_batch(
            sources, targets, h)
        atten = arch_net(cross_pairs)
        return atten, cross_pairs
