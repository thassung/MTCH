"""
Shared GNN models for link prediction.
Used by both benchmark and subgraph modules.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN(torch.nn.Module):
    """
    GCN encoder for link prediction.
    Stacks multiple GCNConv layers with ReLU activation and dropout.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class SAGE(torch.nn.Module):
    """
    GraphSAGE encoder for link prediction.
    Stacks multiple SAGEConv layers with ReLU activation and dropout.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(torch.nn.Module):
    """
    GAT encoder for link prediction.
    Stacks multiple GATConv layers with multi-head attention, ELU activation and dropout.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads=4):
        super(GAT, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                       heads=heads, dropout=dropout))
        # Last layer - single head output
        self.convs.append(GATConv(hidden_channels * heads, out_channels, 
                                   heads=1, concat=False, dropout=dropout))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class LinkPredictor(torch.nn.Module):
    """
    MLP-based link predictor.
    Takes element-wise product of node embeddings and passes through MLP.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()
        
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
    
    def forward(self, x_i, x_j):
        """
        Args:
            x_i: Source node embeddings [batch_size, in_channels]
            x_j: Target node embeddings [batch_size, in_channels]
        Returns:
            Link predictions [batch_size, out_channels]
        """
        x = x_i * x_j  # Element-wise product
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

