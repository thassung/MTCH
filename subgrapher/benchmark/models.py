"""
Link prediction models following PS2 architecture.
Separate GNN encoder and MLP-based LinkPredictor.
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
    MLP-based link predictor following PS2 architecture.
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


class LinkPredictionModel(torch.nn.Module):
    """
    Wrapper combining encoder + predictor with plug-and-play subgrapher interface.
    This enables seamless integration of subgraph extraction methods in the future.
    """
    def __init__(self, encoder, predictor, device='cpu'):
        super(LinkPredictionModel, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.device = device
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()
    
    def forward(self, x, edge_index, edge_label_index):
        """
        Standard forward pass without subgraphing.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_label_index: Edges to predict [2, num_pred_edges]
        Returns:
            Predictions [num_pred_edges, 1]
        """
        # Encode all nodes
        h = self.encoder(x, edge_index)
        
        # Get source and target embeddings
        src, dst = edge_label_index[0], edge_label_index[1]
        
        # Predict links
        return self.predictor(h[src], h[dst])
    
    def forward_with_subgraph(self, x, edge_index, edge_label_index, subgraph_fn=None):
        """
        Forward pass with optional subgraph extraction (plug-and-play interface).
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_label_index: Edges to predict [2, num_pred_edges]
            subgraph_fn: Optional function(x, edge_index, u, v) -> (sub_x, sub_edge_index, node_map)
        Returns:
            Predictions [num_pred_edges, 1]
        """
        if subgraph_fn is None:
            # No subgraphing: use full graph
            return self.forward(x, edge_index, edge_label_index)
        else:
            # With subgraphing: extract subgraph per edge and predict
            predictions = []
            for i in range(edge_label_index.size(1)):
                u, v = edge_label_index[0, i].item(), edge_label_index[1, i].item()
                
                # Extract subgraph using provided function
                sub_x, sub_edge_index, node_map = subgraph_fn(x, edge_index, u, v)
                
                # Encode on subgraph
                h = self.encoder(sub_x, sub_edge_index)
                
                # Map u, v to subgraph node indices
                u_sub = (node_map == u).nonzero(as_tuple=True)[0][0]
                v_sub = (node_map == v).nonzero(as_tuple=True)[0][0]
                
                # Predict link
                pred = self.predictor(h[u_sub].unsqueeze(0), h[v_sub].unsqueeze(0))
                predictions.append(pred)
            
            return torch.cat(predictions, dim=0)
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        predictor_params = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        return encoder_params + predictor_params

