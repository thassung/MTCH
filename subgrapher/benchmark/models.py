"""
Link prediction models.
Imports shared models from utils and adds wrapper class.
"""

import torch

# Import shared models from utils
from ..utils.models import GCN, SAGE, GAT, LinkPredictor


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

