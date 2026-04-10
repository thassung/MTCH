"""
Static k-hop subgraph extractor with preprocessing.
Precomputes k-hop neighborhoods once for fast lookup.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from ..utils.khop_preprocessor import KHopPreprocessor


class StaticKHopExtractor:
    """
    Static k-hop neighborhood extractor with preprocessing.
    
    Args:
        data: PyG Data object containing the full graph
        k: Number of hops (default: 2)
        preprocessor: Optional KHopPreprocessor instance (if None, creates new one)
    """
    
    def __init__(self, data, k=2, preprocessor=None):
        self.data = data
        self.k = k
        
        print(f"StaticKHopExtractor initialized: k={k}")
        print(f"  Graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
        
        # Use provided preprocessor or create new one
        if preprocessor is not None:
            print(f"  Using provided preprocessor")
            self.preprocessor = preprocessor
        else:
            print(f"  Creating new preprocessor...")
            self.preprocessor = KHopPreprocessor(data.edge_index, data.num_nodes, k=k)
    
    def extract_subgraph(self, u, v):
        """
        Extract subgraph as union of k-hop neighborhoods around u and v.
        Uses precomputed k-hop neighborhoods for fast lookup.
        
        Args:
            u: Source node index
            v: Target node index
        
        Returns:
            subgraph_data: PyG Data object with remapped indices
            selected_nodes: Original node indices selected for subgraph
            metadata: Dictionary with extraction metadata
        """
        # Get precomputed k-hop neighborhoods (fast!)
        nodes_u = self.preprocessor.get_khop_nodes(u)
        nodes_v = self.preprocessor.get_khop_nodes(v)
        
        # Union of nodes
        selected_nodes = torch.unique(torch.cat([nodes_u, nodes_v]))
        
        # Extract subgraph using PyG utils
        edge_index_sub, edge_attr_sub = subgraph(
            selected_nodes,
            self.data.edge_index,
            edge_attr=self.data.edge_attr if hasattr(self.data, 'edge_attr') and self.data.edge_attr is not None else None,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes
        )
        
        # Create node mapping (old_idx -> new_idx)
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(selected_nodes)}
        
        # Get remapped indices for u and v
        u_sub = node_mapping.get(u, -1)
        v_sub = node_mapping.get(v, -1)
        
        # Extract node features for selected nodes
        x_sub = self.data.x[selected_nodes]
        
        # Create subgraph Data object
        subgraph_data = Data(
            x=x_sub,
            edge_index=edge_index_sub,
            edge_attr=edge_attr_sub,
            num_nodes=len(selected_nodes)
        )
        
        # Metadata
        metadata = {
            'u_original': u,
            'v_original': v,
            'u_subgraph': u_sub,
            'v_subgraph': v_sub,
            'num_nodes_selected': len(selected_nodes),
            'num_edges_subgraph': edge_index_sub.size(1),
            'k': self.k,
            'nodes_from_u': len(nodes_u),
            'nodes_from_v': len(nodes_v)
        }
        
        return subgraph_data, selected_nodes, metadata
    
    def __repr__(self):
        return f"StaticKHopExtractor(k={self.k}, preprocessor={self.preprocessor})"
