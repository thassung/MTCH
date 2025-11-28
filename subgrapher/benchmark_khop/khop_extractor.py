"""
Static k-hop Subgraph Extractor

Implements non-learnable k-hop neighborhood extraction for baseline comparison.
Uses torch_geometric's k_hop_subgraph utility.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


class StaticKHopExtractor:
    """
    Static (non-learnable) k-hop neighborhood extraction.
    
    Args:
        data: PyG Data object containing the full graph
        k: Number of hops (default 2)
    """
    
    def __init__(self, data, k=2):
        self.data = data
        self.k = k
        self.num_nodes = data.num_nodes
        
    def extract_subgraph(self, u, v):
        """
        Extract k-hop subgraph around nodes u and v.
        
        Args:
            u: Source node index
            v: Target node index
            
        Returns:
            subgraph_data: PyG Data object for the subgraph
            selected_nodes: Tensor of selected node indices
            metadata: Dict with extraction statistics
        """
        # Extract k-hop neighborhood around u
        subset_u, edge_index_u, mapping_u, edge_mask_u = k_hop_subgraph(
            node_idx=u,
            num_hops=self.k,
            edge_index=self.data.edge_index,
            relabel_nodes=False,
            num_nodes=self.num_nodes
        )
        
        # Extract k-hop neighborhood around v
        subset_v, edge_index_v, mapping_v, edge_mask_v = k_hop_subgraph(
            node_idx=v,
            num_hops=self.k,
            edge_index=self.data.edge_index,
            relabel_nodes=False,
            num_nodes=self.num_nodes
        )
        
        # Take union of nodes
        selected_nodes = torch.unique(torch.cat([subset_u, subset_v]))
        
        # Extract subgraph with selected nodes (relabel now)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=selected_nodes,
            num_hops=0,  # Already selected nodes, just get their edges
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.num_nodes
        )
        
        # Create mapping for u and v
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset)}
        u_new = node_mapping.get(u, -1)
        v_new = node_mapping.get(v, -1)
        
        # Create subgraph data object
        subgraph_data = Data(
            x=self.data.x[subset] if hasattr(self.data, 'x') else None,
            edge_index=edge_index,
            edge_attr=self.data.edge_attr[edge_mask] if hasattr(self.data, 'edge_attr') else None,
            num_nodes=len(subset)
        )
        
        # Metadata
        metadata = {
            'num_selected': len(subset),
            'k': self.k,
            'u_original': u,
            'v_original': v,
            'u_subgraph': u_new,
            'v_subgraph': v_new,
            'num_u_neighbors': len(subset_u),
            'num_v_neighbors': len(subset_v),
            'overlap': len(set(subset_u.tolist()) & set(subset_v.tolist()))
        }
        
        return subgraph_data, subset, metadata

