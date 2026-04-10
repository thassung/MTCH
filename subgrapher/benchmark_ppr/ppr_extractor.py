"""
Static PPR-based subgraph extractor with preprocessing.
Precomputes PPR scores once for fast lookup.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from ..utils.ppr_preprocessor import PPRPreprocessor


class StaticPPRExtractor:
    """
    Static PPR-based subgraph extractor with fixed alpha and top-k selection.
    
    Args:
        data: PyG Data object containing the full graph
        alpha: Fixed weight for combining PPR scores (default: 0.5)
        top_k: Number of nodes to select based on PPR scores
        ppr_alpha: Teleport probability for PPR algorithm (default: 0.85)
        preprocessor: Optional PPRPreprocessor instance (if None, creates new one)
    """
    
    def __init__(self, data, alpha=0.5, top_k=100, ppr_alpha=0.85, preprocessor=None):
        self.data = data
        self.alpha = alpha
        self.top_k = top_k
        self.ppr_alpha = ppr_alpha
        
        print(f"StaticPPRExtractor initialized: alpha={alpha}, top_k={top_k}, ppr_alpha={ppr_alpha}")
        print(f"  Graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
        
        # Use provided preprocessor or create new one
        if preprocessor is not None:
            print(f"  Using provided preprocessor")
            self.preprocessor = preprocessor
        else:
            print(f"  Creating new preprocessor...")
            self.preprocessor = PPRPreprocessor(data, ppr_alpha=ppr_alpha)
    
    def extract_subgraph(self, u, v):
        """
        Extract subgraph around edge (u, v) using PPR-based selection.
        Uses precomputed PPR scores for fast lookup.
        
        Args:
            u: Source node index
            v: Target node index
        
        Returns:
            subgraph_data: PyG Data object with remapped indices
            selected_nodes: Original node indices selected for subgraph
            metadata: Dictionary with extraction metadata
        """
        # Get precomputed PPR scores (fast!)
        ppr_u = self.preprocessor.get_ppr(u)
        ppr_v = self.preprocessor.get_ppr(v)
        
        # Combine PPR scores (fixed alpha)
        combined_scores = self.alpha * ppr_u + (1 - self.alpha) * ppr_v
        
        # Select top-k nodes
        top_k_actual = min(self.top_k, len(combined_scores))
        _, top_indices = torch.topk(combined_scores, top_k_actual)
        selected_nodes = top_indices
        
        # Always include u and v
        if u not in selected_nodes:
            selected_nodes = torch.cat([selected_nodes, torch.tensor([u])])
        if v not in selected_nodes:
            selected_nodes = torch.cat([selected_nodes, torch.tensor([v])])
        
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
            'alpha': self.alpha,
            'top_k': self.top_k,
            'ppr_score_u': combined_scores[u].item(),
            'ppr_score_v': combined_scores[v].item()
        }
        
        return subgraph_data, selected_nodes, metadata
    
    def get_cache_stats(self):
        """Get PPR cache statistics."""
        stats = self.preprocessor.get_stats()
        return {
            'cache_size': stats['num_cached'],
            'cache_memory_mb': stats['memory_mb']
        }
    
    def __repr__(self):
        return (f"StaticPPRExtractor(alpha={self.alpha}, top_k={self.top_k}, "
                f"preprocessor={self.preprocessor})")
