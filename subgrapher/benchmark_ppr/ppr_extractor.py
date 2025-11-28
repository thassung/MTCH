"""
Static PPR-based Subgraph Extractor

Implements non-learnable (fixed) PPR-based subgraph extraction for baseline comparison.
Uses fixed alpha=0.5 and grid search over top-k values.
"""

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph
from ..utils.ppr_scorer import personalized_pagerank


class StaticPPRExtractor:
    """
    Static (non-learnable) PPR-based subgraph extraction.
    
    Args:
        data: PyG Data object containing the full graph
        alpha: Fixed combination weight (default 0.5 for balanced)
        ppr_alpha: PPR damping factor (default 0.85)
        top_k: Number of top nodes to select (default 100)
        use_cache: Whether to cache PPR computations (default True)
    """
    
    def __init__(self, data, alpha=0.5, ppr_alpha=0.85, top_k=100, use_cache=True):
        self.data = data
        self.alpha = alpha
        self.ppr_alpha = ppr_alpha
        self.top_k = top_k
        self.use_cache = use_cache
        
        # Convert to NetworkX for PPR computation
        self.G = to_networkx(data, to_undirected=True)
        self.nodes = sorted(self.G.nodes())
        self.num_nodes = len(self.nodes)
        
        # PPR cache
        self.ppr_cache = {} if use_cache else None
        
    def compute_ppr(self, seed_node):
        """Compute PPR scores from a seed node."""
        if self.ppr_cache is not None and seed_node in self.ppr_cache:
            return self.ppr_cache[seed_node]
        
        # Create personalization vector
        seeds = {n: 1.0 if n == seed_node else 0.0 for n in self.nodes}
        
        # Compute PPR
        ppr_dict = personalized_pagerank(self.G, seeds, alpha=self.ppr_alpha)
        
        # Convert to tensor
        ppr_tensor = torch.tensor([ppr_dict[n] for n in self.nodes], dtype=torch.float32)
        
        # Cache result
        if self.ppr_cache is not None:
            self.ppr_cache[seed_node] = ppr_tensor
            
        return ppr_tensor
    
    def extract_subgraph(self, u, v):
        """
        Extract subgraph around nodes u and v using static PPR.
        
        Args:
            u: Source node index
            v: Target node index
            
        Returns:
            subgraph_data: PyG Data object for the subgraph
            selected_nodes: Tensor of selected node indices
            metadata: Dict with extraction statistics
        """
        # Compute PPR from both nodes
        ppr_u = self.compute_ppr(u)
        ppr_v = self.compute_ppr(v)
        
        # Combine with fixed alpha
        combined_scores = self.alpha * ppr_u + (1 - self.alpha) * ppr_v
        
        # Select top-k nodes
        top_k_actual = min(self.top_k, self.num_nodes)
        top_k_values, top_k_indices = torch.topk(combined_scores, top_k_actual)
        
        # Ensure u and v are included
        selected_nodes = top_k_indices
        if u not in selected_nodes:
            selected_nodes = torch.cat([selected_nodes, torch.tensor([u])])
        if v not in selected_nodes:
            selected_nodes = torch.cat([selected_nodes, torch.tensor([v])])
        
        # Remove duplicates and sort
        selected_nodes = torch.unique(selected_nodes)
        
        # Extract subgraph
        edge_index, edge_attr = subgraph(
            selected_nodes,
            self.data.edge_index,
            edge_attr=self.data.edge_attr if hasattr(self.data, 'edge_attr') else None,
            relabel_nodes=True,
            num_nodes=self.num_nodes
        )
        
        # Create new mapping for u and v
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(selected_nodes)}
        u_new = node_mapping.get(u, -1)
        v_new = node_mapping.get(v, -1)
        
        # Create subgraph data object
        subgraph_data = Data(
            x=self.data.x[selected_nodes] if hasattr(self.data, 'x') else None,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(selected_nodes)
        )
        
        # Metadata
        metadata = {
            'num_selected': len(selected_nodes),
            'alpha': self.alpha,
            'top_k': self.top_k,
            'u_original': u,
            'v_original': v,
            'u_subgraph': u_new,
            'v_subgraph': v_new,
            'avg_ppr_score': combined_scores[selected_nodes].mean().item(),
            'min_ppr_score': combined_scores[selected_nodes].min().item(),
            'max_ppr_score': combined_scores[selected_nodes].max().item()
        }
        
        return subgraph_data, selected_nodes, metadata
    
    def clear_cache(self):
        """Clear PPR cache to free memory."""
        if self.ppr_cache is not None:
            self.ppr_cache.clear()
    
    def get_cache_size(self):
        """Get number of cached PPR computations."""
        return len(self.ppr_cache) if self.ppr_cache is not None else 0

