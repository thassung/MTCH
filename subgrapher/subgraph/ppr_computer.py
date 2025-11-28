"""
PPR (Personalized PageRank) computation module with caching.
Reuses existing PPR implementation from utils.
"""

import torch
import networkx as nx
from torch_geometric.utils import to_networkx

# Import existing PPR function
import sys
import os
from ..utils.ppr_scorer import personalized_pagerank


class PPRComputer:
    """
    Efficient PPR computation with caching for subgraph selection.
    
    Args:
        data: PyG Data object containing the graph
        ppr_alpha: Teleport probability for PPR (default: 0.85)
        use_cache: Whether to cache computed PPR vectors
    """
    
    def __init__(self, data, ppr_alpha=0.85, use_cache=True):
        # Convert PyG data to NetworkX once (expensive operation)
        self.G = to_networkx(data, to_undirected=True)
        self.ppr_alpha = ppr_alpha
        self.use_cache = use_cache
        
        # Cache for PPR vectors
        self.cache = {} if use_cache else None
        
        # Store node list for consistent ordering
        self.nodes = sorted(self.G.nodes())
        self.num_nodes = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
    
    def compute_ppr(self, seed_node):
        """
        Compute PPR from a single seed node.
        
        Args:
            seed_node: Node index to use as seed
        
        Returns:
            ppr_tensor: PPR scores as tensor [num_nodes]
        """
        # Check cache
        if self.cache is not None and seed_node in self.cache:
            return self.cache[seed_node]
        
        # Prepare seed distribution
        seeds = {n: 1.0 if n == seed_node else 0.0 for n in self.nodes}
        
        # Compute PPR using existing function
        ppr_dict = personalized_pagerank(self.G, seeds, alpha=self.ppr_alpha)
        
        # Convert to tensor in consistent order
        ppr_tensor = torch.tensor([ppr_dict[n] for n in self.nodes], dtype=torch.float32)
        
        # Cache result
        if self.cache is not None:
            self.cache[seed_node] = ppr_tensor
        
        return ppr_tensor
    
    def compute_ppr_pair(self, u, v):
        """
        Compute PPR from both nodes u and v.
        
        Args:
            u: First seed node index
            v: Second seed node index
        
        Returns:
            ppr_u: PPR scores from u [num_nodes]
            ppr_v: PPR scores from v [num_nodes]
        """
        ppr_u = self.compute_ppr(u)
        ppr_v = self.compute_ppr(v)
        return ppr_u, ppr_v
    
    def compute_ppr_batch(self, seed_nodes):
        """
        Compute PPR for multiple seed nodes.
        
        Args:
            seed_nodes: List of node indices
        
        Returns:
            ppr_matrix: PPR scores [len(seed_nodes), num_nodes]
        """
        ppr_vectors = []
        for seed in seed_nodes:
            ppr_vectors.append(self.compute_ppr(seed))
        return torch.stack(ppr_vectors)
    
    def clear_cache(self):
        """Clear the PPR cache."""
        if self.cache is not None:
            self.cache.clear()
    
    def get_cache_size(self):
        """Get number of cached PPR vectors."""
        return len(self.cache) if self.cache is not None else 0
    
    def get_cache_memory_mb(self):
        """Estimate cache memory usage in MB."""
        if self.cache is None or len(self.cache) == 0:
            return 0.0
        # Each PPR vector is num_nodes * 4 bytes (float32)
        bytes_per_vector = self.num_nodes * 4
        total_bytes = len(self.cache) * bytes_per_vector
        return total_bytes / (1024 * 1024)
    
    def __repr__(self):
        cache_info = f", cached={self.get_cache_size()}" if self.use_cache else ""
        return (f"PPRComputer(nodes={self.num_nodes}, "
                f"alpha={self.ppr_alpha}{cache_info})")

