"""
PPR (Personalized PageRank) preprocessor.
Precomputes PPR scores for all nodes once, stores for fast lookup.

Uses the approximate push algorithm (Andersen et al. 2006) for speed.
"""

import torch
import numpy as np
from tqdm import tqdm
from subgrapher.utils.local_ppr import approximate_ppr, build_sparse_adj


class PPRPreprocessor:
    """
    Precomputes and caches PPR scores for all nodes.
    
    Args:
        data: PyG Data object
        ppr_alpha: Teleport probability for PPR (default: 0.85)
        epsilon: Approximate PPR precision (default: 1e-4).
                 Smaller = more accurate but slower.
    """
    
    def __init__(self, data, ppr_alpha=0.85, epsilon=1e-4, log_file=None):
        self.num_nodes = data.num_nodes
        self.ppr_alpha = ppr_alpha
        self.epsilon = epsilon
        self.ppr_cache = {}
        
        print(f"PPRPreprocessor: Precomputing PPR for {data.num_nodes} nodes...")
        print(f"  Building sparse adjacency...")
        self.adj_csr = build_sparse_adj(data.edge_index, data.num_nodes)
        self.nodes = list(range(data.num_nodes))
        
        print(f"  Computing approximate PPR (alpha={ppr_alpha}, epsilon={epsilon})...")
        if log_file:
            print(f"  Logging PPR arrays to: {log_file}")
        self._precompute_all(log_file=log_file)
        print(f"  Cached {len(self.ppr_cache)} PPR vectors")
        
        memory_mb = len(self.ppr_cache) * self.num_nodes * 4 / (1024 * 1024)
        print(f"  Cache memory: ~{memory_mb:.1f} MB")
    
    def _precompute_all(self, log_file=None):
        """Precompute PPR for all nodes using approximate push."""
        log_handle = None
        if log_file:
            log_handle = open(log_file, 'w')
            log_handle.write(f"# PPR Preprocessing Log (alpha={self.ppr_alpha}, "
                             f"epsilon={self.epsilon})\n")
            log_handle.write(f"# Format: node_id | ppr_scores (space-separated)\n\n")
        
        try:
            for node in tqdm(range(self.num_nodes), desc="  Computing PPR",
                             leave=False, mininterval=10):
                ppr_array = approximate_ppr(
                    self.adj_csr, {node},
                    alpha=self.ppr_alpha, epsilon=self.epsilon)
                
                ppr_tensor = torch.from_numpy(ppr_array).float()
                self.ppr_cache[node] = ppr_tensor
                
                if log_handle:
                    ppr_values = ' '.join(f"{v:.6f}" for v in ppr_tensor.tolist())
                    log_handle.write(f"{node} | {ppr_values}\n")
        finally:
            if log_handle:
                log_handle.close()
    
    def get_ppr(self, node):
        """
        Get precomputed PPR scores for a node.
        
        Args:
            node: Node index (int)
        
        Returns:
            ppr_scores: Tensor of PPR scores [num_nodes]
        """
        if node in self.ppr_cache:
            return self.ppr_cache[node]
        
        # Fallback: return uniform distribution
        return torch.ones(self.num_nodes, dtype=torch.float32) / self.num_nodes
    
    def get_ppr_pair(self, u, v):
        """
        Get precomputed PPR scores for two nodes.
        
        Args:
            u: First node index
            v: Second node index
        
        Returns:
            ppr_u: PPR scores from u [num_nodes]
            ppr_v: PPR scores from v [num_nodes]
        """
        return self.get_ppr(u), self.get_ppr(v)
    
    def get_top_k_nodes(self, node, top_k, alpha=0.5, other_node=None):
        """
        Get top-k nodes by PPR score (optionally combined with another node).
        
        Args:
            node: Primary node index
            top_k: Number of top nodes to return
            alpha: Weight for combining PPR scores (default: 0.5)
            other_node: Optional second node to combine with
        
        Returns:
            top_nodes: Tensor of top-k node indices
        """
        ppr = self.get_ppr(node)
        
        if other_node is not None:
            ppr_other = self.get_ppr(other_node)
            ppr = alpha * ppr + (1 - alpha) * ppr_other
        
        top_k_actual = min(top_k, len(ppr))
        _, top_indices = torch.topk(ppr, top_k_actual)
        
        return top_indices
    
    def get_stats(self):
        """Get cache statistics."""
        if not self.ppr_cache:
            return {
                'num_cached': 0,
                'memory_mb': 0
            }
        
        memory_bytes = len(self.ppr_cache) * self.num_nodes * 4
        
        return {
            'num_cached': len(self.ppr_cache),
            'memory_mb': memory_bytes / (1024 * 1024)
        }
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"PPRPreprocessor(alpha={self.ppr_alpha}, cached={stats['num_cached']}, "
                f"memory={stats['memory_mb']:.1f}MB)")
    
    def save(self, path):
        """Save preprocessor cache to disk."""
        torch.save({
            'ppr_alpha': self.ppr_alpha,
            'epsilon': self.epsilon,
            'num_nodes': self.num_nodes,
            'ppr_cache': self.ppr_cache,
        }, path)
        print(f"Saved PPRPreprocessor to {path}")
    
    @classmethod
    def load(cls, path, data=None):
        """Load preprocessor cache from disk."""
        saved_data = torch.load(path, weights_only=False)
        preprocessor = cls.__new__(cls)
        preprocessor.ppr_alpha = saved_data['ppr_alpha']
        preprocessor.num_nodes = saved_data['num_nodes']
        preprocessor.ppr_cache = saved_data['ppr_cache']
        preprocessor.epsilon = saved_data.get('epsilon', 1e-4)
        preprocessor.nodes = list(range(preprocessor.num_nodes))

        if data is not None:
            preprocessor.adj_csr = build_sparse_adj(
                data.edge_index, data.num_nodes)
        else:
            preprocessor.adj_csr = None
        
        print(f"Loaded PPRPreprocessor from {path}")
        return preprocessor


if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    from subgrapher.utils.loader import load_txt_to_pyg
    
    # Default datasets to process
    ALL_DATASETS = ['FB15K237', 'WN18RR', 'NELL-995']
    
    parser = argparse.ArgumentParser(description="Preprocess PPR scores for all datasets")
    parser.add_argument('--alpha', type=float, default=0.85,
                        help='PPR alpha/teleport probability (default: 0.85)')
    
    args = parser.parse_args()
    
    print(f"Will preprocess {len(ALL_DATASETS)} datasets with alpha={args.alpha}")
    print(f"Datasets: {', '.join(ALL_DATASETS)}\n")
    
    # Process each dataset
    for dataset in ALL_DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset} with alpha={args.alpha}")
        print(f"{'='*60}")
        
        try:
            # Load data
            print(f"\nLoading {dataset}...")
            data_path = f"data/{dataset}/train.txt"
            data, node2idx, idx2node = load_txt_to_pyg(data_path)
            
            # Setup paths
            output_dir = f"preprocessed/{dataset}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/ppr_alpha{args.alpha}.pt"
            log_path = f"{output_dir}/ppr_alpha{args.alpha}_log.txt"
            
            # Create preprocessor (automatically computes cache with logging)
            preprocessor = PPRPreprocessor(data, ppr_alpha=args.alpha, log_file=log_path)
            
            # Display stats
            stats = preprocessor.get_stats()
            print(f"\nPreprocessing Statistics:")
            print(f"  Nodes cached: {stats['num_cached']}")
            print(f"  Memory usage: {stats['memory_mb']:.1f} MB")
            
            # Save cache
            preprocessor.save(output_path)
            
            # Save summary statistics
            summary_path = f"{output_dir}/ppr_alpha{args.alpha}_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"PPR Preprocessing Summary\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Dataset: {dataset}\n")
                f.write(f"PPR Alpha: {args.alpha}\n")
                f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Statistics:\n")
                f.write(f"  Total nodes: {data.num_nodes:,}\n")
                f.write(f"  Nodes cached: {stats['num_cached']:,}\n")
                f.write(f"  Cache memory: {stats['memory_mb']:.1f} MB\n\n")
                f.write(f"Files:\n")
                f.write(f"  Cache: {output_path}\n")
                f.write(f"  Log: {log_path}\n")
                f.write(f"  Summary: {summary_path}\n")
            
            print(f"\n✓ {dataset} alpha={args.alpha} complete!")
            print(f"  Cache:   {output_path}")
            print(f"  Log:     {log_path}")
            print(f"  Summary: {summary_path}")
            
        except Exception as e:
            print(f"\n✗ Error processing {dataset} alpha={args.alpha}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ All preprocessing complete!")
    print(f"  Processed {len(ALL_DATASETS)} dataset(s) with alpha={args.alpha}")
    print(f"  Files saved to: preprocessed/{{dataset}}/ppr_alpha{args.alpha}.pt")
