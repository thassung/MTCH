"""
k-hop neighborhood preprocessor.
Precomputes k-hop neighborhoods for all nodes once, stores for fast lookup.
"""

import torch
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm


class KHopPreprocessor:
    """
    Precomputes and caches k-hop neighborhoods for all nodes.
    
    Args:
        edge_index: Graph edge index [2, num_edges]
        num_nodes: Number of nodes in graph
        k: Number of hops (default: 1, 2, 3)
    """
    
    def __init__(self, edge_index, num_nodes, k=2, log_file=None):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.k = k
        self.khop_cache = {}
        
        print(f"KHopPreprocessor: Precomputing {k}-hop neighborhoods for {num_nodes} nodes...")
        if log_file:
            print(f"  Logging k-hop neighborhoods to: {log_file}")
        self._precompute_all(log_file=log_file)
        print(f"  ✓ Cached {len(self.khop_cache)} k-hop neighborhoods")
        
        # Memory estimate
        total_nodes = sum(len(nodes) for nodes in self.khop_cache.values())
        avg_nodes = total_nodes / len(self.khop_cache) if self.khop_cache else 0
        memory_mb = total_nodes * 8 / (1024 * 1024)  # 8 bytes per int64
        print(f"  Average k-hop size: {avg_nodes:.1f} nodes")
        print(f"  Cache memory: ~{memory_mb:.1f} MB")
    
    def _precompute_all(self, log_file=None):
        """Precompute k-hop neighborhoods for all nodes."""
        log_handle = None
        if log_file:
            log_handle = open(log_file, 'w')
            log_handle.write(f"# K-hop Preprocessing Log (k={self.k})\n")
            log_handle.write(f"# Format: node_id | neighbor_count | neighbor_ids (space-separated)\n\n")
        
        try:
            for node in tqdm(range(self.num_nodes), desc=f"  Computing {self.k}-hop", leave=False):
                nodes, _, _, _ = k_hop_subgraph(
                    node_idx=[node],
                    num_hops=self.k,
                    edge_index=self.edge_index,
                    relabel_nodes=False,
                    num_nodes=self.num_nodes
                )
                self.khop_cache[node] = nodes
                
                # Log to file if enabled
                if log_handle:
                    neighbor_ids = ' '.join(str(n.item()) for n in nodes)
                    log_handle.write(f"{node} | {len(nodes)} | {neighbor_ids}\n")
        finally:
            if log_handle:
                log_handle.close()
    
    def get_khop_nodes(self, node):
        """
        Get precomputed k-hop neighborhood for a node.
        
        Args:
            node: Node index (int)
        
        Returns:
            nodes: Tensor of node indices in k-hop neighborhood
        """
        return self.khop_cache.get(node, torch.tensor([node], dtype=torch.long))
    
    def get_khop_union(self, nodes):
        """
        Get union of k-hop neighborhoods for multiple nodes.
        
        Args:
            nodes: List or tensor of node indices
        
        Returns:
            union_nodes: Tensor of unique node indices
        """
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
        
        all_nodes = []
        for node in nodes:
            khop_nodes = self.get_khop_nodes(node)
            all_nodes.append(khop_nodes)
        
        if not all_nodes:
            return torch.tensor([], dtype=torch.long)
        
        return torch.unique(torch.cat(all_nodes))
    
    def get_stats(self):
        """Get cache statistics."""
        if not self.khop_cache:
            return {
                'num_cached': 0,
                'avg_size': 0,
                'min_size': 0,
                'max_size': 0,
                'memory_mb': 0
            }
        
        sizes = [len(nodes) for nodes in self.khop_cache.values()]
        total_nodes = sum(sizes)
        
        return {
            'num_cached': len(self.khop_cache),
            'avg_size': total_nodes / len(self.khop_cache),
            'min_size': min(sizes),
            'max_size': max(sizes),
            'memory_mb': total_nodes * 8 / (1024 * 1024)
        }
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"KHopPreprocessor(k={self.k}, cached={stats['num_cached']}, "
                f"avg_size={stats['avg_size']:.1f})")
    
    def save(self, path):
        """Save preprocessor cache to disk."""
        torch.save({
            'k': self.k,
            'num_nodes': self.num_nodes,
            'khop_cache': self.khop_cache,
        }, path)
        print(f"Saved KHopPreprocessor to {path}")
    
    @classmethod
    def load(cls, path, edge_index):
        """Load preprocessor cache from disk."""
        data = torch.load(path)
        preprocessor = cls.__new__(cls)
        preprocessor.edge_index = edge_index
        preprocessor.num_nodes = data['num_nodes']
        preprocessor.k = data['k']
        preprocessor.khop_cache = data['khop_cache']
        print(f"Loaded KHopPreprocessor from {path}")
        return preprocessor


if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    from subgrapher.utils.loader import load_txt_to_pyg
    
    # Default datasets to process
    ALL_DATASETS = ['FB15K237', 'WN18RR', 'NELL-995']
    
    parser = argparse.ArgumentParser(description="Preprocess k-hop neighborhoods for all datasets")
    parser.add_argument('--k', type=int, nargs='*',
                        help='Number of hops (default: [1, 2, 3])')
    
    args = parser.parse_args()
    
    # Default k values
    k_values = args.k if args.k else [1, 2, 3]
    
    print(f"Will preprocess {len(ALL_DATASETS)} dataset(s) with k={k_values}")
    print(f"Datasets: {', '.join(ALL_DATASETS)}\n")
    
    # Process each dataset and k value
    for dataset in ALL_DATASETS:
        for k in k_values:
            print(f"\n{'='*60}")
            print(f"Processing: {dataset} with k={k}")
            print(f"{'='*60}")
            
            try:
                # Load data
                print(f"\nLoading {dataset}...")
                data_path = f"data/{dataset}/train.txt"
                data, node2idx, idx2node = load_txt_to_pyg(data_path)
                
                # Setup paths
                output_dir = f"preprocessed/{dataset}"
                os.makedirs(output_dir, exist_ok=True)
                output_path = f"{output_dir}/khop_k{k}.pt"
                log_path = f"{output_dir}/khop_k{k}_log.txt"
                
                # Create preprocessor (automatically computes cache with logging)
                preprocessor = KHopPreprocessor(data.edge_index, data.num_nodes, k=k, log_file=log_path)
                
                # Display stats
                stats = preprocessor.get_stats()
                print(f"\nPreprocessing Statistics:")
                print(f"  Nodes cached: {stats['num_cached']}")
                print(f"  Avg k-hop size: {stats['avg_size']:.1f} nodes")
                print(f"  Min k-hop size: {stats['min_size']} nodes")
                print(f"  Max k-hop size: {stats['max_size']} nodes")
                print(f"  Memory usage: {stats['memory_mb']:.1f} MB")
                
                # Save cache
                preprocessor.save(output_path)
                
                # Save summary statistics
                summary_path = f"{output_dir}/khop_k{k}_summary.txt"
                with open(summary_path, 'w') as f:
                    f.write(f"K-hop Neighborhood Preprocessing Summary\n")
                    f.write(f"=" * 50 + "\n\n")
                    f.write(f"Dataset: {dataset}\n")
                    f.write(f"K-hops: {k}\n")
                    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"Statistics:\n")
                    f.write(f"  Total nodes: {data.num_nodes:,}\n")
                    f.write(f"  Nodes cached: {stats['num_cached']:,}\n")
                    f.write(f"  Avg k-hop size: {stats['avg_size']:.1f} nodes\n")
                    f.write(f"  Min k-hop size: {stats['min_size']:,} nodes\n")
                    f.write(f"  Max k-hop size: {stats['max_size']:,} nodes\n")
                    f.write(f"  Cache memory: {stats['memory_mb']:.1f} MB\n\n")
                    f.write(f"Files:\n")
                    f.write(f"  Cache: {output_path}\n")
                    f.write(f"  Log: {log_path}\n")
                    f.write(f"  Summary: {summary_path}\n")
                
                print(f"\n✓ {dataset} k={k} complete!")
                print(f"  Cache:   {output_path}")
                print(f"  Log:     {log_path}")
                print(f"  Summary: {summary_path}")
                
            except Exception as e:
                print(f"\n✗ Error processing {dataset} k={k}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*60}")
    print(f"✓ All preprocessing complete!")
    print(f"  Processed {len(ALL_DATASETS)} datasets × {len(k_values)} k-values")
    print(f"  Files saved to: preprocessed/{{dataset}}/khop_k{{k}}.pt")
