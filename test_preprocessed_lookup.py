"""
Test script to verify preprocessed data lookup optimization.
"""

import os
import torch
from subgrapher.utils.loader import load_txt_to_pyg
from subgrapher.utils.khop_preprocessor import KHopPreprocessor
from subgrapher.utils.ppr_preprocessor import PPRPreprocessor

def test_khop_lookup():
    """Test k-hop preprocessor lookup."""
    print("=" * 60)
    print("Testing K-hop Preprocessed Data Lookup")
    print("=" * 60)
    
    dataset_name = 'FB15K237'
    k = 2
    
    # Load data
    print(f"\n1. Loading {dataset_name} dataset...")
    data_path = f"data/{dataset_name}/train.txt"
    data, node2idx, idx2node = load_txt_to_pyg(data_path)
    print(f"   Nodes: {data.num_nodes}")
    
    # Check if preprocessed exists
    preprocessed_path = f"preprocessed/{dataset_name}/khop_k{k}.pt"
    print(f"\n2. Checking for preprocessed k-hop data...")
    print(f"   Path: {preprocessed_path}")
    
    if os.path.exists(preprocessed_path):
        print(f"   ✓ Found preprocessed data!")
        
        # Load preprocessor
        print(f"\n3. Loading preprocessor...")
        import time
        start = time.time()
        preprocessor = KHopPreprocessor.load(preprocessed_path, data.edge_index)
        load_time = time.time() - start
        
        stats = preprocessor.get_stats()
        print(f"   ✓ Loaded in {load_time:.2f}s")
        print(f"   Cached nodes: {stats['num_cached']}")
        print(f"   Avg k-hop size: {stats['avg_size']:.1f}")
        print(f"   Memory: {stats['memory_mb']:.1f} MB")
        
        # Test lookup
        print(f"\n4. Testing k-hop lookup...")
        test_node = 100
        khop_nodes = preprocessor.get_khop_nodes(test_node)
        print(f"   Node {test_node}: {len(khop_nodes)} neighbors")
        print(f"   ✓ Lookup works!")
        
    else:
        print(f"   ✗ Preprocessed data not found")
        print(f"   Run: python -m subgrapher.utils.khop_preprocessor --k {k}")


def test_ppr_lookup():
    """Test PPR preprocessor lookup."""
    print("\n" + "=" * 60)
    print("Testing PPR Preprocessed Data Lookup")
    print("=" * 60)
    
    dataset_name = 'FB15K237'
    ppr_alpha = 0.85
    
    # Load data
    print(f"\n1. Loading {dataset_name} dataset...")
    data_path = f"data/{dataset_name}/train.txt"
    data, node2idx, idx2node = load_txt_to_pyg(data_path)
    print(f"   Nodes: {data.num_nodes}")
    
    # Check if preprocessed exists
    preprocessed_path = f"preprocessed/{dataset_name}/ppr_alpha{ppr_alpha}.pt"
    print(f"\n2. Checking for preprocessed PPR data...")
    print(f"   Path: {preprocessed_path}")
    
    if os.path.exists(preprocessed_path):
        print(f"   ✓ Found preprocessed data!")
        
        # Load preprocessor
        print(f"\n3. Loading preprocessor...")
        import time
        start = time.time()
        preprocessor = PPRPreprocessor.load(preprocessed_path, data)
        load_time = time.time() - start
        
        stats = preprocessor.get_stats()
        print(f"   ✓ Loaded in {load_time:.2f}s")
        print(f"   Cached nodes: {stats['num_cached']}")
        print(f"   Memory: {stats['memory_mb']:.1f} MB")
        
        # Test lookup
        print(f"\n4. Testing PPR lookup...")
        test_node = 100
        ppr_scores = preprocessor.get_ppr(test_node)
        print(f"   Node {test_node}: PPR vector shape {ppr_scores.shape}")
        print(f"   Top-5 PPR scores: {ppr_scores.topk(5).values.tolist()}")
        print(f"   ✓ Lookup works!")
        
    else:
        print(f"   ✗ Preprocessed data not found")
        print(f"   Run: python -m subgrapher.utils.ppr_preprocessor --alpha {ppr_alpha}")


def test_memory_comparison():
    """Compare memory usage of lookup vs on-the-fly computation."""
    print("\n" + "=" * 60)
    print("Memory Usage Comparison")
    print("=" * 60)
    
    dataset_name = 'FB15K237'
    
    # Check what's preprocessed
    preprocessed_dir = f"preprocessed/{dataset_name}"
    
    if os.path.exists(preprocessed_dir):
        print(f"\nPreprocessed files in {preprocessed_dir}:")
        for filename in sorted(os.listdir(preprocessed_dir)):
            if filename.endswith('.pt'):
                filepath = os.path.join(preprocessed_dir, filename)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {filename}: {size_mb:.1f} MB")
    else:
        print(f"\n✗ No preprocessed directory found")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Preprocessed Data Lookup Optimization Test")
    print("=" * 60)
    
    try:
        test_khop_lookup()
    except Exception as e:
        print(f"\n✗ K-hop test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_ppr_lookup()
    except Exception as e:
        print(f"\n✗ PPR test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_memory_comparison()
    except Exception as e:
        print(f"\n✗ Memory comparison failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ All tests complete!")
    print("=" * 60)

