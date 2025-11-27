"""
Data preprocessing for link prediction benchmark.
Handles edge splitting, negative sampling, and feature generation.
"""

import torch
import numpy as np
from torch_geometric.utils import negative_sampling, to_undirected
import sys
import os

# Add parent directory to path to import loader
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.loader import load_txt_to_pyg


def add_node_features(data, method='random', feature_dim=128):
    """
    Add node features to PyG Data object.
    
    Args:
        data: PyG Data object
        method: 'one_hot' or 'random'
        feature_dim: Dimension for random embeddings
    Returns:
        Data with x attribute
    """
    num_nodes = data.num_nodes
    
    if method == 'one_hot':
        data.x = torch.eye(num_nodes)
    elif method == 'random':
        torch.manual_seed(42)  # For reproducibility
        data.x = torch.randn(num_nodes, feature_dim)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return data


def split_edges(edge_index, num_nodes, val_ratio=0.1, test_ratio=0.1):
    """
    Split edges into train/val/test sets.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    Returns:
        Dictionary with train/val/test edge splits
    """
    num_edges = edge_index.size(1)
    
    # Create edge pairs as tuples for easier handling
    edge_list = edge_index.t().tolist()
    edge_set = set(map(tuple, edge_list))
    
    # Shuffle edges
    indices = torch.randperm(num_edges)
    
    # Calculate split sizes
    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)
    num_train = num_edges - num_val - num_test
    
    # Split indices
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    
    # Create edge splits
    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]
    
    return {
        'train': train_edge_index,
        'val': val_edge_index,
        'test': test_edge_index,
        'num_nodes': num_nodes
    }


def generate_negative_samples(pos_edge_index, num_nodes, num_neg_samples=None):
    """
    Generate negative edge samples.
    
    Args:
        pos_edge_index: Positive edge indices [2, num_pos]
        num_nodes: Number of nodes
        num_neg_samples: Number of negative samples (defaults to num_pos)
    Returns:
        Negative edge indices [2, num_neg_samples]
    """
    if num_neg_samples is None:
        num_neg_samples = pos_edge_index.size(1)
    
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples
    )
    
    return neg_edge_index


def prepare_link_prediction_data(dataset_path, feature_method='random', feature_dim=128,
                                  val_ratio=0.1, test_ratio=0.1, num_neg_samples_eval=100):
    """
    Prepare complete link prediction dataset from .txt file.
    
    Args:
        dataset_path: Path to .txt file with triples
        feature_method: 'one_hot' or 'random'
        feature_dim: Feature dimension for random method
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        num_neg_samples_eval: Number of negative samples per positive edge for evaluation
    Returns:
        Dictionary with all prepared data
    """
    print(f"Loading data from {dataset_path}...")
    data, node2idx, idx2node = load_txt_to_pyg(dataset_path)
    
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    
    # Add node features
    print(f"Adding node features ({feature_method})...")
    data = add_node_features(data, method=feature_method, feature_dim=feature_dim)
    
    # Split edges
    print("Splitting edges into train/val/test...")
    edge_splits = split_edges(data.edge_index, data.num_nodes, val_ratio, test_ratio)
    
    # Generate negative samples for evaluation
    print(f"Generating negative samples for evaluation ({num_neg_samples_eval} per positive edge)...")
    val_neg_edge_index = generate_negative_samples(
        edge_splits['val'], data.num_nodes, 
        num_neg_samples=edge_splits['val'].size(1) * num_neg_samples_eval
    )
    test_neg_edge_index = generate_negative_samples(
        edge_splits['test'], data.num_nodes,
        num_neg_samples=edge_splits['test'].size(1) * num_neg_samples_eval
    )
    
    # Prepare split_edge dictionary (PS2 style)
    split_edge = {
        'train': {
            'source_node': edge_splits['train'][0],
            'target_node': edge_splits['train'][1]
        },
        'valid': {
            'source_node': edge_splits['val'][0],
            'target_node': edge_splits['val'][1],
            'target_node_neg': val_neg_edge_index[1].view(edge_splits['val'].size(1), -1)
        },
        'test': {
            'source_node': edge_splits['test'][0],
            'target_node': edge_splits['test'][1],
            'target_node_neg': test_neg_edge_index[1].view(edge_splits['test'].size(1), -1)
        }
    }
    
    print(f"Train edges: {edge_splits['train'].size(1)}")
    print(f"Val edges: {edge_splits['val'].size(1)}")
    print(f"Test edges: {edge_splits['test'].size(1)}")
    
    return {
        'data': data,
        'split_edge': split_edge,
        'node2idx': node2idx,
        'idx2node': idx2node,
        'train_edge_index': edge_splits['train']
    }

