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


def _ensure_planetoid_raw(name, root):
    """Pre-download Planetoid raw files using urllib (bypasses aiohttp issues)."""
    import urllib.request
    raw_dir = os.path.join(root, name, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    base_url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    endings = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    for ending in endings:
        fname = f'ind.{name.lower()}.{ending}'
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            url = f'{base_url}/{fname}'
            print(f'  Downloading {fname}...', end=' ', flush=True)
            urllib.request.urlretrieve(url, fpath)
            print('done')


def prepare_planetoid_data(name, root='data/planetoid', val_ratio=0.1, test_ratio=0.1,
                           num_neg_samples_eval=100, seed=42):
    """Prepare Cora / CiteSeer / PubMed for link prediction.

    Uses the dataset's real node features (no random embeddings needed).
    Applies the same split_edges + negative sampling logic as prepare_link_prediction_data
    so the rest of the pipeline is unchanged.
    """
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import remove_self_loops

    print(f"Loading {name} from PyG Planetoid...")
    _ensure_planetoid_raw(name, root)
    dataset = Planetoid(root=root, name=name)
    data = dataset[0].clone()
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    feature_dim = int(data.x.size(1))
    print(f"  Nodes: {data.num_nodes}  Edges: {data.edge_index.size(1)}  "
          f"Features: {feature_dim}")

    torch.manual_seed(seed)
    edge_splits = split_edges(data.edge_index, data.num_nodes, val_ratio, test_ratio)

    val_neg = generate_negative_samples(
        edge_splits['val'], data.num_nodes,
        num_neg_samples=edge_splits['val'].size(1) * num_neg_samples_eval)
    test_neg = generate_negative_samples(
        edge_splits['test'], data.num_nodes,
        num_neg_samples=edge_splits['test'].size(1) * num_neg_samples_eval)

    split_edge = {
        'train': {
            'source_node': edge_splits['train'][0],
            'target_node': edge_splits['train'][1],
        },
        'valid': {
            'source_node': edge_splits['val'][0],
            'target_node': edge_splits['val'][1],
            'target_node_neg': val_neg[1].view(edge_splits['val'].size(1), -1),
        },
        'test': {
            'source_node': edge_splits['test'][0],
            'target_node': edge_splits['test'][1],
            'target_node_neg': test_neg[1].view(edge_splits['test'].size(1), -1),
        },
    }

    print(f"  Train: {edge_splits['train'].size(1):,}  "
          f"Val: {edge_splits['val'].size(1):,}  "
          f"Test: {edge_splits['test'].size(1):,}")

    return {
        'data': data,
        'split_edge': split_edge,
        'train_edge_index': edge_splits['train'],
        'feature_dim': feature_dim,
        'node2idx': None,
        'idx2node': None,
    }


def prepare_ogbl_link_data(name='ogbl-ddi', root='data/ogb',
                            feature_method='random', feature_dim=128,
                            num_neg_samples_eval=100, seed=42,
                            max_train_edges=None):
    """Prepare an OGB link-prediction dataset for our pipeline.

    Returns the same dict shape as `prepare_planetoid_data` (data, split_edge,
    train_edge_index, feature_dim, ...). We deliberately do NOT use OGB's
    flat-negative eval format (one shared neg pool per split) — instead we
    generate per-positive negatives so the metrics are directly comparable to
    our Planetoid runs. Trade-off: numbers won't match the OGB leaderboard.

    Currently tested for ogbl-ddi (4267 nodes, ~1.3M edges, no node features
    in raw data — random features added).

    Larger OGB datasets (collab, ppa, citation2) need sparse PPR — out of
    scope for the current dense-N×N pipeline. They will OOM here.

    Args:
        name: 'ogbl-ddi' (others may work but untested in this pipeline)
        root: where to download
        feature_method: 'random' (default) or 'one_hot' if N small enough
        feature_dim: random feature dim
        num_neg_samples_eval: per-positive negatives for val/test
        seed: reproducibility
        max_train_edges: subsample train positives to this many (None = all)
    """
    from ogb.linkproppred import PygLinkPropPredDataset

    print(f"Loading {name} from OGB...")
    dataset = PygLinkPropPredDataset(name=name, root=root)
    data = dataset[0]
    split_edge_ogb = dataset.get_edge_split()

    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    print(f"  Nodes: {data.num_nodes:,}  Edges: {data.edge_index.size(1):,}  "
          f"raw_features: {data.x is not None}")

    # Add features if missing
    if data.x is None or data.x.size(1) == 0:
        data = add_node_features(data, method=feature_method, feature_dim=feature_dim)
    feature_dim = int(data.x.size(1))
    print(f"  Final feature_dim: {feature_dim}")

    # Convert OGB's [E, 2] edge tensors to our [2, E] convention
    train_pos = split_edge_ogb['train']['edge'].t().contiguous()
    valid_pos = split_edge_ogb['valid']['edge'].t().contiguous()
    test_pos  = split_edge_ogb['test']['edge'].t().contiguous()

    # Optional: subsample train edges (ogbl-ddi has 1.07M which is a lot)
    if max_train_edges is not None and train_pos.size(1) > max_train_edges:
        torch.manual_seed(seed)
        idx = torch.randperm(train_pos.size(1))[:max_train_edges]
        train_pos = train_pos[:, idx]
        print(f"  Subsampled train edges: {train_pos.size(1):,}")

    # Per-positive negatives for val/test (matches our Planetoid eval protocol)
    torch.manual_seed(seed)
    val_neg = generate_negative_samples(
        valid_pos, data.num_nodes,
        num_neg_samples=valid_pos.size(1) * num_neg_samples_eval)
    test_neg = generate_negative_samples(
        test_pos, data.num_nodes,
        num_neg_samples=test_pos.size(1) * num_neg_samples_eval)

    split_edge = {
        'train': {
            'source_node': train_pos[0],
            'target_node': train_pos[1],
        },
        'valid': {
            'source_node': valid_pos[0],
            'target_node': valid_pos[1],
            'target_node_neg': val_neg[1].view(valid_pos.size(1), -1),
        },
        'test': {
            'source_node': test_pos[0],
            'target_node': test_pos[1],
            'target_node_neg': test_neg[1].view(test_pos.size(1), -1),
        },
    }

    print(f"  Train: {train_pos.size(1):,}  Val: {valid_pos.size(1):,}  "
          f"Test: {test_pos.size(1):,}")

    return {
        'data': data,
        'split_edge': split_edge,
        'train_edge_index': train_pos,
        'feature_dim': feature_dim,
        'node2idx': None,
        'idx2node': None,
    }


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
    
    # Prepare split_edge dictionary
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
        'train_edge_index': edge_splits['train'],
        'feature_dim': feature_dim,
    }

