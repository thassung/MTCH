"""
Evaluation metrics for k-hop subgraph link prediction.
Modified from benchmark/evaluator.py to extract subgraphs before encoding.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_khop(encoder, predictor, data, split_edge, khop_extractor,
                  split='valid', batch_size=65536, device='cpu', K_values=[1, 3, 10, 50, 100]):
    """
    Comprehensive evaluation with k-hop subgraph extraction.
    
    Args:
        encoder: GNN encoder model
        predictor: Link predictor model
        data: PyG Data object
        split_edge: Dictionary with edge splits
        khop_extractor: StaticKHopExtractor instance
        split: 'valid' or 'test'
        batch_size: Batch size for evaluation
        device: Device
        K_values: List of K values for Hit@K metric
    Returns:
        Dictionary with all metrics
    """
    encoder.eval()
    predictor.eval()
    
    # Get positive and negative edges
    source = split_edge[split]['source_node']
    target = split_edge[split]['target_node']
    target_neg = split_edge[split]['target_node_neg']
    
    num_pos_edges = source.size(0)
    num_neg_per_pos = target_neg.size(1)
    
    # Compute positive and negative predictions
    pos_preds = []
    neg_preds = []
    
    # Process with progress bar
    for idx in tqdm(range(num_pos_edges), desc=f'Evaluating {split}', leave=False):
        u = source[idx].item()
        v_pos = target[idx].item()
        v_negs = target_neg[idx]
        
        # Extract subgraph for this edge
        subgraph_data, selected_nodes, metadata = khop_extractor.extract_subgraph(u, v_pos)
        subgraph_data = subgraph_data.to(device)
        
        u_sub = metadata['u_subgraph']
        v_sub = metadata['v_subgraph']
        
        if u_sub == -1 or v_sub == -1:
            # Fallback: use random score
            pos_preds.append(0.5)
            neg_preds.append([0.5] * num_neg_per_pos)
            continue
        
        # Encode subgraph
        h = encoder(subgraph_data.x, subgraph_data.edge_index)
        
        # Positive prediction
        pos_pred = predictor(h[u_sub].unsqueeze(0), h[v_sub].unsqueeze(0))
        pos_preds.append(pos_pred.item())
        
        # Negative predictions
        neg_pred_list = []
        for v_neg in v_negs:
            v_neg_item = v_neg.item()
            
            # Check if negative node is in subgraph
            node_mask = (selected_nodes == v_neg_item)
            if node_mask.any():
                v_neg_sub = node_mask.nonzero(as_tuple=True)[0][0]
                neg_pred = predictor(h[u_sub].unsqueeze(0), h[v_neg_sub].unsqueeze(0))
                neg_pred_list.append(neg_pred.item())
            else:
                # Negative node not in subgraph, assign low score
                neg_pred_list.append(0.1)
        
        neg_preds.append(neg_pred_list)
    
    # Convert to tensors
    pos_preds = torch.tensor(pos_preds)
    neg_preds = torch.tensor(neg_preds)
    
    # Compute metrics
    results = compute_metrics(pos_preds, neg_preds, K_values)
    
    return results


def compute_metrics(pos_pred, neg_pred, K_values=[1, 3, 10, 50, 100]):
    """
    Compute evaluation metrics from predictions.
    
    Args:
        pos_pred: Positive predictions [num_pos]
        neg_pred: Negative predictions [num_pos, num_neg]
        K_values: List of K values for Hit@K
    Returns:
        Dictionary with metrics
    """
    # MRR (Mean Reciprocal Rank)
    mrr = compute_mrr(pos_pred, neg_pred)
    
    # Hit@K for multiple K values
    hits = {}
    for k in K_values:
        if k <= neg_pred.size(1):
            hits[f'hits@{k}'] = compute_hits_at_k(pos_pred, neg_pred, k)
    
    # AUC and AP
    auc, ap = compute_auc_ap(pos_pred, neg_pred)
    
    results = {
        'mrr': mrr,
        'auc': auc,
        'ap': ap,
        **hits
    }
    
    return results


def compute_mrr(pos_pred, neg_pred):
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        pos_pred: Positive predictions [num_pos]
        neg_pred: Negative predictions [num_pos, num_neg]
    Returns:
        MRR score
    """
    # For each positive, count how many negatives score higher
    # Rank = 1 + number of negatives with higher score
    ranks = 1 + (neg_pred >= pos_pred.unsqueeze(1)).sum(dim=1).float()
    
    # Reciprocal rank
    reciprocal_ranks = 1.0 / ranks
    
    # Mean reciprocal rank
    mrr = reciprocal_ranks.mean().item()
    
    return mrr


def compute_hits_at_k(pos_pred, neg_pred, k):
    """
    Compute Hit@K metric.
    
    Args:
        pos_pred: Positive predictions [num_pos]
        neg_pred: Negative predictions [num_pos, num_neg]
        k: K value
    Returns:
        Hit@K score
    """
    # For each positive, count how many of top-k negatives score higher
    # Hit if positive is in top-k (rank <= k)
    ranks = 1 + (neg_pred >= pos_pred.unsqueeze(1)).sum(dim=1).float()
    hits = (ranks <= k).float()
    
    return hits.mean().item()


def compute_auc_ap(pos_pred, neg_pred):
    """
    Compute AUC-ROC and Average Precision.
    
    Args:
        pos_pred: Positive predictions [num_pos]
        neg_pred: Negative predictions [num_pos, num_neg]
    Returns:
        AUC, AP scores
    """
    # Flatten predictions
    pos_pred_flat = pos_pred.numpy()
    neg_pred_flat = neg_pred.numpy().flatten()
    
    # Create labels (1 for positive, 0 for negative)
    y_true = np.concatenate([
        np.ones(len(pos_pred_flat)),
        np.zeros(len(neg_pred_flat))
    ])
    
    # Concatenate predictions
    y_pred = np.concatenate([pos_pred_flat, neg_pred_flat])
    
    # Compute metrics
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    
    return auc, ap


def print_evaluation_results(results, split='valid'):
    """
    Print evaluation results in a nice format.
    
    Args:
        results: Dictionary with evaluation metrics
        split: 'valid' or 'test'
    """
    print(f"\n{split.capitalize()} Results:")
    print(f"  MRR:      {results['mrr']:.4f}")
    print(f"  AUC:      {results['auc']:.4f}")
    print(f"  AP:       {results['ap']:.4f}")
    
    for k in [1, 3, 10, 50, 100]:
        key = f'hits@{k}'
        if key in results:
            print(f"  Hits@{k:<3}: {results[key]:.4f}")

