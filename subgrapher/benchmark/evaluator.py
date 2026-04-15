"""
Evaluation metrics for link prediction.
Includes Hit@K, MRR, AUC-ROC, and Average Precision.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_link_prediction(encoder, predictor, data, split_edge, split='valid', 
                              batch_size=65536, K_values=[1, 3, 10, 50, 100]):
    """    
    Evaluate link prediction with chunked negative scoring to limit GPU memory.
    """
    encoder.eval()
    predictor.eval()
    
    device = next(encoder.parameters()).device
    use_amp = (device.type == 'cuda')
    
    with torch.amp.autocast('cuda', enabled=use_amp):
        h = encoder(data.x.to(device), data.edge_index.to(device))
    h = h.detach()
    
    source = split_edge[split]['source_node']
    target = split_edge[split]['target_node']
    target_neg = split_edge[split]['target_node_neg']
    
    num_pos_edges = source.size(0)
    num_neg_per_pos = target_neg.size(1)
    
    eval_bs = min(batch_size, 2048)
    
    pos_preds = []
    neg_preds = []
    
    for perm in DataLoader(torch.arange(num_pos_edges), eval_bs):
        src = source[perm].to(device)
        dst = target[perm].to(device)
        pos_preds.append(predictor(h[src], h[dst]).squeeze().cpu())
        
        dst_neg_chunk = target_neg[perm].to(device)
        src_rep = src.unsqueeze(1).expand_as(dst_neg_chunk).reshape(-1)
        dst_neg_flat = dst_neg_chunk.reshape(-1)
        
        chunk_neg = []
        for neg_start in range(0, src_rep.size(0), eval_bs):
            neg_end = min(neg_start + eval_bs, src_rep.size(0))
            chunk_neg.append(
                predictor(h[src_rep[neg_start:neg_end]],
                          h[dst_neg_flat[neg_start:neg_end]]).squeeze().cpu())
        neg_preds.append(torch.cat(chunk_neg).view(len(perm), num_neg_per_pos))
        
        del src, dst, dst_neg_chunk, src_rep, dst_neg_flat, chunk_neg
    
    del h
    torch.cuda.empty_cache()
    
    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)
    
    # Compute MRR (Mean Reciprocal Rank)
    mrr = compute_mrr(pos_pred, neg_pred)
    
    # Compute Hit@K for multiple K values
    hits = {}
    for k in K_values:
        if k <= num_neg_per_pos:
            hits[f'hits@{k}'] = compute_hits_at_k(pos_pred, neg_pred, k)
    
    # Compute AUC and AP
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
        pos_pred: Positive predictions [num_edges]
        neg_pred: Negative predictions [num_edges, num_neg_samples]
    Returns:
        MRR score
    """
    # For each positive edge, count how many negatives score higher
    pos_pred = pos_pred.unsqueeze(1)  # [num_edges, 1]
    ranks = (neg_pred >= pos_pred).sum(dim=1) + 1  # +1 for the positive edge itself
    
    mrr = (1.0 / ranks.float()).mean().item()
    return mrr


def compute_hits_at_k(pos_pred, neg_pred, k):
    """
    Compute Hit@K metric.
    Percentage of positive edges ranked in top-K.
    
    Args:
        pos_pred: Positive predictions [num_edges]
        neg_pred: Negative predictions [num_edges, num_neg_samples]
        k: K value
    Returns:
        Hit@K score
    """
    pos_pred = pos_pred.unsqueeze(1)  # [num_edges, 1]
    ranks = (neg_pred >= pos_pred).sum(dim=1) + 1
    
    hits = (ranks <= k).float().mean().item()
    return hits


def compute_auc_ap(pos_pred, neg_pred):
    """
    Compute AUC-ROC and Average Precision.
    
    Args:
        pos_pred: Positive predictions [num_edges]
        neg_pred: Negative predictions [num_edges, num_neg_samples]
    Returns:
        (auc, ap) tuple
    """
    # Flatten predictions and create labels
    pos_pred_np = pos_pred.cpu().numpy()
    neg_pred_np = neg_pred.cpu().numpy().flatten()
    
    predictions = np.concatenate([pos_pred_np, neg_pred_np])
    labels = np.concatenate([np.ones(len(pos_pred_np)), np.zeros(len(neg_pred_np))])
    
    # Compute metrics
    try:
        auc = roc_auc_score(labels, predictions)
        ap = average_precision_score(labels, predictions)
    except:
        # In case of errors (e.g., all same predictions)
        auc = 0.5
        ap = 0.5
    
    return auc, ap


def print_evaluation_results(results, model_name='Model', split='Test'):
    """
    Pretty print evaluation results.
    
    Args:
        results: Dictionary with evaluation metrics
        model_name: Name of the model
        split: 'Valid' or 'Test'
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - {split} Set Results")
    print(f"{'='*60}")
    print(f"MRR:             {results['mrr']:.4f}")
    print(f"AUC:             {results['auc']:.4f}")
    print(f"AP:              {results['ap']:.4f}")
    
    # Print Hit@K metrics
    for key in sorted(results.keys()):
        if key.startswith('hits@'):
            k = key.split('@')[1]
            print(f"Hit@{k:<3}:         {results[key]:.4f}")
    print(f"{'='*60}")

