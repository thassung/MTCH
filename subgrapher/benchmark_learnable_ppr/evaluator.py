"""
Evaluation for learnable PPR subgraph link prediction.

Two evaluation modes:
  - **subgraph**: per-edge subgraph encoding (fast, used during training).
    Out-of-subgraph negatives receive a fixed low score, which inflates MRR.
  - **fullgraph**: full-graph encoding with all negatives scored by the model.
    This is the gold-standard for comparing against baselines.

Use ``evaluate_learnable_ppr_fullgraph`` for any reported / saved metrics.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph


# ---------------------------------------------------------------------------
# Full-graph evaluation (fair, comparable to benchmark/evaluator.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_learnable_ppr_fullgraph(encoder, predictor, data, split_edge,
                                      split='test', batch_size=4096,
                                      device='cpu',
                                      K_values=None):
    """Evaluate using full-graph encoding — identical methodology to benchmarks.

    The fine-tuned encoder+predictor are applied to the **full graph**, and
    every negative is scored by the model (no 0.1 fallback).  This makes
    results directly comparable to Full Graph / Static PPR / k-hop baselines.

    Args:
        encoder: Fine-tuned GNN encoder
        predictor: Fine-tuned LinkPredictor
        data: PyG Data (full graph with features)
        split_edge: Edge split dictionary
        split: 'valid' or 'test'
        batch_size: Evaluation chunk size
        device: Device
        K_values: Hit@K values to compute

    Returns:
        Dictionary with MRR, AUC, AP, Hits@K
    """
    if K_values is None:
        K_values = [1, 3, 10, 50, 100]

    encoder.eval()
    predictor.eval()

    h = encoder(data.x.to(device), data.edge_index.to(device))

    source = split_edge[split]['source_node']
    target = split_edge[split]['target_node']
    target_neg = split_edge[split]['target_node_neg']

    num_pos = source.size(0)
    num_neg_per_pos = target_neg.size(1)

    eval_bs = min(batch_size, 4096)

    pos_preds = []
    neg_preds = []

    for perm in DataLoader(torch.arange(num_pos), eval_bs):
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

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)

    return _compute_all_metrics(pos_pred, neg_pred, K_values)


# ---------------------------------------------------------------------------
# Subgraph evaluation (used during fine-tuning for speed)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_learnable_ppr(encoder, predictor, data, split_edge,
                            multi_scale_ppr, config_indices,
                            split='valid', alpha=None,
                            epsilon=1e-3, window=10,
                            batch_size=65536, device='cpu',
                            K_values=None,
                            cache_dir=None):
    """
    Evaluate link prediction with learned per-edge PPR configurations.

    Uses LCILP-style approximate PPR + conductance sweep cut for
    subgraph extraction (same as training in Phase 2).

    If *cache_dir* is provided the function loads (or builds + saves)
    a SubgraphCache so repeated evaluations are fast.

    Args:
        encoder: GNN encoder
        predictor: LinkPredictor
        data: PyG Data (full graph)
        split_edge: Edge split dictionary
        multi_scale_ppr: MultiScalePPR instance
        config_indices: Tensor of config indices (for the evaluated split)
        split: 'valid' or 'test'
        alpha: PPR combination weights (list). See resolve_alpha_weights().
        epsilon: Approximate PPR precision (default: 1e-3)
        window: Conductance sweep cut window (default: 10)
        batch_size: Not used for per-edge eval but kept for API compat
        device: Device
        K_values: Hit@K values to compute
        cache_dir: If set, loads/builds SubgraphCache on disk for this split.

    Returns:
        Dictionary with MRR, AUC, AP, Hits@K
    """
    from . import resolve_alpha_weights
    from .finetuner import (LearnablePPRExtractor, SubgraphCache,
                            build_or_load_cache, _long_running_tqdm)
    from ..utils.local_ppr import combine_ppr_and_sweep, build_sparse_adj

    if alpha is None:
        alpha = [0.5]
    if K_values is None:
        K_values = [1, 3, 10, 50, 100]

    w_u, _ = resolve_alpha_weights(alpha)

    encoder.eval()
    predictor.eval()

    source = split_edge[split]['source_node']
    target = split_edge[split]['target_node']
    target_neg = split_edge[split]['target_node_neg']

    num_pos = source.size(0)
    num_neg_per_pos = target_neg.size(1)

    cache = None
    if cache_dir is not None:
        extractor = LearnablePPRExtractor(
            data, multi_scale_ppr, config_indices,
            alpha=alpha, epsilon=epsilon, window=window)
        cache = build_or_load_cache(
            extractor, split_edge, split, cache_dir, verbose=False)

    adj_csr = None
    if cache is None:
        adj_csr = build_sparse_adj(data.edge_index, data.num_nodes)

    pos_preds = []
    neg_preds = []

    for idx in _long_running_tqdm(
            range(num_pos), desc=f'Eval {split}', leave=False):
        u = source[idx].item()
        v_pos = target[idx].item()
        v_negs = target_neg[idx]

        if cache is not None:
            sub_data, u_sub, v_sub = cache.make_data(idx, data.x)
            if sub_data is None:
                pos_preds.append(0.5)
                neg_preds.append([0.5] * num_neg_per_pos)
                continue
            selected_nodes = cache.selected_nodes[idx]
        else:
            if config_indices is not None and idx < len(config_indices):
                cfg_idx = config_indices[idx].item()
            else:
                cfg_idx = 0
            teleport_u, teleport_v = multi_scale_ppr.get_config_for_index(cfg_idx)

            community = combine_ppr_and_sweep(
                adj_csr, u, v_pos,
                alpha_u=teleport_u, alpha_v=teleport_v,
                epsilon=epsilon, blend=w_u, window=window)

            selected_nodes = torch.tensor(sorted(community), dtype=torch.long)

            edge_index_sub, _ = subgraph(
                selected_nodes, data.edge_index,
                relabel_nodes=True, num_nodes=data.num_nodes)

            node_mapping = {node.item(): new_idx
                            for new_idx, node in enumerate(selected_nodes)}
            u_sub = node_mapping.get(u, -1)
            v_sub = node_mapping.get(v_pos, -1)

            if u_sub == -1 or v_sub == -1:
                pos_preds.append(0.5)
                neg_preds.append([0.5] * num_neg_per_pos)
                continue

            x_sub = data.x[selected_nodes]
            sub_data = Data(x=x_sub, edge_index=edge_index_sub)

        sub_data = sub_data.to(device)
        h = encoder(sub_data.x, sub_data.edge_index)

        pos_pred = predictor(h[u_sub].unsqueeze(0), h[v_sub].unsqueeze(0))
        pos_preds.append(pos_pred.item())

        # Fallback embedding for out-of-subgraph negatives: mean of all node
        # embeddings in this subgraph, scored through the predictor. This
        # avoids the pathological 0.1 constant floor that produced bimodal
        # rank distributions (h@3 == h@10 == h@100) and inflated MRR.
        mean_h = h.mean(dim=0, keepdim=True)
        h_u_row = h[u_sub].unsqueeze(0)

        neg_pred_list = []
        for v_neg in v_negs:
            v_neg_item = v_neg.item()
            node_mask = (selected_nodes == v_neg_item)
            if node_mask.any():
                v_neg_sub = node_mask.nonzero(as_tuple=True)[0][0]
                neg_pred = predictor(h_u_row, h[v_neg_sub].unsqueeze(0))
            else:
                neg_pred = predictor(h_u_row, mean_h)
            neg_pred_list.append(neg_pred.item())
        neg_preds.append(neg_pred_list)

        del h, sub_data

    pos_preds = torch.tensor(pos_preds)
    neg_preds = torch.tensor(neg_preds)

    return _compute_all_metrics(pos_preds, neg_preds, K_values)


@torch.no_grad()
def evaluate_search_phase(model, arch_net, multi_scale_ppr, data,
                           split_edge, split='valid', batch_size=1024,
                           device='cpu'):
    """
    Evaluate the architecture search model (Phase 1) on a split.
    Uses full-graph encoding + arch_net attention, no subgraph extraction.

    Returns:
        Dictionary with AUC and AP
    """
    model.eval()
    arch_net.eval()

    source = split_edge[split]['source_node'].to(device)
    target = split_edge[split]['target_node'].to(device)
    target_neg = split_edge[split]['target_node_neg'].to(device)

    num_pos = source.size(0)
    data = data.to(device)
    h = model(data.x, data.edge_index)

    pos_preds = []
    neg_preds_flat = []

    for perm in DataLoader(torch.arange(num_pos), batch_size, shuffle=False):
        src = source[perm]
        dst = target[perm]
        edges = torch.stack([src, dst], dim=0)
        pred = model.compute_pred(h, arch_net, multi_scale_ppr, edges)
        pos_preds.append(pred.squeeze().cpu())

        for i, idx in enumerate(perm):
            v_negs = target_neg[idx]
            src_rep = src[i].unsqueeze(0).expand(len(v_negs))
            neg_edge = torch.stack([src_rep, v_negs], dim=0)
            neg_pred = model.compute_pred(h, arch_net, multi_scale_ppr,
                                           neg_edge)
            neg_preds_flat.append(neg_pred.squeeze().cpu())

    pos_preds = torch.cat(pos_preds).numpy()
    neg_preds_flat = torch.cat(neg_preds_flat).numpy()

    y_true = np.concatenate([np.ones(len(pos_preds)),
                              np.zeros(len(neg_preds_flat))])
    y_pred = np.concatenate([pos_preds, neg_preds_flat])

    return {
        'auc': roc_auc_score(y_true, y_pred),
        'ap': average_precision_score(y_true, y_pred),
    }


def _compute_all_metrics(pos_pred, neg_pred, K_values):
    """Compute MRR, AUC, AP, Hits@K from prediction tensors."""
    ranks = 1 + (neg_pred >= pos_pred.unsqueeze(1)).sum(dim=1).float()
    mrr = (1.0 / ranks).mean().item()

    hits = {}
    for k in K_values:
        if k <= neg_pred.size(1):
            hits[f'hits@{k}'] = (ranks <= k).float().mean().item()

    pos_np = pos_pred.numpy()
    neg_np = neg_pred.numpy().flatten()
    y_true = np.concatenate([np.ones(len(pos_np)), np.zeros(len(neg_np))])
    y_pred = np.concatenate([pos_np, neg_np])
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    return {'mrr': mrr, 'auc': auc, 'ap': ap, **hits}


def print_evaluation_results(results, split='valid'):
    """Print evaluation results in a readable format."""
    print(f"\n{split.capitalize()} Results:")
    print(f"  MRR:      {results['mrr']:.4f}")
    print(f"  AUC:      {results['auc']:.4f}")
    print(f"  AP:       {results['ap']:.4f}")
    for k in [1, 3, 10, 50, 100]:
        key = f'hits@{k}'
        if key in results:
            print(f"  Hits@{k:<3}: {results[key]:.4f}")
