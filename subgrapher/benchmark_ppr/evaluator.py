"""
Evaluation metrics for PPR-based subgraph link prediction.
Modified from benchmark/evaluator.py to extract subgraphs before encoding.
"""

import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm


class PPREvalCache:
    """Pre-extracted subgraph topology for validation/test edges.

    Stores selected_nodes, edge_index, u_sub, v_sub per positive edge.
    Built once, reused every eval call within the same training run.
    """

    def __init__(self, selected_nodes, edge_index, u_sub, v_sub, num_nodes):
        self.selected_nodes = selected_nodes
        self.edge_index = edge_index
        self.u_sub = u_sub
        self.v_sub = v_sub
        self.num_nodes = num_nodes

    def __len__(self):
        return len(self.selected_nodes)

    @classmethod
    def build(cls, source, target, data, ppr_extractor, verbose=True):
        from torch_geometric.utils import subgraph as pyg_subgraph
        n = source.size(0)
        sel_list, ei_list, u_list, v_list, nn_list = [], [], [], [], []

        it = tqdm(range(n), desc='Building eval cache', leave=False,
                  mininterval=10) if verbose else range(n)
        for i in it:
            u = source[i].item()
            v = target[i].item()

            _, selected_nodes, metadata = ppr_extractor.extract_subgraph(u, v)
            u_sub = metadata['u_subgraph']
            v_sub = metadata['v_subgraph']

            edge_index_sub, _ = pyg_subgraph(
                selected_nodes, data.edge_index,
                relabel_nodes=True, num_nodes=data.num_nodes)

            sel_list.append(selected_nodes.cpu())
            ei_list.append(edge_index_sub.cpu())
            u_list.append(u_sub)
            v_list.append(v_sub)
            nn_list.append(len(selected_nodes))

        return cls(sel_list, ei_list, u_list, v_list, nn_list)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'selected_nodes': self.selected_nodes,
            'edge_index': self.edge_index,
            'u_sub': self.u_sub, 'v_sub': self.v_sub,
            'num_nodes': self.num_nodes,
        }, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location='cpu', weights_only=False)
        return cls(d['selected_nodes'], d['edge_index'],
                   d['u_sub'], d['v_sub'], d['num_nodes'])

    def make_data(self, idx, x_full):
        sel = self.selected_nodes[idx]
        if len(sel) == 0:
            return None, -1, -1, sel
        x_sub = x_full[sel]
        return (Data(x=x_sub, edge_index=self.edge_index[idx],
                     num_nodes=self.num_nodes[idx]),
                self.u_sub[idx], self.v_sub[idx], sel)


def build_or_load_eval_cache(source, target, data, ppr_extractor,
                              cache_dir=None, split='valid', verbose=True):
    if cache_dir:
        path = os.path.join(cache_dir, f'{split}_subgraphs.pt')
        if os.path.isfile(path):
            if verbose:
                print(f'[EvalCache] Loading {split} subgraphs from {path}')
            return PPREvalCache.load(path)

    if verbose:
        print(f'[EvalCache] Extracting {split} subgraphs (one-time cost)...')
    cache = PPREvalCache.build(source, target, data, ppr_extractor, verbose)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f'{split}_subgraphs.pt')
        cache.save(path)
        mb = os.path.getsize(path) / 1e6
        if verbose:
            print(f'[EvalCache] Saved: {path} ({mb:.0f} MB)')
    return cache


@torch.no_grad()
def evaluate_ppr(encoder, predictor, data, split_edge, ppr_extractor,
                 split='valid', batch_size=65536, device='cpu',
                 K_values=[1, 3, 10, 50, 100], max_edges=None,
                 eval_cache=None, cache_dir=None):
    """
    Evaluation with PPR-based subgraph extraction.
    Pass eval_cache (PPREvalCache) to skip redundant subgraph extraction.
    Pass cache_dir to auto-build and persist the eval cache to disk.
    Set max_edges to subsample for fast validation during training.
    """
    encoder.eval()
    predictor.eval()

    source = split_edge[split]['source_node']
    target = split_edge[split]['target_node']
    target_neg = split_edge[split]['target_node_neg']

    num_pos_edges = source.size(0)
    num_neg_per_pos = target_neg.size(1)

    if max_edges and max_edges < num_pos_edges:
        perm = torch.randperm(num_pos_edges)[:max_edges]
        source = source[perm]
        target = target[perm]
        target_neg = target_neg[perm]
        num_pos_edges = max_edges
        use_cache = False
    else:
        use_cache = True

    if use_cache and eval_cache is None and cache_dir:
        eval_cache = build_or_load_eval_cache(
            source, target, data, ppr_extractor,
            cache_dir=cache_dir, split=split, verbose=True)

    pos_preds = []
    neg_preds = []

    for idx in tqdm(range(num_pos_edges), desc=f'Eval {split}',
                    leave=False, mininterval=30):
        u = source[idx].item()
        v_negs = target_neg[idx]

        if use_cache and eval_cache is not None:
            sub_data, u_sub, v_sub, selected_nodes = eval_cache.make_data(idx, data.x)
            if sub_data is None:
                pos_preds.append(0.5)
                neg_preds.append([0.5] * num_neg_per_pos)
                continue
            sub_data = sub_data.to(device)
        else:
            v_pos = target[idx].item()
            subgraph_data, selected_nodes, metadata = ppr_extractor.extract_subgraph(u, v_pos)
            sub_data = subgraph_data.to(device)
            u_sub = metadata['u_subgraph']
            v_sub = metadata['v_subgraph']
            if u_sub == -1 or v_sub == -1:
                pos_preds.append(0.5)
                neg_preds.append([0.5] * num_neg_per_pos)
                del sub_data
                continue

        h = encoder(sub_data.x, sub_data.edge_index)

        pos_pred = predictor(h[u_sub].unsqueeze(0), h[v_sub].unsqueeze(0))
        pos_preds.append(pos_pred.item())

        neg_pred_list = []
        for v_neg in v_negs:
            v_neg_item = v_neg.item()
            node_mask = (selected_nodes == v_neg_item)
            if node_mask.any():
                v_neg_sub = node_mask.nonzero(as_tuple=True)[0][0]
                neg_pred = predictor(h[u_sub].unsqueeze(0), h[v_neg_sub].unsqueeze(0))
                neg_pred_list.append(neg_pred.item())
            else:
                neg_pred_list.append(0.1)

        neg_preds.append(neg_pred_list)
        del h, sub_data

    pos_preds = torch.tensor(pos_preds)
    neg_preds = torch.tensor(neg_preds)

    results = compute_metrics(pos_preds, neg_preds, K_values)
    return results


def compute_metrics(pos_pred, neg_pred, K_values=[1, 3, 10, 50, 100]):
    mrr = compute_mrr(pos_pred, neg_pred)
    hits = {}
    for k in K_values:
        if k <= neg_pred.size(1):
            hits[f'hits@{k}'] = compute_hits_at_k(pos_pred, neg_pred, k)
    auc, ap = compute_auc_ap(pos_pred, neg_pred)
    return {'mrr': mrr, 'auc': auc, 'ap': ap, **hits}


def compute_mrr(pos_pred, neg_pred):
    ranks = 1 + (neg_pred >= pos_pred.unsqueeze(1)).sum(dim=1).float()
    return (1.0 / ranks).mean().item()


def compute_hits_at_k(pos_pred, neg_pred, k):
    ranks = 1 + (neg_pred >= pos_pred.unsqueeze(1)).sum(dim=1).float()
    return (ranks <= k).float().mean().item()


def compute_auc_ap(pos_pred, neg_pred):
    pos_pred_flat = pos_pred.numpy()
    neg_pred_flat = neg_pred.numpy().flatten()
    y_true = np.concatenate([np.ones(len(pos_pred_flat)),
                             np.zeros(len(neg_pred_flat))])
    y_pred = np.concatenate([pos_pred_flat, neg_pred_flat])
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return auc, ap


def print_evaluation_results(results, split='valid'):
    print(f"\n{split.capitalize()} Results:")
    print(f"  MRR:      {results['mrr']:.4f}")
    print(f"  AUC:      {results['auc']:.4f}")
    print(f"  AP:       {results['ap']:.4f}")
    for k in [1, 3, 10, 50, 100]:
        key = f'hits@{k}'
        if key in results:
            print(f"  Hits@{k:<3}: {results[key]:.4f}")
