"""
Evaluation for PPR subgraph link prediction.

Fixed version: scores every (u, v) pair (positive AND each negative) with a
fresh joint-seed PPR subgraph extraction around that pair. NO hardcoded
`0.1` floor for out-of-subgraph negatives.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.subgraph_csr import SubgraphCSR


def _build_pair_csr_cache(all_u: torch.Tensor, all_v: torch.Tensor,
                          data, ppr_extractor,
                          progress_desc='Building eval CSR',
                          verbose=True) -> SubgraphCSR:
    def extract(i: int):
        u = int(all_u[i].item())
        v = int(all_v[i].item())

        sub_data, selected_nodes, metadata = ppr_extractor.extract_subgraph(u, v)
        u_sub = metadata.get('u_subgraph', -1)
        v_sub = metadata.get('v_subgraph', -1)
        if u_sub == -1 or v_sub == -1:
            return None

        return (selected_nodes.to(torch.long),
                sub_data.edge_index.to(torch.long),
                int(u_sub), int(v_sub))

    return SubgraphCSR.build(
        num_edges=int(all_u.size(0)),
        extract_fn=extract,
        progress_desc=progress_desc,
        verbose=verbose,
    )


@torch.no_grad()
def evaluate_ppr(encoder, predictor, data, split_edge, ppr_extractor,
                 split='valid', batch_size=1024, device='cpu',
                 K_values=(1, 3, 10, 50, 100),
                 max_edges=None, num_negs_per_pos=None,
                 cache_dir=None, x_full_gpu=None, verbose=True):
    encoder.eval()
    predictor.eval()

    source = split_edge[split]['source_node']
    target = split_edge[split]['target_node']
    target_neg = split_edge[split]['target_node_neg']

    N_total = source.size(0)
    K_total = target_neg.size(1)

    if max_edges is not None and max_edges < N_total:
        gen = torch.Generator().manual_seed(0)
        pos_perm = torch.randperm(N_total, generator=gen)[:max_edges]
        source = source[pos_perm]
        target = target[pos_perm]
        target_neg = target_neg[pos_perm]
        N = int(max_edges)
    else:
        N = N_total

    if num_negs_per_pos is not None and num_negs_per_pos < K_total:
        neg_perm = torch.randperm(K_total)[:num_negs_per_pos]
        target_neg = target_neg[:, neg_perm]
        K = int(num_negs_per_pos)
    else:
        K = K_total

    all_u = source.unsqueeze(1).expand(N, K + 1).reshape(-1).contiguous()
    all_v_mat = torch.cat([target.unsqueeze(1), target_neg], dim=1)
    all_v = all_v_mat.reshape(-1).contiguous()
    M = int(all_u.size(0))

    if cache_dir is not None:
        suffix = f'_me{max_edges}' if max_edges is not None else ''
        suffix += f'_nk{num_negs_per_pos}' if num_negs_per_pos is not None else ''
        path = os.path.join(cache_dir, f'{split}_pair_csr{suffix}.pt')
        use_disk = True
    else:
        use_disk = False
        path = None
    if use_disk:
        if os.path.isfile(path):
            if verbose:
                print(f'[EvalCache] Loading {split} pair CSR from {path}')
            cache = SubgraphCSR.load(path)
        else:
            cache = _build_pair_csr_cache(
                all_u, all_v, data, ppr_extractor,
                progress_desc=f'Building {split} pair CSR', verbose=verbose,
            )
            os.makedirs(cache_dir, exist_ok=True)
            cache.save(path)
            if verbose:
                mb = os.path.getsize(path) / 1e6
                print(f'[EvalCache] Saved: {path} ({mb:.0f} MB)')
    else:
        cache = _build_pair_csr_cache(
            all_u, all_v, data, ppr_extractor,
            progress_desc=f'Building {split} pair CSR (ephemeral)',
            verbose=verbose,
        )

    cache = cache.to(device)
    if x_full_gpu is None:
        x_full_gpu = data.x.to(device)

    all_scores = torch.full((M,), float('nan'), device=device)

    iterator = DataLoader(torch.arange(M).tolist(), batch_size, shuffle=False)
    if verbose:
        iterator = tqdm(iterator, desc=f'Eval {split}',
                        leave=False, mininterval=30)

    for perm in iterator:
        idx = torch.as_tensor(perm, dtype=torch.long, device=device)
        batch = cache.make_batch(idx, x_full_gpu)

        B = int(batch['u_idx'].size(0))
        if B == 0:
            continue

        h = encoder(batch['x'], batch['edge_index'])
        s = predictor(h[batch['u_idx']], h[batch['v_idx']]).view(-1)
        all_scores[batch['valid_idx']] = s.to(all_scores.dtype)

    all_scores = torch.nan_to_num(all_scores, nan=0.0).cpu()
    score_mat = all_scores.view(N, K + 1)
    pos_scores = score_mat[:, 0]
    neg_scores = score_mat[:, 1:]

    return compute_metrics(pos_scores, neg_scores, list(K_values))


def compute_metrics(pos_pred, neg_pred, K_values=(1, 3, 10, 50, 100)) -> dict:
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
    pos_flat = pos_pred.numpy()
    neg_flat = neg_pred.numpy().flatten()
    y_true = np.concatenate([np.ones(len(pos_flat)), np.zeros(len(neg_flat))])
    y_pred = np.concatenate([pos_flat, neg_flat])
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
