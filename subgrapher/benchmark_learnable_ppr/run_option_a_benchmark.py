"""
Option A benchmark runner.

Unified PS2-style learnable PPR: soft PPR adjacency within subgraphs.
- Selector (theta) trained on val loss
- PPR-diffusion GNN (w) trained on train loss
- Evaluation: per-subgraph scoring with hard selector (argmax alpha)
"""

import torch
import torch.nn as nn
import argparse
import json
import os
import time
import copy
import numpy as np
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader
from tqdm import tqdm

from ..benchmark.data_prep import prepare_planetoid_data
from .multi_scale_ppr import MultiScalePPR
from .option_a_model import OptionAGNN, PPRScaleSelector
from .option_a_extractor import OptionAExtractor, build_or_load_cache, sample_neg_subgraphs
from .option_a_searcher import OptionASearcher
from .artifacts import save_learnable_ppr_experiment

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'feature_method': 'random',
    'feature_dim': 128,
    'hidden_channels': 256,
    'num_layers': 3,
    'dropout': 0.3,
    # Selector
    'selector_hidden': 256,
    'selector_layers': 3,
    'selector_scale_emb_dim': 32,
    # Search
    'search_epochs': 50,
    'search_batch_size': 32,       # small — each batch is B subgraph GNN forwards
    'search_lr': 0.01,
    'search_lr_selector': 3e-4,
    'search_lr_min': 1e-3,
    'temperature_start': 1.0,
    'temperature_end': 0.2,
    'edges_per_search_epoch': 10_000,
    'search_val_every': 5,
    # Fine-tune
    'finetune_epochs': 100,
    'finetune_batch_size': 32,
    'finetune_lr': 0.005,
    'finetune_patience': 20,
    'finetune_eval_steps': 5,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    # PPR / extraction
    'teleport_values': [0.90, 0.50, 0.15],   # classic restart (high=local); empirical sizes ~12/42/98 at τ=1e-3
    # Coarse-coarse eps: same precision for train pos cache AND live negs+val
    # avoids the subgraph-size-as-label-leak that asymmetric eps creates.
    # 5e-4 is fast enough to extract live during search/eval.
    'push_epsilon': 5e-4,
    'score_tau': 1e-3,
    'extraction_alpha': 0.15,                # widest scale for envelope subgraph
    # Datasets / encoders
    'datasets': ['Cora'],
    'save_run_artifacts': True,
    'use_checkpoint_cache': True,
    # Full-graph eval (apples-to-apples vs full-graph baselines)
    'full_graph_eval': True,
    'full_graph_eval_max_nodes': 8000,        # skip on PubMed (~19.7k), runs on Cora/CiteSeer
    'cache_root': 'cache',                    # repo-root-relative cache layout
    'results_root': 'results/benchmark-option-a',
}


# ---------------------------------------------------------------------------
# Evaluation (per-subgraph)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_option_a(model, selector, extractor, multi_scale_ppr,
                      data, split_edge, split='valid',
                      batch_size=32, device='cpu',
                      K_values=None, max_neg_per_pos=1000):
    """
    Per-subgraph evaluation for Option A.

    For each positive edge (u,v): extract subgraph, run model, get score.
    For each negative (u, neg_j): same.

    NOTE: Slow for large negative sets. Set max_neg_per_pos to cap.
    """
    if K_values is None:
        K_values = [1, 3, 10, 50, 100]

    model.eval()
    selector.eval()

    ppr_dense = multi_scale_ppr.ppr_dense
    alphas = model.alphas
    x_full = data.x.float().to(device)

    source = split_edge[split]['source_node']
    target = split_edge[split]['target_node']
    target_neg = split_edge[split]['target_node_neg']  # [num_pos, num_neg]

    num_pos = source.size(0)
    num_neg_per_pos = min(target_neg.size(1), max_neg_per_pos)

    pos_preds, neg_preds_all = [], []

    it = tqdm(range(num_pos), desc=f'Eval {split}', mininterval=5)
    for i in it:
        u = source[i].item()
        v = target[i].item()

        # Positive
        nodes_S, u_loc, v_loc = extractor.extract(u, v)
        cross_i = model.compute_selector_input(
            torch.tensor([u], device=device),
            torch.tensor([v], device=device),
            x_full, ppr_dense, alphas)
        w_i = selector(cross_i)[0]
        h_u, h_v = model.forward_subgraph(nodes_S, u_loc, v_loc, x_full, w_i, ppr_dense)
        pos_preds.append(model.predict(h_u.unsqueeze(0), h_v.unsqueeze(0)).item())

        # Negatives
        neg_scores = []
        for j in range(num_neg_per_pos):
            neg_v = target_neg[i, j].item()
            n_S, n_ul, n_vl = extractor.extract(u, neg_v)
            cross_n = model.compute_selector_input(
                torch.tensor([u], device=device),
                torch.tensor([neg_v], device=device),
                x_full, ppr_dense, alphas)
            w_n = selector(cross_n)[0]
            h_u2, h_neg = model.forward_subgraph(n_S, n_ul, n_vl, x_full, w_n, ppr_dense)
            neg_scores.append(model.predict(h_u2.unsqueeze(0), h_neg.unsqueeze(0)).item())
        neg_preds_all.append(neg_scores)

    # Compute MRR and Hits@K
    pos_arr = np.array(pos_preds)          # [num_pos]
    neg_arr = np.array(neg_preds_all)      # [num_pos, num_neg_per_pos]

    # rank of positive among (1 positive + num_neg_per_pos negatives)
    ranks = (neg_arr >= pos_arr[:, None]).sum(axis=1) + 1  # 1-indexed

    mrr = float((1.0 / ranks).mean())
    hits = {f'hits@{k}': float((ranks <= k).mean()) for k in K_values}

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        flat_pos = np.repeat(pos_arr, num_neg_per_pos)
        flat_neg = neg_arr.flatten()
        all_preds = np.concatenate([flat_pos, flat_neg])
        all_labels = np.concatenate([np.ones(len(flat_pos)), np.zeros(len(flat_neg))])
        auc = float(roc_auc_score(all_labels, all_preds))
        ap = float(average_precision_score(all_labels, all_preds))
    except Exception:
        auc, ap = 0.0, 0.0

    results = {'mrr': mrr, 'auc': auc, 'ap': ap, **hits}
    return results


@torch.no_grad()
def evaluate_option_a_full_graph(model, selector, multi_scale_ppr,
                                 data, split_edge, split='test',
                                 batch_size=2048, device='cpu',
                                 K_values=None, max_nodes=8000):
    """
    Full-graph 1000-neg evaluation for LPPR — apples-to-apples vs the
    Full-Graph / Static-PPR / k-hop baselines.

    Computes a single P_soft over the full graph using the train-set average
    selector weights (one alpha mixture, not per-edge), runs the encoder
    once, then scores all eval edges via dot/predict.

    Memory: dense [N,N] tensors per scale. Skipped if N > max_nodes.

    Returns:
        results dict (mrr, auc, ap, hits@K) or None if skipped.
    """
    if K_values is None:
        K_values = [1, 3, 10, 50, 100]

    N = data.num_nodes
    if N > max_nodes:
        print(f'  [full-graph eval] SKIPPED: N={N} > max_nodes={max_nodes} '
              f'(would OOM). Per-subgraph result still produced.')
        return None

    model.eval()
    selector.eval()

    ppr_dense = multi_scale_ppr.ppr_dense
    alphas = model.alphas
    x_full = data.x.float().to(device)

    # 1. Get average selector weights over train edges (single forward over train)
    train_src = split_edge['train']['source_node'].to(device)
    train_dst = split_edge['train']['target_node'].to(device)
    avg_w = torch.zeros(len(alphas), device=device)
    cnt = 0
    for perm in DataLoader(torch.arange(train_src.size(0)), 1024):
        cross = model.compute_selector_input(
            train_src[perm], train_dst[perm], x_full, ppr_dense, alphas)
        w = selector(cross)  # [B, K]; eval mode → one-hot
        avg_w = avg_w + w.sum(dim=0)
        cnt += len(perm)
    avg_w = avg_w / max(cnt, 1)

    # 2. Build full-graph P_soft: per-scale sym-norm + I, then weighted mix
    eye = torch.eye(N, device=device)
    P_full = torch.zeros(N, N, device=device)
    for i, alpha in enumerate(alphas):
        mat = ppr_dense[alpha]
        P_slice = mat.to(device) if mat.device != device else mat
        A_tilde = eye + P_slice
        d_inv_sqrt = A_tilde.sum(dim=1).clamp(min=1e-6).pow(-0.5)
        P_hat = d_inv_sqrt.unsqueeze(1) * A_tilde * d_inv_sqrt.unsqueeze(0)
        P_full = P_full + avg_w[i] * P_hat

    # 3. Single encoder forward
    h = model.encoder(x_full, P_full)

    # 4. Score eval edges (chunked, mirroring evaluate_link_prediction)
    source = split_edge[split]['source_node']
    target = split_edge[split]['target_node']
    target_neg = split_edge[split]['target_node_neg']
    num_pos = source.size(0)
    num_neg = target_neg.size(1)

    eval_bs = min(batch_size, 2048)
    pos_preds, neg_preds = [], []
    for perm in DataLoader(torch.arange(num_pos), eval_bs):
        src = source[perm].to(device)
        dst = target[perm].to(device)
        pos_preds.append(model.predict(h[src], h[dst]).squeeze(-1).cpu())

        dst_neg_chunk = target_neg[perm].to(device)
        src_rep = src.unsqueeze(1).expand_as(dst_neg_chunk).reshape(-1)
        dst_neg_flat = dst_neg_chunk.reshape(-1)
        chunk_neg = []
        for ns in range(0, src_rep.size(0), eval_bs):
            ne = min(ns + eval_bs, src_rep.size(0))
            chunk_neg.append(
                model.predict(h[src_rep[ns:ne]],
                              h[dst_neg_flat[ns:ne]]).squeeze(-1).cpu())
        neg_preds.append(torch.cat(chunk_neg).view(len(perm), num_neg))

    del P_full, h, eye
    if device != 'cpu':
        torch.cuda.empty_cache()

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)

    from ..benchmark.evaluator import compute_mrr, compute_hits_at_k, compute_auc_ap
    mrr = compute_mrr(pos_pred, neg_pred)
    hits = {f'hits@{k}': compute_hits_at_k(pos_pred, neg_pred, k)
            for k in K_values if k <= num_neg}
    auc, ap = compute_auc_ap(pos_pred, neg_pred)

    return {
        'mrr': mrr, 'auc': auc, 'ap': ap,
        'avg_alpha_weights': [float(x) for x in avg_w.cpu().tolist()],
        **hits,
    }


# ---------------------------------------------------------------------------
# Fine-tuning with frozen selector
# ---------------------------------------------------------------------------

def finetune_option_a(model, selector, extractor, multi_scale_ppr,
                      data, split_edge, train_cache,
                      epochs=100, batch_size=32, lr=0.005,
                      weight_decay=1e-5, grad_clip=1.0, patience=20,
                      eval_steps=5, device='cpu', verbose=True,
                      checkpoint_dir=None, neg_extractor=None):
    """
    Fine-tune OptionAGNN with frozen selector (hard alpha selection).

    The selector is set to eval mode (hard argmax) and NOT updated.
    Only model (encoder + predictor) weights are trained.
    """
    from sklearn.metrics import roc_auc_score
    from torch_geometric.utils import negative_sampling, add_self_loops

    _neg_ext = neg_extractor if neg_extractor is not None else extractor

    model.to(device)
    selector.to(device)
    selector.eval()  # hard selection for fine-tuning
    # Explicitly freeze selector params (PS2 convention). Optimizer is
    # already scoped to model.parameters() only, but disabling grad here
    # also saves autograd memory during the train backward.
    for p in selector.parameters():
        p.requires_grad_(False)

    ppr_dense = multi_scale_ppr.ppr_dense
    alphas = model.alphas
    x_full = data.x.float().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    train_src = split_edge['train']['source_node'].to(device)
    train_dst = split_edge['train']['target_node'].to(device)
    pos_edge = torch.stack([train_src, train_dst], dim=0)
    neg_idx, _ = add_self_loops(pos_edge)
    neg_idx = neg_idx.to(device)

    n_train = len(train_cache)
    history = {
        'train_loss': [], 'val_loss': [],
        'best_val_loss': float('inf'), 'best_epoch': 0,
        'stopped_early': False, 'total_time': 0.0,
    }
    best_val = float('inf')
    no_improve = 0
    best_state = None
    _start = time.time()

    iterator = tqdm(range(1, epochs + 1), desc='Fine-tune',
                    mininterval=10) if verbose else range(1, epochs + 1)

    for epoch in iterator:
        model.train()
        indices = torch.randperm(n_train)
        epoch_loss = 0.0
        steps = 0

        for perm in DataLoader(indices.tolist(), batch_size, shuffle=False):
            pos_subs = [train_cache[i] for i in perm]
            train_edge = torch.stack(
                [train_src[perm], train_dst[perm]], dim=0)
            train_neg = negative_sampling(
                neg_idx, num_nodes=data.num_nodes,
                num_neg_samples=len(perm)).to(device)
            neg_subs = sample_neg_subgraphs(_neg_ext, train_neg)

            optimizer.zero_grad()
            loss = model.compute_loss(
                selector, train_edge, train_neg, x_full,
                ppr_dense, pos_subs, neg_subs)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        history['train_loss'].append(avg_loss)

        if epoch % eval_steps == 0 or epoch == epochs:
            # Quick val loss (not full MRR — too slow during training)
            model.eval()
            with torch.no_grad():
                val_src = split_edge['valid']['source_node'].to(device)
                val_dst = split_edge['valid']['target_node'].to(device)
                n_val_sample = min(256, val_src.size(0))
                vi = torch.randperm(val_src.size(0))[:n_val_sample]
                val_edge = torch.stack([val_src[vi], val_dst[vi]], dim=0)
                val_neg = negative_sampling(
                    neg_idx, num_nodes=data.num_nodes,
                    num_neg_samples=n_val_sample).to(device)
                vp_subs = sample_neg_subgraphs(_neg_ext, val_edge)
                vn_subs = sample_neg_subgraphs(_neg_ext, val_neg)
                vl = model.compute_loss(
                    selector, val_edge, val_neg, x_full,
                    ppr_dense, vp_subs, vn_subs).item()

            history['val_loss'].append(vl)
            scheduler.step(vl)

            if vl < best_val - 1e-4:
                best_val = vl
                history['best_val_loss'] = best_val
                history['best_epoch'] = epoch
                no_improve = 0
                best_state = {
                    'model': copy.deepcopy(model.state_dict()),
                }
            else:
                no_improve += eval_steps

            if verbose:
                iterator.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'val': f'{vl:.4f}',
                    'best': f'{best_val:.4f}',
                    'pat': f'{no_improve}/{patience}',
                })

            if no_improve >= patience:
                history['stopped_early'] = True
                if verbose:
                    print(f'\n[Early Stop] epoch {epoch}')
                break

    history['total_time'] = time.time() - _start

    if best_state:
        model.load_state_dict(best_state['model'])
        if verbose:
            print(f'Restored best from epoch {history["best_epoch"]}')

    return history


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_option_a_experiment(dataset_name, dataset_path, config,
                            device='cuda'):
    tqdm.write(f"\n{'=' * 70}")
    tqdm.write(f"Option A: {dataset_name}")
    tqdm.write(f"Teleport values: {config['teleport_values']}")
    tqdm.write(f"Extraction alpha: {config['extraction_alpha']}, tau: {config['score_tau']}")

    # ---- Data ---------------------------------------------------------------
    # Use the Planetoid-specific loader (real node features, train/val/test split).
    # This mirrors what learnable_ppr_planetoid.ipynb does in cell 4.
    dd = prepare_planetoid_data(dataset_name, root=dataset_path)
    data = dd['data']
    split_edge = dd['split_edge']
    # Swap to train-only edges so val/test edges aren't visible during message passing.
    if not getattr(data, '_edge_index_train_only', False):
        data._orig_edge_index = data.edge_index
        data.edge_index = dd['train_edge_index']
        data._edge_index_train_only = True
        tqdm.write(f"  [train-only edges] swapped: "
                   f"{data._orig_edge_index.size(1):,} -> {data.edge_index.size(1):,}")
    tqdm.write(f"  {data}")

    # ---- PPR (dense matrices) -----------------------------------------------
    # Cache PPR vectors at the same root as other caches; mirrors cache/benchmark-ppr/ layout.
    preprocessed_dir = os.path.join(
        config.get('cache_root', 'cache'), 'option-a', 'ppr')
    gpu_device = device if config.get('gpu_ppr', True) else None
    tqdm.write('Loading / computing multi-scale PPR...')
    multi_scale_ppr = MultiScalePPR(
        dataset_name, data=data,
        teleport_values=config['teleport_values'],
        preprocessed_dir=preprocessed_dir,
        device=gpu_device)
    tqdm.write(f'  {multi_scale_ppr}')

    # ---- Extractor + subgraph caches ----------------------------------------
    extractor = OptionAExtractor(
        data,
        push_epsilon=config['push_epsilon'],
        score_tau=config['score_tau'],
        extraction_alpha=config['extraction_alpha'])

    cache_dir = None
    if config.get('use_checkpoint_cache'):
        # Repo-root-relative cache layout (matches benchmark-ppr / benchmark-khop)
        cache_dir = os.path.join(
            config.get('cache_root', 'cache'), 'option-a', dataset_name,
            f'eps{config["push_epsilon"]}_tau{config["score_tau"]}')

    tqdm.write('Building/loading train subgraph cache...')
    train_cache = build_or_load_cache(
        extractor, split_edge, 'train', cache_dir, verbose=True)
    tqdm.write(f'  Train cache: {len(train_cache)} subgraphs')

    # ---- Model & Selector ---------------------------------------------------
    feat_dim = data.x.size(1)
    model = OptionAGNN(
        feat_dim=feat_dim,
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        alphas=config['teleport_values'],
        selector_hidden=config['selector_hidden'],
        selector_layers=config['selector_layers'])

    selector = PPRScaleSelector(
        in_channels=feat_dim,
        hidden_channels=config['selector_hidden'],
        num_layers=config['selector_layers'],
        num_scales=len(config['teleport_values']),
        temperature=config['temperature_start'],
        scale_emb_dim=config.get('selector_scale_emb_dim', 32))

    tqdm.write(f'  OptionAGNN params: {sum(p.numel() for p in model.parameters()):,}')
    tqdm.write(f'  PPRScaleSelector params: {sum(p.numel() for p in selector.parameters()):,}')

    # ---- Architecture search ------------------------------------------------
    tqdm.write('\n[Phase 1] Bi-level architecture search...')
    searcher = OptionASearcher(
        model=model,
        selector=selector,
        multi_scale_ppr=multi_scale_ppr,
        data=data,
        split_edge=split_edge,
        extractor=extractor,
        train_cache=train_cache,
        device=device,
        lr=config['search_lr'],
        lr_selector=config['search_lr_selector'],
        lr_min=config['search_lr_min'],
        temperature_start=config['temperature_start'],
        temperature_end=config['temperature_end'],
        edges_per_epoch=config['edges_per_search_epoch'])

    search_history = searcher.search(
        epochs=config['search_epochs'],
        batch_size=config['search_batch_size'],
        val_every=config['search_val_every'])

    # Alpha distribution after search
    alpha_counts = torch.zeros(len(config['teleport_values']), dtype=torch.long)
    with torch.no_grad():
        alpha_indices = searcher.get_edge_alpha_indices('train')
        for k in range(len(config['teleport_values'])):
            alpha_counts[k] = (alpha_indices == k).sum()
    tqdm.write(f'  Alpha distribution (train): {alpha_counts.tolist()}')
    tqdm.write(f'  Dominant alpha: {config["teleport_values"][alpha_counts.argmax()]}')

    # ---- Fine-tuning --------------------------------------------------------
    tqdm.write('\n[Phase 2] Fine-tuning with frozen selector...')
    finetune_history = finetune_option_a(
        model=model,
        selector=selector,
        extractor=extractor,
        multi_scale_ppr=multi_scale_ppr,
        data=data,
        split_edge=split_edge,
        train_cache=train_cache,
        epochs=config['finetune_epochs'],
        batch_size=config['finetune_batch_size'],
        lr=config['finetune_lr'],
        weight_decay=config['weight_decay'],
        grad_clip=config['grad_clip'],
        patience=config['finetune_patience'],
        eval_steps=config['finetune_eval_steps'],
        device=device,
        verbose=True)

    # ---- Evaluation ---------------------------------------------------------
    tqdm.write('\n[Eval] Running per-subgraph evaluation (100-neg)...')
    val_subgraph = evaluate_option_a(
        model, selector, extractor, multi_scale_ppr,
        data, split_edge, split='valid',
        batch_size=config['search_batch_size'], device=device,
        max_neg_per_pos=100)
    test_subgraph = evaluate_option_a(
        model, selector, extractor, multi_scale_ppr,
        data, split_edge, split='test',
        batch_size=config['search_batch_size'], device=device,
        max_neg_per_pos=100)
    tqdm.write(f'  [per-subgraph] Val  MRR={val_subgraph["mrr"]:.4f}  '
               f'AUC={val_subgraph["auc"]:.4f}  AP={val_subgraph["ap"]:.4f}')
    tqdm.write(f'  [per-subgraph] Test MRR={test_subgraph["mrr"]:.4f}  '
               f'AUC={test_subgraph["auc"]:.4f}  AP={test_subgraph["ap"]:.4f}')

    # Full-graph eval (apples-to-apples vs Full-Graph / Static-PPR / k-hop)
    val_full, test_full = None, None
    if config.get('full_graph_eval', True):
        tqdm.write('\n[Eval] Running full-graph evaluation (1000-neg)...')
        max_nodes = config.get('full_graph_eval_max_nodes', 8000)
        val_full = evaluate_option_a_full_graph(
            model, selector, multi_scale_ppr, data, split_edge,
            split='valid', device=device, max_nodes=max_nodes)
        test_full = evaluate_option_a_full_graph(
            model, selector, multi_scale_ppr, data, split_edge,
            split='test', device=device, max_nodes=max_nodes)
        if test_full is not None:
            tqdm.write(f'  [full-graph]   Val  MRR={val_full["mrr"]:.4f}  '
                       f'AUC={val_full["auc"]:.4f}  AP={val_full["ap"]:.4f}')
            tqdm.write(f'  [full-graph]   Test MRR={test_full["mrr"]:.4f}  '
                       f'AUC={test_full["auc"]:.4f}  AP={test_full["ap"]:.4f}')

    # Headline test_results = full-graph if available, else per-subgraph
    headline_val = val_full if val_full is not None else val_subgraph
    headline_test = test_full if test_full is not None else test_subgraph

    result = {
        'dataset': dataset_name,
        'encoder': 'PPRDiff',
        'method': 'Option A',
        'eval_mode': 'full-graph' if test_full is not None else 'per-subgraph',
        'val_results': headline_val,
        'test_results': headline_test,
        'subgraph_val_results': val_subgraph,
        'subgraph_test_results': test_subgraph,
        'full_graph_val_results': val_full,
        'full_graph_test_results': test_full,
        'alpha_distribution': alpha_counts.tolist(),
        'dominant_alpha': config['teleport_values'][int(alpha_counts.argmax())],
        'teleport_values': config['teleport_values'],
        'extraction_alpha': config['extraction_alpha'],
        'push_epsilon': config['push_epsilon'],
        'score_tau': config['score_tau'],
        'search_time': float(search_history.get('total_time', 0)),
        'finetune_time': float(finetune_history.get('total_time', 0)),
        'train_time': float(search_history.get('total_time', 0)
                            + finetune_history.get('total_time', 0)),
        'best_val_loss_search': float(search_history.get('best_val_loss', 0)),
        'best_epoch_search': int(search_history.get('best_epoch', 0)),
        'best_val_loss_finetune': float(finetune_history.get('best_val_loss', 0)),
        'best_epoch_finetune': int(finetune_history.get('best_epoch', 0)),
        'stopped_early': bool(finetune_history.get('stopped_early', False)),
        'config': {k: v for k, v in config.items()
                   if k not in ('datasets',)},
        'seed': config.get('seed', 42),
        'run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Save JSON if requested
    if config.get('save_results', False):
        results_root = config.get('results_root', 'results/benchmark-option-a')
        save_dir = os.path.join(results_root, dataset_name)
        run_dir = os.path.join(save_dir, 'runs', result['run_id'])
        os.makedirs(run_dir, exist_ok=True)
        for p in [os.path.join(run_dir, 'full_results.json'),
                  os.path.join(save_dir, 'full_results.json')]:
            with open(p, 'w') as f:
                json.dump({k: v for k, v in result.items()
                           if k not in ('config',) or True}, f,
                          indent=2, default=str)
        tqdm.write(f'\nSaved: {save_dir}/full_results.json')

    return result


# Convenience alias matching the run_one(...) idiom from the rest of the codebase
def run_one(dataset_name, dataset_path='data/Planetoid', config=None,
            device=None, **overrides):
    """
    Single-dataset entry point. Used by both the notebook cell loop and the
    CLI script in scripts/run_lppr.py — same code path.

    Args:
        dataset_name: 'Cora' / 'CiteSeer' / 'PubMed'
        dataset_path: Planetoid root
        config: full config dict; if None, DEFAULT_CONFIG is used
        device: 'cuda' / 'cpu' / None (auto)
        **overrides: any DEFAULT_CONFIG key to override
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return run_option_a_experiment(dataset_name, dataset_path, config, device=device)


def run_option_a_benchmark(config=None, device=None):
    """Entry point: run Option A on all configured datasets."""
    if config is None:
        config = DEFAULT_CONFIG.copy()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {}
    for dataset_name in config['datasets']:
        dataset_path = os.path.join('data', 'Planetoid')
        res = run_option_a_experiment(
            dataset_name, dataset_path, config, device=device)
        results[dataset_name] = res

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Option A benchmark')
    parser.add_argument('--datasets', nargs='+', default=['Cora'])
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--search_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg['datasets'] = args.datasets
    cfg['hidden_channels'] = args.hidden
    cfg['num_layers'] = args.layers
    cfg['search_epochs'] = args.search_epochs
    cfg['finetune_epochs'] = args.finetune_epochs

    run_option_a_benchmark(cfg, device=args.device)
