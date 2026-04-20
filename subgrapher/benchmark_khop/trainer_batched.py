"""
Training pipeline for k-hop subgraph link prediction.

CSR in-memory subgraph cache + fully vectorised batch construction.
Replaces the old Python-list-of-tensors cache that pickled to 50 GB for k=1
FB15K237 and thrashed disk on every batch.
"""

import copy
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm

from ..utils.subgraph_csr import SubgraphCSR
from .evaluator import evaluate_khop


# ---------------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------------


def build_or_load_khop_csr_cache(source_edge, target_edge, data, khop_extractor,
                                  cache_dir=None, verbose=True):
    """Build (or load) a SubgraphCSR cache for k-hop training edges."""
    if cache_dir:
        path = os.path.join(cache_dir, 'train_subgraphs_csr.pt')
        if os.path.isfile(path):
            if verbose:
                print(f'[Cache] Loading CSR k-hop cache from {path}')
            return SubgraphCSR.load(path)

    if verbose:
        print('[Cache] Extracting k-hop subgraphs (one-time cost)...')

    edge_index_full = data.edge_index
    num_nodes_full = data.num_nodes

    def extract(i: int):
        u = int(source_edge[i].item())
        v = int(target_edge[i].item())

        nodes_u = khop_extractor.preprocessor.get_khop_nodes(u)
        nodes_v = khop_extractor.preprocessor.get_khop_nodes(v)
        selected_nodes = torch.unique(torch.cat([nodes_u, nodes_v]))

        edge_index_sub, _ = subgraph(
            selected_nodes, edge_index_full,
            relabel_nodes=True, num_nodes=num_nodes_full,
        )

        # selected_nodes is sorted by torch.unique.
        pos_u = torch.searchsorted(selected_nodes, torch.tensor([u]))[0].item()
        pos_v = torch.searchsorted(selected_nodes, torch.tensor([v]))[0].item()
        if pos_u >= selected_nodes.numel() or selected_nodes[pos_u].item() != u:
            return None
        if pos_v >= selected_nodes.numel() or selected_nodes[pos_v].item() != v:
            return None

        return selected_nodes, edge_index_sub, pos_u, pos_v

    cache = SubgraphCSR.build(
        num_edges=int(source_edge.size(0)),
        extract_fn=extract,
        progress_desc='Building k-hop CSR',
        verbose=verbose,
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, 'train_subgraphs_csr.pt')
        cache.save(path)
        if verbose:
            mb = os.path.getsize(path) / 1e6
            print(f'[Cache] Saved CSR: {path} ({mb:.0f} MB)')
            print(f'[Cache] {cache.summary()}')

    return cache


# ---------------------------------------------------------------------------
# One training epoch over the CSR cache
# ---------------------------------------------------------------------------


def train_epoch_khop_csr(encoder, predictor, cache, x_full_gpu, optimizer,
                          batch_size, device, grad_clip=None,
                          edges_per_epoch=None, verbose=False):
    encoder.train()
    predictor.train()

    n_total = len(cache)
    if edges_per_epoch and edges_per_epoch < n_total:
        indices = torch.randperm(n_total, device=device)[:edges_per_epoch]
    else:
        indices = torch.randperm(n_total, device=device)

    dataloader = DataLoader(indices.tolist(), batch_size, shuffle=False)
    if verbose:
        dataloader = tqdm(dataloader, desc='  Batches', leave=False,
                          mininterval=30)

    total_loss = 0.0
    total_examples = 0

    for perm in dataloader:
        idx = torch.as_tensor(perm, dtype=torch.long, device=device)
        batch = cache.make_batch(idx, x_full_gpu)

        B = int(batch["u_idx"].size(0))
        if B == 0:
            continue
        if batch["total_edges"] == 0:
            continue

        optimizer.zero_grad()
        h = encoder(batch["x"], batch["edge_index"])

        u_idx = batch["u_idx"]
        v_idx = batch["v_idx"]
        nn_b = batch["num_nodes_vec"]
        node_offsets = batch["batch_node_offsets"][:B]

        nn_f = nn_b.to(torch.float32)
        neg_local = (torch.rand(B, device=device) * nn_f).long()
        neg_local = neg_local.clamp_(max=nn_b - 1)
        neg_idx = node_offsets + neg_local

        pos_out = predictor(h[u_idx], h[v_idx])
        neg_out = predictor(h[u_idx], h[neg_idx])

        loss = (-torch.log(pos_out + 1e-15)
                - torch.log(1 - neg_out + 1e-15)).mean()
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()),
                grad_clip,
            )
        optimizer.step()

        total_loss += loss.item() * B
        total_examples += B

    return total_loss / max(total_examples, 1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


def train_model_khop_batched(encoder, predictor, data, split_edge, khop_extractor,
                              epochs=150, batch_size=512, lr=0.005,
                              eval_steps=5, device='cpu', verbose=True,
                              patience=10, min_delta=0.0001, weight_decay=1e-5,
                              lr_scheduler='reduce_on_plateau', grad_clip=1.0,
                              edges_per_epoch=None, cache_dir=None,
                              max_eval_edges=2000,
                              eval_num_negs=None):
    encoder = encoder.to(device)
    predictor = predictor.to(device)

    source_edge = split_edge['train']['source_node']
    target_edge = split_edge['train']['target_node']

    cache = build_or_load_khop_csr_cache(
        source_edge, target_edge, data, khop_extractor,
        cache_dir=cache_dir, verbose=verbose,
    )
    cache = cache.to(device)
    x_full_gpu = data.x.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    scheduler = None
    if lr_scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5,
        )
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01,
        )

    history = {
        'train_loss': [], 'val_results': [], 'epoch_times': [],
        'learning_rates': [], 'best_val_mrr': 0.0, 'best_epoch': 0,
        'stopped_early': False, 'stop_reason': None,
    }

    best_val_mrr = 0.0
    epochs_without_improvement = 0
    best_model_state = None

    start_time = time.time()
    iterator = tqdm(range(1, epochs + 1), desc='Training',
                    mininterval=10, maxinterval=60) if verbose else range(1, epochs + 1)

    for epoch in iterator:
        epoch_start = time.time()
        show_batch = verbose and (epoch <= 2)
        loss = train_epoch_khop_csr(
            encoder, predictor, cache, x_full_gpu, optimizer,
            batch_size, device, grad_clip=grad_clip,
            edges_per_epoch=edges_per_epoch, verbose=show_batch,
        )

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(loss)
        history['epoch_times'].append(epoch_time)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        if epoch % eval_steps == 0 or epoch == epochs:
            me = None if epoch == epochs else max_eval_edges
            val_results = evaluate_khop(
                encoder, predictor, data, split_edge, khop_extractor,
                split='valid', batch_size=batch_size, device=device,
                max_edges=me, cache_dir=cache_dir,
                num_negs_per_pos=eval_num_negs,
                x_full_gpu=x_full_gpu,
            )
            history['val_results'].append(val_results)
            current_val_mrr = val_results['mrr']

            if current_val_mrr > best_val_mrr + min_delta:
                best_val_mrr = current_val_mrr
                history['best_val_mrr'] = best_val_mrr
                history['best_epoch'] = epoch
                epochs_without_improvement = 0
                best_model_state = {
                    'encoder': copy.deepcopy(encoder.state_dict()),
                    'predictor': copy.deepcopy(predictor.state_dict()),
                }
            else:
                epochs_without_improvement += eval_steps

            if scheduler is not None:
                if lr_scheduler == 'reduce_on_plateau':
                    scheduler.step(current_val_mrr)
                else:
                    scheduler.step()

            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss:.4f}',
                    'val_mrr': f'{current_val_mrr:.4f}',
                    'best': f'{best_val_mrr:.4f}',
                    'patience': f'{epochs_without_improvement}/{patience}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'time/ep': f'{epoch_time:.1f}s',
                })

            if epochs_without_improvement >= patience:
                history['stopped_early'] = True
                history['stop_reason'] = f'No improvement for {patience} epochs'
                if verbose:
                    print(f"\n[Early Stop] No improvement for {patience} epochs")
                    print(f"Best MRR: {best_val_mrr:.4f} at epoch {history['best_epoch']}")
                break

    total_time = time.time() - start_time
    history['total_time'] = total_time

    if best_model_state is not None:
        encoder.load_state_dict(best_model_state['encoder'])
        predictor.load_state_dict(best_model_state['predictor'])
        if verbose:
            print(f"Restored best model from epoch {history['best_epoch']}")

    return history
