"""
Training pipeline for PPR-based subgraph link prediction.
Uses pre-cached subgraphs + batched GNN forward with vectorised loss.
"""

import torch
import torch.nn as nn
import os
import time
import copy
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from tqdm import tqdm
from .evaluator import evaluate_ppr


# ---------------------------------------------------------------------------
# Subgraph cache: extract once, reuse every epoch
# ---------------------------------------------------------------------------

class PPRSubgraphCache:
    """Pre-extracted subgraph topology for every training edge.

    Stores topology only (selected_nodes, edge_index, u_sub, v_sub).
    Node features are sliced from ``data.x`` at training time.
    Cache is shared across encoders for the same (dataset, top_k).
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
    def build(cls, source_edge, target_edge, data, ppr_extractor, verbose=True):
        n = source_edge.size(0)
        sel_list, ei_list, u_list, v_list, nn_list = [], [], [], [], []
        skipped = 0

        it = tqdm(range(n), desc='Building PPR cache', leave=False) if verbose else range(n)
        for i in it:
            u = source_edge[i].item()
            v = target_edge[i].item()

            ppr_u = ppr_extractor.preprocessor.get_ppr(u)
            ppr_v = ppr_extractor.preprocessor.get_ppr(v)
            combined = ppr_extractor.alpha * ppr_u + (1 - ppr_extractor.alpha) * ppr_v
            top_k_actual = min(ppr_extractor.top_k, len(combined))
            _, selected_nodes = torch.topk(combined, top_k_actual)

            if u not in selected_nodes:
                selected_nodes = torch.cat([selected_nodes, torch.tensor([u])])
            if v not in selected_nodes:
                selected_nodes = torch.cat([selected_nodes, torch.tensor([v])])

            edge_index_sub, _ = subgraph(
                selected_nodes, data.edge_index,
                relabel_nodes=True, num_nodes=data.num_nodes)

            node_mapping = {node.item(): new_idx for new_idx, node in enumerate(selected_nodes)}
            u_sub = node_mapping.get(u, -1)
            v_sub = node_mapping.get(v, -1)

            if u_sub == -1 or v_sub == -1:
                sel_list.append(torch.zeros(0, dtype=torch.long))
                ei_list.append(torch.zeros(2, 0, dtype=torch.long))
                u_list.append(-1); v_list.append(-1); nn_list.append(0)
                skipped += 1
            else:
                sel_list.append(selected_nodes.cpu())
                ei_list.append(edge_index_sub.cpu())
                u_list.append(u_sub); v_list.append(v_sub)
                nn_list.append(len(selected_nodes))

        if verbose and skipped:
            print(f'  ({skipped}/{n} edges skipped)')
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
            return None, -1, -1
        x_sub = x_full[sel]
        return (Data(x=x_sub, edge_index=self.edge_index[idx],
                     num_nodes=self.num_nodes[idx]),
                self.u_sub[idx], self.v_sub[idx])


def build_or_load_ppr_cache(source_edge, target_edge, data, ppr_extractor,
                             cache_dir=None, verbose=True):
    if cache_dir:
        path = os.path.join(cache_dir, 'train_subgraphs.pt')
        if os.path.isfile(path):
            if verbose:
                print(f'[Cache] Loading PPR subgraphs from {path}')
            return PPRSubgraphCache.load(path)

    if verbose:
        print('[Cache] Extracting PPR subgraphs (one-time cost)...')
    cache = PPRSubgraphCache.build(source_edge, target_edge, data,
                                    ppr_extractor, verbose)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, 'train_subgraphs.pt')
        cache.save(path)
        mb = os.path.getsize(path) / 1e6
        if verbose:
            print(f'[Cache] Saved: {path} ({mb:.0f} MB)')
    return cache


# ---------------------------------------------------------------------------
# Training epoch using cache
# ---------------------------------------------------------------------------

def train_epoch_ppr_cached(encoder, predictor, data, cache, optimizer,
                            batch_size, device, grad_clip=None,
                            verbose=False, edges_per_epoch=None,
                            max_subgraphs_per_forward=512):
    encoder.train()
    predictor.train()

    n_total = len(cache)
    if edges_per_epoch and edges_per_epoch < n_total:
        indices = torch.randperm(n_total)[:edges_per_epoch]
    else:
        indices = torch.randperm(n_total)

    total_loss = 0.0
    total_examples = 0

    dataloader = DataLoader(indices.tolist(), batch_size, shuffle=False)
    if verbose:
        dataloader = tqdm(dataloader, desc='  Batches', leave=False,
                          mininterval=30)

    x_full = data.x

    for perm in dataloader:
        subgraph_list, u_subs, v_subs, nn_list = [], [], [], []
        for idx in perm:
            sub_data, u_s, v_s = cache.make_data(idx, x_full)
            if sub_data is None:
                continue
            subgraph_list.append(sub_data)
            u_subs.append(u_s)
            v_subs.append(v_s)
            nn_list.append(cache.num_nodes[idx])

        if len(subgraph_list) == 0:
            continue

        optimizer.zero_grad()

        mb = max_subgraphs_per_forward
        n = len(subgraph_list)
        accum_loss = 0.0
        accum_ex = 0

        for start in range(0, n, mb):
            end = min(start + mb, n)
            batched = Batch.from_data_list(subgraph_list[start:end]).to(device)
            h = encoder(batched.x, batched.edge_index)

            B = end - start
            offsets = torch.zeros(B, dtype=torch.long, device=device)
            nn_t = torch.tensor(nn_list[start:end], dtype=torch.long, device=device)
            if B > 1:
                offsets[1:] = nn_t[:-1].cumsum(0)

            u_idx = offsets + torch.tensor(u_subs[start:end], dtype=torch.long, device=device)
            v_idx = offsets + torch.tensor(v_subs[start:end], dtype=torch.long, device=device)
            neg_local = torch.stack([torch.randint(0, nn, (1,)) for nn in nn_list[start:end]]).squeeze(1).to(device)
            neg_idx = offsets + neg_local

            pos_out = predictor(h[u_idx], h[v_idx])
            neg_out = predictor(h[u_idx], h[neg_idx])
            mb_loss = (-torch.log(pos_out + 1e-15) - torch.log(1 - neg_out + 1e-15)).mean()
            mb_loss.backward()
            accum_loss += mb_loss.item() * B
            accum_ex += B

        if accum_ex > 0:
            n_micro = (n + mb - 1) // mb
            if n_micro > 1:
                scale = 1.0 / n_micro
                for p in list(encoder.parameters()) + list(predictor.parameters()):
                    if p.grad is not None:
                        p.grad.mul_(scale)

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()), grad_clip)
            optimizer.step()
            total_loss += accum_loss
            total_examples += accum_ex

    return total_loss / max(total_examples, 1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_model_ppr_batched(encoder, predictor, data, split_edge, ppr_extractor,
                             epochs=500, batch_size=512, lr=0.005,
                             eval_steps=5, device='cpu', verbose=True,
                             patience=30, min_delta=0.0001, weight_decay=1e-5,
                             lr_scheduler='reduce_on_plateau', grad_clip=1.0,
                             edges_per_epoch=None, cache_dir=None,
                             max_eval_edges=2000):
    encoder = encoder.to(device)
    predictor = predictor.to(device)

    source_edge = split_edge['train']['source_node']
    target_edge = split_edge['train']['target_node']
    cache = build_or_load_ppr_cache(source_edge, target_edge, data,
                                     ppr_extractor, cache_dir, verbose)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=weight_decay)

    scheduler = None
    if lr_scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)

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
        loss = train_epoch_ppr_cached(
            encoder, predictor, data, cache,
            optimizer, batch_size, device, grad_clip=grad_clip,
            verbose=show_batch, edges_per_epoch=edges_per_epoch)

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(loss)
        history['epoch_times'].append(epoch_time)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        if epoch % eval_steps == 0 or epoch == epochs:
            me = None if epoch == epochs else max_eval_edges
            val_results = evaluate_ppr(
                encoder, predictor, data, split_edge, ppr_extractor,
                split='valid', batch_size=batch_size, device=device,
                max_edges=me, cache_dir=cache_dir)
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
