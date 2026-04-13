"""
Phase 2: Fine-tune a GNN on subgraphs extracted with learned PPR configurations.

Uses the per-edge (teleport_u, teleport_v) configs from architecture search
to extract actual subgraphs, then trains with the existing batched infrastructure.

Optimizations over naive per-edge-per-epoch extraction:
  - SubgraphCache: extract every subgraph once, reuse across all epochs
  - Vectorized loss: single batched predictor call instead of Python for-loop
  - Edge subsampling: use a random subset of training edges per epoch
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
from datetime import datetime


def _long_running_tqdm(iterable, desc, heartbeat_mins=10, **kwargs):
    """tqdm generator: capped refresh rate + heartbeat timestamp every N mins."""
    interval = float(os.environ.get('MTCH_TQDM_MININTERVAL', '5'))
    has_len = hasattr(iterable, '__len__')
    miniters = max(200, len(iterable) // 200) if has_len else 200
    bar = tqdm(
        iterable, desc=desc,
        mininterval=interval, maxinterval=60, miniters=miniters,
        **kwargs,
    )
    heartbeat_secs = heartbeat_mins * 60
    last_hb = time.time()
    try:
        for item in bar:
            yield item
            now = time.time()
            if now - last_hb >= heartbeat_secs:
                last_hb = now
                pct = bar.n / bar.total * 100 if bar.total else 0
                tqdm.write(
                    f'  [{datetime.now().strftime("%H:%M:%S")}] '
                    f'{desc}: {bar.n}/{bar.total} ({pct:.0f}%)')
    finally:
        bar.close()


class LearnablePPRExtractor:
    """
    Subgraph extractor that uses per-edge learned PPR configurations.

    For each edge (u,v), uses the arch_net-selected (teleport_u, teleport_v)
    to determine which PPR vectors to combine, then extracts top-k nodes.

    Args:
        data: PyG Data object (full graph)
        multi_scale_ppr: MultiScalePPR instance
        config_indices: Tensor [num_edges] of learned config index per edge
        alpha: Combination weights for PPR_u and PPR_v.
               List of length 1: w_u = alpha[0], w_v = 1 - alpha[0].
               List of length 2: (w_u, w_v) normalized to sum 1.
               Scalar accepted for backwards compatibility.
        top_k: Number of nodes to select per subgraph
    """

    def __init__(self, data, multi_scale_ppr, config_indices,
                 alpha=None, top_k=100):
        from . import resolve_alpha_weights
        self.data = data
        self.multi_scale_ppr = multi_scale_ppr
        self.config_indices = config_indices
        if alpha is None:
            alpha = [0.5]
        self.alpha = alpha
        self.w_u, self.w_v = resolve_alpha_weights(alpha)
        self.top_k = top_k

    def extract_subgraph(self, u, v, edge_idx):
        """
        Extract subgraph for edge (u,v) using its learned PPR config.

        Returns:
            subgraph_data: PyG Data with remapped indices
            selected_nodes: Original node indices
            metadata: Dict with u_subgraph, v_subgraph, config info
        """
        config_idx = self.config_indices[edge_idx].item()
        teleport_u, teleport_v = self.multi_scale_ppr.get_config_for_index(
            config_idx)

        ppr_u = self.multi_scale_ppr.get_ppr(u, teleport_u)
        ppr_v = self.multi_scale_ppr.get_ppr(v, teleport_v)

        combined = self.w_u * ppr_u + self.w_v * ppr_v
        top_k_actual = min(self.top_k, len(combined))
        _, selected_nodes = torch.topk(combined, top_k_actual)

        dev = self.data.edge_index.device
        selected_nodes = selected_nodes.detach().to(dev).long()

        if not (selected_nodes == u).any():
            selected_nodes = torch.cat(
                [selected_nodes, torch.tensor([u], device=dev, dtype=torch.long)])
        if not (selected_nodes == v).any():
            selected_nodes = torch.cat(
                [selected_nodes, torch.tensor([v], device=dev, dtype=torch.long)])

        edge_index_sub, _ = subgraph(
            selected_nodes, self.data.edge_index,
            relabel_nodes=True, num_nodes=self.data.num_nodes)

        node_mapping = {node.item(): new_idx
                        for new_idx, node in enumerate(selected_nodes)}
        u_sub = node_mapping.get(u, -1)
        v_sub = node_mapping.get(v, -1)

        x_sub = self.data.x[selected_nodes]
        subgraph_data = Data(x=x_sub, edge_index=edge_index_sub,
                             num_nodes=len(selected_nodes))

        metadata = {
            'u_subgraph': u_sub,
            'v_subgraph': v_sub,
            'config_idx': config_idx,
            'teleport_u': teleport_u,
            'teleport_v': teleport_v,
            'num_nodes_selected': len(selected_nodes),
            'num_edges_subgraph': edge_index_sub.size(1),
        }
        return subgraph_data, selected_nodes, metadata


# ---------------------------------------------------------------------------
# SubgraphCache: pre-extract once, reuse every epoch
# ---------------------------------------------------------------------------

class SubgraphCache:
    """Pre-extracted subgraph topology for every edge in a split.

    Stores only the topology (selected_nodes, edge_index_sub, u_sub, v_sub)
    so the cache is compact (~2-6 GB for ~400K edges).  Node features are
    sliced from ``data.x`` at training time.

    Usage::

        cache = SubgraphCache.build(extractor, split_edge, 'train', verbose=True)
        cache.save('cache/train.pt')
        cache = SubgraphCache.load('cache/train.pt')
    """

    def __init__(self, selected_nodes_list, edge_index_list,
                 u_sub_list, v_sub_list, num_nodes_list):
        self.selected_nodes = selected_nodes_list   # list[Tensor]
        self.edge_index = edge_index_list            # list[Tensor [2, E_i]]
        self.u_sub = u_sub_list                      # list[int]
        self.v_sub = v_sub_list                      # list[int]
        self.num_nodes = num_nodes_list              # list[int]

    def __len__(self):
        return len(self.selected_nodes)

    # ---- factory ----------------------------------------------------------

    @classmethod
    def build_fast(cls, extractor, split_edge, split='train', verbose=True):
        """Extract subgraphs storing topology compactly (CPU tensors)."""
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        n = source.size(0)

        sel_list, ei_list, u_list, v_list, nn_list = [], [], [], [], []
        skipped = 0

        edge_index_cpu = extractor.data.edge_index.cpu()
        num_graph_nodes = extractor.data.num_nodes

        it = (_long_running_tqdm(range(n), desc=f'Caching {split} subgraphs')
              if verbose else range(n))
        for i in it:
            u = source[i].item()
            v = target[i].item()

            config_idx = extractor.config_indices[i].item()
            teleport_u, teleport_v = extractor.multi_scale_ppr.get_config_for_index(config_idx)

            ppr_u = extractor.multi_scale_ppr.get_ppr(u, teleport_u)
            ppr_v = extractor.multi_scale_ppr.get_ppr(v, teleport_v)

            combined = extractor.w_u * ppr_u + extractor.w_v * ppr_v
            top_k_actual = min(extractor.top_k, len(combined))
            _, selected_nodes = torch.topk(combined, top_k_actual)
            selected_nodes = selected_nodes.cpu().long()

            if not (selected_nodes == u).any():
                selected_nodes = torch.cat(
                    [selected_nodes, torch.tensor([u], dtype=torch.long)])
            if not (selected_nodes == v).any():
                selected_nodes = torch.cat(
                    [selected_nodes, torch.tensor([v], dtype=torch.long)])

            edge_index_sub, _ = subgraph(
                selected_nodes, edge_index_cpu,
                relabel_nodes=True, num_nodes=num_graph_nodes)

            node_mapping = {node.item(): new_idx
                            for new_idx, node in enumerate(selected_nodes)}
            u_sub = node_mapping.get(u, -1)
            v_sub = node_mapping.get(v, -1)

            if u_sub == -1 or v_sub == -1:
                sel_list.append(torch.zeros(0, dtype=torch.long))
                ei_list.append(torch.zeros(2, 0, dtype=torch.long))
                u_list.append(-1)
                v_list.append(-1)
                nn_list.append(0)
                skipped += 1
            else:
                sel_list.append(selected_nodes)
                ei_list.append(edge_index_sub)
                u_list.append(u_sub)
                v_list.append(v_sub)
                nn_list.append(len(selected_nodes))

        if verbose and skipped:
            print(f'  ({skipped}/{n} edges skipped: u or v not in subgraph)')

        return cls(sel_list, ei_list, u_list, v_list, nn_list)

    # ---- persistence ------------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'selected_nodes': self.selected_nodes,
            'edge_index': self.edge_index,
            'u_sub': self.u_sub,
            'v_sub': self.v_sub,
            'num_nodes': self.num_nodes,
        }, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location='cpu')
        return cls(d['selected_nodes'], d['edge_index'],
                   d['u_sub'], d['v_sub'], d['num_nodes'])

    # ---- helpers ----------------------------------------------------------

    def make_data(self, idx, x_full):
        """Build a PyG Data for edge *idx*, slicing features from *x_full*."""
        sel = self.selected_nodes[idx]
        if len(sel) == 0:
            return None, -1, -1
        x_sub = x_full[sel.to(x_full.device)]
        return (Data(x=x_sub, edge_index=self.edge_index[idx],
                     num_nodes=self.num_nodes[idx]),
                self.u_sub[idx], self.v_sub[idx])


def build_or_load_cache(extractor, split_edge, split, cache_dir,
                        verbose=True):
    """Build a SubgraphCache or load from disk if it already exists."""
    if cache_dir:
        path = os.path.join(cache_dir, f'{split}_subgraphs.pt')
        if os.path.isfile(path):
            if verbose:
                print(f'[Cache] Loading {split} subgraphs from {path}')
            return SubgraphCache.load(path)

    if verbose:
        print(f'[Cache] Extracting {split} subgraphs (one-time cost)...')
    cache = SubgraphCache.build_fast(extractor, split_edge, split, verbose)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f'{split}_subgraphs.pt')
        cache.save(path)
        mb = os.path.getsize(path) / 1e6
        if verbose:
            print(f'[Cache] Saved {split} cache: {path} ({mb:.0f} MB)')
    return cache


# ---------------------------------------------------------------------------
# Vectorized forward + loss
# ---------------------------------------------------------------------------

def _forward_micro_batch(encoder, predictor, subgraph_list, u_subs, v_subs,
                         num_nodes_list, device):
    """Vectorized forward + loss on a micro-batch of subgraphs."""
    batched = Batch.from_data_list(subgraph_list).to(device)
    h = encoder(batched.x, batched.edge_index)

    B = len(u_subs)
    offsets = torch.zeros(B, dtype=torch.long, device=device)
    nn_t = torch.tensor(num_nodes_list, dtype=torch.long, device=device)
    if B > 1:
        offsets[1:] = nn_t[:-1].cumsum(0)

    u_indices = offsets + torch.tensor(u_subs, dtype=torch.long, device=device)
    v_indices = offsets + torch.tensor(v_subs, dtype=torch.long, device=device)
    neg_local = torch.stack([
        torch.randint(0, nn, (1,)) for nn in num_nodes_list
    ]).squeeze(1).to(device)
    neg_indices = offsets + neg_local

    pos_out = predictor(h[u_indices], h[v_indices])
    neg_out = predictor(h[u_indices], h[neg_indices])

    loss = (-torch.log(pos_out + 1e-15) - torch.log(1 - neg_out + 1e-15)).mean()
    return loss, B


# ---------------------------------------------------------------------------
# Training loop (uses cache + subsampling)
# ---------------------------------------------------------------------------

def train_epoch_finetune(encoder, predictor, data, cache, optimizer,
                         batch_size, device, grad_clip=None, verbose=False,
                         max_subgraphs_per_forward=256,
                         edges_per_epoch=None):
    """Train one epoch using pre-cached subgraphs.

    Args:
        cache: SubgraphCache for the training split.
        edges_per_epoch: If set, subsample this many edges per epoch.
    """
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
        dataloader = tqdm(dataloader, desc='  Fine-tune batches',
                          leave=False, mininterval=2)

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
        accum_loss = 0.0
        accum_examples = 0
        n = len(subgraph_list)
        mb = max_subgraphs_per_forward

        for start in range(0, n, mb):
            end = min(start + mb, n)
            mb_loss, mb_ex = _forward_micro_batch(
                encoder, predictor,
                subgraph_list[start:end],
                u_subs[start:end], v_subs[start:end],
                nn_list[start:end], device)
            mb_loss.backward()
            accum_loss += mb_loss.item() * mb_ex
            accum_examples += mb_ex

        if accum_examples > 0:
            n_micro = (n + mb - 1) // mb
            if n_micro > 1:
                scale = 1.0 / n_micro
                for p in list(encoder.parameters()) + list(predictor.parameters()):
                    if p.grad is not None:
                        p.grad.mul_(scale)

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) +
                    list(predictor.parameters()), grad_clip)
            optimizer.step()
            total_loss += accum_loss
            total_examples += accum_examples

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / max(total_examples, 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def finetune_on_subgraphs(encoder, predictor, data, split_edge,
                           multi_scale_ppr, config_indices,
                           alpha=None, top_k=100, epochs=200,
                           batch_size=8192, lr=0.005, eval_steps=5,
                           device='cpu', verbose=True, patience=20,
                           weight_decay=1e-5, grad_clip=1.0,
                           max_subgraphs_per_forward=256,
                           checkpoint_dir=None,
                           cache_dir=None,
                           edges_per_epoch=None):
    """
    Phase 2: Fine-tune encoder+predictor on subgraphs with learned configs.

    Args:
        encoder: GNN encoder
        predictor: LinkPredictor
        data: PyG Data (full graph, with features)
        split_edge: Edge split dictionary
        multi_scale_ppr: MultiScalePPR instance
        config_indices: Tensor [num_train_edges] of learned config per edge
        alpha: PPR combination weights (list). See resolve_alpha_weights().
        top_k: Subgraph size
        epochs: Max training epochs
        batch_size: Edges per batch
        lr: Learning rate
        eval_steps: Evaluate every N epochs
        device: Device
        verbose: Print progress
        patience: Early stopping patience
        weight_decay: L2 regularization
        grad_clip: Gradient clipping norm
        max_subgraphs_per_forward: GPU micro-batch size (subgraphs per
            forward pass). Lower this if OOM occurs.
        checkpoint_dir: Saves best model + periodic checkpoints here.
        cache_dir: Directory for pre-extracted subgraph caches.
            If None, caches are built in memory only (not persisted).
        edges_per_epoch: If set, subsample this many training edges per
            epoch instead of using all.  Greatly reduces epoch time.

    Returns:
        history: Training history dict
    """
    if alpha is None:
        alpha = [0.5]
    from .evaluator import evaluate_learnable_ppr

    encoder = encoder.to(device)
    predictor = predictor.to(device)

    # ---- build / load subgraph caches ------------------------------------
    extractor = LearnablePPRExtractor(
        data, multi_scale_ppr, config_indices,
        alpha=alpha, top_k=top_k)

    train_cache = build_or_load_cache(
        extractor, split_edge, 'train', cache_dir, verbose)

    # ---- optimizer / scheduler -------------------------------------------
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10)

    history = {
        'train_loss': [],
        'val_results': [],
        'best_val_mrr': 0.0,
        'best_epoch': 0,
        'stopped_early': False,
    }

    start_epoch = 1
    best_val_mrr = 0.0
    epochs_no_improve = 0
    best_state = None

    # ---- resume from checkpoint ------------------------------------------
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            encoder.load_state_dict(ckpt['encoder'])
            predictor.load_state_dict(ckpt['predictor'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            history = ckpt['history']
            start_epoch = ckpt['epoch'] + 1
            best_val_mrr = ckpt['best_val_mrr']
            epochs_no_improve = ckpt.get('epochs_no_improve', 0)
            best_state = ckpt.get('best_state', None)
            if verbose:
                print(f"[Checkpoint] Resumed from epoch {ckpt['epoch']} "
                      f"(best MRR {best_val_mrr:.4f})")

    start = time.time()
    last_heartbeat = time.time()

    iterator = range(start_epoch, epochs + 1)
    if verbose:
        iterator = tqdm(iterator, desc='Fine-tuning',
                        initial=start_epoch - 1, total=epochs)

    for epoch in iterator:
        show_batch = (epoch <= start_epoch + 2) if verbose else False
        loss = train_epoch_finetune(
            encoder, predictor, data, train_cache,
            optimizer, batch_size, device, grad_clip=grad_clip,
            verbose=show_batch,
            max_subgraphs_per_forward=max_subgraphs_per_forward,
            edges_per_epoch=edges_per_epoch)
        history['train_loss'].append(loss)

        if time.time() - last_heartbeat >= 600:
            last_heartbeat = time.time()
            tqdm.write(
                f'  [{datetime.now().strftime("%H:%M:%S")}] '
                f'Fine-tuning epoch {epoch}/{epochs}, loss={loss:.4f}')

        if epoch % eval_steps == 0 or epoch == epochs:
            val_results = evaluate_learnable_ppr(
                encoder, predictor, data, split_edge,
                multi_scale_ppr, config_indices,
                split='valid', alpha=alpha, top_k=top_k,
                batch_size=batch_size, device=device,
                cache_dir=cache_dir)
            history['val_results'].append(val_results)

            mrr = val_results['mrr']
            scheduler.step(mrr)

            if mrr > best_val_mrr + 0.0001:
                best_val_mrr = mrr
                history['best_val_mrr'] = best_val_mrr
                history['best_epoch'] = epoch
                epochs_no_improve = 0
                best_state = {
                    'encoder': copy.deepcopy(encoder.state_dict()),
                    'predictor': copy.deepcopy(predictor.state_dict()),
                }
                if checkpoint_dir:
                    torch.save(best_state,
                               os.path.join(checkpoint_dir, 'best_model.pt'))
            else:
                epochs_no_improve += eval_steps

            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss:.4f}',
                    'mrr': f'{mrr:.4f}',
                    'best': f'{best_val_mrr:.4f}',
                    'pat': f'{epochs_no_improve}/{patience}',
                })

            if checkpoint_dir:
                torch.save({
                    'epoch': epoch,
                    'encoder': encoder.state_dict(),
                    'predictor': predictor.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'history': history,
                    'best_val_mrr': best_val_mrr,
                    'epochs_no_improve': epochs_no_improve,
                    'best_state': best_state,
                }, os.path.join(checkpoint_dir, 'latest_checkpoint.pt'))

            if epochs_no_improve >= patience:
                history['stopped_early'] = True
                if verbose:
                    print(f"\n[Early Stop] epoch {epoch}")
                break

    history['total_time'] = history.get('total_time', 0) + (time.time() - start)

    if best_state:
        encoder.load_state_dict(best_state['encoder'])
        predictor.load_state_dict(best_state['predictor'])
        if verbose:
            print(f"Restored best from epoch {history['best_epoch']}")

    return history
