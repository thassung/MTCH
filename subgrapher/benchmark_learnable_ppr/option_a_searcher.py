"""
Bi-level architecture search for Option A (PS2-style learnable PPR).

Parameters:
  w (model weights):  PPRDiffusionEncoder + LinkPredictor  -> trained on train loss
  theta (selector):   PPRScaleSelector                     -> trained on val loss

Uses first-order DARTS approximation (no HVP) for simplicity.
Temperature annealing drives selector toward hard selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, add_self_loops
from tqdm import tqdm

from .option_a_extractor import OptionASubgraphCache, sample_neg_subgraphs


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class OptionASearcher:
    """
    Bi-level searcher for Option A.

    Args:
        model: OptionAGNN instance
        selector: PPRScaleSelector instance
        multi_scale_ppr: MultiScalePPR (for ppr_dense access)
        data: PyG Data (full graph)
        split_edge: edge split dict
        extractor: OptionAExtractor
        train_cache: OptionASubgraphCache for train edges
        device: torch device
        lr: learning rate for model (w)
        lr_selector: learning rate for selector (theta)
        lr_min: min LR for cosine schedule
        temperature_start/end: annealing range
        edges_per_epoch: subsample training edges (None = all)
    """

    def __init__(self, model, selector, multi_scale_ppr, data, split_edge,
                 extractor, train_cache, device='cuda',
                 lr=0.01, lr_selector=3e-4, lr_min=1e-3,
                 temperature_start=1.0, temperature_end=0.2,
                 edges_per_epoch=None, search_extractor=None):
        self.model = model.to(device)
        self.selector = selector.to(device)
        self.multi_scale_ppr = multi_scale_ppr
        self.ppr_dense = multi_scale_ppr.ppr_dense
        self.alphas = model.alphas
        self.data = data.to(device)
        self.split_edge = split_edge
        self.extractor = extractor
        # search_extractor is used for all LIVE extractions during search
        # (negatives, val positives, val negatives).  Defaults to extractor.
        # Pass a coarser-epsilon extractor here for a large speed-up.
        self.search_extractor = search_extractor if search_extractor is not None else extractor
        self.train_cache = train_cache
        self.device = device
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.edges_per_epoch = edges_per_epoch

        self.optimizer_model = torch.optim.Adam(
            model.parameters(), lr=lr)
        self.optimizer_selector = torch.optim.Adam(
            selector.parameters(), lr=lr_selector, betas=(0.5, 0.999))
        self.lr = lr
        self.lr_min = lr_min

        # pre-stack positive edges
        pos_edge = torch.stack([
            split_edge['train']['source_node'],
            split_edge['train']['target_node'],
        ], dim=0)
        neg_edge_idx, _ = add_self_loops(pos_edge)
        self._neg_edge_idx = neg_edge_idx.to(device)

        pos_val = torch.stack([
            split_edge['valid']['source_node'],
            split_edge['valid']['target_node'],
        ], dim=0)
        neg_val_idx, _ = add_self_loops(pos_val)
        self._neg_val_idx = neg_val_idx.to(device)

    def _neg(self, ref_idx, n):
        return negative_sampling(
            ref_idx, num_nodes=self.data.num_nodes,
            num_neg_samples=n).to(self.device)

    def _temperature(self, epoch, total):
        if total <= 1:
            return self.temperature_end
        frac = epoch / (total - 1)
        return self.temperature_start + frac * (
            self.temperature_end - self.temperature_start)

    def _get_subgraphs_from_cache(self, cache, indices):
        """Retrieve (nodes_S, u_local, v_local) tuples for a batch of indices."""
        return [cache[i.item()] for i in indices]

    def _compute_loss_batch(self, model, selector, edges, neg_edges,
                            pos_subs, neg_subs):
        x_full = self.data.x.float()
        return model.compute_loss(
            selector, edges, neg_edges, x_full,
            self.ppr_dense, pos_subs, neg_subs)

    def search(self, epochs=50, batch_size=64, verbose=True, val_every=5):
        """
        Run bi-level architecture search.

        Args:
            epochs: search epochs
            batch_size: number of edges per batch (small due to per-subgraph GNN)
            verbose: show tqdm
            val_every: evaluate val MRR every N epochs

        Returns:
            history dict
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_model, T_max=epochs, eta_min=self.lr_min)

        train_src = self.split_edge['train']['source_node'].to(self.device)
        train_dst = self.split_edge['train']['target_node'].to(self.device)
        num_train = train_src.size(0)

        val_src = self.split_edge['valid']['source_node'].to(self.device)
        val_dst = self.split_edge['valid']['target_node'].to(self.device)
        num_valid = val_src.size(0)
        val_cycler = _CyclicPermuter(num_valid, batch_size)

        history = {
            'search_loss': [], 'val_loss': [],
            'best_val_loss': float('inf'), 'best_epoch': 0,
            'temperature': [], 'arch_entropy': [], 'top1_mass': [],
            'total_time': 0.0,
        }
        best_val_loss = float('inf')
        start_time = time.time()

        iterator = (tqdm(range(epochs), desc='OptionA Search',
                         mininterval=10, maxinterval=60)
                    if verbose else range(epochs))

        for epoch in iterator:
            temp = self._temperature(epoch, epochs)
            self.selector.set_temperature(temp)

            if self.edges_per_epoch and self.edges_per_epoch < num_train:
                epoch_idx = torch.randperm(num_train)[:self.edges_per_epoch]
            else:
                epoch_idx = torch.randperm(num_train)

            self.model.train()
            self.selector.train()
            epoch_loss = 0.0
            steps = 0

            for perm in DataLoader(epoch_idx, batch_size, shuffle=True):
                train_edge = torch.stack(
                    [train_src[perm], train_dst[perm]], dim=0)
                train_neg = self._neg(self._neg_edge_idx, len(perm))

                pos_subs = self._get_subgraphs_from_cache(
                    self.train_cache, perm)
                neg_subs = sample_neg_subgraphs(
                    self.search_extractor, train_neg)

                # ---- Step 1: update selector on val loss (theta step) ----
                val_perm = val_cycler.next(self.device)
                val_edge = torch.stack([val_src[val_perm], val_dst[val_perm]], dim=0)
                val_neg = self._neg(self._neg_val_idx, len(val_perm))

                val_pos_subs = sample_neg_subgraphs(
                    self.search_extractor, val_edge)
                val_neg_subs = sample_neg_subgraphs(
                    self.search_extractor, val_neg)

                self.optimizer_selector.zero_grad()
                val_loss = self._compute_loss_batch(
                    self.model, self.selector,
                    val_edge, val_neg,
                    val_pos_subs, val_neg_subs)

                if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                    val_loss.backward()
                    nn.utils.clip_grad_norm_(self.selector.parameters(), 1.0)
                    self.optimizer_selector.step()

                # ---- Step 2: update model on train loss (w step) ---------
                # Keep selector in train mode (soft weights) so the GNN always
                # sees a smooth P_soft mixture — hard argmax during early random
                # selection would give contradictory gradients across batches.
                # Selector grads from train_loss are computed but discarded
                # (only optimizer_model.step() is called).
                self.optimizer_model.zero_grad()
                train_loss = self._compute_loss_batch(
                    self.model, self.selector,
                    train_edge, train_neg,
                    pos_subs, neg_subs)

                if not (torch.isnan(train_loss) or torch.isinf(train_loss)):
                    train_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer_model.step()
                    epoch_loss += train_loss.item()
                    steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            history['search_loss'].append(avg_loss)
            history['temperature'].append(temp)

            ent, top1 = self._arch_diagnostics(
                train_src, train_dst, batch_size)
            history['arch_entropy'].append(ent)
            history['top1_mass'].append(top1)

            do_val = (epoch % val_every == 0) or (epoch == epochs - 1)
            if do_val:
                vl = self._val_loss(batch_size)
                history['val_loss'].append(vl)
                if vl < best_val_loss:
                    best_val_loss = vl
                    history['best_val_loss'] = best_val_loss
                    history['best_epoch'] = epoch
                    history['best_model_state'] = copy.deepcopy(
                        self.model.state_dict())
                    history['best_selector_state'] = copy.deepcopy(
                        self.selector.state_dict())
            else:
                history['val_loss'].append(None)

            if verbose:
                postfix = {
                    'loss': f'{avg_loss:.4f}',
                    'tau': f'{temp:.3f}',
                    'ent': f'{ent:.2f}',
                }
                if history['val_loss'][-1] is not None:
                    postfix['val'] = f'{history["val_loss"][-1]:.4f}'
                postfix['best'] = f'{best_val_loss:.4f}'
                iterator.set_postfix(postfix)

            scheduler.step()

        history['total_time'] = time.time() - start_time

        if 'best_model_state' in history:
            self.model.load_state_dict(history['best_model_state'])
            self.selector.load_state_dict(history['best_selector_state'])
            if verbose:
                print(f'Restored best from epoch {history["best_epoch"]}')

        return history

    @torch.no_grad()
    def _arch_diagnostics(self, train_src, train_dst, batch_size,
                          sample_size=512):
        self.model.eval()
        self.selector.train()
        n = train_src.size(0)
        idx = torch.randperm(n)[:min(sample_size, n)]
        src, dst = train_src[idx], train_dst[idx]
        x_full = self.data.x.float()
        cross = self.model.compute_selector_input(
            src, dst, x_full, self.ppr_dense, self.alphas)
        atten = self.selector(cross)
        entropy = -(atten * (atten + 1e-15).log()).sum(dim=-1).mean().item()
        top1 = atten.max(dim=-1).values.mean().item()
        self.model.train()
        return entropy, top1

    @torch.no_grad()
    def _val_loss(self, batch_size, sample=256):
        self.model.eval()
        self.selector.eval()
        val_src = self.split_edge['valid']['source_node'].to(self.device)
        val_dst = self.split_edge['valid']['target_node'].to(self.device)
        n = min(sample, val_src.size(0))
        idx = torch.randperm(val_src.size(0))[:n]
        edges = torch.stack([val_src[idx], val_dst[idx]], dim=0)
        neg = self._neg(self._neg_val_idx, n)
        pos_subs = sample_neg_subgraphs(self.extractor, edges)
        neg_subs = sample_neg_subgraphs(self.extractor, neg)
        x_full = self.data.x.float()
        loss = self.model.compute_loss(
            self.selector, edges, neg, x_full,
            self.ppr_dense, pos_subs, neg_subs)
        return loss.item()

    @torch.no_grad()
    def get_edge_alpha_indices(self, split='train', batch_size=256):
        """Get hard alpha index for every edge in a split."""
        self.model.eval()
        self.selector.eval()
        src = self.split_edge[split]['source_node'].to(self.device)
        dst = self.split_edge[split]['target_node'].to(self.device)
        x_full = self.data.x.float()
        all_indices = []
        for perm in DataLoader(torch.arange(src.size(0)), batch_size):
            cross = self.model.compute_selector_input(
                src[perm], dst[perm], x_full, self.ppr_dense, self.alphas)
            indices = self.selector.get_alpha_indices(cross)
            all_indices.append(indices.cpu())
        return torch.cat(all_indices)


class _CyclicPermuter:
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size
        self._perm = torch.randperm(n)
        self._pos = 0

    def next(self, device=None):
        if self._pos + self.batch_size > self.n:
            self._perm = torch.randperm(self.n)
            self._pos = 0
        batch = self._perm[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return batch.to(device) if device is not None else batch
