"""
Bi-level architecture search for learnable PPR configuration.

Phase 1: Learns per-edge (teleport_u, teleport_v) selection on the full graph.
Temperature annealing controls the training horizon -- no early stopping needed.
"""

import torch
import torch.nn as nn
import copy
import time
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, add_self_loops
from tqdm import tqdm


def _concat(xs):
    """Flatten and concatenate a list of tensors."""
    return torch.cat([x.view(-1) for x in xs])


def _hessian_vector_product(vector, dalpha_, model, arch_net,
                            multi_scale_ppr, data, train_edge, train_edge_neg,
                            r=1e-2):
    """
    Approximate Hessian-vector product for second-order gradient.

    Computes:  d^2 L_train / (d_w d_alpha) * v
    using finite difference: [grad_alpha L(w+Rv, alpha) - grad_alpha L(w-Rv, alpha)] / 2R

    Always runs in float32 for numerical stability (even when AMP is active).
    """
    all_vector = vector + dalpha_
    R = r / (_concat(all_vector).norm() + 1e-8)

    for p, v in zip(model.parameters(), vector):
        p.data.add_(v, alpha=R)
    for p, v in zip(arch_net.parameters(), dalpha_):
        p.data.add_(v, alpha=R)

    with torch.cuda.amp.autocast(enabled=False):
        h = model(data.x.float(), data.edge_index)
        loss = model.compute_loss(h, arch_net, multi_scale_ppr,
                                  train_edge, train_edge_neg)

    variable_list = [param for param in arch_net.parameters()]
    grads_p = torch.autograd.grad(loss, variable_list)

    for p, v in zip(model.parameters(), vector):
        p.data.sub_(v, alpha=2 * R)
    for p, v in zip(arch_net.parameters(), dalpha_):
        p.data.sub_(v, alpha=2 * R)

    with torch.cuda.amp.autocast(enabled=False):
        h = model(data.x.float(), data.edge_index)
        loss = model.compute_loss(h, arch_net, multi_scale_ppr,
                                  train_edge, train_edge_neg)

    grads_n = torch.autograd.grad(loss, variable_list)

    for p, v in zip(model.parameters(), vector):
        p.data.add_(v, alpha=R)
    for p, v in zip(arch_net.parameters(), dalpha_):
        p.data.add_(v, alpha=R)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


class ArchitectureSearcher:
    """
    Bi-level architecture search for PPR configuration.

    Phase 1 training loop:
    For each batch:
      1. v_model = copy(model), v_arch = copy(arch_net)
      2. Inner step: train v_model + v_arch on train batch
      3. Compute val loss with v_model + v_arch (gradient only, no optimizer step)
      4. Hessian-vector product for second-order gradients (skipped if first_order)
      5. Update arch_net with implicit gradients
      6. Update model on train batch with frozen arch_net

    Temperature annealing controls the training horizon. No early stopping is used;
    the full epoch budget runs to allow the temperature to anneal completely.

    Args:
        model: AutoLinkPPR instance
        arch_net: PPRSearchNet instance
        multi_scale_ppr: MultiScalePPR instance
        data: PyG Data object (full graph)
        split_edge: Edge split dictionary
        device: torch device
        lr: Learning rate for model
        lr_arch: Learning rate for arch_net
        lr_min: Minimum learning rate for cosine scheduler
        temperature_start: Starting temperature for annealing (default 1.0)
        temperature_end: Ending temperature for annealing (default 0.2)
        edges_per_search_epoch: Subsample training edges per epoch (None = all)
        first_order: If True, skip HVP (first-order DARTS approximation)
    """

    def __init__(self, model, arch_net, multi_scale_ppr, data, split_edge,
                 device='cuda', lr=0.01, lr_arch=0.01, lr_min=0.001,
                 temperature_start=1.0, temperature_end=0.2,
                 edges_per_search_epoch=None, first_order=False):
        self.model = model.to(device)
        self.arch_net = arch_net.to(device)
        self.multi_scale_ppr = multi_scale_ppr
        self.data = data.to(device)
        self.split_edge = split_edge
        self.device = device

        self.v_model = copy.deepcopy(model).to(device)
        self.v_arch_net = copy.deepcopy(arch_net).to(device)

        self.optimizer = torch.optim.Adam(
            list(model.parameters()), lr=lr)
        self.v_optimizer = torch.optim.SGD(
            list(self.v_model.parameters()) +
            list(self.v_arch_net.parameters()),
            lr=lr)
        self.optimizer_arch = torch.optim.Adam(
            arch_net.parameters(), lr=lr_arch, betas=(0.5, 0.999))

        self.lr = lr
        self.lr_min = lr_min
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.edges_per_search_epoch = edges_per_search_epoch
        self.first_order = first_order

        pos_edge = torch.stack([
            split_edge['train']['source_node'],
            split_edge['train']['target_node']
        ], dim=0)
        self._neg_edge_index, _ = add_self_loops(pos_edge)
        self._neg_edge_index = self._neg_edge_index.to(device)

    def _generate_neg_edges(self, sample_size):
        """Generate negative edge samples using pre-computed self-loop edge index."""
        neg_edge = negative_sampling(
            self._neg_edge_index, num_nodes=self.data.num_nodes,
            num_neg_samples=sample_size)
        return neg_edge.to(self.device)

    def _get_temperature(self, epoch, total_epochs):
        """Linear temperature annealing from start to end."""
        if total_epochs <= 1:
            return self.temperature_end
        frac = epoch / (total_epochs - 1)
        return self.temperature_start + frac * (self.temperature_end - self.temperature_start)

    def search(self, epochs=50, batch_size=1024, verbose=True,
               val_every=5, use_amp=False, **kwargs):
        """
        Run architecture search.

        Args:
            epochs: Number of search epochs
            batch_size: Edges per batch
            verbose: Show tqdm progress
            val_every: Run full validation every N epochs (default 5)
            use_amp: Enable mixed-precision (float16) for forward/backward

        Returns:
            history: Dict with search metrics and diagnostics
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr_min)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        pos_train_source = self.split_edge['train']['source_node'].to(
            self.device)
        pos_train_target = self.split_edge['train']['target_node'].to(
            self.device)
        num_train = pos_train_source.size(0)

        pos_valid_source = self.split_edge['valid']['source_node'].to(
            self.device)
        pos_valid_target = self.split_edge['valid']['target_node'].to(
            self.device)
        num_valid = pos_valid_source.size(0)
        val_perm_gen = _CyclicPermuter(num_valid, batch_size)

        history = {
            'search_loss': [],
            'val_loss': [],
            'config_distributions': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'total_time': 0.0,
            'temperature': [],
            'arch_entropy': [],
            'softmax_mass_top1': [],
            'embedding_norm_mean': [],
            'embedding_norm_max': [],
        }

        best_val_loss = float('inf')
        start_time = time.time()

        iterator = tqdm(range(epochs), desc='Arch Search') if verbose else range(epochs)

        for epoch in iterator:
            temp = self._get_temperature(epoch, epochs)
            self.arch_net.set_temperature(temp)
            self.v_arch_net.set_temperature(temp)

            lr_current = self.optimizer.param_groups[0]['lr']
            epoch_loss = 0.0
            steps = 0

            if (self.edges_per_search_epoch is not None
                    and self.edges_per_search_epoch < num_train):
                epoch_indices = torch.randperm(num_train)[:self.edges_per_search_epoch]
            else:
                epoch_indices = torch.arange(num_train)

            self.model.train()
            for perm in DataLoader(epoch_indices, batch_size, shuffle=True):
                self.arch_net.train()

                train_src = pos_train_source[perm]
                train_dst = pos_train_target[perm]
                train_edge = torch.stack([train_src, train_dst], dim=0)
                train_edge_neg = self._generate_neg_edges(len(perm))

                # Step 1: Copy model -> v_model
                self.v_model.load_state_dict(self.model.state_dict())
                self.v_arch_net.load_state_dict(self.arch_net.state_dict())

                # Step 2: Inner step - train v_model + v_arch on train batch
                self.v_optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    h = self.v_model(self.data.x, self.data.edge_index)
                    loss_inner = self.v_model.compute_loss(
                        h, self.v_arch_net, self.multi_scale_ppr,
                        train_edge, train_edge_neg)

                loss_inner.backward()
                nn.utils.clip_grad_norm_(self.v_model.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.v_arch_net.parameters(), 1.0)
                self.v_optimizer.step()

                # Step 3: Val loss for meta-gradient (no optimizer step, no scaler)
                val_perm = val_perm_gen.next(self.device)
                val_src = pos_valid_source[val_perm]
                val_dst = pos_valid_target[val_perm]
                valid_edge = torch.stack([val_src, val_dst], dim=0)
                valid_edge_neg = self._generate_neg_edges(len(val_perm))

                self.v_optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    h_v = self.v_model(self.data.x, self.data.edge_index)
                    loss_val = self.v_model.compute_loss(
                        h_v, self.v_arch_net, self.multi_scale_ppr,
                        valid_edge, valid_edge_neg)

                loss_val.backward()
                nn.utils.clip_grad_norm_(self.v_model.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.v_arch_net.parameters(), 1.0)

                # Step 4: Arch_net update
                dalpha = [v.grad for v in self.v_arch_net.parameters()]
                dalpha_ = [v.grad.data for v in self.v_arch_net.parameters()]
                vector = [v.grad.data for v in self.v_model.parameters()]

                if self.first_order:
                    pass
                else:
                    implicit_grads = _hessian_vector_product(
                        vector, dalpha_, self.model, self.arch_net,
                        self.multi_scale_ppr, self.data,
                        train_edge, train_edge_neg)

                    for g, ig in zip(dalpha, implicit_grads):
                        g.data.sub_(ig.data, alpha=lr_current)

                i = 0
                for name, params in self.arch_net.named_parameters():
                    if params.requires_grad:
                        if params.grad is None:
                            params.grad = dalpha[i].data.clone()
                        else:
                            params.grad.data.copy_(dalpha[i].data)
                        i += 1
                nn.utils.clip_grad_norm_(self.arch_net.parameters(), 1.0)
                self.optimizer_arch.step()

                # Step 5: Update model on train batch with frozen arch_net
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    h = self.model(self.data.x, self.data.edge_index)
                    self.arch_net.eval()
                    loss_model = self.model.compute_loss(
                        h, self.arch_net, self.multi_scale_ppr,
                        train_edge, train_edge_neg)

                scaler.scale(loss_model).backward()
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss_model.item()
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            history['search_loss'].append(avg_loss)
            history['temperature'].append(temp)

            with torch.no_grad():
                h_diag = self.model(self.data.x, self.data.edge_index)
                h_norms = h_diag.norm(dim=-1)
                history['embedding_norm_mean'].append(h_norms.mean().item())
                history['embedding_norm_max'].append(h_norms.max().item())

            arch_entropy, mass_top1 = self._compute_arch_diagnostics(
                pos_train_source, pos_train_target, batch_size)
            history['arch_entropy'].append(arch_entropy)
            history['softmax_mass_top1'].append(mass_top1)

            snapshot_epochs = {0, epochs // 3, 2 * epochs // 3, epochs - 1}
            if epoch in snapshot_epochs:
                _, counts = self.get_edge_configs('train', batch_size=4096)
                history['config_distributions'].append({
                    'epoch': epoch,
                    'train_counts': counts.tolist(),
                })

            do_val = (epoch % val_every == 0) or (epoch == epochs - 1) or (epoch == 0)
            if do_val:
                val_loss = self._evaluate_val_loss(batch_size)
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    history['best_val_loss'] = best_val_loss
                    history['best_epoch'] = epoch
                    history['best_model_state'] = copy.deepcopy(
                        self.model.state_dict())
                    history['best_arch_state'] = copy.deepcopy(
                        self.arch_net.state_dict())
            else:
                history['val_loss'].append(None)

            if verbose:
                postfix = {
                    'loss': f'{avg_loss:.4f}',
                    'tau': f'{temp:.3f}',
                    'ent': f'{arch_entropy:.2f}',
                }
                if history['val_loss'][-1] is not None:
                    postfix['val'] = f'{history["val_loss"][-1]:.4f}'
                postfix['best'] = f'{best_val_loss:.4f}'
                iterator.set_postfix(postfix)

            scheduler.step()

        history['total_time'] = time.time() - start_time

        if 'best_model_state' in history:
            self.model.load_state_dict(history['best_model_state'])
            self.arch_net.load_state_dict(history['best_arch_state'])
            if verbose:
                print(f"Restored best from epoch {history['best_epoch']}")

        return history

    @torch.no_grad()
    def _compute_arch_diagnostics(self, train_source, train_target,
                                  batch_size, sample_size=4096):
        """Compute mean arch entropy and top-1 softmax mass on a training sample."""
        self.model.eval()
        self.arch_net.train()

        n = train_source.size(0)
        sample_n = min(sample_size, n)
        idx = torch.randperm(n)[:sample_n]
        src = train_source[idx]
        dst = train_target[idx]

        h = self.model(self.data.x, self.data.edge_index)
        cross = self.multi_scale_ppr.get_ppr_cross_pair_batch(src, dst, h)
        atten = self.arch_net(cross)

        entropy = -(atten * (atten + 1e-15).log()).sum(dim=-1).mean().item()
        top1_mass = atten.max(dim=-1).values.mean().item()

        self.model.train()
        return entropy, top1_mass

    @torch.no_grad()
    def _evaluate_val_loss(self, batch_size):
        """Compute validation loss with current model + arch_net."""
        self.model.eval()
        self.arch_net.eval()

        val_src = self.split_edge['valid']['source_node'].to(self.device)
        val_dst = self.split_edge['valid']['target_node'].to(self.device)
        num_val = val_src.size(0)

        h = self.model(self.data.x, self.data.edge_index)

        total_loss = 0.0
        count = 0
        for perm in DataLoader(range(num_val), batch_size, shuffle=False):
            src = val_src[perm]
            dst = val_dst[perm]
            edge = torch.stack([src, dst], dim=0)
            edge_neg = self._generate_neg_edges(len(perm))

            loss = self.model.compute_loss(
                h, self.arch_net, self.multi_scale_ppr, edge, edge_neg)
            total_loss += loss.item() * len(perm)
            count += len(perm)

        return total_loss / max(count, 1)

    @torch.no_grad()
    def get_edge_configs(self, split='train', batch_size=4096):
        """
        Get learned config index for every edge in a split.

        Returns:
            config_indices: Tensor [num_edges] of config indices
            config_counts: Tensor [num_configs] count per config
        """
        self.model.eval()
        self.arch_net.eval()

        src = self.split_edge[split]['source_node'].to(self.device)
        dst = self.split_edge[split]['target_node'].to(self.device)
        num_edges = src.size(0)

        all_indices = []
        h = self.model(self.data.x, self.data.edge_index)

        for perm in DataLoader(range(num_edges), batch_size, shuffle=False):
            edges = torch.stack([src[perm], dst[perm]], dim=0)
            cross = self.multi_scale_ppr.get_ppr_cross_pair_batch(
                edges[0], edges[1], h)
            indices = self.arch_net.get_config_indices(cross)
            all_indices.append(indices.cpu())

        config_indices = torch.cat(all_indices)
        config_counts = torch.bincount(
            config_indices, minlength=self.multi_scale_ppr.num_configs)

        return config_indices, config_counts


class _CyclicPermuter:
    """Yields random permutation batches, cycling when exhausted."""

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
        if device is not None:
            batch = batch.to(device)
        return batch
