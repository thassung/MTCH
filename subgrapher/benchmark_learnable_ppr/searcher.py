"""
Bi-level architecture search for learnable PPR configuration.
Ported from PS2's train_arch() with Hessian-vector products.

Phase 1: Learns per-edge (teleport_u, teleport_v) selection on the full graph.
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
    Ported from PS2's _hessian_vector_product().

    Computes:  d^2 L_train / (d_w d_alpha) * v
    using finite difference: [grad_alpha L(w+Rv, alpha) - grad_alpha L(w-Rv, alpha)] / 2R
    """
    all_vector = vector + dalpha_
    R = r / _concat(all_vector).norm()

    # w+ = w + R * v
    for p, v in zip(model.parameters(), vector):
        p.data.add_(v, alpha=R)
    for p, v in zip(arch_net.parameters(), dalpha_):
        p.data.add_(v, alpha=R)

    h = model(data.x, data.edge_index)
    loss = model.compute_loss(h, arch_net, multi_scale_ppr,
                              train_edge, train_edge_neg)

    variable_list = [param for param in arch_net.parameters()]
    grads_p = torch.autograd.grad(loss, variable_list)

    # w- = w - 2R * v (from w+ back to w-R*v)
    for p, v in zip(model.parameters(), vector):
        p.data.sub_(v, alpha=2 * R)
    for p, v in zip(arch_net.parameters(), dalpha_):
        p.data.sub_(v, alpha=2 * R)

    h = model(data.x, data.edge_index)
    loss = model.compute_loss(h, arch_net, multi_scale_ppr,
                              train_edge, train_edge_neg)

    grads_n = torch.autograd.grad(loss, variable_list)

    # Restore w = w- + R * v
    for p, v in zip(model.parameters(), vector):
        p.data.add_(v, alpha=R)
    for p, v in zip(arch_net.parameters(), dalpha_):
        p.data.add_(v, alpha=R)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


def _generate_neg_edges(split_edge, num_nodes, device, sample_size):
    """Generate negative edge samples for training."""
    pos_edge = torch.stack([
        split_edge['train']['source_node'],
        split_edge['train']['target_node']
    ], dim=0)
    new_edge_index, _ = add_self_loops(pos_edge)
    neg_edge = negative_sampling(
        new_edge_index, num_nodes=num_nodes,
        num_neg_samples=sample_size)
    return neg_edge.to(device)


class ArchitectureSearcher:
    """
    Bi-level architecture search for PPR configuration.

    Phase 1 training loop:
    For each batch:
      1. v_model = copy(model), v_arch = copy(arch_net)
      2. Inner step: train v_model + v_arch on train batch
      3. Compute val loss with v_model + v_arch
      4. Hessian-vector product for second-order gradients
      5. Update arch_net with implicit gradients
      6. Update model on train batch with frozen arch_net

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
    """

    def __init__(self, model, arch_net, multi_scale_ppr, data, split_edge,
                 device='cuda', lr=0.01, lr_arch=0.01, lr_min=0.001):
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
        self.v_optimizer = torch.optim.Adam(
            list(self.v_model.parameters()) +
            list(self.v_arch_net.parameters()),
            lr=lr)
        self.optimizer_arch = torch.optim.Adam(
            arch_net.parameters(), lr=lr_arch, betas=(0.5, 0.999))

        self.lr = lr
        self.lr_min = lr_min

    def search(self, epochs=50, batch_size=1024, patience=50, verbose=True):
        """
        Run architecture search.

        Returns:
            history: Dict with search metrics
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr_min)

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
        }

        best_val_loss = float('inf')
        cnt_wait = 0
        start_time = time.time()

        iterator = tqdm(range(epochs), desc='Arch Search') if verbose else range(epochs)

        for epoch in iterator:
            scheduler.step()
            lr_current = scheduler.get_last_lr()[-1]
            epoch_loss = 0.0
            steps = 0

            self.model.train()
            for perm in DataLoader(range(num_train), batch_size, shuffle=True):
                self.arch_net.train()

                # Edges for this batch
                train_src = pos_train_source[perm]
                train_dst = pos_train_target[perm]
                train_edge = torch.stack([train_src, train_dst], dim=0)
                train_edge_neg = _generate_neg_edges(
                    self.split_edge, self.data.num_nodes, self.device,
                    len(perm))

                # Step 1: Copy model -> v_model
                self.v_model.load_state_dict(self.model.state_dict())
                self.v_arch_net.load_state_dict(self.arch_net.state_dict())

                # Step 2: Inner step - train v_model + v_arch on train batch
                h = self.v_model(self.data.x, self.data.edge_index)
                loss_inner = self.v_model.compute_loss(
                    h, self.v_arch_net, self.multi_scale_ppr,
                    train_edge, train_edge_neg)

                self.v_optimizer.zero_grad()
                loss_inner.backward()
                nn.utils.clip_grad_norm_(self.v_model.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.v_arch_net.parameters(), 1.0)
                self.v_optimizer.step()

                # Step 3: Validation loss with v_model + v_arch
                val_perm = val_perm_gen.next(self.device)
                val_src = pos_valid_source[val_perm]
                val_dst = pos_valid_target[val_perm]
                valid_edge = torch.stack([val_src, val_dst], dim=0)
                valid_edge_neg = _generate_neg_edges(
                    self.split_edge, self.data.num_nodes, self.device,
                    len(val_perm))

                h_v = self.v_model(self.data.x, self.data.edge_index)
                loss_val = self.v_model.compute_loss(
                    h_v, self.v_arch_net, self.multi_scale_ppr,
                    valid_edge, valid_edge_neg)

                self.v_optimizer.zero_grad()
                loss_val.backward()
                nn.utils.clip_grad_norm_(self.v_model.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.v_arch_net.parameters(), 1.0)

                # Step 4: Hessian-vector product for arch_net update
                dalpha = [v.grad for v in self.v_arch_net.parameters()]
                dalpha_ = [v.grad.data for v in self.v_arch_net.parameters()]
                vector = [v.grad.data for v in self.v_model.parameters()]

                implicit_grads = _hessian_vector_product(
                    vector, dalpha_, self.model, self.arch_net,
                    self.multi_scale_ppr, self.data,
                    train_edge, train_edge_neg)

                for g, ig in zip(dalpha, implicit_grads):
                    g.data.sub_(ig.data, alpha=lr_current)

                # Apply gradients to arch_net
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
                h = self.model(self.data.x, self.data.edge_index)
                self.arch_net.eval()
                loss_model = self.model.compute_loss(
                    h, self.arch_net, self.multi_scale_ppr,
                    train_edge, train_edge_neg)
                self.optimizer.zero_grad()
                loss_model.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss_model.item()
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            history['search_loss'].append(avg_loss)

            # Periodic validation and config distribution tracking
            val_loss = self._evaluate_val_loss(batch_size)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history['best_val_loss'] = best_val_loss
                history['best_epoch'] = epoch
                cnt_wait = 0
                history['best_model_state'] = copy.deepcopy(
                    self.model.state_dict())
                history['best_arch_state'] = copy.deepcopy(
                    self.arch_net.state_dict())
            else:
                cnt_wait += 1

            if verbose:
                iterator.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'val': f'{val_loss:.4f}',
                    'best': f'{best_val_loss:.4f}',
                    'wait': f'{cnt_wait}/{patience}',
                })

            if cnt_wait >= patience:
                if verbose:
                    print(f"\n[Early Stop] epoch {epoch}")
                break

        history['total_time'] = time.time() - start_time

        # Restore best
        if 'best_model_state' in history:
            self.model.load_state_dict(history['best_model_state'])
            self.arch_net.load_state_dict(history['best_arch_state'])
            if verbose:
                print(f"Restored best from epoch {history['best_epoch']}")

        return history

    @torch.no_grad()
    def _evaluate_val_loss(self, batch_size):
        """Compute validation loss with current model + arch_net."""
        self.model.eval()
        self.arch_net.eval()

        val_src = self.split_edge['valid']['source_node'].to(self.device)
        val_dst = self.split_edge['valid']['target_node'].to(self.device)
        num_val = val_src.size(0)

        total_loss = 0.0
        count = 0
        for perm in DataLoader(range(num_val), batch_size, shuffle=False):
            src = val_src[perm]
            dst = val_dst[perm]
            edge = torch.stack([src, dst], dim=0)
            edge_neg = _generate_neg_edges(
                self.split_edge, self.data.num_nodes, self.device, len(perm))

            h = self.model(self.data.x, self.data.edge_index)
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
        config_counts = torch.zeros(self.multi_scale_ppr.num_configs,
                                    dtype=torch.long)
        for idx in config_indices:
            config_counts[idx] += 1

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
