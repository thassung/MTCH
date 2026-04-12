"""
Phase 2: Fine-tune a GNN on subgraphs extracted with learned PPR configurations.

Uses the per-edge (teleport_u, teleport_v) configs from architecture search
to extract actual subgraphs, then trains with the existing batched infrastructure.
"""

import torch
import torch.nn as nn
import time
import copy
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from tqdm import tqdm


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

        Args:
            u: Source node index
            v: Target node index
            edge_idx: Index into config_indices for this edge

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


def _forward_micro_batch(encoder, predictor, subgraph_list, edge_mappings, device):
    """Run forward + loss on a micro-batch of subgraphs. Returns (loss, n_examples)."""
    batched = Batch.from_data_list(subgraph_list).to(device)
    h = encoder(batched.x, batched.edge_index)

    mb_loss = torch.tensor(0.0, device=device)
    mb_examples = 0
    node_offset = 0

    for u_sub, v_sub, num_nodes in edge_mappings:
        u_idx = node_offset + u_sub
        v_idx = node_offset + v_sub

        pos_out = predictor(h[u_idx].unsqueeze(0), h[v_idx].unsqueeze(0))
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        neg_offset = torch.randint(0, num_nodes, (1,), device=device)
        neg_idx = node_offset + neg_offset
        neg_out = predictor(h[u_idx].unsqueeze(0), h[neg_idx])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        mb_loss = mb_loss + pos_loss + neg_loss
        mb_examples += 1
        node_offset += num_nodes

    return mb_loss, mb_examples


def train_epoch_finetune(encoder, predictor, data, split_edge,
                         extractor, optimizer, batch_size, device,
                         grad_clip=None, verbose=False,
                         max_subgraphs_per_forward=256):
    """
    Train one epoch on subgraphs extracted with learned PPR configs.

    Subgraphs are extracted on CPU, then forwarded in micro-batches of
    *max_subgraphs_per_forward* to avoid GPU OOM.  Gradients are
    accumulated across micro-batches before a single optimizer step per
    outer batch.
    """
    encoder.train()
    predictor.train()

    source = split_edge['train']['source_node']
    target = split_edge['train']['target_node']

    total_loss = 0.0
    total_examples = 0

    dataloader = DataLoader(range(source.size(0)), batch_size, shuffle=True)
    if verbose:
        dataloader = tqdm(dataloader, desc='  Fine-tune batches', leave=False)

    for perm in dataloader:
        # --- extract all subgraphs for this batch (CPU-side) ---
        subgraph_list = []
        edge_mappings = []

        for i in perm:
            u = source[i].item()
            v = target[i].item()
            sub_data, _, meta = extractor.extract_subgraph(u, v, i.item())
            if meta['u_subgraph'] == -1 or meta['v_subgraph'] == -1:
                continue
            subgraph_list.append(sub_data)
            edge_mappings.append((meta['u_subgraph'], meta['v_subgraph'],
                                  meta['num_nodes_selected']))

        if len(subgraph_list) == 0:
            continue

        # --- micro-batch forward with gradient accumulation ---
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_examples = 0
        n = len(subgraph_list)
        mb = max_subgraphs_per_forward

        for start in range(0, n, mb):
            end = min(start + mb, n)
            mb_loss, mb_ex = _forward_micro_batch(
                encoder, predictor,
                subgraph_list[start:end], edge_mappings[start:end], device)
            if mb_ex == 0:
                continue
            (mb_loss / mb_ex).backward()
            accum_loss += mb_loss.item()
            accum_examples += mb_ex

        if accum_examples > 0:
            # Scale gradients so the effective loss = mean over all micro-batches
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
            total_loss += accum_loss / accum_examples * accum_examples
            total_examples += accum_examples

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / max(total_examples, 1)


def finetune_on_subgraphs(encoder, predictor, data, split_edge,
                           multi_scale_ppr, config_indices,
                           alpha=None, top_k=100, epochs=500,
                           batch_size=8192, lr=0.005, eval_steps=5,
                           device='cpu', verbose=True, patience=30,
                           weight_decay=1e-5, grad_clip=1.0,
                           max_subgraphs_per_forward=256,
                           checkpoint_dir=None):
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
        batch_size: Edges per batch (subgraph extraction batch)
        lr: Learning rate
        eval_steps: Evaluate every N epochs
        device: Device
        verbose: Print progress
        patience: Early stopping patience
        weight_decay: L2 regularization
        grad_clip: Gradient clipping norm
        max_subgraphs_per_forward: GPU micro-batch size (subgraphs per
            forward pass). Lower this if OOM occurs.
        checkpoint_dir: If set, saves best model + periodic checkpoints
            to this directory so progress survives crashes.

    Returns:
        history: Training history dict
    """
    import os as _os
    if alpha is None:
        alpha = [0.5]
    from .evaluator import evaluate_learnable_ppr

    encoder = encoder.to(device)
    predictor = predictor.to(device)

    extractor = LearnablePPRExtractor(
        data, multi_scale_ppr, config_indices,
        alpha=alpha, top_k=top_k)

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

    # --- resume from checkpoint if available ---
    start_epoch = 1
    best_val_mrr = 0.0
    epochs_no_improve = 0
    best_state = None

    if checkpoint_dir:
        _os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = _os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if _os.path.isfile(ckpt_path):
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

    iterator = range(start_epoch, epochs + 1)
    if verbose:
        iterator = tqdm(iterator, desc='Fine-tuning',
                        initial=start_epoch - 1, total=epochs)

    for epoch in iterator:
        show_batch = (epoch <= start_epoch + 2) if verbose else False
        loss = train_epoch_finetune(
            encoder, predictor, data, split_edge, extractor,
            optimizer, batch_size, device, grad_clip=grad_clip,
            verbose=show_batch,
            max_subgraphs_per_forward=max_subgraphs_per_forward)
        history['train_loss'].append(loss)

        if epoch % eval_steps == 0 or epoch == epochs:
            val_results = evaluate_learnable_ppr(
                encoder, predictor, data, split_edge,
                multi_scale_ppr, config_indices,
                split='valid', alpha=alpha, top_k=top_k,
                batch_size=batch_size, device=device)
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
                               _os.path.join(checkpoint_dir, 'best_model.pt'))
            else:
                epochs_no_improve += eval_steps

            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss:.4f}',
                    'mrr': f'{mrr:.4f}',
                    'best': f'{best_val_mrr:.4f}',
                    'pat': f'{epochs_no_improve}/{patience}',
                })

            # periodic checkpoint every eval_steps
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
                }, _os.path.join(checkpoint_dir, 'latest_checkpoint.pt'))

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
