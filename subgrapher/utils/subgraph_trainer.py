"""
Shared training utilities for subgraph-based link prediction.

Used by both benchmark-ppr and benchmark-khop modules to avoid code duplication.
"""

import torch
import torch.nn as nn
import time
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..benchmark.evaluator import evaluate_link_prediction, print_evaluation_results


def train_epoch_with_subgraphs(encoder, predictor, full_data, split_edge, 
                                 extractor, optimizer, batch_size, device, grad_clip=None):
    """
    Train for one epoch with subgraph extraction.
    
    Args:
        encoder: GNN encoder model
        predictor: Link predictor model
        full_data: Full PyG Data object
        split_edge: Dictionary with edge splits
        extractor: Subgraph extractor (PPR or k-hop)
        optimizer: Optimizer
        batch_size: Batch size
        device: Device (cpu/cuda)
        grad_clip: Max gradient norm (None to disable)
    Returns:
        Average loss for the epoch
    """
    encoder.train()
    predictor.train()
    
    # split_edge['train'] is a tensor [2, num_edges]
    source_edge = split_edge['train'][0].to(device)
    target_edge = split_edge['train'][1].to(device)
    
    total_loss = 0
    total_examples = 0
    
    # Mini-batch training
    for perm in DataLoader(range(source_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        
        # Get batch edges
        src_batch = source_edge[perm]
        dst_batch = target_edge[perm]
        
        # Accumulate loss across batch (per-edge subgraph)
        batch_loss = 0
        for i in range(len(src_batch)):
            u = src_batch[i].item()
            v = dst_batch[i].item()
            
            # Extract subgraph for this edge
            subgraph_data, selected_nodes, metadata = extractor.extract_subgraph(u, v)
            subgraph_data = subgraph_data.to(device)
            
            # Get u, v indices in subgraph
            u_sub = metadata['u_subgraph']
            v_sub = metadata['v_subgraph']
            
            if u_sub == -1 or v_sub == -1:
                # Edge nodes not in subgraph (should not happen)
                continue
            
            # Encode subgraph nodes
            h = encoder(subgraph_data.x, subgraph_data.edge_index)
            
            # Positive prediction
            pos_out = predictor(h[u_sub].unsqueeze(0), h[v_sub].unsqueeze(0))
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            
            # Negative sampling from subgraph
            dst_neg = torch.randint(0, subgraph_data.num_nodes, (1,), 
                                   dtype=torch.long, device=device)
            neg_out = predictor(h[u_sub].unsqueeze(0), h[dst_neg])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            
            # Accumulate loss
            batch_loss += (pos_loss + neg_loss)
        
        # Average over batch
        if len(src_batch) > 0:
            batch_loss = batch_loss / len(src_batch)
            batch_loss.backward()
            
            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()), 
                    grad_clip
                )
            
            optimizer.step()
            
            total_loss += batch_loss.item() * len(src_batch)
            total_examples += len(src_batch)
    
    return total_loss / total_examples if total_examples > 0 else 0.0


def evaluate_with_subgraphs(encoder, predictor, full_data, split_edge,
                             extractor, split='val', batch_size=65536, device='cpu'):
    """
    Evaluate model with subgraph extraction.
    
    Args:
        encoder: GNN encoder
        predictor: Link predictor
        full_data: Full graph data
        split_edge: Edge splits
        extractor: Subgraph extractor
        split: 'valid' or 'test'
        batch_size: Batch size
        device: Device
    Returns:
        Dictionary with evaluation metrics
    """
    encoder.eval()
    predictor.eval()

    # source_edge = split_edge[split]['source_node'].to(device)
    # target_edge = split_edge[split]['target_node'].to(device)
    # neg_edge = split_edge[split]['target_node_neg'].to(device)
    
    source_edge = split_edge[split][0].to(device)
    target_edge = split_edge[split][1].to(device)
    # Generate negative samples (500 per positive edge for ranking)
    num_pos = source_edge.size(0)
    neg_edge = torch.randint(0, full_data.num_nodes, (num_pos, 500), device=device)
    
    pos_preds = []
    neg_preds = []
    
    with torch.no_grad():
        # Process in batches
        for perm in DataLoader(range(source_edge.size(0)), batch_size, shuffle=False):
            src_batch = source_edge[perm]
            dst_batch = target_edge[perm]
            neg_batch = neg_edge[perm]
            
            # Evaluate each edge in batch
            for i in range(len(src_batch)):
                u = src_batch[i].item()
                v_pos = dst_batch[i].item()
                v_negs = neg_batch[i]
                
                # Extract subgraph
                subgraph_data, selected_nodes, metadata = extractor.extract_subgraph(u, v_pos)
                subgraph_data = subgraph_data.to(device)
                
                u_sub = metadata['u_subgraph']
                v_sub = metadata['v_subgraph']
                
                if u_sub == -1 or v_sub == -1:
                    # Fallback: use random score
                    pos_preds.append(0.5)
                    neg_preds.append([0.5] * len(v_negs))
                    continue
                
                # Encode subgraph
                h = encoder(subgraph_data.x, subgraph_data.edge_index)
                
                # Positive prediction
                pos_pred = predictor(h[u_sub].unsqueeze(0), h[v_sub].unsqueeze(0))
                pos_preds.append(pos_pred.item())
                
                # Negative predictions
                neg_pred_list = []
                for v_neg in v_negs:
                    v_neg_item = v_neg.item()
                    
                    # Check if negative node is in subgraph
                    if v_neg_item in selected_nodes:
                        v_neg_sub = (selected_nodes == v_neg_item).nonzero(as_tuple=True)[0][0]
                        neg_pred = predictor(h[u_sub].unsqueeze(0), h[v_neg_sub].unsqueeze(0))
                        neg_pred_list.append(neg_pred.item())
                    else:
                        # Negative node not in subgraph, assign low score
                        neg_pred_list.append(0.1)
                
                neg_preds.append(neg_pred_list)
    
    # Convert to tensors
    pos_preds = torch.tensor(pos_preds)
    neg_preds = torch.tensor(neg_preds)
    
    # Compute metrics (same as existing evaluator)
    from ..benchmark.evaluator import compute_metrics
    results = compute_metrics(pos_preds, neg_preds)
    
    return results


def train_with_subgraph_extraction(encoder, predictor, full_data, split_edge, extractor,
                                    epochs=500, batch_size=65536, lr=0.001,
                                    eval_steps=5, device='cpu', verbose=True,
                                    patience=30, min_delta=0.0001, weight_decay=0.0,
                                    lr_scheduler='reduce_on_plateau', grad_clip=1.0):
    """
    Complete training loop with subgraph extraction, early stopping, and LR scheduling.
    
    Args:
        encoder: GNN encoder model
        predictor: Link predictor model
        full_data: Full PyG Data object
        split_edge: Dictionary with edge splits
        extractor: Subgraph extractor (PPR or k-hop)
        epochs: Maximum training epochs (with early stopping)
        batch_size: Batch size
        lr: Initial learning rate
        eval_steps: Evaluate every N epochs
        device: Device (cpu/cuda)
        verbose: Print progress
        patience: Early stopping patience
        min_delta: Minimum improvement threshold
        weight_decay: L2 regularization
        lr_scheduler: 'reduce_on_plateau', 'cosine', or None
        grad_clip: Gradient clipping max norm
    Returns:
        Dictionary with training history
    """
    # Move models to device
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    full_data = full_data.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = None
    if lr_scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
    
    # Training history
    history = {
        'train_loss': [],
        'val_results': [],
        'epoch_times': [],
        'learning_rates': [],
        'best_val_mrr': 0.0,
        'best_epoch': 0,
        'stopped_early': False,
        'stop_reason': None
    }
    
    # Early stopping tracking
    best_val_mrr = 0.0
    epochs_without_improvement = 0
    best_model_state = None
    
    # Training loop
    start_time = time.time()
    iterator = tqdm(range(1, epochs + 1), desc='Training') if verbose else range(1, epochs + 1)
    
    for epoch in iterator:
        epoch_start = time.time()
        
        # Train one epoch with subgraph extraction
        loss = train_epoch_with_subgraphs(
            encoder, predictor, full_data, split_edge, extractor,
            optimizer, batch_size, device, grad_clip=grad_clip
        )
        
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(loss)
        history['epoch_times'].append(epoch_time)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Evaluate on validation set
        if epoch % eval_steps == 0 or epoch == epochs:
            val_results = evaluate_with_subgraphs(
                encoder, predictor, full_data, split_edge, extractor,
                split='val', batch_size=batch_size, device=device
            )
            history['val_results'].append(val_results)
            
            current_val_mrr = val_results['mrr']
            
            # Check for improvement
            if current_val_mrr > best_val_mrr + min_delta:
                best_val_mrr = current_val_mrr
                history['best_val_mrr'] = best_val_mrr
                history['best_epoch'] = epoch
                epochs_without_improvement = 0
                
                # Save best model
                best_model_state = {
                    'encoder': copy.deepcopy(encoder.state_dict()),
                    'predictor': copy.deepcopy(predictor.state_dict())
                }
            else:
                epochs_without_improvement += eval_steps
            
            # Learning rate scheduling
            if scheduler is not None:
                if lr_scheduler == 'reduce_on_plateau':
                    scheduler.step(current_val_mrr)
                else:
                    scheduler.step()
            
            # Progress display
            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss:.4f}',
                    'val_mrr': f'{current_val_mrr:.4f}',
                    'best': f'{best_val_mrr:.4f}',
                    'patience': f'{epochs_without_improvement}/{patience}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # Early stopping
            if epochs_without_improvement >= patience:
                history['stopped_early'] = True
                history['stop_reason'] = f'No improvement for {patience} epochs'
                if verbose:
                    print(f"\n[Early Stop] No improvement for {patience} epochs")
                    print(f"Best MRR: {best_val_mrr:.4f} at epoch {history['best_epoch']}")
                break
    
    total_time = time.time() - start_time
    history['total_time'] = total_time
    
    # Restore best model
    if best_model_state is not None:
        encoder.load_state_dict(best_model_state['encoder'])
        predictor.load_state_dict(best_model_state['predictor'])
        if verbose:
            print(f"[Checkpoint] Restored best model from epoch {history['best_epoch']}")
    
    # Final summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Summary:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Epochs run: {epoch}/{epochs}")
        print(f"  Best val MRR: {history['best_val_mrr']:.4f} (epoch {history['best_epoch']})")
        if history['stopped_early']:
            print(f"  Early stopped: {history['stop_reason']}")
        print(f"{'='*60}")
    
    return history

