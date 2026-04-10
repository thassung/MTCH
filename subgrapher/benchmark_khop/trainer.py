"""
Training pipeline for k-hop subgraph link prediction.
Modified from benchmark/trainer.py to extract subgraphs before encoding.
"""

import torch
import torch.nn as nn
import time
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evaluator import evaluate_khop


def train_epoch_khop(encoder, predictor, data, split_edge, khop_extractor,
                     optimizer, batch_size, device, grad_clip=None, verbose=False):
    """
    Train for one epoch with k-hop subgraph extraction.
    Uses per-edge subgraphs to prevent information leakage, with preprocessing for speed.
    
    Args:
        encoder: GNN encoder model
        predictor: Link predictor model
        data: PyG Data object (full graph)
        split_edge: Dictionary with edge splits
        khop_extractor: StaticKHopExtractor instance
        optimizer: Optimizer
        batch_size: Batch size
        device: Device (cpu/cuda)
        grad_clip: Max gradient norm (None to disable)
        verbose: Show batch-level progress bar
    Returns:
        Average loss for the epoch
    """
    encoder.train()
    predictor.train()
    
    source_edge = split_edge['train']['source_node'].to(device)
    target_edge = split_edge['train']['target_node'].to(device)
    
    total_loss = 0
    total_examples = 0
    
    # Mini-batch training
    dataloader = DataLoader(range(source_edge.size(0)), batch_size, shuffle=True)
    if verbose:
        dataloader = tqdm(dataloader, desc='  Training batches', leave=False)
    
    for perm in dataloader:
        optimizer.zero_grad()
        batch_loss = 0
        batch_examples = 0
        
        # Process each edge with its OWN isolated subgraph
        for i in perm:
            u = source_edge[i].item()
            v = target_edge[i].item()
            
            # Get precomputed k-hop neighborhoods (fast O(1) lookup!)
            nodes_u = khop_extractor.preprocessor.get_khop_nodes(u)
            nodes_v = khop_extractor.preprocessor.get_khop_nodes(v)
            
            # Create subgraph for THIS EDGE ONLY (no information leakage)
            selected_nodes = torch.unique(torch.cat([nodes_u, nodes_v]))
            
            # Extract subgraph edges
            from torch_geometric.utils import subgraph
            edge_index_sub, _ = subgraph(
                selected_nodes, data.edge_index,
                relabel_nodes=True, num_nodes=data.num_nodes
            )
            
            # Create node mapping
            node_mapping = {node.item(): idx for idx, node in enumerate(selected_nodes)}
            u_sub = node_mapping.get(u, -1)
            v_sub = node_mapping.get(v, -1)
            
            if u_sub == -1 or v_sub == -1:
                continue
            
            # Encode THIS edge's subgraph (isolated!)
            x_sub = data.x[selected_nodes].to(device)
            edge_index_sub = edge_index_sub.to(device)
            h = encoder(x_sub, edge_index_sub)
            
            # Positive prediction
            pos_out = predictor(h[u_sub].unsqueeze(0), h[v_sub].unsqueeze(0))
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            
            # Negative sampling within THIS subgraph
            neg_idx = torch.randint(0, len(selected_nodes), (1,), device=device)
            neg_out = predictor(h[u_sub].unsqueeze(0), h[neg_idx])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            
            # Edge loss
            edge_loss = pos_loss + neg_loss
            batch_loss += edge_loss
            batch_examples += 1
        
        if batch_examples > 0:
            # Average and backprop
            batch_loss = batch_loss / batch_examples
            batch_loss.backward()
            
            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()), 
                    grad_clip
                )
            
            optimizer.step()
            
            total_loss += batch_loss.item() * batch_examples
            total_examples += batch_examples
    
    return total_loss / total_examples if total_examples > 0 else 0.0


def train_model_khop(encoder, predictor, data, split_edge, khop_extractor,
                     epochs=500, batch_size=65536, lr=0.005, 
                     eval_steps=5, device='cpu', verbose=True,
                     patience=30, min_delta=0.0001, weight_decay=1e-5,
                     lr_scheduler='reduce_on_plateau', grad_clip=1.0):
    """
    Complete training loop with early stopping for k-hop subgraph model.
    
    Args:
        encoder: GNN encoder model
        predictor: Link predictor model
        data: PyG Data object
        split_edge: Dictionary with edge splits
        khop_extractor: StaticKHopExtractor instance
        epochs: Maximum training epochs
        batch_size: Batch size
        lr: Initial learning rate
        eval_steps: Evaluate every N epochs
        device: Device (cpu/cuda)
        verbose: Print progress
        patience: Early stopping patience
        min_delta: Minimum improvement to count as progress
        weight_decay: L2 regularization strength
        lr_scheduler: 'reduce_on_plateau', 'cosine', or None
        grad_clip: Gradient clipping max norm
    Returns:
        Dictionary with training history
    """
    # Move models to device (data stays on CPU for subgraph extraction)
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    
    # Optimizer with weight decay
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
        
        # Train one epoch (with batch progress bar for first 3 epochs)
        show_batch_progress = (epoch <= 3) if verbose else False
        loss = train_epoch_khop(encoder, predictor, data, split_edge, khop_extractor,
                               optimizer, batch_size, device, grad_clip=grad_clip,
                               verbose=show_batch_progress)
        
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(loss)
        history['epoch_times'].append(epoch_time)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Evaluate on validation set
        if epoch % eval_steps == 0 or epoch == epochs:
            val_results = evaluate_khop(
                encoder, predictor, data, split_edge, khop_extractor,
                split='valid', batch_size=batch_size, device=device
            )
            history['val_results'].append(val_results)
            
            current_val_mrr = val_results['mrr']
            
            # Check for improvement
            if current_val_mrr > best_val_mrr + min_delta:
                best_val_mrr = current_val_mrr
                history['best_val_mrr'] = best_val_mrr
                history['best_epoch'] = epoch
                epochs_without_improvement = 0
                
                # Save best model checkpoint
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
            
            # Early stopping check
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
            print(f"Restored best model from epoch {history['best_epoch']}")
    
    return history

