"""
Training pipeline for link prediction models.
Includes early stopping, learning rate scheduling, gradient clipping, and checkpointing.
"""

import torch
import torch.nn as nn
import time
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evaluator import evaluate_link_prediction, print_evaluation_results


def train_epoch(encoder, predictor, data, split_edge, optimizer, batch_size, device, grad_clip=None):
    """
    Train for one epoch with gradient clipping.
    
    Args:
        encoder: GNN encoder model
        predictor: Link predictor model
        data: PyG Data object
        split_edge: Dictionary with edge splits
        optimizer: Optimizer
        batch_size: Batch size
        device: Device (cpu/cuda)
        grad_clip: Max gradient norm (None to disable)
    Returns:
        Average loss for the epoch
    """
    encoder.train()
    predictor.train()
    
    source_edge = split_edge['train']['source_node']
    target_edge = split_edge['train']['target_node']
    
    total_loss = 0
    total_examples = 0
    
    # Mini-batch training
    for perm in DataLoader(torch.arange(source_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        
        # Encode all nodes (data already on device via train_model)
        h = encoder(data.x, data.edge_index)
        
        # Get positive edges (move batch slice to device)
        src = source_edge[perm].to(device)
        dst = target_edge[perm].to(device)
        
        # Positive predictions
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        
        # Random negative sampling
        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                               dtype=torch.long, device=device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        
        # Total loss
        loss = pos_loss + neg_loss
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 
                grad_clip
            )
        
        optimizer.step()
        
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    
    return total_loss / total_examples


def train_model(encoder, predictor, data, split_edge, 
                epochs=500, batch_size=65536, lr=0.001, 
                eval_steps=5, device='cpu', verbose=True,
                patience=20, min_delta=0.0001, weight_decay=0.0,
                lr_scheduler='reduce_on_plateau', grad_clip=1.0):
    """
    Complete training loop with early stopping, LR scheduling, and checkpointing.
    
    Args:
        encoder: GNN encoder model
        predictor: Link predictor model
        data: PyG Data object
        split_edge: Dictionary with edge splits
        epochs: Maximum training epochs (with early stopping, may stop earlier)
        batch_size: Batch size
        lr: Initial learning rate
        eval_steps: Evaluate every N epochs
        device: Device (cpu/cuda)
        verbose: Print progress
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum improvement to count as progress
        weight_decay: L2 regularization strength
        lr_scheduler: 'reduce_on_plateau', 'cosine', or None
        grad_clip: Gradient clipping max norm (None to disable)
    Returns:
        Dictionary with training history
    """
    # Move models to device; keep only graph tensors on GPU
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    
    # Optimizer with weight decay (L2 regularization)
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
        
        # Train one epoch
        loss = train_epoch(encoder, predictor, data, split_edge, 
                          optimizer, batch_size, device, grad_clip=grad_clip)
        
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(loss)
        history['epoch_times'].append(epoch_time)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Evaluate on validation set
        if epoch % eval_steps == 0 or epoch == epochs:
            val_results = evaluate_link_prediction(
                encoder, predictor, data, split_edge, 
                split='valid', batch_size=batch_size
            )
            history['val_results'].append(val_results)
            
            current_val_mrr = val_results['mrr']
            
            # Check for improvement
            if current_val_mrr > best_val_mrr + min_delta:
                # Significant improvement
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
                # No improvement
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
            print(f"[Checkpoint] Restored best model from epoch {history['best_epoch']}")
    
    # Move graph data back to CPU to free GPU memory
    data.x = data.x.cpu()
    data.edge_index = data.edge_index.cpu()
    torch.cuda.empty_cache()
    
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


def benchmark_model(model_name, encoder, predictor, data, split_edge,
                    epochs=500, batch_size=65536, lr=0.001,
                    eval_steps=5, device='cpu', patience=20, 
                    weight_decay=1e-5, lr_scheduler='reduce_on_plateau',
                    grad_clip=1.0):
    """
    Complete benchmark for a single model with early stopping.
    
    Args:
        model_name: Name of the model (for display)
        encoder: GNN encoder
        predictor: Link predictor
        data: PyG Data object
        split_edge: Dictionary with edge splits
        epochs: Maximum training epochs (early stopping may end sooner)
        batch_size: Batch size
        lr: Initial learning rate
        eval_steps: Evaluate every N epochs
        device: Device
        patience: Early stopping patience
        weight_decay: L2 regularization
        lr_scheduler: Learning rate scheduler type
        grad_clip: Gradient clipping max norm
    Returns:
        Dictionary with complete benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*60}")
    
    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    num_params += sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Early stopping: enabled (patience={patience})")
    print(f"Learning rate: {lr} (scheduler={lr_scheduler})")
    print(f"Regularization: weight_decay={weight_decay}, grad_clip={grad_clip}")
    
    # Train with early stopping
    history = train_model(
        encoder, predictor, data, split_edge,
        epochs=epochs, batch_size=batch_size, lr=lr,
        eval_steps=eval_steps, device=device, verbose=True,
        patience=patience, weight_decay=weight_decay,
        lr_scheduler=lr_scheduler, grad_clip=grad_clip
    )
    
    # Final test evaluation (uses best checkpoint)
    print(f"\nEvaluating on test set (best checkpoint)...")
    test_results = evaluate_link_prediction(
        encoder, predictor, data, split_edge,
        split='test', batch_size=batch_size
    )
    
    print_evaluation_results(test_results, model_name=model_name, split='Test')
    
    # Release GPU memory
    encoder.cpu()
    predictor.cpu()
    torch.cuda.empty_cache()
    
    return {
        'model_name': model_name,
        'num_params': num_params,
        'train_time': history['total_time'],
        'best_val_mrr': history['best_val_mrr'],
        'best_epoch': history['best_epoch'],
        'stopped_early': history['stopped_early'],
        'test_results': test_results,
        'history': history
    }

