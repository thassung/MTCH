"""
Training pipeline for link prediction models.
"""

import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evaluator import evaluate_link_prediction, print_evaluation_results


def train_epoch(encoder, predictor, data, split_edge, optimizer, batch_size, device):
    """
    Train for one epoch following PS2 style.
    
    Args:
        encoder: GNN encoder model
        predictor: Link predictor model
        data: PyG Data object
        split_edge: Dictionary with edge splits
        optimizer: Optimizer
        batch_size: Batch size
        device: Device (cpu/cuda)
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
    for perm in DataLoader(range(source_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        
        # Encode all nodes
        h = encoder(data.x, data.edge_index)
        
        # Get positive edges
        src, dst = source_edge[perm], target_edge[perm]
        
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
        optimizer.step()
        
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    
    return total_loss / total_examples


def train_model(encoder, predictor, data, split_edge, 
                epochs=100, batch_size=65536, lr=0.001, 
                eval_steps=10, device='cpu', verbose=True):
    """
    Complete training loop with evaluation.
    
    Args:
        encoder: GNN encoder model
        predictor: Link predictor model
        data: PyG Data object
        split_edge: Dictionary with edge splits
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        eval_steps: Evaluate every N epochs
        device: Device (cpu/cuda)
        verbose: Print progress
    Returns:
        Dictionary with training history
    """
    # Move models and data to device
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    data = data.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_results': [],
        'epoch_times': [],
        'best_val_mrr': 0.0,
        'best_epoch': 0
    }
    
    # Training loop
    start_time = time.time()
    
    iterator = tqdm(range(1, epochs + 1), desc='Training') if verbose else range(1, epochs + 1)
    
    for epoch in iterator:
        epoch_start = time.time()
        
        # Train
        loss = train_epoch(encoder, predictor, data, split_edge, 
                          optimizer, batch_size, device)
        
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(loss)
        history['epoch_times'].append(epoch_time)
        
        # Evaluate
        if epoch % eval_steps == 0 or epoch == epochs:
            val_results = evaluate_link_prediction(
                encoder, predictor, data, split_edge, 
                split='valid', batch_size=batch_size
            )
            history['val_results'].append(val_results)
            
            # Track best model
            if val_results['mrr'] > history['best_val_mrr']:
                history['best_val_mrr'] = val_results['mrr']
                history['best_epoch'] = epoch
            
            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss:.4f}',
                    'val_mrr': f'{val_results["mrr"]:.4f}',
                    'val_h@10': f'{val_results.get("hits@10", 0):.4f}'
                })
    
    total_time = time.time() - start_time
    history['total_time'] = total_time
    
    if verbose:
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation MRR: {history['best_val_mrr']:.4f} at epoch {history['best_epoch']}")
    
    return history


def benchmark_model(model_name, encoder, predictor, data, split_edge,
                    epochs=100, batch_size=65536, lr=0.001,
                    eval_steps=10, device='cpu'):
    """
    Complete benchmark for a single model.
    
    Args:
        model_name: Name of the model (for display)
        encoder: GNN encoder
        predictor: Link predictor
        data: PyG Data object
        split_edge: Dictionary with edge splits
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        eval_steps: Evaluate every N epochs
        device: Device
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
    
    # Train
    history = train_model(
        encoder, predictor, data, split_edge,
        epochs=epochs, batch_size=batch_size, lr=lr,
        eval_steps=eval_steps, device=device, verbose=True
    )
    
    # Final test evaluation
    print(f"\nEvaluating on test set...")
    test_results = evaluate_link_prediction(
        encoder, predictor, data, split_edge,
        split='test', batch_size=batch_size
    )
    
    print_evaluation_results(test_results, model_name=model_name, split='Test')
    
    return {
        'model_name': model_name,
        'num_params': num_params,
        'train_time': history['total_time'],
        'best_val_mrr': history['best_val_mrr'],
        'test_results': test_results,
        'history': history
    }

