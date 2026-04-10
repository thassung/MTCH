"""
Static PPR-based subgraph benchmark runner.
Grid search over top-k values: [50, 100, 200, 300, 500]
"""

import torch
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from ..utils.models import GCN, SAGE, GAT, LinkPredictor
from ..benchmark.data_prep import prepare_link_prediction_data
from .ppr_extractor import StaticPPRExtractor
from .trainer_batched import train_model_ppr_batched  # OPTIMIZED: batched processing
from .evaluator import evaluate_ppr, print_evaluation_results
from ..utils.ppr_preprocessor import PPRPreprocessor


DEFAULT_CONFIG = {
    'feature_method': 'random',
    'feature_dim': 128,
    'hidden_channels': 256,
    'num_layers': 3,
    'dropout': 0.3,
    'epochs': 500,
    'batch_size': 8192,  # OPTIMIZED: Large batch with batched processing (disjoint union)
    'lr': 0.005,  # Match inner_lr of learnable PPR
    'eval_steps': 5,
    'patience': 30,
    'weight_decay': 1e-5,
    'lr_scheduler': 'reduce_on_plateau',
    'grad_clip': 1.0,
    'alpha': 0.5,  # Fixed alpha for PPR combination
    'ppr_alpha': 0.85,  # Teleport probability for PPR
    'top_k_values': [50, 100, 200, 300, 500]  # Grid search values
}


def load_or_create_ppr_preprocessor(dataset_name, data, ppr_alpha):
    """
    Load PPR preprocessor from disk if available, otherwise create and save it.
    
    Args:
        dataset_name: Name of dataset (e.g., 'FB15K237')
        data: PyG Data object
        ppr_alpha: PPR teleport probability
    
    Returns:
        PPRPreprocessor instance
    """
    preprocessed_dir = f"preprocessed/{dataset_name}"
    preprocessed_path = f"{preprocessed_dir}/ppr_alpha{ppr_alpha}.pt"
    
    # Check if preprocessed file exists
    if os.path.exists(preprocessed_path):
        print(f"  Loading preprocessed PPR data from: {preprocessed_path}")
        try:
            preprocessor = PPRPreprocessor.load(preprocessed_path, data)
            stats = preprocessor.get_stats()
            print(f"    ✓ Loaded {stats['num_cached']} PPR vectors")
            print(f"    Memory: {stats['memory_mb']:.1f} MB")
            return preprocessor
        except Exception as e:
            print(f"    ✗ Failed to load preprocessed data: {e}")
            print(f"    Will create new preprocessor...")
    
    # Create new preprocessor
    print(f"  Creating new PPR preprocessor (alpha={ppr_alpha})...")
    os.makedirs(preprocessed_dir, exist_ok=True)
    log_path = f"{preprocessed_dir}/ppr_alpha{ppr_alpha}_log.txt"
    preprocessor = PPRPreprocessor(data, ppr_alpha=ppr_alpha, log_file=log_path)
    
    # Save for future use
    print(f"  Saving preprocessed data to: {preprocessed_path}")
    preprocessor.save(preprocessed_path)
    
    # Save summary
    stats = preprocessor.get_stats()
    summary_path = f"{preprocessed_dir}/ppr_alpha{ppr_alpha}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"PPR Preprocessing Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"PPR Alpha: {ppr_alpha}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Total nodes: {data.num_nodes:,}\n")
        f.write(f"  Nodes cached: {stats['num_cached']:,}\n")
        f.write(f"  Cache memory: {stats['memory_mb']:.1f} MB\n")
    
    return preprocessor


def create_encoder(encoder_type, in_channels, hidden_channels, num_layers, dropout):
    """Create encoder model."""
    if encoder_type == 'GCN':
        return GCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_type == 'SAGE':
        return SAGE(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_type == 'GAT':
        return GAT(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def save_visualization_data(ppr_extractor, split_edge, split, save_dir, num_samples=10):
    """
    Save subgraph node arrays for visualization.
    
    Args:
        ppr_extractor: StaticPPRExtractor instance
        split_edge: Edge splits
        split: 'valid' or 'test'
        save_dir: Directory to save visualization data
        num_samples: Number of sample edges to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    
    source = split_edge[split]['source_node']
    target = split_edge[split]['target_node']
    
    num_edges = min(num_samples, source.size(0))
    
    visualization_data = []
    for i in range(num_edges):
        u = source[i].item()
        v = target[i].item()
        
        # Extract subgraph
        subgraph_data, selected_nodes, metadata = ppr_extractor.extract_subgraph(u, v)
        
        visualization_data.append({
            'edge': [u, v],
            'selected_nodes': selected_nodes.cpu().numpy().tolist(),
            'num_nodes': len(selected_nodes),
            'metadata': {
                'u_subgraph': metadata['u_subgraph'],
                'v_subgraph': metadata['v_subgraph'],
                'num_edges_subgraph': metadata['num_edges_subgraph'],
                'ppr_score_u': metadata['ppr_score_u'],
                'ppr_score_v': metadata['ppr_score_v']
            }
        })
    
    # Save to JSON
    viz_path = os.path.join(save_dir, 'subgraph_visualization.json')
    with open(viz_path, 'w') as f:
        json.dump(visualization_data, f, indent=2)
    
    print(f"  Saved visualization data: {viz_path}")


def save_results_summary(save_dir, result, dataset_name, encoder_type, top_k, train_time, history):
    """Save results summary to text file."""
    summary_path = os.path.join(save_dir, 'results_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write(f"Static PPR Subgraph Benchmark Results\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Encoder: {encoder_type}\n")
        f.write(f"Top-K: {top_k}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Model Configuration:\n")
        f.write(f"  Hidden channels: {result['config']['hidden_channels']}\n")
        f.write(f"  Num layers: {result['config']['num_layers']}\n")
        f.write(f"  Dropout: {result['config']['dropout']}\n")
        f.write(f"  Total parameters: {result['num_params']:,}\n\n")
        
        f.write(f"Training Configuration:\n")
        f.write(f"  Epochs (max): {result['config']['epochs']}\n")
        f.write(f"  Actual epochs: {len(history['train_loss'])}\n")
        f.write(f"  Early stopped: {history.get('stopped_early', False)}\n")
        f.write(f"  Best epoch: {history.get('best_epoch', 'N/A')}\n")
        f.write(f"  Learning rate: {result['config']['lr']}\n")
        f.write(f"  Batch size: {result['config']['batch_size']}\n")
        f.write(f"  Patience: {result['config']['patience']}\n")
        f.write(f"  Training time: {train_time:.2f}s\n\n")
        
        f.write(f"PPR Configuration:\n")
        f.write(f"  Alpha (fixed): {result['config']['alpha']}\n")
        f.write(f"  PPR alpha: {result['config']['ppr_alpha']}\n")
        f.write(f"  Top-K: {top_k}\n\n")
        
        f.write(f"Test Results:\n")
        test = result['test_results']
        f.write(f"  MRR:      {test['mrr']:.6f}\n")
        f.write(f"  AUC:      {test['auc']:.6f}\n")
        f.write(f"  AP:       {test['ap']:.6f}\n")
        f.write(f"  Hits@1:   {test.get('hits@1', 0):.6f}\n")
        f.write(f"  Hits@3:   {test.get('hits@3', 0):.6f}\n")
        f.write(f"  Hits@10:  {test.get('hits@10', 0):.6f}\n")
        f.write(f"  Hits@50:  {test.get('hits@50', 0):.6f}\n")
        f.write(f"  Hits@100: {test.get('hits@100', 0):.6f}\n\n")
        
        f.write(f"Validation Results (Best Epoch):\n")
        if history['val_results']:
            best_val = history['val_results'][-1]
            f.write(f"  MRR:      {best_val['mrr']:.6f}\n")
            f.write(f"  AUC:      {best_val['auc']:.6f}\n")
            f.write(f"  AP:       {best_val['ap']:.6f}\n")
    
    print(f"  Results saved to: {summary_path}")


def run_single_experiment(dataset_name, dataset_path, encoder_type, top_k, config, device='cuda'):
    """Run a single experiment for one dataset, encoder, and top_k."""
    print(f"\n{'='*80}")
    print(f"Experiment: {dataset_name} | {encoder_type} | top_k={top_k}")
    print(f"{'='*80}")
    
    # Prepare dataset
    print(f"\nLoading dataset: {dataset_path}")
    dataset_dict = prepare_link_prediction_data(
        dataset_path,
        feature_method=config['feature_method'],
        feature_dim=config['feature_dim']
    )
    
    data = dataset_dict['data']
    split_edge = dataset_dict['split_edge']
    node2idx = dataset_dict['node2idx']
    idx2node = dataset_dict['idx2node']
    
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Train edges: {split_edge['train']['source_node'].size(0)}")
    print(f"  Val edges: {split_edge['valid']['source_node'].size(0)}")
    print(f"  Test edges: {split_edge['test']['source_node'].size(0)}")
    
    # Load or create PPR preprocessor
    print(f"\nLoading/creating PPR preprocessor (ppr_alpha={config['ppr_alpha']})...")
    ppr_preprocessor = load_or_create_ppr_preprocessor(dataset_name, data, config['ppr_alpha'])
    
    # Create PPR extractor with preprocessor
    print(f"\nInitializing PPR extractor (top_k={top_k})...")
    ppr_extractor = StaticPPRExtractor(
        data,
        alpha=config['alpha'],
        top_k=top_k,
        ppr_alpha=config['ppr_alpha'],
        preprocessor=ppr_preprocessor
    )
    
    # Create models
    print(f"\nCreating models...")
    encoder = create_encoder(
        encoder_type,
        in_channels=config['feature_dim'],
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    predictor = LinkPredictor(config['hidden_channels'], config['hidden_channels'], 1, config['num_layers'], config['dropout'])
    
    num_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in predictor.parameters())
    print(f"  Total parameters: {num_params:,}")
    
    # Train with BATCHED processing (optimized!)
    print(f"\nTraining (BATCHED mode - processes all edges in one GNN pass)...")
    import time
    train_start = time.time()
    
    history = train_model_ppr_batched(
        encoder, predictor, data, split_edge, ppr_extractor,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        eval_steps=config['eval_steps'],
        device=device,
        verbose=True,
        patience=config['patience'],
        weight_decay=config['weight_decay'],
        lr_scheduler=config['lr_scheduler'],
        grad_clip=config['grad_clip']
    )
    
    train_time = time.time() - train_start
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_results = evaluate_ppr(
        encoder, predictor, data, split_edge, ppr_extractor,
        split='test',
        batch_size=config['batch_size'],
        device=device
    )
    
    print_evaluation_results(test_results, 'test')
    
    # Create results directory
    exp_name = f"{encoder_type}_topk{top_k}"
    save_dir = f"results/benchmark-ppr/{dataset_name}/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save visualization data
    print(f"\nSaving visualization data...")
    save_visualization_data(ppr_extractor, split_edge, 'test', save_dir, num_samples=10)
    
    # Prepare result dictionary
    result = {
        'dataset': dataset_name,
        'encoder': encoder_type,
        'top_k': top_k,
        'num_params': num_params,
        'train_time': train_time,
        'test_results': test_results,
        'history': {
            'best_epoch': history.get('best_epoch', 0),
            'best_val_mrr': history.get('best_val_mrr', 0.0),
            'stopped_early': history.get('stopped_early', False),
            'total_time': history.get('total_time', train_time)
        },
        'config': config,
        'ppr_cache_stats': ppr_extractor.get_cache_stats()
    }
    
    # Save results summary
    save_results_summary(save_dir, result, dataset_name, encoder_type, top_k, train_time, history)
    
    # Save full results as JSON
    results_path = os.path.join(save_dir, 'full_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_result = {
            'dataset': result['dataset'],
            'encoder': result['encoder'],
            'top_k': result['top_k'],
            'num_params': result['num_params'],
            'train_time': result['train_time'],
            'test_results': {k: float(v) for k, v in result['test_results'].items()},
            'history': result['history'],
            'config': result['config'],
            'ppr_cache_stats': result['ppr_cache_stats']
        }
        json.dump(json_result, f, indent=2)
    
    print(f"\n✓ Experiment completed: {exp_name}")
    print(f"  Test MRR: {test_results['mrr']:.4f}")
    print(f"  Results saved to: {save_dir}")
    
    return result


def run_ppr_benchmark(config=None):
    """
    Run full PPR benchmark with grid search.
    
    Args:
        config: Configuration dictionary (uses DEFAULT_CONFIG if None)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Dataset paths
    dataset_paths = {
        'FB15K237': 'data/FB15K237/train.txt',
        'WN18RR': 'data/WN18RR/train.txt',
        'NELL-995': 'data/NELL-995/train.txt'
    }
    
    # Get datasets and encoders from config
    dataset_names = config.get('datasets', ['FB15K237', 'WN18RR', 'NELL-995'])
    datasets = {name: dataset_paths[name] for name in dataset_names if name in dataset_paths}
    encoders = config.get('encoders', ['GCN', 'SAGE', 'GAT'])
    top_k_values = config.get('top_k_values', [50, 100, 200, 300, 500])
    device = config.get('device', 'cuda')
    
    # Run all experiments
    all_results = []
    
    for dataset_name, dataset_path in datasets.items():
        for encoder_type in encoders:
            for top_k in top_k_values:
                try:
                    result = run_single_experiment(
                        dataset_name, dataset_path, encoder_type, top_k, config, device
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"\n✗ Error in {dataset_name}_{encoder_type}_topk{top_k}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Save aggregated results
    agg_results_path = 'results/benchmark-ppr/all_results.json'
    os.makedirs(os.path.dirname(agg_results_path), exist_ok=True)
    
    with open(agg_results_path, 'w') as f:
        json_results = []
        for r in all_results:
            json_results.append({
                'dataset': r['dataset'],
                'encoder': r['encoder'],
                'top_k': r['top_k'],
                'test_mrr': float(r['test_results']['mrr']),
                'test_auc': float(r['test_results']['auc']),
                'test_ap': float(r['test_results']['ap']),
                'train_time': r['train_time']
            })
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"PPR Benchmark Complete!")
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Results saved to: results/benchmark-ppr/")
    print(f"{'='*80}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Static PPR Subgraph Benchmark')
    parser.add_argument('--datasets', nargs='+', default=['FB15K237', 'WN18RR', 'NELL-995'],
                       help='Datasets to run (default: all)')
    parser.add_argument('--encoders', nargs='+', default=['GCN', 'SAGE', 'GAT'],
                       help='Encoders to run (default: all)')
    parser.add_argument('--top_k', type=int, nargs='+', default=[50, 100, 200, 300, 500],
                       help='Top-K values for grid search')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Max epochs')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8192,
                       help='Batch size: number of edges per batch (BATCHED mode processes all in ONE GNN pass! 8192=default, 16384=faster, 65536=max speed)')
    
    args = parser.parse_args()
    
    # Create config from args
    config = DEFAULT_CONFIG.copy()
    config.update({
        'datasets': args.datasets,
        'encoders': args.encoders,
        'top_k_values': args.top_k,
        'device': args.device,
        'epochs': args.epochs,
        'patience': args.patience,
        'lr': args.lr,
        'batch_size': args.batch_size
    })
    
    # Run benchmark
    run_ppr_benchmark(config)


if __name__ == '__main__':
    main()

