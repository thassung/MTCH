"""
Static PPR Subgraph Baseline Benchmark

Runs grid search over top-k values with fixed alpha=0.5.
Tests all combinations of:
- 3 datasets (FB15K237, WN18RR, NELL-995)
- 3 encoders (GCN, SAGE, GAT)
- 5 top_k values (50, 100, 200, 300, 500)

Total: 45 experiments
"""

import os
import sys
import json
import csv
import time
import datetime
import argparse
import traceback

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from subgrapher.benchmark_ppr.ppr_extractor import StaticPPRExtractor
from subgrapher.utils.models import GCN, SAGE, GAT, LinkPredictor
from subgrapher.utils.subgraph_trainer import train_with_subgraph_extraction, evaluate_with_subgraphs
from subgrapher.utils.loader import load_txt_to_pyg
from subgrapher.benchmark.data_prep import add_node_features, split_edges


# Default configuration (updated with user's request)
DEFAULT_CONFIG = {
    'feature_method': 'random',
    'feature_dim': 128,
    'hidden_channels': 256,
    'num_layers': 3,
    'dropout': 0.3,
    'epochs': 500,              # Updated from 3000
    'batch_size': 65536,
    'lr': 0.005,
    'eval_steps': 5,
    'patience': 30,             
    'weight_decay': 1e-5,
    'lr_scheduler': 'reduce_on_plateau',
    'grad_clip': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # PPR-specific
    'alpha': 0.5,               # Fixed alpha for baseline
    'ppr_alpha': 0.85,          # PPR damping factor
    'top_k_values': [50, 100, 200, 300, 500]  # Grid search
}


def setup_directories():
    """Create output directory structure."""
    base_dir = 'results/benchmark-ppr'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'visualizations'), exist_ok=True)
    return base_dir


def prepare_dataset(dataset_path, config):
    """Load and prepare dataset."""
    # Load data
    data, node2idx, idx2node = load_txt_to_pyg(dataset_path)
    
    # Add features
    data = add_node_features(
        data, 
        method=config['feature_method'], 
        feature_dim=config['feature_dim']
    )
    
    # Split edges
    split_edge = split_edges(
        data.edge_index, 
        data.num_nodes, 
        val_ratio=0.1, 
        test_ratio=0.1
    )
    
    # Use only training edges for graph structure
    data.edge_index = split_edge['train']
    
    return data, split_edge, (node2idx, idx2node)


def create_models(encoder_type, in_channels, hidden_channels, num_layers, dropout):
    """Create encoder and predictor models."""
    if encoder_type == 'GCN':
        encoder = GCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_type == 'SAGE':
        encoder = SAGE(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_type == 'GAT':
        encoder = GAT(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    predictor = LinkPredictor(hidden_channels, hidden_channels, 1, num_layers=3, dropout=dropout)
    
    return encoder, predictor


def run_single_experiment(dataset_name, dataset_path, encoder_type, top_k, config):
    """
    Run a single experiment: one dataset + one encoder + one top_k value.
    
    Returns:
        Dictionary with results or None if failed
    """
    exp_name = f"{dataset_name}_{encoder_type}_topk{top_k}"
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}")
    
    try:
        # Prepare dataset
        print("Loading dataset...")
        data, split_edge, (node2idx, idx2node) = prepare_dataset(dataset_path, config)
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Train edges: {split_edge['train'].size(1)}")
        print(f"  Val edges: {split_edge['val'].size(1)}")
        print(f"  Test edges: {split_edge['test'].size(1)}")
        
        # Create PPR extractor
        print(f"Creating PPR extractor (alpha={config['alpha']}, top_k={top_k})...")
        extractor = StaticPPRExtractor(
            data, 
            alpha=config['alpha'], 
            ppr_alpha=config['ppr_alpha'], 
            top_k=top_k,
            use_cache=True
        )
        
        # Create models
        print(f"Creating {encoder_type} encoder...")
        encoder, predictor = create_models(
            encoder_type, 
            data.x.size(1), 
            config['hidden_channels'], 
            config['num_layers'], 
            config['dropout']
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        num_params += sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        print(f"Total parameters: {num_params:,}")
        
        # Train
        print("\nTraining...")
        start_time = time.time()
        history = train_with_subgraph_extraction(
            encoder, predictor, data, split_edge, extractor,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['lr'],
            eval_steps=config['eval_steps'],
            device=config['device'],
            verbose=True,
            patience=config['patience'],
            weight_decay=config['weight_decay'],
            lr_scheduler=config['lr_scheduler'],
            grad_clip=config['grad_clip']
        )
        train_time = time.time() - start_time
        
        # Test
        print("\nEvaluating on test set...")
        test_results = evaluate_with_subgraphs(
            encoder, predictor, data, split_edge, extractor,
            split='test', 
            batch_size=config['batch_size'],
            device=config['device']
        )
        
        print(f"\nTest Results:")
        print(f"  MRR: {test_results['mrr']:.4f}")
        print(f"  AUC: {test_results['auc']:.4f}")
        print(f"  AP: {test_results['ap']:.4f}")
        print(f"  Hits@10: {test_results.get('hits@10', 0):.4f}")
        
        # Collect results
        result = {
            'dataset': dataset_name,
            'encoder': encoder_type,
            'top_k': top_k,
            'alpha': config['alpha'],
            'num_params': num_params,
            'train_time': train_time,
            'best_val_mrr': history['best_val_mrr'],
            'best_epoch': history['best_epoch'],
            'stopped_early': history['stopped_early'],
            'test_results': test_results,
            'history': history
        }
        
        # Clear cache
        extractor.clear_cache()
        
        print(f"\n✓ {exp_name} completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"✗ Error in {exp_name}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None


def save_experiment_results(result, base_dir):
    """Save individual experiment results."""
    if result is None:
        return
    
    dataset_name = result['dataset']
    encoder_type = result['encoder']
    top_k = result['top_k']
    
    # Create directory
    exp_dir = os.path.join(base_dir, dataset_name, f"{encoder_type}_topk{top_k}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'encoder': encoder_type,
            'top_k': top_k,
            'alpha': result['alpha'],
            'num_params': result['num_params'],
            'training_info': {
                'best_epoch': result['best_epoch'],
                'stopped_early': result['stopped_early'],
                'total_time_seconds': result['train_time']
            }
        }, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(exp_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(result['test_results'], f, indent=2)
    
    # Save summary
    summary_path = os.path.join(exp_dir, 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Experiment: {encoder_type} on {dataset_name} (top_k={top_k})\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Encoder: {encoder_type}\n")
        f.write(f"  Parameters: {result['num_params']:,}\n")
        f.write(f"  Top-k: {top_k}\n")
        f.write(f"  Alpha: {result['alpha']}\n\n")
        f.write(f"Training:\n")
        f.write(f"  Time: {result['train_time']:.2f}s\n")
        f.write(f"  Best Epoch: {result['best_epoch']}\n")
        f.write(f"  Early Stopped: {'Yes' if result['stopped_early'] else 'No'}\n")
        f.write(f"  Best Val MRR: {result['best_val_mrr']:.4f}\n\n")
        f.write(f"Test Results:\n")
        f.write(f"  MRR: {result['test_results']['mrr']:.4f}\n")
        f.write(f"  AUC: {result['test_results']['auc']:.4f}\n")
        f.write(f"  AP: {result['test_results']['ap']:.4f}\n")
        for key in sorted(result['test_results'].keys()):
            if key.startswith('hits@'):
                f.write(f"  {key}: {result['test_results'][key]:.4f}\n")
    
    print(f"  ✓ Results saved to {exp_dir}")


def create_comparison_table(all_results, base_dir):
    """Create CSV comparison table."""
    csv_path = os.path.join(base_dir, 'comparison_table.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Encoder', 'Top_K', 'Alpha', 'Params', 'Train_Time_s', 
                        'MRR', 'AUC', 'AP', 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@50'])
        
        for result in all_results:
            if result is None:
                continue
            test = result['test_results']
            writer.writerow([
                result['dataset'],
                result['encoder'],
                result['top_k'],
                result['alpha'],
                result['num_params'],
                f"{result['train_time']:.2f}",
                f"{test['mrr']:.4f}",
                f"{test['auc']:.4f}",
                f"{test['ap']:.4f}",
                f"{test.get('hits@1', 0):.4f}",
                f"{test.get('hits@3', 0):.4f}",
                f"{test.get('hits@10', 0):.4f}",
                f"{test.get('hits@50', 0):.4f}"
            ])
    
    print(f"✓ Comparison table saved to {csv_path}")


def create_visualizations(all_results, base_dir):
    """Create comparison visualizations."""
    viz_dir = os.path.join(base_dir, 'visualizations')
    
    # Filter successful results
    results = [r for r in all_results if r is not None]
    if not results:
        print("Warning: No results for visualizations")
        return
    
    # Group by dataset and encoder
    datasets = sorted(set(r['dataset'] for r in results))
    encoders = sorted(set(r['encoder'] for r in results))
    top_k_values = sorted(set(r['top_k'] for r in results))
    
    # 1. MRR vs Top-K for each dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        for encoder in encoders:
            encoder_results = [r for r in results if r['dataset'] == dataset and r['encoder'] == encoder]
            encoder_results = sorted(encoder_results, key=lambda x: x['top_k'])
            
            if encoder_results:
                ks = [r['top_k'] for r in encoder_results]
                mrrs = [r['test_results']['mrr'] for r in encoder_results]
                ax.plot(ks, mrrs, marker='o', label=encoder, linewidth=2)
        
        ax.set_xlabel('Top-K')
        ax.set_ylabel('MRR')
        ax.set_title(f'{dataset}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('MRR vs Top-K Selection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mrr_vs_topk.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Best Top-K per Dataset/Encoder
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, encoder in enumerate(encoders):
        best_mrrs = []
        for dataset in datasets:
            dataset_results = [r for r in results if r['dataset'] == dataset and r['encoder'] == encoder]
            if dataset_results:
                best = max(dataset_results, key=lambda x: x['test_results']['mrr'])
                best_mrrs.append(best['test_results']['mrr'])
            else:
                best_mrrs.append(0)
        
        ax.bar(x + i * width, best_mrrs, width, label=encoder)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Best MRR')
    ax.set_title('Best Performance Across Top-K Values', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'best_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to {viz_dir}/")


def run_full_benchmark(config=None):
    """
    Run full PPR baseline benchmark.
    
    Args:
        config: Configuration dict containing datasets, encoders, top_k_values, etc.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Dataset paths directory
    dataset_paths = {
        'FB15K237': 'data/FB15K237/train.txt',
        'WN18RR': 'data/WN18RR/train.txt',
        'NELL-995': 'data/NELL-995/train.txt'
    }
    
    # Extract from config
    dataset_names = config.get('datasets', ['FB15K237', 'WN18RR', 'NELL-995'])
    datasets = {name: dataset_paths[name] for name in dataset_names if name in dataset_paths}
    encoders = config.get('encoders', ['GCN', 'SAGE', 'GAT'])
    top_k_values = config.get('top_k_values', [50, 100, 200, 300, 500])
    
    print("="*80)
    print("STATIC PPR SUBGRAPH BASELINE BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Datasets: {list(datasets.keys())}")
    print(f"  Encoders: {encoders}")
    print(f"  Top-K values: {top_k_values}")
    print(f"  Fixed Alpha: {config['alpha']}")
    print(f"  Total experiments: {len(datasets) * len(encoders) * len(top_k_values)}")
    print(f"  Device: {config['device']}")
    print("\n")
    
    # Setup directories
    base_dir = setup_directories()
    print(f"✓ Output directory: {base_dir}/")
    
    # Run experiments
    all_results = []
    start_time = time.time()
    
    for dataset_name, dataset_path in datasets.items():
        for encoder_type in encoders:
            for top_k in top_k_values:
                result = run_single_experiment(
                    dataset_name, dataset_path, encoder_type, top_k, config
                )
                all_results.append(result)
                
                # Save immediately
                if result is not None:
                    save_experiment_results(result, base_dir)
    
    total_time = time.time() - start_time
    
    # Generate reports
    print(f"\n{'='*80}")
    print("GENERATING REPORTS")
    print(f"{'='*80}\n")
    
    # Save full results
    json_path = os.path.join(base_dir, 'full_results.json')
    with open(json_path, 'w') as f:
        serializable_results = []
        for r in all_results:
            if r is not None:
                r_copy = r.copy()
                if 'history' in r_copy:
                    del r_copy['history']
                serializable_results.append(r_copy)
        json.dump(serializable_results, f, indent=2)
    print(f"✓ Full results saved to {json_path}")
    
    # Create comparison table
    create_comparison_table(all_results, base_dir)
    
    # Create visualizations
    create_visualizations(all_results, base_dir)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time:.1f}s)")
    print(f"Results saved to: {base_dir}/")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run static PPR subgraph baseline benchmark'
    )
    
    # Dataset and model selection
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['FB15K237', 'WN18RR', 'NELL-995'],
                       help='Datasets to benchmark')
    parser.add_argument('--encoders', type=str, nargs='+',
                       default=['GCN', 'SAGE', 'GAT'],
                       choices=['GCN', 'SAGE', 'GAT'],
                       help='Encoder types to benchmark')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top_k_values', type=int, nargs='+', default=[50, 100, 200, 300, 500])
    
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['datasets'] = args.datasets
    config['encoders'] = args.encoders
    config['epochs'] = args.epochs
    config['patience'] = args.patience
    config['device'] = args.device
    config['top_k_values'] = args.top_k_values
    
    # Run benchmark
    run_full_benchmark(config=config)


if __name__ == '__main__':
    main()

