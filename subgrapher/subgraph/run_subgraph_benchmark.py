"""
Benchmark runner for learnable PPR subgraph models.
Compares different encoders and configurations on link prediction datasets.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import argparse
import time
from pathlib import Path

from subgrapher.subgraph import SubgraphLinkPrediction
from subgrapher.utils.loader import load_txt_to_pyg
from subgrapher.benchmark.data_prep import add_node_features, split_edges
from subgrapher.subgraph.results_manager import SubgraphResultsManager
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def _save_results_summary(exp_dir, result, dataset_name, encoder_type, train_time, history):
    """Save a text summary of results (similar to benchmark format)."""
    summary_path = os.path.join(exp_dir, 'results_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write(f"Model: LearnablePPR_{encoder_type}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Parameters: {result['num_params']:,}\n")
        f.write(f"\n")
        f.write(f"Training:\n")
        f.write(f"  Time: {train_time:.2f}s\n")
        f.write(f"  Best Epoch: {result['best_epoch']}\n")
        f.write(f"  Best Val Loss: {result['best_val_loss']:.4f}\n")
        f.write(f"\n")
        
        # Test Results (if available)
        if 'test_metrics' in result and result['test_metrics']:
            metrics = result['test_metrics']
            f.write(f"Test Results:\n")
            if 'mrr' in metrics:
                f.write(f"  MRR:     {metrics['mrr']:.4f}\n")
            if 'auc' in metrics:
                f.write(f"  AUC:     {metrics['auc']:.4f}\n")
            if 'ap' in metrics:
                f.write(f"  AP:      {metrics['ap']:.4f}\n")
            # hits@k in order: 1, 10, 100, 3, 50
            for k in [1, 10, 100, 3, 50]:
                key = f'hits@{k}'
                if key in metrics:
                    f.write(f"  {key}:  {metrics[key]:.4f}\n")
            f.write(f"\n")
        
        f.write(f"Learned Selector Parameters:\n")
        f.write(f"  Alpha: {result['selector_params']['alpha']:.6f}\n")
        f.write(f"  Threshold: {result['selector_params']['threshold']:.6f}\n")
        if 'threshold_percentile' in result['selector_params'] and result['selector_params']['threshold_percentile'] is not None:
            f.write(f"  Threshold Percentile: {result['selector_params']['threshold_percentile']:.6f}\n")
        f.write(f"\n")
        f.write(f"Subgraph Statistics:\n")
        f.write(f"  Average Size: {result['subgraph_stats']['avg_size']:.1f} nodes\n")
        f.write(f"  Min Size: {result['subgraph_stats']['min_size']} nodes\n")
        f.write(f"  Max Size: {result['subgraph_stats']['max_size']} nodes\n")


def prepare_dataset(dataset_path, feature_method='random', feature_dim=128, 
                   val_ratio=0.1, test_ratio=0.1):
    """Load and prepare dataset for link prediction."""
    print(f"\nLoading {dataset_path}...")
    data, node2idx, idx2node = load_txt_to_pyg(dataset_path)
    
    # Add features
    data = add_node_features(data, method=feature_method, feature_dim=feature_dim)
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"  Features: {data.x.size(1)}D ({feature_method})")
    
    # Split edges
    edge_splits = split_edges(data.edge_index, data.num_nodes, 
                             val_ratio=val_ratio, test_ratio=test_ratio)
    
    # Convert to edge lists (edge_splits returns tensors [2, num_edges])
    train_edges = list(zip(
        edge_splits['train'][0].tolist(),  # source nodes
        edge_splits['train'][1].tolist()   # target nodes
    ))
    val_edges = list(zip(
        edge_splits['val'][0].tolist(),
        edge_splits['val'][1].tolist()
    ))
    test_edges = list(zip(
        edge_splits['test'][0].tolist(),
        edge_splits['test'][1].tolist()
    ))
    
    # Use training edges for graph structure
    data.edge_index = edge_splits['train']
    
    print(f"  Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)}")
    
    return data, train_edges, val_edges, test_edges, (node2idx, idx2node)


def run_single_experiment(dataset_name, dataset_path, encoder_type, config):
    """Run single subgraph experiment."""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {dataset_name} with {encoder_type}")
    print("="*80)
    
    # Prepare data
    data, train_edges, val_edges, test_edges, _ = prepare_dataset(
        dataset_path,
        feature_method=config['feature_method'],
        feature_dim=config['feature_dim']
    )
    
    # Create model with results saving enabled
    model = SubgraphLinkPrediction(
        encoder_type=encoder_type,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        ppr_alpha=config['ppr_alpha'],
        adaptive_threshold=config['adaptive_threshold'],
        device=config['device'],
        save_results=True,
        dataset_name=dataset_name
    )
    
    print(f"\nModel: LearnablePPR with {encoder_type}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {model.device}")
    
    # Train
    start_time = time.time()
    history = model.fit(
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        epochs=config['epochs'],
        inner_steps=config['inner_steps'],
        inner_lr=config['inner_lr'],
        outer_lr=config['outer_lr'],
        eval_steps=config['eval_steps'],
        patience=config['patience'],
        use_full_graph=False,
        meta_learning_order=config.get('meta_learning_order', 'first'),
        verbose=True
    )
    train_time = time.time() - start_time
    
    # Test - Get positive predictions
    print(f"\nEvaluating on test set ({len(test_edges)} edges)...")
    pos_predictions, pred_metadata = model.predict(test_edges, data, batch_size=64)
    
    # Generate negative samples and get predictions
    print("Generating negative samples for test evaluation...")
    num_neg_per_pos = 500  # Standard for ranking metrics
    neg_predictions_all = []
    
    import random
    for u, v in test_edges[:min(1000, len(test_edges))]:  # Sample subset for speed
        neg_edges = []
        for _ in range(num_neg_per_pos):
            neg_v = random.randint(0, data.num_nodes - 1)
            neg_edges.append((u, neg_v))
        
        neg_preds, _ = model.predict(neg_edges, data, batch_size=64)
        neg_predictions_all.append(neg_preds)
    
    # Compute test metrics (MRR, AUC, AP, hits@k)
    print("Computing test metrics...")
    test_metrics = {}
    
    # For AUC and AP, we need binary labels and predictions
    if len(pos_predictions) > 0:
        pos_preds_array = np.array([p.item() if hasattr(p, 'item') else p for p in pos_predictions])
        
        # Compute MRR and Hits@K using ranking
        if neg_predictions_all:
            mrr_list = []
            hits_at_k = {1: [], 3: [], 10: [], 50: [], 100: []}
            
            for i, neg_preds in enumerate(neg_predictions_all):
                if i >= len(pos_preds_array):
                    break
                pos_pred = pos_preds_array[i]
                neg_preds_array = np.array([p.item() if hasattr(p, 'item') else p for p in neg_preds])
                
                # Rank: how many negatives score higher than positive
                rank = (neg_preds_array >= pos_pred).sum() + 1
                
                mrr_list.append(1.0 / rank)
                for k in hits_at_k.keys():
                    hits_at_k[k].append(1.0 if rank <= k else 0.0)
            
            test_metrics['mrr'] = np.mean(mrr_list)
            for k, v in hits_at_k.items():
                test_metrics[f'hits@{k}'] = np.mean(v)
        
        # Compute AUC and AP (need negative samples)
        if neg_predictions_all:
            # Sample a subset for AUC/AP computation
            sample_size = min(len(pos_preds_array), len(neg_predictions_all), 1000)
            pos_sample = pos_preds_array[:sample_size]
            neg_sample = np.array([neg_predictions_all[i][0].item() if hasattr(neg_predictions_all[i][0], 'item') 
                                   else neg_predictions_all[i][0] for i in range(sample_size)])
            
            y_true = np.concatenate([np.ones(len(pos_sample)), np.zeros(len(neg_sample))])
            y_pred = np.concatenate([pos_sample, neg_sample])
            
            test_metrics['auc'] = roc_auc_score(y_true, y_pred)
            test_metrics['ap'] = average_precision_score(y_true, y_pred)
        
        print(f"  MRR: {test_metrics.get('mrr', 0):.4f}")
        print(f"  AUC: {test_metrics.get('auc', 0):.4f}")
        print(f"  AP: {test_metrics.get('ap', 0):.4f}")
        print(f"  Hits@1: {test_metrics.get('hits@1', 0):.4f}")
        print(f"  Hits@10: {test_metrics.get('hits@10', 0):.4f}")
    
    # Visualize sample subgraphs
    if config.get('save_visualizations', True) and len(test_edges) > 0:
        print("\nGenerating subgraph visualizations...")
        for i in range(min(3, len(test_edges))):
            u, v = test_edges[i]
            try:
                fig = model.visualize_subgraph(data, u, v)
                if model.results_manager and model.exp_dir:
                    model.results_manager.save_visualized_subgraph(model.exp_dir, fig, i)
            except Exception as e:
                print(f"  Warning: Could not visualize subgraph for edge {i}: {e}")
    
    # Collect results
    result = {
        'dataset': dataset_name,
        'model_name': 'LearnablePPR',
        'encoder_type': encoder_type,
        'num_params': model.count_parameters(),
        'train_time': train_time,
        'best_val_loss': history.get('best_val_loss', 0),
        'best_epoch': history.get('best_epoch', 0),
        'selector_params': model.get_selector_params(),
        'test_metrics': test_metrics,  # Add computed metrics
        'test_predictions': {
            'mean': pred_metadata['pred_mean'],
            'std': pred_metadata['pred_std']
        },
        'subgraph_stats': {
            'avg_size': pred_metadata['avg_subgraph_size'],
            'min_size': pred_metadata['min_subgraph_size'],
            'max_size': pred_metadata['max_subgraph_size']
        }
    }
    
    print(f"\n✓ Experiment completed in {train_time:.1f}s")
    print(f"  Best val loss: {result['best_val_loss']:.4f}")
    print(f"  Test MRR: {test_metrics.get('mrr', 0):.4f}")
    print(f"  Test AUC: {test_metrics.get('auc', 0):.4f}")
    print(f"  Learned α: {result['selector_params']['alpha']:.3f}")
    print(f"  Avg subgraph: {result['subgraph_stats']['avg_size']:.1f} nodes")
    
    # Save results summary right after test evaluation
    if model.results_manager and model.exp_dir:
        _save_results_summary(model.exp_dir, result, dataset_name, encoder_type, train_time, history)
        print(f"  Results saved to: {model.exp_dir}/results_summary.txt")
    
    return result


def run_full_benchmark(datasets=None, encoders=None, config=None):
    """
    Run full benchmark on multiple datasets and encoders.
    
    Args:
        datasets: Dict of {name: path} for datasets
        encoders: List of encoder types ['GCN', 'SAGE', 'GAT']
        config: Configuration dictionary
    
    Returns:
        Dictionary of all results
    """    
    print("="*80)
    print("LEARNABLE PPR SUBGRAPH BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nDatasets: {list(datasets.keys())}")
    print(f"Encoders: {encoders}")
    print(f"Total experiments: {len(datasets) * len(encoders)}")
    
    # Run experiments
    all_results = {}
    results_manager = SubgraphResultsManager()
    
    total_start = time.time()
    
    for dataset_name, dataset_path in datasets.items():
        dataset_results = []
        
        for encoder_type in encoders:
            try:
                result = run_single_experiment(
                    dataset_name, dataset_path, encoder_type, config
                )
                dataset_results.append(result)
            except Exception as e:
                print(f"\n✗ Experiment failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Save comparison table for this dataset
        if dataset_results:
            results_manager.create_comparison_table(dataset_name, dataset_results)
            all_results[dataset_name] = dataset_results
    
    total_time = time.time() - total_start
    
    # Save full results
    results_manager.save_full_results(all_results)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"Results saved to: results/subgraph/")
    print(f"\nGenerated files:")
    print(f"  - full_results.json")
    print(f"  - {{dataset}}/comparison_table.csv")
    print(f"  - {{dataset}}/{{model}}/...")
    
    return all_results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run learnable PPR subgraph benchmark'
    )
    
    # Dataset arguments
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['FB15K237', 'WN18RR', 'NELL-995'],
                       help='Datasets to benchmark')
    parser.add_argument('--encoders', type=str, nargs='+',
                       default=['SAGE', 'GCN', 'GAT'],
                       choices=['GCN', 'SAGE', 'GAT'],
                       help='Encoder types to benchmark')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--ppr_alpha', type=float, default=0.85,
                       help='PPR teleport probability')
    
    # Training arguments (defaults match benchmark for fair comparison)
    parser.add_argument('--epochs', type=int, default=500,
                       help='Maximum training epochs (default: 500)')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience (default: 30)')
    parser.add_argument('--eval_steps', type=int, default=5,
                       help='Evaluation frequency (default: 5)')
    
    # Meta-learning specific arguments
    parser.add_argument('--inner_steps', type=int, default=5,
                       help='Inner loop steps (commonly 1 or 5)')
    parser.add_argument('--inner_lr', type=float, default=0.005,
                       help='Inner loop learning rate (GNN)')
    parser.add_argument('--outer_lr', type=float, default=0.0005,
                       help='Outer loop learning rate (selector)')
    parser.add_argument('--meta_learning_order', type=str, default='first',
                       choices=['first', 'second'],
                       help='Meta-learning order: first=FOMAML (memory efficient), second=full MAML (high memory)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--no_viz', action='store_true',
                       help='Disable subgraph visualizations')
    
    args = parser.parse_args()
    
    # Build datasets dict
    dataset_paths = {
        'FB15K237': 'data/FB15K237/train.txt',
        'WN18RR': 'data/WN18RR/train.txt',
        'NELL-995': 'data/NELL-995/train.txt'
    }
    datasets = {name: dataset_paths[name] for name in args.datasets if name in dataset_paths}
    
    # Build config
    config = {
        'feature_method': 'random',
        'feature_dim': 128,
        'hidden_dim': args.hidden_dim,          # 256
        'num_layers': args.num_layers,          # 3
        'dropout': args.dropout,                # 0.3
        'ppr_alpha': args.ppr_alpha,            # 0.85
        'adaptive_threshold': True,
        'epochs': args.epochs,                  # 500
        'inner_steps': args.inner_steps,        # 5
        'inner_lr': args.inner_lr,              # 0.005
        'outer_lr': args.outer_lr,              # 0.0005
        'eval_steps': args.eval_steps,          # 5
        'patience': args.patience,              # 30
        'meta_learning_order': args.meta_learning_order,  # 'first'=FOMAML, 'second'=full MAML
        'device': args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu',
        'save_visualizations': not args.no_viz, # True
    }
    
    # Run benchmark
    results = run_full_benchmark(
        datasets=datasets,
        encoders=args.encoders,
        config=config
    )
    
    return results


if __name__ == '__main__':
    main()

