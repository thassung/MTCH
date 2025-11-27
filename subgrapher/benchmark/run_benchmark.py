"""
Main benchmark runner for link prediction models.
Compares GCN, GraphSAGE, and GAT on link prediction task.
"""

import torch
import argparse

from .models import GCN, SAGE, GAT, LinkPredictor
from .data_prep import prepare_link_prediction_data
from .trainer import benchmark_model


def print_comparison_table(results_list):
    """
    Print comparison table for all models.
    
    Args:
        results_list: List of result dictionaries
    """
    print("\n" + "="*90)
    print("LINK PREDICTION BENCHMARK COMPARISON")
    print("="*90)
    
    # Header
    header = f"{'Model':<12} | {'Params':<8} | {'Time(s)':<8} | {'MRR':<7} | {'AUC':<7} | {'AP':<7} | {'H@1':<7} | {'H@10':<7} | {'H@50':<7}"
    print(header)
    print("-"*90)
    
    # Rows
    for result in results_list:
        test = result['test_results']
        row = (f"{result['model_name']:<12} | "
               f"{result['num_params']/1000:.1f}K"[:8].ljust(8) + " | "
               f"{result['train_time']:<8.1f} | "
               f"{test['mrr']:<7.4f} | "
               f"{test['auc']:<7.4f} | "
               f"{test['ap']:<7.4f} | "
               f"{test.get('hits@1', 0):<7.4f} | "
               f"{test.get('hits@10', 0):<7.4f} | "
               f"{test.get('hits@50', 0):<7.4f}")
        print(row)
    
    print("="*90)
    
    # Summary
    print("\nSummary:")
    best_mrr = max(results_list, key=lambda x: x['test_results']['mrr'])
    fastest = min(results_list, key=lambda x: x['train_time'])
    
    print(f"  Best MRR:        {best_mrr['model_name']} ({best_mrr['test_results']['mrr']:.4f})")
    print(f"  Fastest:         {fastest['model_name']} ({fastest['train_time']:.1f}s)")
    print(f"  Speed ratio:     {max(r['train_time'] for r in results_list) / min(r['train_time'] for r in results_list):.2f}x")


def run_benchmark(dataset_path, 
                  feature_method='random',
                  feature_dim=128,
                  hidden_channels=256,
                  num_layers=3,
                  dropout=0.3,
                  epochs=100,
                  batch_size=65536,
                  lr=0.001,
                  eval_steps=10,
                  device='cuda',
                  models_to_run=None):
    """
    Run complete benchmark on dataset.
    
    Args:
        dataset_path: Path to dataset .txt file
        feature_method: 'random' or 'one_hot'
        feature_dim: Feature dimension
        hidden_channels: Hidden dimension for GNN
        num_layers: Number of GNN layers
        dropout: Dropout rate
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        eval_steps: Evaluation frequency
        device: 'cpu' or 'cuda'
        models_to_run: List of model names or None for all
    Returns:
        List of result dictionaries
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Prepare data
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    prepared_data = prepare_link_prediction_data(
        dataset_path,
        feature_method=feature_method,
        feature_dim=feature_dim
    )
    
    data = prepared_data['data']
    split_edge = prepared_data['split_edge']
    train_edge_index = prepared_data['train_edge_index']
    
    # Update data for training (only use training edges)
    data.edge_index = train_edge_index
    
    in_channels = data.x.size(1)
    
    # Models to benchmark
    if models_to_run is None:
        models_to_run = ['GCN', 'SAGE', 'GAT']
    
    results_list = []
    
    # Benchmark each model
    for model_name in models_to_run:
        # Create encoder
        if model_name == 'GCN':
            encoder = GCN(in_channels, hidden_channels, hidden_channels, 
                         num_layers, dropout)
        elif model_name == 'SAGE':
            encoder = SAGE(in_channels, hidden_channels, hidden_channels,
                          num_layers, dropout)
        elif model_name == 'GAT':
            encoder = GAT(in_channels, hidden_channels, hidden_channels,
                         num_layers, dropout, heads=4)
        else:
            print(f"Unknown model: {model_name}, skipping...")
            continue
        
        # Create predictor
        predictor = LinkPredictor(hidden_channels, hidden_channels, 1,
                                  num_layers, dropout)
        
        # Reset parameters
        encoder.reset_parameters()
        predictor.reset_parameters()
        
        # Benchmark
        result = benchmark_model(
            model_name, encoder, predictor, data, split_edge,
            epochs=epochs, batch_size=batch_size, lr=lr,
            eval_steps=eval_steps, device=device
        )
        
        results_list.append(result)
    
    # Print comparison
    print_comparison_table(results_list)
    
    return results_list


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Link Prediction Benchmark')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='data/FB15K237/train.txt',
                       help='Path to dataset .txt file')
    parser.add_argument('--feature_method', type=str, default='random',
                       choices=['random', 'one_hot'],
                       help='Node feature method')
    parser.add_argument('--feature_dim', type=int, default=128,
                       help='Feature dimension for random method')
    
    # Model arguments
    parser.add_argument('--hidden_channels', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=65536,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--eval_steps', type=int, default=10,
                       help='Evaluation frequency (epochs)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to run (default: all)')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        dataset_path=args.dataset,
        feature_method=args.feature_method,
        feature_dim=args.feature_dim,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_steps=args.eval_steps,
        device=args.device,
        models_to_run=args.models
    )
    
    return results


if __name__ == '__main__':
    main()

