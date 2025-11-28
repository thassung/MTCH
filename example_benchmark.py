"""
Example script demonstrating how to use the link prediction benchmark.

This script shows how to:
1. Run a complete benchmark on a knowledge graph dataset
2. Compare multiple GNN models (GCN, GraphSAGE, GAT)
3. Get detailed performance metrics and training times
"""

import torch
from subgrapher.benchmark.run_benchmark import run_benchmark

def main():
    """Run example benchmark."""
    
    print("="*70)
    print("Link Prediction Benchmark - Example Usage")
    print("="*70)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    if device == 'cpu':
        print("WARNING: Running on CPU. This will be slow!")
        print("For faster training, use a GPU-enabled system.")
    
    # Configure benchmark
    config = {
        'dataset_path': 'data/FB15K237/train.txt',  # Knowledge graph dataset
        'feature_method': 'random',                  # Use random node features
        'feature_dim': 128,                          # 128-dimensional embeddings
        'hidden_channels': 256,                      # Hidden dimension for GNN
        'num_layers': 3,                             # 3-layer GNN
        'dropout': 0.3,                              # Dropout rate
        'epochs': 50,                                # Training epochs (reduced for demo)
        'batch_size': 65536,                         # Batch size
        'lr': 0.001,                                 # Learning rate
        'eval_steps': 10,                            # Evaluate every 10 epochs
        'device': device,
        'models_to_run': ['GCN', 'SAGE', 'GAT']     # All three models
    }
    
    print("\nBenchmark Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run benchmark
    print("\n" + "="*70)
    print("Starting Benchmark...")
    print("="*70)
    
    results = run_benchmark(**config)
    
    # Additional analysis
    print("\n" + "="*70)
    print("Additional Analysis")
    print("="*70)
    
    # Accuracy vs Speed tradeoff
    print("\nAccuracy vs Speed Trade-off:")
    for result in results:
        mrr = result['test_results']['mrr']
        time = result['train_time']
        efficiency = mrr / time  # MRR per second
        print(f"  {result['model_name']:<12}: {mrr:.4f} MRR in {time:.1f}s "
              f"(efficiency: {efficiency:.6f} MRR/s)")
    
    # Memory footprint
    print("\nModel Size Comparison:")
    for result in results:
        params = result['num_params']
        print(f"  {result['model_name']:<12}: {params:,} parameters "
              f"({params/1000:.1f}K)")
    
    # Best model per metric
    print("\nBest Model per Metric:")
    metrics = ['mrr', 'auc', 'hits@10']
    for metric in metrics:
        best = max(results, key=lambda x: x['test_results'].get(metric, 0))
        value = best['test_results'].get(metric, 0)
        print(f"  {metric.upper():<10}: {best['model_name']} ({value:.4f})")
    
    print("\n" + "="*70)
    print("Benchmark Complete!")
    print("="*70)
    
    return results


if __name__ == '__main__':
    # Run the example
    results = main()
    
    # # Save results 
    # import json
    # with open('benchmark_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)

