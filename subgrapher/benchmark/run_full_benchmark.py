"""
Full benchmark runner for link prediction models.
Runs all 3 models (GCN, GraphSAGE, GAT) on all 3 datasets (FB15K237, WN18RR, NELL-995).
Saves comprehensive results, logs, and visualizations to results/benchmark/.
"""

import os
import sys
import json
import csv
import time
import datetime
import traceback
from contextlib import redirect_stdout
import io

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import existing benchmark function from same directory
from .run_benchmark import run_benchmark


# Default configuration with early stopping
DEFAULT_CONFIG = {
    'feature_method': 'random',
    'feature_dim': 128,
    'hidden_channels': 256,
    'num_layers': 3,
    'dropout': 0.3,
    'epochs': 500,             # Max epochs
    'batch_size': 65536,        # Batch size 2^16
    'lr': 0.005,
    'eval_steps': 5,            # More frequent evaluation for early stopping
    'patience': 200,            # Early stopping patience
    'weight_decay': 1e-5,       # L2 regularization
    'lr_scheduler': 'reduce_on_plateau',  # Learning rate scheduling
    'grad_clip': 1.0,           # Gradient clipping
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def setup_directories():
    """Create output directory structure."""
    base_dir = 'results/benchmark'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'visualizations'), exist_ok=True)
    
    datasets = ['FB15K237', 'WN18RR', 'NELL-995']
    models = ['GCN', 'SAGE', 'GAT']
    
    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        for model in models:
            model_dir = os.path.join(dataset_dir, model)
            os.makedirs(model_dir, exist_ok=True)
    
    return base_dir


def save_config(config, output_path):
    """Save configuration to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def save_experiment_results(results, dataset_name, base_dir):
    """
    Save results from one dataset (3 models) to individual directories.
    
    Args:
        results: List of result dicts from run_benchmark()
        dataset_name: Name of the dataset
        base_dir: Base results directory
    """
    dataset_dir = os.path.join(base_dir, dataset_name)
    
    # Save results for each model
    for result in results:
        model_name = result['model_name']
        model_dir = os.path.join(dataset_dir, model_name)
        
        # Save config
        config_path = os.path.join(model_dir, 'config.json')
        save_config({
            'model': model_name,
            'dataset': dataset_name,
            'num_params': result['num_params'],
            'training_info': {
                'best_epoch': result.get('best_epoch', 0),
                'stopped_early': result.get('stopped_early', False),
                'total_time_seconds': result['train_time']
            },
            'hyperparameters': {
                'hidden_channels': DEFAULT_CONFIG['hidden_channels'],
                'num_layers': DEFAULT_CONFIG['num_layers'],
                'dropout': DEFAULT_CONFIG['dropout'],
                'max_epochs': DEFAULT_CONFIG['epochs'],
                'patience': DEFAULT_CONFIG['patience'],
                'lr_initial': DEFAULT_CONFIG['lr'],
                'lr_scheduler': DEFAULT_CONFIG['lr_scheduler'],
                'weight_decay': DEFAULT_CONFIG['weight_decay'],
                'grad_clip': DEFAULT_CONFIG['grad_clip'],
                'batch_size': DEFAULT_CONFIG['batch_size'],
                'eval_steps': DEFAULT_CONFIG['eval_steps']
            }
        }, config_path)
        
        # Save metrics
        metrics_path = os.path.join(model_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(result['test_results'], f, indent=2)
        
        # Save summary
        summary_path = os.path.join(model_dir, 'results_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Parameters: {result['num_params']:,}\n")
            f.write(f"\nTraining:\n")
            f.write(f"  Time: {result['train_time']:.2f}s\n")
            f.write(f"  Best Epoch: {result.get('best_epoch', 'N/A')}\n")
            f.write(f"  Early Stopped: {'Yes' if result.get('stopped_early', False) else 'No'}\n")
            f.write(f"  Best Val MRR: {result['best_val_mrr']:.4f}\n")
            f.write(f"\nTest Results:\n")
            f.write(f"  MRR:     {result['test_results']['mrr']:.4f}\n")
            f.write(f"  AUC:     {result['test_results']['auc']:.4f}\n")
            f.write(f"  AP:      {result['test_results']['ap']:.4f}\n")
            for key in sorted(result['test_results'].keys()):
                if key.startswith('hits@'):
                    f.write(f"  {key}:  {result['test_results'][key]:.4f}\n")
        
        # Save training curve if history available
        if 'history' in result and result['history']:
            try:
                plot_training_curve(result['history'], model_name, dataset_name, model_dir)
            except Exception as e:
                print(f"Warning: Could not plot training curve: {e}")


def plot_training_curve(history, model_name, dataset_name, output_dir):
    """Plot training loss and validation metrics over epochs."""
    if not history.get('train_loss') or not history.get('val_results'):
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{model_name} - Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot validation MRR
    val_epochs = [i * DEFAULT_CONFIG['eval_steps'] for i in range(1, len(history['val_results']) + 1)]
    val_mrr = [res['mrr'] for res in history['val_results']]
    ax2.plot(val_epochs, val_mrr, 'r-', linewidth=2, marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation MRR')
    ax2.set_title(f'{model_name} - Validation MRR')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} on {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_single_dataset(dataset_name, dataset_path, config, base_dir):
    """
    Run all 3 models on one dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to dataset file
        config: Configuration dict
        base_dir: Base results directory
    Returns:
        List of results or None if failed
    """
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*80}")
    
    dataset_dir = os.path.join(base_dir, dataset_name)
    log_path = os.path.join(dataset_dir, 'experiment_log.txt')
    
    start_time = time.time()
    
    try:
        # Capture stdout to log file
        log_buffer = io.StringIO()
        
        with redirect_stdout(log_buffer):
            # Call existing run_benchmark function with early stopping
            results = run_benchmark(
                dataset_path=dataset_path,
                feature_method=config['feature_method'],
                feature_dim=config['feature_dim'],
                hidden_channels=config['hidden_channels'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                lr=config['lr'],
                eval_steps=config['eval_steps'],
                patience=config['patience'],
                weight_decay=config['weight_decay'],
                lr_scheduler=config['lr_scheduler'],
                grad_clip=config['grad_clip'],
                device=config['device'],
                models_to_run=['GCN', 'SAGE', 'GAT']
            )
        
        # Save log
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_buffer.getvalue())
        
        # Also print to console
        print(log_buffer.getvalue())
        
        # Save results
        save_experiment_results(results, dataset_name, base_dir)
        
        elapsed = time.time() - start_time
        print(f"\n✓ {dataset_name} completed in {elapsed:.1f}s")
        
        return results
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"✗ Error in {dataset_name}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # Save error log
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(error_msg)
        
        return None


def create_comparison_table(all_results, base_dir):
    """Create CSV comparison table."""
    csv_path = os.path.join(base_dir, 'comparison_table.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Model', 'Params', 'Train_Time_s', 'MRR', 'AUC', 
                        'AP', 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@50'])
        
        for dataset_name, results in all_results.items():
            if results is None:
                continue
            for result in results:
                test = result['test_results']
                writer.writerow([
                    dataset_name,
                    result['model_name'],
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


def create_visualizations(all_results, base_dir):
    """Create comparison visualizations."""
    viz_dir = os.path.join(base_dir, 'visualizations')
    
    # Extract data
    datasets = []
    models = []
    mrr_matrix = []
    time_data = []
    
    for dataset_name in sorted(all_results.keys()):
        results = all_results[dataset_name]
        if results is None:
            continue
        
        datasets.append(dataset_name)
        dataset_mrr = []
        
        for result in sorted(results, key=lambda x: x['model_name']):
            if dataset_name == sorted(all_results.keys())[0]:  # Only add models once
                models.append(result['model_name'])
            
            dataset_mrr.append(result['test_results']['mrr'])
            time_data.append({
                'Dataset': dataset_name,
                'Model': result['model_name'],
                'Time': result['train_time']
            })
        
        mrr_matrix.append(dataset_mrr)
    
    if not datasets or not models:
        print("Warning: No data for visualizations")
        return
    
    # 1. MRR Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(mrr_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=models, yticklabels=datasets, cbar_kws={'label': 'MRR'})
    plt.title('Mean Reciprocal Rank (MRR) - Model × Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mrr_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Training Time Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, model in enumerate(models):
        times = [next((d['Time'] for d in time_data 
                      if d['Dataset'] == ds and d['Model'] == model), 0) 
                for ds in datasets]
        ax.bar(x + i * width, times, width, label=model)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'training_time_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Performance Comparison (MRR bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        mrrs = [next((r['test_results']['mrr'] for r in all_results[ds] 
                     if r['model_name'] == model), 0) 
               for ds in datasets if all_results[ds] is not None]
        ax.bar(x + i * width, mrrs, width, label=model)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('MRR')
    ax.set_title('Model Performance Comparison (MRR)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'performance_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Hits@K Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    k_values = [1, 10, 50]
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        for i, model in enumerate(models):
            hits = [next((r['test_results'].get(f'hits@{k}', 0) for r in all_results[ds] 
                         if r['model_name'] == model), 0) 
                   for ds in datasets if all_results[ds] is not None]
            ax.bar(x + i * width, hits, width, label=model)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel(f'Hits@{k}')
        ax.set_title(f'Hits@{k} Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Hit@K Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'hits_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to {viz_dir}/")


def create_summary_report(all_results, base_dir, total_time):
    """Create human-readable summary report."""
    report_path = os.path.join(base_dir, 'summary_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LINK PREDICTION BENCHMARK - FULL RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Run Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Runtime: {total_time/3600:.2f} hours ({total_time:.1f}s)\n")
        f.write(f"Device: {DEFAULT_CONFIG['device']}\n")
        f.write(f"Configuration:\n")
        for key, value in DEFAULT_CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS BY DATASET\n")
        f.write("="*80 + "\n\n")
        
        for dataset_name in sorted(all_results.keys()):
            results = all_results[dataset_name]
            f.write(f"\n{dataset_name}:\n")
            f.write("-" * 80 + "\n")
            
            if results is None:
                f.write("  ✗ Failed - see error log\n")
                continue
            
            for result in sorted(results, key=lambda x: x['model_name']):
                f.write(f"  {result['model_name']}:\n")
                f.write(f"    Parameters:    {result['num_params']:,}\n")
                f.write(f"    Training Time: {result['train_time']:.2f}s\n")
                f.write(f"    MRR:           {result['test_results']['mrr']:.4f}\n")
                f.write(f"    AUC:           {result['test_results']['auc']:.4f}\n")
                f.write(f"    Hits@10:       {result['test_results'].get('hits@10', 0):.4f}\n")
                f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("BEST RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Find best models
        all_flat = []
        for dataset_name, results in all_results.items():
            if results:
                for r in results:
                    all_flat.append({**r, 'dataset': dataset_name})
        
        if all_flat:
            best_mrr = max(all_flat, key=lambda x: x['test_results']['mrr'])
            fastest = min(all_flat, key=lambda x: x['train_time'])
            
            f.write(f"Best MRR: {best_mrr['model_name']} on {best_mrr['dataset']} "
                   f"({best_mrr['test_results']['mrr']:.4f})\n")
            f.write(f"Fastest:  {fastest['model_name']} on {fastest['dataset']} "
                   f"({fastest['train_time']:.1f}s)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FILES GENERATED\n")
        f.write("="*80 + "\n\n")
        f.write("- comparison_table.csv: All results in CSV format\n")
        f.write("- full_results.json: Complete results in JSON\n")
        f.write("- visualizations/: Comparison plots\n")
        f.write("- {dataset}/{model}/: Individual experiment results\n")
    
    print(f"✓ Summary report saved to {report_path}")


def run_full_benchmark(config=None):
    """
    Run full benchmark: 3 models × 3 datasets = 9 experiments.
    
    Args:
        config: Optional config dict (uses DEFAULT_CONFIG if None)
    Returns:
        Dictionary of all results
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    print("="*80)
    print("FULL LINK PREDICTION BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n")
    
    # Setup directories
    base_dir = setup_directories()
    print(f"✓ Output directory: {base_dir}/")
    
    # Define datasets
    datasets = {
        'FB15K237': 'data/FB15K237/train.txt',
        'WN18RR': 'data/WN18RR/train.txt',
        'NELL-995': 'data/NELL-995/train.txt'
    }
    
    # Run experiments
    all_results = {}
    start_time = time.time()
    
    for dataset_name, dataset_path in datasets.items():
        results = run_single_dataset(dataset_name, dataset_path, config, base_dir)
        all_results[dataset_name] = results
    
    total_time = time.time() - start_time
    
    # Save aggregated results
    print(f"\n{'='*80}")
    print("GENERATING REPORTS AND VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # Save full results JSON
    json_path = os.path.join(base_dir, 'full_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        # Convert results to serializable format
        serializable_results = {}
        for dataset_name, results in all_results.items():
            if results is not None:
                serializable_results[dataset_name] = []
                for r in results:
                    # Remove non-serializable history details
                    r_copy = r.copy()
                    if 'history' in r_copy:
                        del r_copy['history']
                    serializable_results[dataset_name].append(r_copy)
        json.dump(serializable_results, f, indent=2)
    print(f"✓ Full results saved to {json_path}")
    
    # Create comparison table
    create_comparison_table(all_results, base_dir)
    print(f"✓ Comparison table saved to {base_dir}/comparison_table.csv")
    
    # Create visualizations
    create_visualizations(all_results, base_dir)
    
    # Create summary report
    create_summary_report(all_results, base_dir, total_time)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time:.1f}s)")
    print(f"Results saved to: {base_dir}/")
    print("\nGenerated files:")
    print(f"  - summary_report.txt")
    print(f"  - full_results.json")
    print(f"  - comparison_table.csv")
    print(f"  - visualizations/ (4 plots)")
    print(f"  - {{dataset}}/{{model}}/ (individual results)")
    
    return all_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run full benchmark: 3 models × 3 datasets'
    )

    # Add custom arguments here
    ###
    
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    for key, value in vars(args).items():
        if value is not None:    
            config[key] = value
    
    # Run benchmark
    run_full_benchmark(config)


if __name__ == '__main__':
    main()

