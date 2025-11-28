"""
Unified Comparison Script for All Link Prediction Methods

Aggregates and compares results from all benchmark systems:
1. Full Graph (results/benchmark/)
2. Static PPR (results/benchmark-ppr/)
3. Static k-hop (results/benchmark-khop/)
4. Learnable PPR (results/subgraph/)

Creates unified comparison tables and visualizations.
"""

import os
import json
import csv
import glob
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def load_full_graph_results(base_dir='results/benchmark'):
    """Load results from full graph benchmark."""
    results = []
    
    if not os.path.exists(base_dir):
        print(f"Warning: {base_dir} not found")
        return results
    
    # Read comparison table if available
    csv_path = os.path.join(base_dir, 'comparison_table.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({
                    'method': 'Full Graph',
                    'dataset': row['Dataset'],
                    'encoder': row['Model'],
                    'config': '-',
                    'mrr': float(row['MRR']),
                    'auc': float(row['AUC']),
                    'ap': float(row['AP']),
                    'hits@10': float(row['Hits@10']),
                    'train_time': float(row['Train_Time_s'])
                })
    
    return results


def load_ppr_results(base_dir='results/benchmark-ppr'):
    """Load results from static PPR benchmark."""
    results = []
    
    if not os.path.exists(base_dir):
        print(f"Warning: {base_dir} not found")
        return results
    
    # Read comparison table if available
    csv_path = os.path.join(base_dir, 'comparison_table.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({
                    'method': 'Static PPR',
                    'dataset': row['Dataset'],
                    'encoder': row['Encoder'],
                    'config': f"k={row['Top_K']}",
                    'mrr': float(row['MRR']),
                    'auc': float(row['AUC']),
                    'ap': float(row['AP']),
                    'hits@10': float(row['Hits@10']),
                    'train_time': float(row['Train_Time_s'])
                })
    
    return results


def load_khop_results(base_dir='results/benchmark-khop'):
    """Load results from static k-hop benchmark."""
    results = []
    
    if not os.path.exists(base_dir):
        print(f"Warning: {base_dir} not found")
        return results
    
    # Read comparison table if available
    csv_path = os.path.join(base_dir, 'comparison_table.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({
                    'method': 'Static k-hop',
                    'dataset': row['Dataset'],
                    'encoder': row['Encoder'],
                    'config': f"k={row['K']}",
                    'mrr': float(row['MRR']),
                    'auc': float(row['AUC']),
                    'ap': float(row['AP']),
                    'hits@10': float(row['Hits@10']),
                    'train_time': float(row['Train_Time_s'])
                })
    
    return results


def load_learnable_ppr_results(base_dir='results/subgraph'):
    """Load results from learnable PPR benchmark."""
    results = []
    
    if not os.path.exists(base_dir):
        print(f"Warning: {base_dir} not found")
        return results
    
    # Scan for experiment directories
    for dataset_dir in glob.glob(os.path.join(base_dir, '*')):
        if not os.path.isdir(dataset_dir):
            continue
        
        dataset_name = os.path.basename(dataset_dir)
        
        for exp_dir in glob.glob(os.path.join(dataset_dir, '*')):
            if not os.path.isdir(exp_dir):
                continue
            
            # Load metrics
            metrics_path = os.path.join(exp_dir, 'metrics.json')
            config_path = os.path.join(exp_dir, 'config.json')
            
            if os.path.exists(metrics_path) and os.path.exists(config_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                encoder = config.get('encoder', 'Unknown')
                
                results.append({
                    'method': 'Learnable PPR',
                    'dataset': dataset_name,
                    'encoder': encoder,
                    'config': 'learnable',
                    'mrr': metrics.get('mrr', 0),
                    'auc': metrics.get('auc', 0),
                    'ap': metrics.get('ap', 0),
                    'hits@10': metrics.get('hits@10', 0),
                    'train_time': config.get('training_info', {}).get('total_time_seconds', 0)
                })
    
    return results


def create_unified_table(all_results, output_path='results/unified_comparison.csv'):
    """Create unified comparison table."""
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("Warning: No results to create table")
        return
    
    # Sort by method, dataset, encoder
    df = df.sort_values(['dataset', 'encoder', 'method', 'config'])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Unified comparison table saved to {output_path}")
    
    return df


def create_best_methods_table(all_results, output_path='results/best_methods.csv'):
    """Create table showing best method for each dataset/encoder combination."""
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("Warning: No results for best methods table")
        return
    
    # Group by dataset and encoder, find best MRR
    best_results = []
    
    for (dataset, encoder), group in df.groupby(['dataset', 'encoder']):
        best_row = group.loc[group['mrr'].idxmax()]
        best_results.append({
            'dataset': dataset,
            'encoder': encoder,
            'best_method': best_row['method'],
            'best_config': best_row['config'],
            'mrr': best_row['mrr'],
            'auc': best_row['auc'],
            'ap': best_row['ap'],
            'hits@10': best_row['hits@10'],
            'train_time': best_row['train_time']
        })
    
    best_df = pd.DataFrame(best_results)
    best_df = best_df.sort_values(['dataset', 'encoder'])
    best_df.to_csv(output_path, index=False)
    print(f"✓ Best methods table saved to {output_path}")
    
    return best_df


def create_method_summary(all_results, output_path='results/method_summary.txt'):
    """Create human-readable summary of method performance."""
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("Warning: No results for summary")
        return
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("UNIFIED METHOD COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("Methods Compared:\n")
        for method in df['method'].unique():
            count = len(df[df['method'] == method])
            avg_mrr = df[df['method'] == method]['mrr'].mean()
            f.write(f"  {method}: {count} experiments, Avg MRR = {avg_mrr:.4f}\n")
        f.write("\n")
        
        # Best method per dataset
        f.write("="*80 + "\n")
        f.write("BEST METHOD PER DATASET\n")
        f.write("="*80 + "\n\n")
        
        for dataset in sorted(df['dataset'].unique()):
            f.write(f"{dataset}:\n")
            dataset_df = df[df['dataset'] == dataset]
            
            for encoder in sorted(dataset_df['encoder'].unique()):
                encoder_df = dataset_df[dataset_df['encoder'] == encoder]
                best = encoder_df.loc[encoder_df['mrr'].idxmax()]
                
                f.write(f"  {encoder}: {best['method']} ({best['config']}) - MRR={best['mrr']:.4f}\n")
            f.write("\n")
        
        # Method win rate
        f.write("="*80 + "\n")
        f.write("METHOD WIN RATE (Best MRR)\n")
        f.write("="*80 + "\n\n")
        
        win_counts = defaultdict(int)
        total_comparisons = 0
        
        for (dataset, encoder), group in df.groupby(['dataset', 'encoder']):
            best_method = group.loc[group['mrr'].idxmax()]['method']
            win_counts[best_method] += 1
            total_comparisons += 1
        
        for method in sorted(win_counts.keys()):
            win_rate = win_counts[method] / total_comparisons * 100
            f.write(f"  {method}: {win_counts[method]}/{total_comparisons} ({win_rate:.1f}%)\n")
        f.write("\n")
    
    print(f"✓ Method summary saved to {output_path}")


def create_visualizations(all_results, output_dir='results/visualizations'):
    """Create comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("Warning: No results for visualizations")
        return
    
    datasets = sorted(df['dataset'].unique())
    encoders = sorted(df['encoder'].unique())
    methods = sorted(df['method'].unique())
    
    # 1. MRR comparison across methods
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_df = df[df['dataset'] == dataset]
        
        # For each encoder, show best result from each method
        x = np.arange(len(encoders))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            mrrs = []
            for encoder in encoders:
                encoder_method_df = dataset_df[(dataset_df['encoder'] == encoder) & 
                                               (dataset_df['method'] == method)]
                if not encoder_method_df.empty:
                    best_mrr = encoder_method_df['mrr'].max()
                    mrrs.append(best_mrr)
                else:
                    mrrs.append(0)
            
            ax.bar(x + i * width, mrrs, width, label=method)
        
        ax.set_xlabel('Encoder')
        ax.set_ylabel('Best MRR')
        ax.set_title(f'{dataset}')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(encoders)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Method Comparison: Best MRR', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Training time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Average training time per method
    method_times = []
    method_names = []
    for method in methods:
        method_df = df[df['method'] == method]
        if not method_df.empty:
            avg_time = method_df['train_time'].mean()
            method_times.append(avg_time)
            method_names.append(method)
    
    ax.bar(method_names, method_times)
    ax.set_ylabel('Average Training Time (s)')
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap: Method × Dataset (best MRR)
    heatmap_data = []
    for method in methods:
        method_row = []
        for dataset in datasets:
            dataset_method_df = df[(df['dataset'] == dataset) & (df['method'] == method)]
            if not dataset_method_df.empty:
                best_mrr = dataset_method_df['mrr'].max()
                method_row.append(best_mrr)
            else:
                method_row.append(0)
        heatmap_data.append(method_row)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=datasets, yticklabels=methods, cbar_kws={'label': 'Best MRR'})
    plt.title('Method Performance Heatmap (Best MRR)', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to {output_dir}/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare all link prediction methods'
    )
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("UNIFIED METHOD COMPARISON")
    print("="*80)
    print("\nLoading results from all benchmarks...\n")
    
    # Load all results
    all_results = []
    
    print("Loading Full Graph results...")
    full_graph = load_full_graph_results()
    all_results.extend(full_graph)
    print(f"  Found {len(full_graph)} results")
    
    print("Loading Static PPR results...")
    ppr = load_ppr_results()
    all_results.extend(ppr)
    print(f"  Found {len(ppr)} results")
    
    print("Loading Static k-hop results...")
    khop = load_khop_results()
    all_results.extend(khop)
    print(f"  Found {len(khop)} results")
    
    print("Loading Learnable PPR results...")
    learnable = load_learnable_ppr_results()
    all_results.extend(learnable)
    print(f"  Found {len(learnable)} results")
    
    print(f"\nTotal results: {len(all_results)}")
    
    if not all_results:
        print("\n✗ No results found. Please run benchmarks first.")
        return
    
    # Create outputs
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORTS")
    print("="*80 + "\n")
    
    # Unified table
    df = create_unified_table(all_results, 
                             os.path.join(args.output_dir, 'unified_comparison.csv'))
    
    # Best methods table
    create_best_methods_table(all_results,
                              os.path.join(args.output_dir, 'best_methods.csv'))
    
    # Summary report
    create_method_summary(all_results,
                         os.path.join(args.output_dir, 'method_summary.txt'))
    
    # Visualizations
    create_visualizations(all_results,
                         os.path.join(args.output_dir, 'visualizations'))
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"Results saved to {args.output_dir}/")
    print("\nGenerated files:")
    print("  - unified_comparison.csv")
    print("  - best_methods.csv")
    print("  - method_summary.txt")
    print("  - visualizations/")


if __name__ == '__main__':
    main()

