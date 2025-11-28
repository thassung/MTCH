"""
Results management for learnable subgraph experiments.
Saves results in structure similar to benchmark system.
"""

import os
import json
import csv
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path


class SubgraphResultsManager:
    """
    Manages saving and organizing experimental results.
    
    Structure:
        results/subgraph/
        ├── {dataset}/
        │   ├── {model}_{encoder}/
        │   │   ├── config.json
        │   │   ├── metrics.json
        │   │   ├── selector_params.json
        │   │   ├── results_summary.txt
        │   │   ├── training_history.json
        │   │   ├── training_curve.png
        │   │   ├── selector_evolution.png
        │   │   ├── subgraph_sizes.png
        │   │   └── experiment_log.txt
        │   └── comparison_table.csv
        └── full_results.json
    """
    
    def __init__(self, base_dir='results/subgraph'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_experiment_dir(self, dataset_name, model_name, encoder_type):
        """Create directory structure for experiment."""
        model_dir_name = f"{model_name}_{encoder_type}"
        exp_dir = self.base_dir / dataset_name / model_dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def save_config(self, exp_dir, config):
        """Save experiment configuration."""
        config_path = exp_dir / 'config.json'
        
        # Convert non-serializable objects
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, torch.device):
                serializable_config[key] = str(value)
            else:
                serializable_config[key] = value
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2)
    
    def save_metrics(self, exp_dir, metrics):
        """Save evaluation metrics."""
        metrics_path = exp_dir / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
    
    def save_selector_params(self, exp_dir, selector_params, history=None):
        """Save learned selector parameters and evolution."""
        params_path = exp_dir / 'selector_params.json'
        
        data = {
            'final_params': selector_params,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if history and 'selector_alpha' in history:
            data['evolution'] = {
                'alpha': history['selector_alpha'],
                'threshold': history['selector_threshold']
            }
        
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def save_training_history(self, exp_dir, history):
        """Save full training history."""
        history_path = exp_dir / 'training_history.json'
        
        # Convert to serializable format
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, (list, float, int, str, bool, type(None))):
                serializable_history[key] = value
            elif isinstance(value, dict) and key != 'best_selector_state':
                serializable_history[key] = value
            # Skip state dicts
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2)
    
    def save_summary(self, exp_dir, summary_data):
        """Save human-readable summary."""
        summary_path = exp_dir / 'results_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LEARNABLE SUBGRAPH EXPERIMENT RESULTS\n")
            f.write("="*80 + "\n\n")
            
            # Configuration
            f.write("Configuration:\n")
            f.write(f"  Dataset: {summary_data.get('dataset', 'N/A')}\n")
            f.write(f"  Model: {summary_data.get('model_name', 'N/A')}\n")
            f.write(f"  Encoder: {summary_data.get('encoder_type', 'N/A')}\n")
            f.write(f"  Hidden Dim: {summary_data.get('hidden_dim', 'N/A')}\n")
            f.write(f"  Num Layers: {summary_data.get('num_layers', 'N/A')}\n")
            f.write(f"  Total Parameters: {summary_data.get('num_params', 'N/A'):,}\n")
            f.write("\n")
            
            # Training info
            f.write("Training:\n")
            f.write(f"  Time: {summary_data.get('train_time', 0):.2f}s\n")
            f.write(f"  Epochs: {summary_data.get('epochs_trained', 'N/A')}\n")
            f.write(f"  Best Epoch: {summary_data.get('best_epoch', 'N/A')}\n")
            f.write(f"  Early Stopped: {summary_data.get('stopped_early', 'N/A')}\n")
            f.write(f"  Best Val Loss: {summary_data.get('best_val_loss', 0):.4f}\n")
            f.write("\n")
            
            # Selector parameters
            if 'selector_params' in summary_data:
                f.write("Learned Selector Parameters:\n")
                params = summary_data['selector_params']
                f.write(f"  Alpha: {params.get('alpha', 0):.4f}\n")
                if params.get('threshold_percentile') is not None:
                    f.write(f"  Threshold Percentile: {params['threshold_percentile']:.4f}\n")
                f.write(f"  Sharpness: {params.get('sharpness', 0):.2f}\n")
                f.write("\n")
            
            # Subgraph statistics
            if 'subgraph_stats' in summary_data:
                f.write("Subgraph Statistics:\n")
                stats = summary_data['subgraph_stats']
                f.write(f"  Avg Size: {stats.get('avg_size', 0):.1f} nodes\n")
                f.write(f"  Min Size: {stats.get('min_size', 0)} nodes\n")
                f.write(f"  Max Size: {stats.get('max_size', 0)} nodes\n")
                f.write(f"  Std Dev: {stats.get('std_size', 0):.1f} nodes\n")
                f.write("\n")
            
            # Test results
            if 'test_results' in summary_data:
                f.write("Test Results:\n")
                test = summary_data['test_results']
                for key, value in sorted(test.items()):
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
    
    def plot_training_curve(self, exp_dir, history, model_name, dataset_name):
        """Plot training loss and validation loss over epochs."""
        if not history.get('train_loss') or not history.get('val_loss'):
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training loss
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} - Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation loss
        ax2.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        if 'best_epoch' in history:
            ax2.axvline(x=history['best_epoch'], color='g', linestyle='--', 
                       label=f"Best ({history['best_epoch']})")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{model_name} - Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} on {dataset_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = exp_dir / 'training_curve.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_selector_evolution(self, exp_dir, history, model_name):
        """Plot evolution of selector parameters during training."""
        if 'selector_alpha' not in history or 'selector_threshold' not in history:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['selector_alpha']) + 1)
        
        # Alpha evolution
        ax1.plot(epochs, history['selector_alpha'], 'b-', linewidth=2)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balanced (0.5)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Alpha')
        ax1.set_title('Selector Alpha Evolution')
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Threshold evolution
        ax2.plot(epochs, history['selector_threshold'], 'r-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Threshold')
        ax2.set_title('Selector Threshold Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Learned Parameters', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = exp_dir / 'selector_evolution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_subgraph_statistics(self, exp_dir, subgraph_sizes, total_nodes, model_name):
        """Plot subgraph size distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(subgraph_sizes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=sum(subgraph_sizes)/len(subgraph_sizes), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {sum(subgraph_sizes)/len(subgraph_sizes):.1f}')
        ax1.set_xlabel('Subgraph Size (nodes)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Subgraph Size Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Box plot
        ax2.boxplot([subgraph_sizes], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='blue'),
                   medianprops=dict(color='red', linewidth=2))
        ax2.axhline(y=total_nodes, color='orange', linestyle='--', 
                   label=f'Full Graph: {total_nodes} nodes')
        ax2.set_ylabel('Subgraph Size (nodes)')
        ax2.set_title('Subgraph Size Statistics')
        ax2.set_xticklabels(['Selected Subgraphs'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'{model_name} - Subgraph Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = exp_dir / 'subgraph_sizes.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_experiment_log(self, exp_dir, log_text):
        """Save experiment log/stdout."""
        log_path = exp_dir / 'experiment_log.txt'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_text)
    
    def create_comparison_table(self, dataset_name, results_list):
        """Create CSV comparison table for a dataset."""
        csv_path = self.base_dir / dataset_name / 'comparison_table.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Model', 'Encoder', 'Params', 'Train_Time_s', 
                'Val_Loss', 'Test_Loss', 'Avg_Subgraph_Size',
                'Alpha', 'Threshold', 'MRR', 'AUC', 'AP'
            ])
            
            for result in results_list:
                writer.writerow([
                    result.get('model_name', 'N/A'),
                    result.get('encoder_type', 'N/A'),
                    result.get('num_params', 0),
                    f"{result.get('train_time', 0):.2f}",
                    f"{result.get('best_val_loss', 0):.4f}",
                    f"{result.get('test_loss', 0):.4f}",
                    f"{result.get('avg_subgraph_size', 0):.1f}",
                    f"{result.get('alpha', 0):.4f}",
                    f"{result.get('threshold', 0):.4f}",
                    f"{result.get('test_results', {}).get('mrr', 0):.4f}",
                    f"{result.get('test_results', {}).get('auc', 0):.4f}",
                    f"{result.get('test_results', {}).get('ap', 0):.4f}"
                ])
    
    def save_full_results(self, all_results):
        """Save complete results across all experiments."""
        json_path = self.base_dir / 'full_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
    
    def save_visualized_subgraph(self, exp_dir, fig, edge_idx=0):
        """Save subgraph visualization."""
        viz_dir = exp_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        output_path = viz_dir / f'subgraph_example_{edge_idx}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return output_path

