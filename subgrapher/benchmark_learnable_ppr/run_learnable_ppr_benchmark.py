"""
Learnable PPR subgraph benchmark runner.
Phase 1: Architecture search to learn per-edge (teleport_u, teleport_v).
Phase 2: Fine-tune GNN on subgraphs extracted with learned configs.
"""

import torch
import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

from ..utils.models import GCN, SAGE, GAT, LinkPredictor
from ..benchmark.data_prep import prepare_link_prediction_data
from .multi_scale_ppr import MultiScalePPR
from .autolink_ppr import AutoLinkPPR
from .search_net import PPRSearchNet
from .searcher import ArchitectureSearcher
from .finetuner import finetune_on_subgraphs
from .evaluator import (evaluate_learnable_ppr, print_evaluation_results)


DEFAULT_CONFIG = {
    'feature_method': 'random',
    'feature_dim': 128,
    'hidden_channels': 256,
    'num_layers': 3,
    'dropout': 0.3,
    # Phase 1: Architecture search
    'search_epochs': 50,
    'search_batch_size': 1024,
    'search_lr': 0.01,
    'search_lr_arch': 0.01,
    'search_lr_min': 0.001,
    'search_patience': 50,
    'arch_hidden': 256,
    'arch_layers': 3,
    'temperature': 0.07,
    # Phase 2: Fine-tuning
    'finetune_epochs': 500,
    'finetune_batch_size': 8192,
    'finetune_lr': 0.005,
    'finetune_patience': 30,
    'finetune_eval_steps': 5,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    # PPR
    'teleport_values': [0.50, 0.85, 0.95],
    'alpha': [0.5],
    'top_k': 100,
}


def create_encoder(encoder_type, in_channels, hidden_channels, num_layers,
                   dropout):
    if encoder_type == 'GCN':
        return GCN(in_channels, hidden_channels, hidden_channels,
                   num_layers, dropout)
    elif encoder_type == 'SAGE':
        return SAGE(in_channels, hidden_channels, hidden_channels,
                    num_layers, dropout)
    elif encoder_type == 'GAT':
        return GAT(in_channels, hidden_channels, hidden_channels,
                   num_layers, dropout)
    raise ValueError(f"Unknown encoder type: {encoder_type}")


def run_single_experiment(dataset_name, dataset_path, encoder_type, config,
                          device='cuda'):
    """Run full learnable PPR experiment for one dataset + encoder."""
    print(f"\n{'=' * 80}")
    print(f"Learnable PPR: {dataset_name} | {encoder_type}")
    print(f"Teleport values: {config['teleport_values']}")
    print(f"{'=' * 80}")

    # ── Data ──
    print(f"\nLoading dataset: {dataset_path}")
    dataset_dict = prepare_link_prediction_data(
        dataset_path,
        feature_method=config['feature_method'],
        feature_dim=config['feature_dim'])

    data = dataset_dict['data']
    split_edge = dataset_dict['split_edge']

    print(f"  Nodes: {data.num_nodes}")
    print(f"  Train edges: {split_edge['train']['source_node'].size(0)}")

    # ── Multi-Scale PPR ──
    print(f"\nLoading multi-scale PPR...")
    multi_scale_ppr = MultiScalePPR(
        dataset_name, data,
        teleport_values=config['teleport_values'])
    print(f"  {multi_scale_ppr}")

    num_configs = multi_scale_ppr.num_configs

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Architecture Search
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print(f"Phase 1: Architecture Search ({config['search_epochs']} epochs)")
    print(f"{'─' * 60}")

    model = AutoLinkPPR(
        in_channels=config['feature_dim'],
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        gnn_type=encoder_type,
        num_configs=num_configs,
    )

    arch_net = PPRSearchNet(
        in_channels=config['hidden_channels'],
        hidden_channels=config['arch_hidden'],
        num_layers=config['arch_layers'],
        temperature=config['temperature'],
    )

    num_params_search = (sum(p.numel() for p in model.parameters()) +
                         sum(p.numel() for p in arch_net.parameters()))
    print(f"  Search model params: {num_params_search:,}")

    searcher = ArchitectureSearcher(
        model, arch_net, multi_scale_ppr, data, split_edge,
        device=device,
        lr=config['search_lr'],
        lr_arch=config['search_lr_arch'],
        lr_min=config['search_lr_min'],
    )

    search_history = searcher.search(
        epochs=config['search_epochs'],
        batch_size=config['search_batch_size'],
        patience=config['search_patience'],
        verbose=True,
    )

    # Get learned configs for all splits
    print("\nExtracting learned per-edge configurations...")
    train_configs, train_counts = searcher.get_edge_configs('train')
    val_configs, val_counts = searcher.get_edge_configs('valid')
    test_configs, test_counts = searcher.get_edge_configs('test')

    print(f"  Train config distribution: {train_counts.tolist()}")
    print(f"  Val config distribution:   {val_counts.tolist()}")
    print(f"  Test config distribution:  {test_counts.tolist()}")
    for i, (tu, tv) in enumerate(multi_scale_ppr.config_labels):
        print(f"    Config {i}: teleport_u={tu}, teleport_v={tv} "
              f"-> train={train_counts[i].item()}, "
              f"test={test_counts[i].item()}")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Fine-tune on Extracted Subgraphs
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print(f"Phase 2: Fine-tune on Subgraphs "
          f"(top_k={config['top_k']}, {config['finetune_epochs']} epochs)")
    print(f"{'─' * 60}")

    ft_encoder = create_encoder(
        encoder_type,
        in_channels=config['feature_dim'],
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    )
    ft_predictor = LinkPredictor(
        config['hidden_channels'], config['hidden_channels'], 1,
        config['num_layers'], config['dropout'])

    num_params_ft = (sum(p.numel() for p in ft_encoder.parameters()) +
                     sum(p.numel() for p in ft_predictor.parameters()))
    print(f"  Fine-tune model params: {num_params_ft:,}")

    ft_history = finetune_on_subgraphs(
        ft_encoder, ft_predictor, data, split_edge,
        multi_scale_ppr, train_configs,
        alpha=config['alpha'],
        top_k=config['top_k'],
        epochs=config['finetune_epochs'],
        batch_size=config['finetune_batch_size'],
        lr=config['finetune_lr'],
        eval_steps=config['finetune_eval_steps'],
        device=device,
        verbose=True,
        patience=config['finetune_patience'],
        weight_decay=config['weight_decay'],
        grad_clip=config['grad_clip'],
    )

    # ── Test Evaluation ──
    print(f"\nEvaluating on test set...")
    test_results = evaluate_learnable_ppr(
        ft_encoder, ft_predictor, data, split_edge,
        multi_scale_ppr, test_configs,
        split='test', alpha=config['alpha'], top_k=config['top_k'],
        device=device)
    print_evaluation_results(test_results, 'test')

    # ── Save Results ──
    save_dir = f"results/benchmark-learnable-ppr/{dataset_name}/{encoder_type}"
    os.makedirs(save_dir, exist_ok=True)

    result = {
        'dataset': dataset_name,
        'encoder': encoder_type,
        'num_params_search': num_params_search,
        'num_params_finetune': num_params_ft,
        'teleport_values': config['teleport_values'],
        'num_configs': num_configs,
        'top_k': config['top_k'],
        'alpha': config['alpha'],
        'search_time': search_history['total_time'],
        'finetune_time': ft_history.get('total_time', 0),
        'test_results': {k: float(v) for k, v in test_results.items()},
        'search_history': {
            'best_epoch': search_history['best_epoch'],
            'best_val_loss': search_history['best_val_loss'],
        },
        'finetune_history': {
            'best_epoch': ft_history['best_epoch'],
            'best_val_mrr': ft_history['best_val_mrr'],
            'stopped_early': ft_history['stopped_early'],
        },
        'config_distribution': {
            'train': train_counts.tolist(),
            'val': val_counts.tolist(),
            'test': test_counts.tolist(),
            'labels': [f"({tu},{tv})"
                       for tu, tv in multi_scale_ppr.config_labels],
        },
    }

    with open(os.path.join(save_dir, 'full_results.json'), 'w') as f:
        json.dump(result, f, indent=2)

    _save_summary(save_dir, result)

    print(f"\n  Results saved to: {save_dir}")
    print(f"  Test MRR: {test_results['mrr']:.4f}")

    return result


def _save_summary(save_dir, result):
    """Write human-readable results summary."""
    path = os.path.join(save_dir, 'results_summary.txt')
    with open(path, 'w') as f:
        f.write("Learnable PPR Subgraph Benchmark Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset:  {result['dataset']}\n")
        f.write(f"Encoder:  {result['encoder']}\n")
        f.write(f"Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"PPR Configuration:\n")
        f.write(f"  Teleport values: {result['teleport_values']}\n")
        f.write(f"  Num configs:     {result['num_configs']}\n")
        f.write(f"  Alpha (fixed):   {result['alpha']}\n")
        f.write(f"  Top-K:           {result['top_k']}\n\n")

        f.write(f"Timing:\n")
        f.write(f"  Search time:     {result['search_time']:.1f}s\n")
        f.write(f"  Fine-tune time:  {result['finetune_time']:.1f}s\n")
        f.write(f"  Total:           "
                f"{result['search_time'] + result['finetune_time']:.1f}s\n\n")

        f.write(f"Test Results:\n")
        for k, v in result['test_results'].items():
            f.write(f"  {k:<10}: {v:.6f}\n")

        f.write(f"\nConfig Distribution (train):\n")
        labels = result['config_distribution']['labels']
        counts = result['config_distribution']['train']
        total = sum(counts)
        for label, count in zip(labels, counts):
            pct = 100 * count / max(total, 1)
            f.write(f"  {label}: {count:>6} ({pct:5.1f}%)\n")


def run_learnable_ppr_benchmark(config=None):
    """Run full benchmark across datasets and encoders."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    dataset_paths = {
        'FB15K237': 'data/FB15K237/train.txt',
        'WN18RR': 'data/WN18RR/train.txt',
        'NELL-995': 'data/NELL-995/train.txt',
    }

    dataset_names = config.get('datasets', ['FB15K237', 'WN18RR', 'NELL-995'])
    datasets = {n: dataset_paths[n] for n in dataset_names
                if n in dataset_paths}
    encoders = config.get('encoders', ['GCN', 'SAGE', 'GAT'])
    device = config.get('device', 'cuda')

    all_results = []
    for dataset_name, path in datasets.items():
        for enc in encoders:
            try:
                result = run_single_experiment(
                    dataset_name, path, enc, config, device)
                all_results.append(result)
            except Exception as e:
                print(f"\nError in {dataset_name}_{enc}: {e}")
                import traceback
                traceback.print_exc()

    # Aggregated results
    agg_path = 'results/benchmark-learnable-ppr/all_results.json'
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, 'w') as f:
        json.dump([{
            'dataset': r['dataset'],
            'encoder': r['encoder'],
            'test_mrr': r['test_results']['mrr'],
            'test_auc': r['test_results']['auc'],
            'test_ap': r['test_results']['ap'],
            'search_time': r['search_time'],
            'finetune_time': r['finetune_time'],
        } for r in all_results], f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Learnable PPR Benchmark Complete!")
    print(f"  Experiments: {len(all_results)}")
    print(f"{'=' * 80}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Learnable PPR Subgraph Benchmark')
    parser.add_argument('--datasets', nargs='+',
                        default=['FB15K237', 'WN18RR', 'NELL-995'])
    parser.add_argument('--encoders', nargs='+',
                        default=['GCN', 'SAGE', 'GAT'])
    parser.add_argument('--teleport_values', nargs='+', type=float,
                        default=[0.50, 0.85, 0.95],
                        help='Teleport probabilities for PPR search space')
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5],
                        help='PPR combination weights (1 or 2 values)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--search_epochs', type=int, default=50)
    parser.add_argument('--search_batch_size', type=int, default=1024)
    parser.add_argument('--finetune_epochs', type=int, default=500)
    parser.add_argument('--finetune_batch_size', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.07)

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        'datasets': args.datasets,
        'encoders': args.encoders,
        'teleport_values': sorted(args.teleport_values),
        'top_k': args.top_k,
        'alpha': args.alpha,
        'device': args.device,
        'search_epochs': args.search_epochs,
        'search_batch_size': args.search_batch_size,
        'finetune_epochs': args.finetune_epochs,
        'finetune_batch_size': args.finetune_batch_size,
        'temperature': args.temperature,
    })

    run_learnable_ppr_benchmark(config)


if __name__ == '__main__':
    main()
