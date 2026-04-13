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

from tqdm import tqdm

from ..utils.models import GCN, SAGE, GAT, LinkPredictor
from ..benchmark.data_prep import prepare_link_prediction_data
from .multi_scale_ppr import MultiScalePPR
from .autolink_ppr import AutoLinkPPR
from .search_net import PPRSearchNet
from .searcher import ArchitectureSearcher
from .finetuner import finetune_on_subgraphs
from .evaluator import (evaluate_learnable_ppr, print_evaluation_results)
from .artifacts import save_learnable_ppr_experiment


# Defaults match `learnable_ppr.ipynb` / `_gen_notebook.py`
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
    'finetune_epochs': 200,
    'finetune_batch_size': 8192,
    'finetune_lr': 0.005,
    'finetune_patience': 20,
    'finetune_eval_steps': 5,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    'max_subgraphs_per_forward': 256,
    'edges_per_epoch': 100_000,
    'use_checkpoint_cache': True,
    'save_run_artifacts': True,
    # PPR
    'teleport_values': [0.50, 0.85, 0.95],
    'alpha': [0.5],
    'top_k': 100,
    'datasets': ['FB15K237'],
    'encoders': ['SAGE'],
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
                   num_layers, dropout, heads=4)
    raise ValueError(f"Unknown encoder type: {encoder_type}")


def _alpha_for_json(alpha):
    """Notebook stores a single float; API uses a one-element list."""
    if isinstance(alpha, (list, tuple)):
        if len(alpha) == 1:
            return float(alpha[0])
        return list(alpha)
    return float(alpha)


def run_single_experiment(dataset_name, dataset_path, encoder_type, config,
                          device='cuda'):
    """Run full learnable PPR experiment for one dataset + encoder."""
    tqdm.write(f"\n{'=' * 80}")
    tqdm.write(f"Learnable PPR: {dataset_name} | {encoder_type}")
    tqdm.write(f"Teleport values: {config['teleport_values']}")
    tqdm.write(f"{'=' * 80}")

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

    if config.get('use_checkpoint_cache', True):
        ckpt_dir = (
            f'checkpoints/learnable-ppr/{dataset_name}/{encoder_type}')
        cache_dir = f'cache/learnable-ppr/{dataset_name}/{encoder_type}'
    else:
        ckpt_dir = None
        cache_dir = None

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
        max_subgraphs_per_forward=config.get(
            'max_subgraphs_per_forward', 256),
        checkpoint_dir=ckpt_dir,
        cache_dir=cache_dir,
        edges_per_epoch=config.get('edges_per_epoch'),
    )

    # ── Test Evaluation ──
    print(f"\nEvaluating on test set...")
    test_results = evaluate_learnable_ppr(
        ft_encoder, ft_predictor, data, split_edge,
        multi_scale_ppr, test_configs,
        split='test', alpha=config['alpha'], top_k=config['top_k'],
        device=device,
        cache_dir=cache_dir)
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
        'alpha': _alpha_for_json(config['alpha']),
        'search_time': search_history['total_time'],
        'finetune_time': ft_history.get('total_time', 0),
        'finetune_best_mrr': ft_history['best_val_mrr'],
        'finetune_best_epoch': ft_history['best_epoch'],
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

    if config.get('save_run_artifacts', True):
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(
            save_dir, 'runs', run_id)
        exp_bundle = {
            'search_history': search_history,
            'ft_history': ft_history,
            'test_results': test_results,
            'ft_encoder': ft_encoder,
            'ft_predictor': ft_predictor,
            'model': model,
            'arch_net': arch_net,
            'train_configs': train_configs,
            'val_configs': val_configs,
            'test_configs': test_configs,
            'train_counts': train_counts,
            'val_counts': val_counts,
            'test_counts': test_counts,
        }
        extra_hp = {
            'feature_dim': config['feature_dim'],
            'hidden_channels': config['hidden_channels'],
            'num_layers': config['num_layers'],
            'dropout': config['dropout'],
            'search_epochs': config['search_epochs'],
            'search_batch_size': config['search_batch_size'],
            'search_lr': config['search_lr'],
            'search_patience': config['search_patience'],
            'temperature': config['temperature'],
            'finetune_epochs': config['finetune_epochs'],
            'finetune_batch_size': config['finetune_batch_size'],
            'finetune_lr': config['finetune_lr'],
            'finetune_patience': config['finetune_patience'],
            'device': str(device),
        }
        save_learnable_ppr_experiment(
            run_dir, dataset_name, encoder_type, run_id,
            config['teleport_values'], config['alpha'], config['top_k'],
            exp_bundle, multi_scale_ppr, extra_config=extra_hp)
        print(f"  Run artifacts: {run_dir}")

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
        f.write(f"  Alpha (fixed):   {result['alpha']!r}\n")
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

    dataset_names = config.get('datasets', DEFAULT_CONFIG['datasets'])
    encoders = config.get('encoders', DEFAULT_CONFIG['encoders'])
    device = config.get('device', 'cuda')
    show_run_bar = config.get('progress_experiments', True)

    plan = []
    for name in dataset_names:
        if name not in dataset_paths:
            tqdm.write(f"[Skip] Unknown dataset: {name}")
            continue
        for enc in encoders:
            plan.append((name, dataset_paths[name], enc))

    all_results = []
    run_iter = plan
    if show_run_bar and plan:
        run_iter = tqdm(
            plan, desc='Learnable PPR (dataset×encoder)', unit='run')

    for dataset_name, path, enc in run_iter:
        if show_run_bar and plan:
            run_iter.set_postfix_str(f'{dataset_name} | {enc}', refresh=False)
        try:
            result = run_single_experiment(
                dataset_name, path, enc, config, device)
            all_results.append(result)
        except Exception as e:
            tqdm.write(f"\nError in {dataset_name}_{enc}: {e}")
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
        description='Learnable PPR Subgraph Benchmark '
 '(same pipeline as learnable_ppr.ipynb)')
    parser.add_argument('--datasets', nargs='+',
                        default=DEFAULT_CONFIG['datasets'])
    parser.add_argument('--encoders', nargs='+',
                        default=DEFAULT_CONFIG['encoders'])
    parser.add_argument('--teleport_values', nargs='+', type=float,
                        default=[0.50, 0.85, 0.95],
                        help='Teleport probabilities for PPR search space')
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5],
                        help='PPR combination weights (1 or 2 values)')
    parser.add_argument('--device', type=str, default='auto',
                        help="'cuda', 'cpu', or 'auto'")
    parser.add_argument('--search_epochs', type=int, default=50)
    parser.add_argument('--search_batch_size', type=int, default=1024)
    parser.add_argument('--search_patience', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=200)
    parser.add_argument('--finetune_batch_size', type=int, default=8192)
    parser.add_argument('--finetune_patience', type=int, default=20)
    parser.add_argument('--max_subgraphs_per_forward', type=int, default=256)
    parser.add_argument('--edges_per_epoch', type=int, default=100_000,
                        help='Train edges per epoch (subsample);0 = all')
    parser.add_argument('--no_checkpoint_cache', action='store_true',
                        help='Disable checkpoints/ and cache/ writes')
    parser.add_argument('--no_run_progress', action='store_true',
                        help='Disable tqdm bar for dataset×encoder runs')
    parser.add_argument('--no_run_artifacts', action='store_true',
                        help='Skip results/.../runs/<id>/ JSON + checkpoints + config .pt')
    parser.add_argument('--temperature', type=float, default=0.07)

    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    edges_pe = args.edges_per_epoch if args.edges_per_epoch > 0 else None

    config = DEFAULT_CONFIG.copy()
    config.update({
        'datasets': args.datasets,
        'encoders': args.encoders,
        'teleport_values': sorted(args.teleport_values),
        'top_k': args.top_k,
        'alpha': args.alpha,
        'device': device,
        'search_epochs': args.search_epochs,
        'search_batch_size': args.search_batch_size,
        'search_patience': args.search_patience,
        'finetune_epochs': args.finetune_epochs,
        'finetune_batch_size': args.finetune_batch_size,
        'finetune_patience': args.finetune_patience,
        'max_subgraphs_per_forward': args.max_subgraphs_per_forward,
        'edges_per_epoch': edges_pe,
        'use_checkpoint_cache': not args.no_checkpoint_cache,
        'progress_experiments': not args.no_run_progress,
        'save_run_artifacts': not args.no_run_artifacts,
        'temperature': args.temperature,
    })

    tqdm.write(f'Device: {device}')
    n_cfg = len(config['teleport_values']) ** 2
    tqdm.write(
        f"Search space: {len(config['teleport_values'])}×"
        f"{len(config['teleport_values'])} = {n_cfg} configs")

    run_learnable_ppr_benchmark(config)


if __name__ == '__main__':
    main()
