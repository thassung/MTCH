"""
Static PPR-based subgraph benchmark runner.
Uses LCILP-style approximate PPR + conductance sweep cut.
Grid search over alpha and epsilon values.
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
from .ppr_extractor import StaticPPRExtractor
from .trainer_batched import train_model_ppr_batched
from .evaluator import evaluate_ppr, print_evaluation_results


DEFAULT_CONFIG = {
    'feature_method': 'random',
    'feature_dim': 128,
    'hidden_channels': 256,
    'num_layers': 3,
    'dropout': 0.3,
    'epochs': 500,
    'batch_size': 8192,
    'lr': 0.005,
    'eval_steps': 5,
    'patience': 30,
    'weight_decay': 1e-5,
    'lr_scheduler': 'reduce_on_plateau',
    'grad_clip': 1.0,
    'ppr_alphas': [0.80, 0.85, 0.90],
    'ppr_epsilons': [1e-2, 1e-3, 1e-4],
    'ppr_window': 10,
}


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
    """Save subgraph node arrays for visualization."""
    os.makedirs(save_dir, exist_ok=True)

    source = split_edge[split]['source_node']
    target = split_edge[split]['target_node']
    num_edges = min(num_samples, source.size(0))

    visualization_data = []
    for i in range(num_edges):
        u = source[i].item()
        v = target[i].item()
        subgraph_data, selected_nodes, metadata = ppr_extractor.extract_subgraph(u, v)
        visualization_data.append({
            'edge': [u, v],
            'selected_nodes': selected_nodes.cpu().numpy().tolist(),
            'num_nodes': len(selected_nodes),
            'metadata': {
                'u_subgraph': metadata['u_subgraph'],
                'v_subgraph': metadata['v_subgraph'],
                'num_edges_subgraph': metadata['num_edges_subgraph'],
            }
        })

    viz_path = os.path.join(save_dir, 'subgraph_visualization.json')
    with open(viz_path, 'w') as f:
        json.dump(visualization_data, f, indent=2)
    print(f"  Saved visualization data: {viz_path}")


def run_single_experiment(dataset_name, dataset_path, encoder_type,
                          ppr_alpha, ppr_epsilon, config, device='cuda'):
    """Run a single experiment for one dataset, encoder, and (alpha, epsilon) pair."""
    ppr_window = config.get('ppr_window', 10)
    print(f"\n{'='*80}")
    print(f"Experiment: {dataset_name} | {encoder_type} | "
          f"alpha={ppr_alpha}, epsilon={ppr_epsilon}")
    print(f"{'='*80}")

    print(f"\nLoading dataset: {dataset_path}")
    dataset_dict = prepare_link_prediction_data(
        dataset_path,
        feature_method=config['feature_method'],
        feature_dim=config['feature_dim'])

    data = dataset_dict['data']
    split_edge = dataset_dict['split_edge']

    print(f"  Nodes: {data.num_nodes}")
    print(f"  Train edges: {split_edge['train']['source_node'].size(0)}")

    ppr_extractor = StaticPPRExtractor(
        data, alpha=ppr_alpha, epsilon=ppr_epsilon, window=ppr_window)

    encoder = create_encoder(
        encoder_type, config['feature_dim'],
        config['hidden_channels'], config['num_layers'], config['dropout'])
    predictor = LinkPredictor(
        config['hidden_channels'], config['hidden_channels'],
        1, config['num_layers'], config['dropout'])

    num_params = sum(p.numel() for p in encoder.parameters()) + \
                 sum(p.numel() for p in predictor.parameters())
    print(f"  Total parameters: {num_params:,}")

    cache_dir = f'cache/benchmark-ppr/{dataset_name}/a{ppr_alpha}_e{ppr_epsilon}'

    train_start = time.time()
    history = train_model_ppr_batched(
        encoder, predictor, data, split_edge, ppr_extractor,
        epochs=config['epochs'], batch_size=config['batch_size'],
        lr=config['lr'], eval_steps=config['eval_steps'],
        device=device, verbose=True, patience=config['patience'],
        weight_decay=config['weight_decay'],
        lr_scheduler=config['lr_scheduler'],
        grad_clip=config['grad_clip'], cache_dir=cache_dir)
    train_time = time.time() - train_start

    print(f"\nEvaluating on test set...")
    test_results = evaluate_ppr(
        encoder, predictor, data, split_edge, ppr_extractor,
        split='test', batch_size=config['batch_size'],
        device=device, cache_dir=cache_dir)
    print_evaluation_results(test_results, 'test')

    exp_name = f"{encoder_type}_a{ppr_alpha}_e{ppr_epsilon}"
    save_dir = f"results/benchmark-ppr/{dataset_name}/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_visualization_data(ppr_extractor, split_edge, 'test', save_dir)

    result = {
        'dataset': dataset_name,
        'encoder': encoder_type,
        'ppr_alpha': ppr_alpha,
        'ppr_epsilon': ppr_epsilon,
        'ppr_window': ppr_window,
        'num_params': num_params,
        'train_time': train_time,
        'test_results': {k: float(v) for k, v in test_results.items()},
        'history': {
            'best_epoch': history.get('best_epoch', 0),
            'best_val_mrr': history.get('best_val_mrr', 0.0),
            'stopped_early': history.get('stopped_early', False),
            'total_time': history.get('total_time', train_time),
        },
        'config': config,
    }

    results_path = os.path.join(save_dir, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nExperiment completed: {exp_name}")
    print(f"  Test MRR: {test_results['mrr']:.4f}")
    print(f"  Results saved to: {save_dir}")

    return result


def run_ppr_benchmark(config=None):
    """Run full PPR benchmark with grid search over alpha and epsilon."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    dataset_paths = {
        'FB15K237': 'data/FB15K237/train.txt',
        'WN18RR': 'data/WN18RR/train.txt',
        'NELL-995': 'data/NELL-995/train.txt',
    }

    dataset_names = config.get('datasets', ['FB15K237', 'WN18RR', 'NELL-995'])
    datasets = {n: dataset_paths[n] for n in dataset_names if n in dataset_paths}
    encoders = config.get('encoders', ['GCN', 'SAGE', 'GAT'])
    ppr_alphas = config.get('ppr_alphas', [0.80, 0.85, 0.90])
    ppr_epsilons = config.get('ppr_epsilons', [1e-2, 1e-3, 1e-4])
    device = config.get('device', 'cuda')

    all_results = []

    for dataset_name, dataset_path in datasets.items():
        for encoder_type in encoders:
            for alpha in ppr_alphas:
                for eps in ppr_epsilons:
                    try:
                        result = run_single_experiment(
                            dataset_name, dataset_path,
                            encoder_type, alpha, eps, config, device)
                        all_results.append(result)
                    except Exception as e:
                        print(f"\nError in {dataset_name}_{encoder_type}"
                              f"_a{alpha}_e{eps}: {e}")
                        import traceback
                        traceback.print_exc()

    agg_results_path = 'results/benchmark-ppr/all_results.json'
    os.makedirs(os.path.dirname(agg_results_path), exist_ok=True)
    with open(agg_results_path, 'w') as f:
        json.dump([{
            'dataset': r['dataset'],
            'encoder': r['encoder'],
            'ppr_alpha': r['ppr_alpha'],
            'ppr_epsilon': r['ppr_epsilon'],
            'test_mrr': float(r['test_results']['mrr']),
            'test_auc': float(r['test_results']['auc']),
            'test_ap': float(r['test_results']['ap']),
            'train_time': r['train_time'],
        } for r in all_results], f, indent=2)

    print(f"\n{'='*80}")
    print(f"PPR Benchmark Complete!")
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Results saved to: results/benchmark-ppr/")
    print(f"{'='*80}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Static PPR Subgraph Benchmark')
    parser.add_argument('--datasets', nargs='+',
                        default=['FB15K237', 'WN18RR', 'NELL-995'])
    parser.add_argument('--encoders', nargs='+',
                        default=['GCN', 'SAGE', 'GAT'])
    parser.add_argument('--ppr_alphas', type=float, nargs='+',
                        default=[0.80, 0.85, 0.90])
    parser.add_argument('--ppr_epsilons', type=float, nargs='+',
                        default=[1e-2, 1e-3, 1e-4])
    parser.add_argument('--ppr_window', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=8192)

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        'datasets': args.datasets,
        'encoders': args.encoders,
        'ppr_alphas': args.ppr_alphas,
        'ppr_epsilons': args.ppr_epsilons,
        'ppr_window': args.ppr_window,
        'device': args.device,
        'epochs': args.epochs,
        'patience': args.patience,
        'lr': args.lr,
        'batch_size': args.batch_size,
    })
    run_ppr_benchmark(config)


if __name__ == '__main__':
    main()
