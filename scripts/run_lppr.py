"""
Linux-nano-friendly CLI for the Learnable PPR (LPPR / LPPR) framework.

Same code path as `learnable_ppr_planetoid.ipynb` — both call
`subgrapher.benchmark_learnable_ppr.run_option_a_benchmark.run_one`.

Usage:
    python scripts/run_lppr.py --dataset Cora
    python scripts/run_lppr.py --dataset PubMed --search-epochs 30 --finetune-epochs 100
    python scripts/run_lppr.py --dataset Cora --smoke
    python scripts/run_lppr.py --dataset Cora --no-full-graph-eval

Output: results/benchmark-option-a/{dataset}/full_results.json
        + a timestamped copy under results/benchmark-option-a/{dataset}/runs/{run_id}/

Wall-clock estimates (rough, GPU): Cora ~5-15 min, CiteSeer ~5-15 min,
PubMed ~30-90 min depending on full-graph-eval, --search-epochs, etc.
"""

import argparse
import sys
import os

# Make the repo root importable when invoked from any CWD
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from subgrapher.benchmark_learnable_ppr.run_option_a_benchmark import (
    DEFAULT_CONFIG, run_one)


def parse_args():
    p = argparse.ArgumentParser(
        description='Learnable PPR (LPPR) — single-dataset runner')
    p.add_argument('--dataset', required=True,
                   choices=['Cora', 'CiteSeer', 'PubMed'],
                   help='Planetoid dataset')
    p.add_argument('--dataset-path', default='data/Planetoid',
                   help='Planetoid root (downloads here on first use)')
    p.add_argument('--device', default=None,
                   help='cuda | cpu | None (auto-detect)')
    # Encoder backbone (default GCN — matches the GCN baseline cell)
    p.add_argument('--encoder', default=DEFAULT_CONFIG['encoder_type'],
                   choices=['GCN', 'SAGE', 'GAT', 'PPRDiff'],
                   help='Backbone encoder operating on P_soft. GCN/SAGE/GAT '
                        'are PS2-style apples-to-apples vs the same-named '
                        'baseline; PPRDiff is the original ablation.')
    # Training mode (default auto: joint for ≤10k nodes, bilevel for PubMed)
    p.add_argument('--train-mode', default=DEFAULT_CONFIG['train_mode'],
                   choices=['auto', 'joint', 'bilevel'],
                   help='joint = single-phase, recommended for ≤10k nodes; '
                        'bilevel = original PS2 θ-on-val/w-on-train; '
                        'auto picks joint for ≤10k nodes else bilevel.')
    # Joint training knobs
    p.add_argument('--joint-epochs', type=int,
                   default=DEFAULT_CONFIG['joint_epochs'])
    p.add_argument('--joint-patience', type=int,
                   default=DEFAULT_CONFIG['joint_patience'])
    p.add_argument('--entropy-coeff', type=float,
                   default=DEFAULT_CONFIG['joint_entropy_coeff_start'],
                   help='Initial entropy bonus coefficient for joint training')
    # Bi-level knobs (only used when --train-mode=bilevel)
    p.add_argument('--search-epochs', type=int,
                   default=DEFAULT_CONFIG['search_epochs'])
    p.add_argument('--finetune-epochs', type=int,
                   default=DEFAULT_CONFIG['finetune_epochs'])
    # Architecture
    p.add_argument('--hidden', type=int,
                   default=DEFAULT_CONFIG['hidden_channels'])
    p.add_argument('--num-layers', type=int,
                   default=DEFAULT_CONFIG['num_layers'])
    p.add_argument('--gat-heads', type=int,
                   default=DEFAULT_CONFIG['gat_heads'])
    # PPR / extraction
    p.add_argument('--push-epsilon', type=float,
                   default=DEFAULT_CONFIG['push_epsilon'],
                   help='PPR push precision (5e-4 default = coarse-coarse)')
    p.add_argument('--score-tau', type=float,
                   default=DEFAULT_CONFIG['score_tau'])
    p.add_argument('--seed', type=int, default=42)
    # Eval
    p.add_argument('--no-full-graph-eval', action='store_true',
                   help='Skip the full-graph 1000-neg eval pass')
    p.add_argument('--full-graph-eval-max-nodes', type=int,
                   default=DEFAULT_CONFIG['full_graph_eval_max_nodes'])
    # I/O
    p.add_argument('--no-save', action='store_true',
                   help='Do not write full_results.json')
    p.add_argument('--smoke', action='store_true',
                   help='Tiny smoke test: short epochs, small batch. '
                        'Verifies the pipeline runs end-to-end on this machine.')
    p.add_argument('--cache-root', default=DEFAULT_CONFIG['cache_root'])
    p.add_argument('--results-root', default=DEFAULT_CONFIG['results_root'])
    return p.parse_args()


def main():
    import torch
    import numpy as np
    import random

    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    overrides = {
        'encoder_type': args.encoder,
        'gat_heads': args.gat_heads,
        'train_mode': args.train_mode,
        'joint_epochs': args.joint_epochs,
        'joint_patience': args.joint_patience,
        'joint_entropy_coeff_start': args.entropy_coeff,
        'search_epochs': args.search_epochs,
        'finetune_epochs': args.finetune_epochs,
        'hidden_channels': args.hidden,
        'num_layers': args.num_layers,
        'push_epsilon': args.push_epsilon,
        'score_tau': args.score_tau,
        'seed': args.seed,
        'full_graph_eval': not args.no_full_graph_eval,
        'full_graph_eval_max_nodes': args.full_graph_eval_max_nodes,
        'save_results': not args.no_save,
        'cache_root': args.cache_root,
        'results_root': args.results_root,
    }

    if args.smoke:
        overrides.update({
            'joint_epochs': 4,
            'joint_patience': 4,
            'joint_eval_steps': 1,
            'search_epochs': 2,
            'finetune_epochs': 4,
            'edges_per_search_epoch': 512,
            'finetune_patience': 4,
            'search_val_every': 1,
            'finetune_eval_steps': 1,
            'save_results': False,
        })
        print('[SMOKE] Reduced epochs/edges; not writing JSON.')

    print(f'[run_lppr] dataset={args.dataset} encoder={args.encoder} '
          f'train_mode={args.train_mode} device={args.device or "auto"} '
          f'seed={args.seed} push_epsilon={overrides["push_epsilon"]}')

    result = run_one(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        device=args.device,
        **overrides)

    # Print headline
    tr = result.get('test_results') or {}
    print('\n' + '=' * 60)
    print(f'LPPR {args.dataset} — eval_mode={result.get("eval_mode")}')
    print('=' * 60)
    for k in ['mrr', 'auc', 'ap', 'hits@1', 'hits@3', 'hits@10', 'hits@50', 'hits@100']:
        if k in tr:
            print(f'  {k:>10s} = {tr[k]:.4f}')
    if result.get('full_graph_test_results') and result.get('subgraph_test_results'):
        sg = result['subgraph_test_results']
        print(f'  (per-subgraph reference: MRR={sg["mrr"]:.4f}  '
              f'AUC={sg["auc"]:.4f}  AP={sg["ap"]:.4f})')
    print('=' * 60)


if __name__ == '__main__':
    main()
