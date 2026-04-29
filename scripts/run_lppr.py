"""
Linux-nano-friendly CLI for the Learnable PPR (LPPR / Option A) framework.

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
        description='Learnable PPR (Option A) — single-dataset runner')
    p.add_argument('--dataset', required=True,
                   choices=['Cora', 'CiteSeer', 'PubMed'],
                   help='Planetoid dataset')
    p.add_argument('--dataset-path', default='data/Planetoid',
                   help='Planetoid root (downloads here on first use)')
    p.add_argument('--device', default=None,
                   help='cuda | cpu | None (auto-detect)')
    p.add_argument('--search-epochs', type=int,
                   default=DEFAULT_CONFIG['search_epochs'])
    p.add_argument('--finetune-epochs', type=int,
                   default=DEFAULT_CONFIG['finetune_epochs'])
    p.add_argument('--hidden', type=int,
                   default=DEFAULT_CONFIG['hidden_channels'])
    p.add_argument('--num-layers', type=int,
                   default=DEFAULT_CONFIG['num_layers'])
    p.add_argument('--push-epsilon', type=float,
                   default=DEFAULT_CONFIG['push_epsilon'],
                   help='PPR push precision (5e-4 default = coarse-coarse)')
    p.add_argument('--score-tau', type=float,
                   default=DEFAULT_CONFIG['score_tau'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-full-graph-eval', action='store_true',
                   help='Skip the full-graph 1000-neg eval pass')
    p.add_argument('--full-graph-eval-max-nodes', type=int,
                   default=DEFAULT_CONFIG['full_graph_eval_max_nodes'])
    p.add_argument('--no-save', action='store_true',
                   help='Do not write full_results.json')
    p.add_argument('--smoke', action='store_true',
                   help='Tiny smoke test: search=2, finetune=4, edges=512. '
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
            'search_epochs': 2,
            'finetune_epochs': 4,
            'edges_per_search_epoch': 512,
            'finetune_patience': 4,
            'search_val_every': 1,
            'finetune_eval_steps': 1,
            'save_results': False,  # don't pollute results/
        })
        print('[SMOKE] Reduced epochs/edges; not writing JSON.')

    print(f'[run_lppr] dataset={args.dataset} '
          f'device={args.device or "auto"} seed={args.seed}')
    print(f'[run_lppr] search_epochs={overrides["search_epochs"]} '
          f'finetune_epochs={overrides["finetune_epochs"]} '
          f'push_epsilon={overrides["push_epsilon"]}')

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
