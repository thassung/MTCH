"""
Linux-nano-friendly CLI for the baseline runners (Full Graph / Static PPR / Static k-hop).

Mirrors the same helpers as `benchmark_runner_planetoid.ipynb`.

Usage:
    python scripts/run_baselines.py --datasets Cora --methods full-graph
    python scripts/run_baselines.py --datasets Cora CiteSeer PubMed \
        --methods full-graph static-ppr static-khop --encoders GCN SAGE GAT
    python scripts/run_baselines.py --datasets Cora --methods full-graph --smoke

Output JSONs:
    Full Graph:  results/benchmark/{dataset}/{encoder}/full_results.json
    Static PPR:  results/benchmark-ppr/{dataset}/{encoder}_a{a}_e{e}/full_results.json
    Static k-hop: results/benchmark-khop/{dataset}/{encoder}_k{k}/full_results.json

Wall-clock estimates (rough, GPU): Cora <10 min/method, PubMed:
  full-graph ~10-30 min, static-ppr ~2-3 h (incl. neg-cache build, K=3),
  static-khop ~30-60 min.
"""

import argparse
import os
import sys
import time
import json
import random
import gc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Defaults match benchmark_runner_planetoid.ipynb.
HIDDEN_CHANNELS = 256
NUM_LAYERS = 3
DROPOUT = 0.3

FULL_EPOCHS = 500
FULL_BATCH_SIZE = 8192
FULL_LR = 0.005
FULL_PATIENCE = 30

SUB_EPOCHS = 500
SUB_BATCH_SIZE = 1024
SUB_LR = 0.005
SUB_PATIENCE = 30
EVAL_STEPS = 5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

ENCODER_LR_OVERRIDE = {'GAT': 0.001}
ENCODER_CLIP_OVERRIDE = {'GAT': 0.5}

PPR_ALPHAS = [0.85]
PPR_EPSILONS = [1e-3]
PPR_WINDOW = 10
PPR_NUM_NEGS = 3       # K=5 -> K=3 (Yang et al. KDD'20: variance ~ 1/K, +25% noise within seed std on Planetoid; ~40% wall-clock cut)
PPR_STRUCT_DIM = 2
PPR_EPSILON_PRECOMP = 1e-4

K_VALUES = [2]


def parse_args():
    p = argparse.ArgumentParser(
        description='Baseline runner: Full Graph / Static PPR / Static k-hop')
    p.add_argument('--datasets', nargs='+', required=True,
                   choices=['Cora', 'CiteSeer', 'PubMed'])
    p.add_argument('--methods', nargs='+', required=True,
                   choices=['full-graph', 'static-ppr', 'static-khop'])
    p.add_argument('--encoders', nargs='+', default=['GCN', 'SAGE', 'GAT'],
                   choices=['GCN', 'SAGE', 'GAT'])
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true',
                   help='Tiny smoke test: 3 epochs, patience 3.')
    return p.parse_args()


def make_encoder(enc_type, in_ch, hid=HIDDEN_CHANNELS,
                 n_layers=NUM_LAYERS, dropout=DROPOUT):
    from subgrapher.utils.models import GCN, SAGE, GAT
    if enc_type == 'GCN':  return GCN(in_ch, hid, hid, n_layers, dropout)
    if enc_type == 'SAGE': return SAGE(in_ch, hid, hid, n_layers, dropout)
    if enc_type == 'GAT':  return GAT(in_ch, hid, hid, n_layers, dropout, heads=4)
    raise ValueError(enc_type)


def make_predictor(hid=HIDDEN_CHANNELS, n_layers=NUM_LAYERS, dropout=DROPOUT):
    from subgrapher.utils.models import LinkPredictor
    return LinkPredictor(hid, hid, 1, n_layers, dropout)


def enc_lr(enc_type, base=SUB_LR):
    return ENCODER_LR_OVERRIDE.get(enc_type, base)


def enc_clip(enc_type, base=GRAD_CLIP):
    return ENCODER_CLIP_OVERRIDE.get(enc_type, base)


def ensure_train_only_edges(data, dd):
    if not getattr(data, '_edge_index_train_only', False):
        data._orig_edge_index = data.edge_index
        data.edge_index = dd['train_edge_index']
        data._edge_index_train_only = True
        print(f'  [train-only edges] {data._orig_edge_index.size(1):,} -> '
              f'{data.edge_index.size(1):,}')
    return data


def save_full_results(base_dir, result_dict):
    run_id = time.strftime('%Y%m%d_%H%M%S')
    result_dict['run_id'] = run_id
    result_dict['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    run_dir = os.path.join(base_dir, 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)
    run_path = os.path.join(run_dir, 'full_results.json')
    with open(run_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f'  -> {run_path}')
    os.makedirs(base_dir, exist_ok=True)
    latest_path = os.path.join(base_dir, 'full_results.json')
    with open(latest_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f'  -> {latest_path} (latest)')


def run_full_graph(ds_name, dd, enc_types, device, args):
    import torch
    from subgrapher.benchmark.trainer import benchmark_model

    full_epochs = 3 if args.smoke else FULL_EPOCHS
    full_pat = 3 if args.smoke else FULL_PATIENCE

    data = dd['data']; split_edge = dd['split_edge']
    in_ch = dd['feature_dim']

    for enc_type in enc_types:
        print(f'\n=== Full Graph: {ds_name} / {enc_type} ===')
        torch.manual_seed(args.seed)
        encoder = make_encoder(enc_type, in_ch=in_ch)
        predictor = make_predictor()
        encoder.reset_parameters(); predictor.reset_parameters()

        result = benchmark_model(
            enc_type, encoder, predictor, data, split_edge,
            epochs=full_epochs, batch_size=FULL_BATCH_SIZE,
            lr=FULL_LR, eval_steps=EVAL_STEPS, device=device,
            patience=full_pat, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP)

        save_full_results(
            f'results/benchmark/{ds_name}/{enc_type}',
            {
                'dataset': ds_name, 'encoder': enc_type,
                'method': 'Full Graph',
                'test_results': {k: float(v) for k, v in result['test_results'].items()},
                'train_time': float(result['train_time']),
                'num_params': int(result.get('num_params', 0)),
                'best_epoch': int(result.get('best_epoch', 0)),
                'stopped_early': bool(result.get('stopped_early', False)),
                'config': {
                    'epochs': full_epochs, 'batch_size': FULL_BATCH_SIZE,
                    'lr': FULL_LR, 'patience': full_pat,
                    'feature_dim': in_ch, 'hidden_channels': HIDDEN_CHANNELS,
                    'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
                },
                'seed': args.seed,
            })

        del encoder, predictor, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_static_ppr(ds_name, dd, enc_types, device, args, ppr_pre):
    import torch
    from subgrapher.benchmark_ppr.ppr_extractor import StaticPPRExtractor
    from subgrapher.benchmark_ppr.trainer_lcilp import train_model_ppr_lcilp
    from subgrapher.benchmark_ppr.evaluator import evaluate_ppr_lcilp
    from subgrapher.benchmark_ppr.graph_classifier import SubgraphClassifier

    sub_epochs = 3 if args.smoke else SUB_EPOCHS
    sub_pat = 3 if args.smoke else SUB_PATIENCE

    data = dd['data']; split_edge = dd['split_edge']

    for ppr_alpha in PPR_ALPHAS:
        for ppr_eps in PPR_EPSILONS:
            ppr_ext = StaticPPRExtractor(
                data, alpha=ppr_alpha, epsilon=ppr_eps, window=PPR_WINDOW)

            for enc_type in enc_types:
                print(f'\n=== Static PPR (a={ppr_alpha}, e={ppr_eps}, K={PPR_NUM_NEGS}): '
                      f'{ds_name} / {enc_type} ===')
                torch.manual_seed(args.seed)
                classifier = SubgraphClassifier(
                    drnl_dim=PPR_STRUCT_DIM, hidden=HIDDEN_CHANNELS,
                    num_layers=NUM_LAYERS, dropout=DROPOUT,
                    encoder_type=enc_type,
                    feature_dim=dd['feature_dim'])
                classifier.reset_parameters()

                cache_dir = f'cache/benchmark-ppr/{ds_name}/a{ppr_alpha}_e{ppr_eps}'
                t0 = time.time()
                hist = train_model_ppr_lcilp(
                    classifier, data, split_edge, ppr_ext,
                    ppr_preprocessor=ppr_pre,
                    epochs=sub_epochs, batch_size=SUB_BATCH_SIZE,
                    lr=enc_lr(enc_type), eval_steps=EVAL_STEPS, device=device,
                    verbose=True, patience=sub_pat,
                    weight_decay=WEIGHT_DECAY, grad_clip=enc_clip(enc_type),
                    num_negs=PPR_NUM_NEGS,
                    edges_per_epoch=None,
                    cache_dir=cache_dir, max_eval_edges=2000)
                train_time = time.time() - t0

                test_res = evaluate_ppr_lcilp(
                    classifier, data, split_edge, ppr_ext,
                    ppr_preprocessor=ppr_pre,
                    split='test', batch_size=SUB_BATCH_SIZE, device=device,
                    max_edges=None, cache_dir=cache_dir)

                save_full_results(
                    f'results/benchmark-ppr/{ds_name}/{enc_type}_a{ppr_alpha}_e{ppr_eps}',
                    {
                        'dataset': ds_name, 'encoder': enc_type,
                        'method': 'Static PPR',
                        'ppr_alpha': ppr_alpha, 'ppr_epsilon': ppr_eps,
                        'ppr_window': PPR_WINDOW, 'num_negs': PPR_NUM_NEGS,
                        'test_results': {k: float(v) for k, v in test_res.items()},
                        'train_time': float(train_time),
                        'best_epoch': int(hist.get('best_epoch', 0)),
                        'stopped_early': bool(hist.get('stopped_early', False)),
                        'config': {
                            'ppr_alpha': ppr_alpha, 'ppr_epsilon': ppr_eps,
                            'ppr_window': PPR_WINDOW,
                            'ppr_struct_dim': PPR_STRUCT_DIM,
                            'num_negs': PPR_NUM_NEGS,
                            'epochs': sub_epochs, 'batch_size': SUB_BATCH_SIZE,
                            'lr': enc_lr(enc_type), 'patience': sub_pat,
                            'hidden_channels': HIDDEN_CHANNELS,
                            'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
                        },
                        'seed': args.seed,
                    })

                del classifier, hist, test_res
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del ppr_ext


def run_static_khop(ds_name, dd, enc_types, device, args):
    import torch
    from subgrapher.benchmark_khop.run_khop_benchmark import load_or_create_khop_preprocessor
    from subgrapher.benchmark_khop.khop_extractor import StaticKHopExtractor
    from subgrapher.benchmark_khop.trainer_batched import train_model_khop_batched
    from subgrapher.benchmark_khop.evaluator import evaluate_khop
    from subgrapher.benchmark.evaluator import evaluate_link_prediction

    sub_epochs = 3 if args.smoke else SUB_EPOCHS
    sub_pat = 3 if args.smoke else SUB_PATIENCE

    data = dd['data']; split_edge = dd['split_edge']
    in_ch = dd['feature_dim']

    for k in K_VALUES:
        khop_pre = load_or_create_khop_preprocessor(ds_name, data, k)
        khop_ext = StaticKHopExtractor(data, k=k, preprocessor=khop_pre)

        for enc_type in enc_types:
            print(f'\n=== Static k-hop (k={k}): {ds_name} / {enc_type} ===')
            torch.manual_seed(args.seed)
            encoder = make_encoder(enc_type, in_ch=in_ch)
            predictor = make_predictor()
            encoder.reset_parameters(); predictor.reset_parameters()

            cache_dir = f'cache/benchmark-khop/{ds_name}/k{k}'
            t0 = time.time()
            hist = train_model_khop_batched(
                encoder, predictor, data, split_edge, khop_ext,
                epochs=sub_epochs, batch_size=SUB_BATCH_SIZE,
                lr=enc_lr(enc_type), eval_steps=EVAL_STEPS, device=device,
                verbose=True, patience=sub_pat,
                weight_decay=WEIGHT_DECAY, grad_clip=enc_clip(enc_type),
                edges_per_epoch=None, cache_dir=cache_dir)
            train_time = time.time() - t0

            test_res = evaluate_khop(
                encoder, predictor, data, split_edge, khop_ext,
                split='test', batch_size=SUB_BATCH_SIZE, device=device,
                max_edges=None, cache_dir=cache_dir)
            fg_test_res = evaluate_link_prediction(
                encoder, predictor, data, split_edge,
                split='test', batch_size=65536)

            save_full_results(
                f'results/benchmark-khop/{ds_name}/{enc_type}_k{k}',
                {
                    'dataset': ds_name, 'encoder': enc_type, 'k': k,
                    'method': 'Static k-hop',
                    'test_results': {kk: float(v) for kk, v in fg_test_res.items()},
                    'subgraph_test_results': {kk: float(v) for kk, v in test_res.items()},
                    'train_time': float(train_time),
                    'best_epoch': int(hist.get('best_epoch', 0)),
                    'stopped_early': bool(hist.get('stopped_early', False)),
                    'config': {
                        'k': k, 'epochs': sub_epochs,
                        'batch_size': SUB_BATCH_SIZE,
                        'lr': enc_lr(enc_type), 'patience': sub_pat,
                        'feature_dim': in_ch,
                        'hidden_channels': HIDDEN_CHANNELS,
                        'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
                    },
                    'seed': args.seed,
                })

            del encoder, predictor, hist, test_res, fg_test_res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del khop_ext, khop_pre


def main():
    import torch
    import numpy as np

    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[run_baselines] device={device} datasets={args.datasets} '
          f'methods={args.methods} encoders={args.encoders} '
          f'smoke={args.smoke}')

    from subgrapher.benchmark.data_prep import prepare_planetoid_data

    # Load all datasets up front (fast — Planetoid data download cached)
    datasets = {}
    for ds_name in args.datasets:
        print(f'\nLoading {ds_name}...')
        dd = prepare_planetoid_data(ds_name)
        datasets[ds_name] = dd
        ensure_train_only_edges(dd['data'], dd)

    # Static PPR needs the PPR preprocessor (one-time per dataset)
    ppr_preprocessors = {}
    if 'static-ppr' in args.methods:
        from subgrapher.utils.ppr_preprocessor import PPRPreprocessor
        for ds_name in args.datasets:
            dd = datasets[ds_name]
            ppr_path = (f'cache/benchmark-ppr/{ds_name}/'
                        f'ppr_alpha{PPR_ALPHAS[0]}_eps{PPR_EPSILON_PRECOMP}.pt')
            if os.path.isfile(ppr_path):
                print(f'[PPR] Loading {ds_name} from {ppr_path}')
                ppr_preprocessors[ds_name] = PPRPreprocessor.load(
                    ppr_path, dd['data'])
            else:
                print(f'[PPR] Building {ds_name} (one-time cost)...')
                ppr_pre = PPRPreprocessor(dd['data'],
                                          ppr_alpha=PPR_ALPHAS[0],
                                          epsilon=PPR_EPSILON_PRECOMP)
                os.makedirs(os.path.dirname(ppr_path), exist_ok=True)
                ppr_pre.save(ppr_path)
                ppr_preprocessors[ds_name] = ppr_pre

    # Run each (dataset, method) combo
    for ds_name in args.datasets:
        dd = datasets[ds_name]
        for method in args.methods:
            if method == 'full-graph':
                run_full_graph(ds_name, dd, args.encoders, device, args)
            elif method == 'static-ppr':
                run_static_ppr(ds_name, dd, args.encoders, device, args,
                               ppr_preprocessors[ds_name])
            elif method == 'static-khop':
                run_static_khop(ds_name, dd, args.encoders, device, args)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print('\nAll baseline runs complete.')


if __name__ == '__main__':
    main()
