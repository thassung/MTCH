"""Re-run the missing PubMed/GAT k=2 baseline cell.

Mirrors `benchmark_runner_planetoid.ipynb` cell 17 for one (dataset, encoder, k)
triple. Writes `results/benchmark-khop/PubMed/GAT_k2/full_results.json`.
"""
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

SEED = 42
DATASET = 'PubMed'
ENCODER = 'GAT'
K = 2

HIDDEN_CHANNELS = 256
NUM_LAYERS = 3
DROPOUT = 0.3

SUB_EPOCHS = 500
SUB_BATCH_SIZE = 1024
SUB_LR = 0.001  # GAT override from notebook
SUB_PATIENCE = 30
EDGES_PER_EPOCH = None
EVAL_STEPS = 5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 0.5  # GAT override from notebook

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

import numpy as np
import random
import json

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from subgrapher.benchmark.data_prep import prepare_planetoid_data
from subgrapher.utils.models import GAT, LinkPredictor
from subgrapher.benchmark_khop.run_khop_benchmark import load_or_create_khop_preprocessor
from subgrapher.benchmark_khop.khop_extractor import StaticKHopExtractor
from subgrapher.benchmark_khop.trainer_batched import train_model_khop_batched
from subgrapher.benchmark_khop.evaluator import evaluate_khop
from subgrapher.benchmark.evaluator import evaluate_link_prediction


print(f'Loading {DATASET}...')
dd = prepare_planetoid_data(DATASET)
data = dd['data']
split_edge = dd['split_edge']
in_ch = dd['feature_dim']

# train-only edges, mirroring notebook's ensure_train_only_edges
data._orig_edge_index = data.edge_index
data.edge_index = dd['train_edge_index']
data._edge_index_train_only = True
print(f'  train-only edges: {data.edge_index.size(1):,}')

khop_pre = load_or_create_khop_preprocessor(DATASET, data, K)
khop_ext = StaticKHopExtractor(data, k=K, preprocessor=khop_pre)

torch.manual_seed(SEED)
encoder = GAT(in_ch, HIDDEN_CHANNELS, HIDDEN_CHANNELS, NUM_LAYERS, DROPOUT, heads=4)
predictor = LinkPredictor(HIDDEN_CHANNELS, HIDDEN_CHANNELS, 1, NUM_LAYERS, DROPOUT)
encoder.reset_parameters()
predictor.reset_parameters()

cache_dir = f'cache/benchmark-khop/{DATASET}/k{K}'
os.makedirs(cache_dir, exist_ok=True)

print(f'\n=== Static k-hop (k={K}): {DATASET} / {ENCODER} ===')
t0 = time.time()
hist = train_model_khop_batched(
    encoder, predictor, data, split_edge, khop_ext,
    epochs=SUB_EPOCHS, batch_size=SUB_BATCH_SIZE,
    lr=SUB_LR, eval_steps=EVAL_STEPS, device=DEVICE,
    verbose=True, patience=SUB_PATIENCE,
    weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,
    edges_per_epoch=EDGES_PER_EPOCH,
    cache_dir=cache_dir)
train_time = time.time() - t0

test_res = evaluate_khop(
    encoder, predictor, data, split_edge, khop_ext,
    split='test', batch_size=SUB_BATCH_SIZE, device=DEVICE,
    max_edges=None, cache_dir=cache_dir)
fg_test_res = evaluate_link_prediction(
    encoder, predictor, data, split_edge,
    split='test', batch_size=65536)


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


save_full_results(
    f'results/benchmark-khop/{DATASET}/{ENCODER}_k{K}',
    {
        'dataset': DATASET, 'encoder': ENCODER, 'k': K,
        'test_results': {k_: float(v) for k_, v in fg_test_res.items()},
        'subgraph_test_results': {k_: float(v) for k_, v in test_res.items()},
        'train_time': train_time,
        'best_epoch': hist.get('best_epoch', 0),
        'stopped_early': hist.get('stopped_early', False),
        'config': {
            'k': K, 'epochs': SUB_EPOCHS, 'batch_size': SUB_BATCH_SIZE,
            'lr': SUB_LR, 'patience': SUB_PATIENCE,
            'feature_dim': in_ch, 'hidden_channels': HIDDEN_CHANNELS,
            'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
        },
        'seed': SEED,
    })

print(f'\nDone. train_time={train_time:.1f}s '
      f'fg_mrr={fg_test_res["mrr"]:.4f} sub_mrr={test_res["mrr"]:.4f}')
