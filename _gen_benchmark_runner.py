"""Generates benchmark_runner.ipynb — runs Full Graph, Static PPR, and Static k-hop.

Learnable PPR is NOT run here — it comes from learnable_ppr.ipynb.
Each experiment saves a full_results.json with a common schema so
benchmark_analysis.ipynb can load everything uniformly.

Save layout:
  results/benchmark/{dataset}/{encoder}/full_results.json
  results/benchmark-ppr/{dataset}/{encoder}_topk{k}/full_results.json
  results/benchmark-khop/{dataset}/{encoder}_k{k}/full_results.json
  results/benchmark-learnable-ppr/{dataset}/{encoder}/full_results.json   (from learnable_ppr.ipynb)
"""
import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def md(src):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [src]})

def code(src):
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]})


# ═══════════════════════════════════════════════════════════════════════════════
md(
"# Benchmark Runner\n"
"\n"
"Runs three link-prediction methods and saves `full_results.json` per experiment.\n"
"**Learnable PPR** is run separately via `learnable_ppr.ipynb` — not duplicated here.\n"
"\n"
"| Method | Subgraph | Sweep | Output |\n"
"|--------|----------|-------|--------|\n"
"| Full Graph | None | — | `results/benchmark/{ds}/{enc}/` |\n"
"| Static PPR | Top-k PPR | `PPR_TOPK` | `results/benchmark-ppr/{ds}/{enc}_topk{k}/` |\n"
"| Static k-hop | k-hop | `K_VALUES` | `results/benchmark-khop/{ds}/{enc}_k{k}/` |\n"
"\n"
"After running this + `learnable_ppr.ipynb`, open `benchmark_analysis.ipynb` to compare."
)

# ── Config ────────────────────────────────────────────────────────────────────
code(
"# ═══════════════════════════════════════════════════════════════════════════════\n"
"# CONFIGURATION\n"
"# ═══════════════════════════════════════════════════════════════════════════════\n"
"\n"
"SEED = 42\n"
"DATASETS  = ['FB15K237']           # FB15K237, WN18RR, NELL-995\n"
"ENCODERS  = ['GCN', 'SAGE', 'GAT']\n"
"\n"
"# Shared architecture\n"
"FEATURE_DIM     = 128\n"
"HIDDEN_CHANNELS = 256\n"
"NUM_LAYERS      = 3\n"
"DROPOUT         = 0.3\n"
"\n"
"# Full-graph training\n"
"FULL_EPOCHS     = 500\n"
"FULL_BATCH_SIZE = 4096\n"
"FULL_LR         = 0.005\n"
"FULL_PATIENCE   = 30\n"
"\n"
"# Subgraph-based training (PPR, k-hop)\n"
"SUB_EPOCHS          = 500\n"
"SUB_BATCH_SIZE      = 512       # subgraphs per GNN forward\n"
"SUB_LR              = 0.005\n"
"SUB_PATIENCE        = 30\n"
"EDGES_PER_EPOCH     = 100000    # subsample training edges (None = all)\n"
"EVAL_STEPS          = 5\n"
"WEIGHT_DECAY        = 1e-5\n"
"GRAD_CLIP           = 1.0\n"
"\n"
"# Static PPR sweep\n"
"PPR_ALPHA     = 0.85\n"
"PPR_TOPK      = [50, 100, 200, 300, 500]\n"
"ALPHA_COMBINE = 0.5\n"
"\n"
"# Static k-hop sweep\n"
"K_VALUES = [1, 2, 3]\n"
"\n"
"import torch\n"
"DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
"print(f'Device: {DEVICE}  |  Datasets: {DATASETS}  |  Encoders: {ENCODERS}')"
)

# ── Imports + seed ────────────────────────────────────────────────────────────
code(
"import numpy as np, random, json, os, time\n"
"\n"
"torch.manual_seed(SEED)\n"
"torch.cuda.manual_seed_all(SEED)\n"
"np.random.seed(SEED)\n"
"random.seed(SEED)\n"
"torch.backends.cudnn.deterministic = True\n"
"torch.backends.cudnn.benchmark = False\n"
"\n"
"from subgrapher.benchmark.data_prep import prepare_link_prediction_data\n"
"from subgrapher.utils.models import GCN, SAGE, GAT, LinkPredictor\n"
"\n"
"\n"
"DATASET_PATHS = {\n"
"    'FB15K237':  'data/FB15K237/train.txt',\n"
"    'WN18RR':    'data/WN18RR/train.txt',\n"
"    'NELL-995':  'data/NELL-995/train.txt',\n"
"}\n"
"\n"
"def make_encoder(enc_type, in_ch=FEATURE_DIM, hid=HIDDEN_CHANNELS,\n"
"                 n_layers=NUM_LAYERS, dropout=DROPOUT):\n"
"    if enc_type == 'GCN':  return GCN(in_ch, hid, hid, n_layers, dropout)\n"
"    if enc_type == 'SAGE': return SAGE(in_ch, hid, hid, n_layers, dropout)\n"
"    if enc_type == 'GAT':  return GAT(in_ch, hid, hid, n_layers, dropout, heads=4)\n"
"    raise ValueError(enc_type)\n"
"\n"
"def make_predictor(hid=HIDDEN_CHANNELS, n_layers=NUM_LAYERS, dropout=DROPOUT):\n"
"    return LinkPredictor(hid, hid, 1, n_layers, dropout)\n"
"\n"
"def save_full_results(base_dir, result_dict):\n"
"    \"\"\"Save full_results.json inside a timestamped run directory.\n"
"    Layout: base_dir/runs/{timestamp}/full_results.json\n"
"    Also writes base_dir/full_results.json as the latest copy.\n"
"    \"\"\"\n"
"    run_id = time.strftime('%Y%m%d_%H%M%S')\n"
"    result_dict['run_id'] = run_id\n"
"    result_dict['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')\n"
"\n"
"    run_dir = os.path.join(base_dir, 'runs', run_id)\n"
"    os.makedirs(run_dir, exist_ok=True)\n"
"    run_path = os.path.join(run_dir, 'full_results.json')\n"
"    with open(run_path, 'w') as f:\n"
"        json.dump(result_dict, f, indent=2, default=str)\n"
"    print(f'  -> {run_path}')\n"
"\n"
"    os.makedirs(base_dir, exist_ok=True)\n"
"    latest_path = os.path.join(base_dir, 'full_results.json')\n"
"    with open(latest_path, 'w') as f:\n"
"        json.dump(result_dict, f, indent=2, default=str)\n"
"    print(f'  -> {latest_path} (latest)')\n"
"\n"
"print('Ready.')"
)

# ── Data loading ──────────────────────────────────────────────────────────────
md("## 1. Load Datasets")

code(
"datasets = {}\n"
"for ds_name in DATASETS:\n"
"    print(f'\\nLoading {ds_name}...')\n"
"    dd = prepare_link_prediction_data(\n"
"        DATASET_PATHS[ds_name],\n"
"        feature_method='random', feature_dim=FEATURE_DIM)\n"
"    datasets[ds_name] = dd\n"
"    data = dd['data']; se = dd['split_edge']\n"
"    print(f'  Nodes: {data.num_nodes:,}  Edges: {data.edge_index.size(1):,}')\n"
"    print(f'  Train: {se[\"train\"][\"source_node\"].size(0):,}  '\n"
"          f'Val: {se[\"valid\"][\"source_node\"].size(0):,}  '\n"
"          f'Test: {se[\"test\"][\"source_node\"].size(0):,}')"
)

# ── Full Graph ────────────────────────────────────────────────────────────────
md("## 2. Full Graph (No Subgraph)")

code(
"from subgrapher.benchmark.trainer import benchmark_model"
)

code(
"for ds_name in DATASETS:\n"
"    dd = datasets[ds_name]\n"
"    data = dd['data']\n"
"    orig_edge_index = data.edge_index.clone()\n"
"    data.edge_index = dd['train_edge_index']\n"
"    split_edge = dd['split_edge']\n"
"\n"
"    for enc_type in ENCODERS:\n"
"        print(f'\\n=== Full Graph: {ds_name} / {enc_type} ===')\n"
"        torch.manual_seed(SEED)\n"
"        encoder   = make_encoder(enc_type)\n"
"        predictor = make_predictor()\n"
"        encoder.reset_parameters()\n"
"        predictor.reset_parameters()\n"
"\n"
"        result = benchmark_model(\n"
"            enc_type, encoder, predictor, data, split_edge,\n"
"            epochs=FULL_EPOCHS, batch_size=FULL_BATCH_SIZE,\n"
"            lr=FULL_LR, eval_steps=EVAL_STEPS, device=DEVICE,\n"
"            patience=FULL_PATIENCE, weight_decay=WEIGHT_DECAY,\n"
"            grad_clip=GRAD_CLIP)\n"
"\n"
"        save_full_results(\n"
"            f'results/benchmark/{ds_name}/{enc_type}',\n"
"            {\n"
"                'dataset': ds_name,\n"
"                'encoder': enc_type,\n"
"                'test_results': {k: float(v) for k, v in result['test_results'].items()},\n"
"                'train_time': result['train_time'],\n"
"                'num_params': result['num_params'],\n"
"                'best_epoch': result['best_epoch'],\n"
"                'stopped_early': result['stopped_early'],\n"
"                'config': {\n"
"                    'epochs': FULL_EPOCHS,\n"
"                    'batch_size': FULL_BATCH_SIZE,\n"
"                    'lr': FULL_LR,\n"
"                    'patience': FULL_PATIENCE,\n"
"                    'hidden_channels': HIDDEN_CHANNELS,\n"
"                    'num_layers': NUM_LAYERS,\n"
"                    'dropout': DROPOUT,\n"
"                },\n"
"                'seed': SEED,\n"
"            })\n"
"\n"
"        del encoder, predictor, result\n"
"        torch.cuda.empty_cache()\n"
"\n"
"    data.edge_index = orig_edge_index\n"
"    del orig_edge_index"
)

code(
"import gc\n"
"gc.collect()\n"
"torch.cuda.empty_cache()\n"
"print('Full Graph done — GPU memory released.')"
)

# ── Static PPR ────────────────────────────────────────────────────────────────
md("## 3. Static PPR Subgraph")

code(
"from subgrapher.benchmark_ppr.run_ppr_benchmark import load_or_create_ppr_preprocessor\n"
"from subgrapher.benchmark_ppr.ppr_extractor import StaticPPRExtractor\n"
"from subgrapher.benchmark_ppr.trainer_batched import train_model_ppr_batched\n"
"from subgrapher.benchmark_ppr.evaluator import evaluate_ppr"
)

code(
"for ds_name in DATASETS:\n"
"    dd = datasets[ds_name]\n"
"    data = dd['data']; split_edge = dd['split_edge']\n"
"    ppr_pre = load_or_create_ppr_preprocessor(ds_name, data, PPR_ALPHA)\n"
"\n"
"    for top_k in PPR_TOPK:\n"
"        ppr_ext = StaticPPRExtractor(\n"
"            data, alpha=ALPHA_COMBINE, top_k=top_k,\n"
"            ppr_alpha=PPR_ALPHA, preprocessor=ppr_pre)\n"
"\n"
"        for enc_type in ENCODERS:\n"
"            print(f'\\n=== Static PPR (top_k={top_k}): {ds_name} / {enc_type} ===')\n"
"            torch.manual_seed(SEED)\n"
"            encoder   = make_encoder(enc_type)\n"
"            predictor = make_predictor()\n"
"            encoder.reset_parameters()\n"
"            predictor.reset_parameters()\n"
"\n"
"            cache_dir = f'cache/benchmark-ppr/{ds_name}/topk{top_k}'\n"
"            t0 = time.time()\n"
"            hist = train_model_ppr_batched(\n"
"                encoder, predictor, data, split_edge, ppr_ext,\n"
"                epochs=SUB_EPOCHS, batch_size=SUB_BATCH_SIZE,\n"
"                lr=SUB_LR, eval_steps=EVAL_STEPS, device=DEVICE,\n"
"                verbose=True, patience=SUB_PATIENCE,\n"
"                weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,\n"
"                edges_per_epoch=EDGES_PER_EPOCH,\n"
"                cache_dir=cache_dir)\n"
"            train_time = time.time() - t0\n"
"\n"
"            test_res = evaluate_ppr(\n"
"                encoder, predictor, data, split_edge, ppr_ext,\n"
"                split='test', batch_size=SUB_BATCH_SIZE, device=DEVICE,\n"
"                max_edges=5000, cache_dir=cache_dir)\n"
"\n"
"            save_full_results(\n"
"                f'results/benchmark-ppr/{ds_name}/{enc_type}_topk{top_k}',\n"
"                {\n"
"                    'dataset': ds_name,\n"
"                    'encoder': enc_type,\n"
"                    'top_k': top_k,\n"
"                    'test_results': {k: float(v) for k, v in test_res.items()},\n"
"                    'train_time': train_time,\n"
"                    'best_epoch': hist.get('best_epoch', 0),\n"
"                    'stopped_early': hist.get('stopped_early', False),\n"
"                    'config': {\n"
"                        'ppr_alpha': PPR_ALPHA,\n"
"                        'alpha_combine': ALPHA_COMBINE,\n"
"                        'top_k': top_k,\n"
"                        'epochs': SUB_EPOCHS,\n"
"                        'batch_size': SUB_BATCH_SIZE,\n"
"                        'lr': SUB_LR,\n"
"                        'patience': SUB_PATIENCE,\n"
"                        'hidden_channels': HIDDEN_CHANNELS,\n"
"                        'num_layers': NUM_LAYERS,\n"
"                        'dropout': DROPOUT,\n"
"                    },\n"
"                    'seed': SEED,\n"
"                })\n"
"\n"
"            del encoder, predictor, hist, test_res\n"
"            torch.cuda.empty_cache()\n"
"\n"
"        del ppr_ext\n"
"\n"
"    del ppr_pre"
)

code(
"gc.collect()\n"
"torch.cuda.empty_cache()\n"
"print('Static PPR done — GPU memory released.')"
)

# ── Static k-hop ─────────────────────────────────────────────────────────────
md("## 4. Static k-hop Subgraph")

code(
"from subgrapher.benchmark_khop.run_khop_benchmark import load_or_create_khop_preprocessor\n"
"from subgrapher.benchmark_khop.khop_extractor import StaticKHopExtractor\n"
"from subgrapher.benchmark_khop.trainer_batched import train_model_khop_batched\n"
"from subgrapher.benchmark_khop.evaluator import evaluate_khop"
)

code(
"for ds_name in DATASETS:\n"
"    dd = datasets[ds_name]\n"
"    data = dd['data']; split_edge = dd['split_edge']\n"
"\n"
"    for k in K_VALUES:\n"
"        khop_pre = load_or_create_khop_preprocessor(ds_name, data, k)\n"
"        khop_ext = StaticKHopExtractor(data, k=k, preprocessor=khop_pre)\n"
"\n"
"        for enc_type in ENCODERS:\n"
"            print(f'\\n=== Static k-hop (k={k}): {ds_name} / {enc_type} ===')\n"
"            torch.manual_seed(SEED)\n"
"            encoder   = make_encoder(enc_type)\n"
"            predictor = make_predictor()\n"
"            encoder.reset_parameters()\n"
"            predictor.reset_parameters()\n"
"\n"
"            cache_dir = f'cache/benchmark-khop/{ds_name}/k{k}'\n"
"            t0 = time.time()\n"
"            hist = train_model_khop_batched(\n"
"                encoder, predictor, data, split_edge, khop_ext,\n"
"                epochs=SUB_EPOCHS, batch_size=SUB_BATCH_SIZE,\n"
"                lr=SUB_LR, eval_steps=EVAL_STEPS, device=DEVICE,\n"
"                verbose=True, patience=SUB_PATIENCE,\n"
"                weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,\n"
"                edges_per_epoch=EDGES_PER_EPOCH,\n"
"                cache_dir=cache_dir)\n"
"            train_time = time.time() - t0\n"
"\n"
"            test_res = evaluate_khop(\n"
"                encoder, predictor, data, split_edge, khop_ext,\n"
"                split='test', batch_size=SUB_BATCH_SIZE, device=DEVICE,\n"
"                max_edges=5000, cache_dir=cache_dir)\n"
"\n"
"            save_full_results(\n"
"                f'results/benchmark-khop/{ds_name}/{enc_type}_k{k}',\n"
"                {\n"
"                    'dataset': ds_name,\n"
"                    'encoder': enc_type,\n"
"                    'k': k,\n"
"                    'test_results': {k_: float(v) for k_, v in test_res.items()},\n"
"                    'train_time': train_time,\n"
"                    'best_epoch': hist.get('best_epoch', 0),\n"
"                    'stopped_early': hist.get('stopped_early', False),\n"
"                    'config': {\n"
"                        'k': k,\n"
"                        'epochs': SUB_EPOCHS,\n"
"                        'batch_size': SUB_BATCH_SIZE,\n"
"                        'lr': SUB_LR,\n"
"                        'patience': SUB_PATIENCE,\n"
"                        'hidden_channels': HIDDEN_CHANNELS,\n"
"                        'num_layers': NUM_LAYERS,\n"
"                        'dropout': DROPOUT,\n"
"                    },\n"
"                    'seed': SEED,\n"
"                })\n"
"\n"
"            del encoder, predictor, hist, test_res\n"
"            torch.cuda.empty_cache()\n"
"\n"
"        del khop_ext, khop_pre"
)

code(
"gc.collect()\n"
"torch.cuda.empty_cache()\n"
"print('Static k-hop done — GPU memory released.')"
)

# ── Summary ───────────────────────────────────────────────────────────────────
md("## 5. Done")

code(
"print('All experiments complete.')\n"
"print('Results saved to:')\n"
"for d in ['results/benchmark', 'results/benchmark-ppr', 'results/benchmark-khop']:\n"
"    n = sum(1 for root, _, files in os.walk(d) if 'full_results.json' in files)\n"
"    print(f'  {d}: {n} full_results.json files')\n"
"print('\\nRun learnable_ppr.ipynb for Learnable PPR results.')\n"
"print('Then open benchmark_analysis.ipynb to compare all methods.')"
)

# ═══════════════════════════════════════════════════════════════════════════════
with open(r"d:\Work + Project\Thesis\MTCH\benchmark_runner.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook written: benchmark_runner.ipynb")
