"""Generates benchmark_runner_planetoid.ipynb — Full Graph, Static PPR, Static k-hop on Cora/CiteSeer/PubMed.

Uses PyG Planetoid datasets (real node features, no .txt loading needed).
Results saved under the same schema as benchmark_runner.ipynb so benchmark_analysis.ipynb loads both.

Save layout:
  results/benchmark/{dataset}/{encoder}/full_results.json
  results/benchmark-ppr/{dataset}/{encoder}_a{alpha}_e{epsilon}/full_results.json
  results/benchmark-khop/{dataset}/{encoder}_k{k}/full_results.json
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
"# Benchmark Runner — Planetoid (Cora / CiteSeer / PubMed)\n"
"\n"
"Runs Full Graph, Static PPR (LCILP-style), and Static k-hop on Cora, CiteSeer, PubMed.\n"
"Uses real PyG node features — no random embeddings.\n"
"\n"
"| Method | Subgraph | Output |\n"
"|--------|----------|--------|\n"
"| Full Graph | None | `results/benchmark/{ds}/{enc}/` |\n"
"| Static PPR | PPR + sweep cut | `results/benchmark-ppr/{ds}/{enc}_a{a}_e{e}/` |\n"
"| Static k-hop | k-hop | `results/benchmark-khop/{ds}/{enc}_k{k}/` |\n"
"\n"
"After running this + `learnable_ppr_planetoid.ipynb`, open `benchmark_analysis.ipynb` to compare."
)

# ── Config ────────────────────────────────────────────────────────────────────
code(
"SEED = 42\n"
"DATASETS  = ['Cora', 'CiteSeer', 'PubMed']\n"
"ENCODERS  = ['GCN', 'SAGE', 'GAT']\n"
"\n"
"# Architecture (feature dim is read per-dataset from real node features)\n"
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
"# Planetoid graphs are small — EDGES_PER_EPOCH=None uses all training edges\n"
"SUB_EPOCHS          = 500\n"
"SUB_BATCH_SIZE      = 256\n"
"SUB_LR              = 0.005\n"
"SUB_PATIENCE        = 30\n"
"EDGES_PER_EPOCH     = None      # None = all (Cora ~4.7K, CiteSeer ~3.7K, PubMed ~8K)\n"
"EVAL_STEPS          = 5\n"
"WEIGHT_DECAY        = 1e-5\n"
"GRAD_CLIP           = 1.0\n"
"\n"
"ENCODER_LR_OVERRIDE   = {'GAT': 0.001}\n"
"ENCODER_CLIP_OVERRIDE = {'GAT': 0.5}\n"
"\n"
"# Static PPR\n"
"PPR_ALPHAS    = [0.85]\n"
"PPR_EPSILONS  = [1e-3]\n"
"PPR_WINDOW    = 10\n"
"DRNL_MAX_DIST = 6\n"
"DRNL_DIM      = 2 * (DRNL_MAX_DIST + 1)\n"
"PPR_MARGIN    = 10.0\n"
"\n"
"# Static k-hop\n"
"K_VALUES = [2]\n"
"\n"
"import torch\n"
"GPU_ID = 0\n"
"DEVICE = f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu'\n"
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
"from subgrapher.benchmark.data_prep import prepare_planetoid_data\n"
"from subgrapher.utils.models import GCN, SAGE, GAT, LinkPredictor\n"
"\n"
"\n"
"def make_encoder(enc_type, in_ch, hid=HIDDEN_CHANNELS,\n"
"                 n_layers=NUM_LAYERS, dropout=DROPOUT):\n"
"    if enc_type == 'GCN':  return GCN(in_ch, hid, hid, n_layers, dropout)\n"
"    if enc_type == 'SAGE': return SAGE(in_ch, hid, hid, n_layers, dropout)\n"
"    if enc_type == 'GAT':  return GAT(in_ch, hid, hid, n_layers, dropout, heads=4)\n"
"    raise ValueError(enc_type)\n"
"\n"
"def make_predictor(hid=HIDDEN_CHANNELS, n_layers=NUM_LAYERS, dropout=DROPOUT):\n"
"    return LinkPredictor(hid, hid, 1, n_layers, dropout)\n"
"\n"
"def enc_lr(enc_type, base=SUB_LR):\n"
"    return ENCODER_LR_OVERRIDE.get(enc_type, base)\n"
"\n"
"def enc_clip(enc_type, base=GRAD_CLIP):\n"
"    return ENCODER_CLIP_OVERRIDE.get(enc_type, base)\n"
"\n"
"def ensure_train_only_edges(data, dd):\n"
"    if not getattr(data, '_edge_index_train_only', False):\n"
"        data._orig_edge_index = data.edge_index\n"
"        data.edge_index = dd['train_edge_index']\n"
"        data._edge_index_train_only = True\n"
"        print(f'  [train-only edges] swapped: '\n"
"              f\"{data._orig_edge_index.size(1):,} -> {data.edge_index.size(1):,}\")\n"
"    return data\n"
"\n"
"def save_full_results(base_dir, result_dict):\n"
"    run_id = time.strftime('%Y%m%d_%H%M%S')\n"
"    result_dict['run_id'] = run_id\n"
"    result_dict['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')\n"
"    run_dir = os.path.join(base_dir, 'runs', run_id)\n"
"    os.makedirs(run_dir, exist_ok=True)\n"
"    run_path = os.path.join(run_dir, 'full_results.json')\n"
"    with open(run_path, 'w') as f:\n"
"        json.dump(result_dict, f, indent=2, default=str)\n"
"    print(f'  -> {run_path}')\n"
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
"    dd = prepare_planetoid_data(ds_name)\n"
"    datasets[ds_name] = dd\n"
"    data = dd['data']; se = dd['split_edge']\n"
"    print(f'  Nodes: {data.num_nodes:,}  Edges: {data.edge_index.size(1):,}  '\n"
"          f'Feature dim: {dd[\"feature_dim\"]}')\n"
"    print(f'  Train: {se[\"train\"][\"source_node\"].size(0):,}  '\n"
"          f'Val: {se[\"valid\"][\"source_node\"].size(0):,}  '\n"
"          f'Test: {se[\"test\"][\"source_node\"].size(0):,}')"
)

code(
"for ds_name in DATASETS:\n"
"    print(f'[{ds_name}]')\n"
"    ensure_train_only_edges(datasets[ds_name]['data'], datasets[ds_name])"
)

# ── Full Graph ────────────────────────────────────────────────────────────────
md("## 2. Full Graph (No Subgraph)")

code(
"from subgrapher.benchmark.trainer import benchmark_model"
)

code(
"for ds_name in DATASETS:\n"
"    dd = datasets[ds_name]\n"
"    data = dd['data']; split_edge = dd['split_edge']\n"
"    ensure_train_only_edges(data, dd)\n"
"    in_ch = dd['feature_dim']\n"
"\n"
"    for enc_type in ENCODERS:\n"
"        print(f'\\n=== Full Graph: {ds_name} / {enc_type} ===')\n"
"        torch.manual_seed(SEED)\n"
"        encoder   = make_encoder(enc_type, in_ch=in_ch)\n"
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
"                    'epochs': FULL_EPOCHS, 'batch_size': FULL_BATCH_SIZE,\n"
"                    'lr': FULL_LR, 'patience': FULL_PATIENCE,\n"
"                    'feature_dim': in_ch, 'hidden_channels': HIDDEN_CHANNELS,\n"
"                    'num_layers': NUM_LAYERS, 'dropout': DROPOUT,\n"
"                },\n"
"                'seed': SEED,\n"
"            })\n"
"\n"
"        del encoder, predictor, result\n"
"        torch.cuda.empty_cache()"
)

code(
"import gc\n"
"gc.collect(); torch.cuda.empty_cache()\n"
"print('Full Graph done.')"
)

# ── Static PPR ────────────────────────────────────────────────────────────────
md("## 3. Static PPR Subgraph (LCILP-style)")

code(
"from subgrapher.benchmark_ppr.ppr_extractor import StaticPPRExtractor\n"
"from subgrapher.benchmark_ppr.trainer_lcilp import train_model_ppr_lcilp\n"
"from subgrapher.benchmark_ppr.evaluator import evaluate_ppr_lcilp\n"
"from subgrapher.benchmark_ppr.graph_classifier import SubgraphClassifier"
)

code(
"for ds_name in DATASETS:\n"
"    dd = datasets[ds_name]\n"
"    data = dd['data']; split_edge = dd['split_edge']\n"
"    ensure_train_only_edges(data, dd)\n"
"\n"
"    for ppr_alpha in PPR_ALPHAS:\n"
"        for ppr_eps in PPR_EPSILONS:\n"
"            ppr_ext = StaticPPRExtractor(\n"
"                data, alpha=ppr_alpha, epsilon=ppr_eps, window=PPR_WINDOW)\n"
"\n"
"            for enc_type in ENCODERS:\n"
"                print(f'\\n=== Static PPR LCILP (a={ppr_alpha}, e={ppr_eps}): '\n"
"                      f'{ds_name} / {enc_type} ===')\n"
"                torch.manual_seed(SEED)\n"
"                classifier = SubgraphClassifier(\n"
"                    drnl_dim=DRNL_DIM, hidden=HIDDEN_CHANNELS,\n"
"                    num_layers=NUM_LAYERS, dropout=DROPOUT,\n"
"                    encoder_type=enc_type)\n"
"                classifier.reset_parameters()\n"
"\n"
"                cache_dir = f'cache/benchmark-ppr/{ds_name}/a{ppr_alpha}_e{ppr_eps}'\n"
"                t0 = time.time()\n"
"                hist = train_model_ppr_lcilp(\n"
"                    classifier, data, split_edge, ppr_ext,\n"
"                    epochs=SUB_EPOCHS, batch_size=SUB_BATCH_SIZE,\n"
"                    lr=enc_lr(enc_type), eval_steps=EVAL_STEPS, device=DEVICE,\n"
"                    verbose=True, patience=SUB_PATIENCE,\n"
"                    weight_decay=WEIGHT_DECAY, grad_clip=enc_clip(enc_type),\n"
"                    margin=PPR_MARGIN, drnl_max_dist=DRNL_MAX_DIST,\n"
"                    edges_per_epoch=EDGES_PER_EPOCH,\n"
"                    cache_dir=cache_dir, max_eval_edges=2000)\n"
"                train_time = time.time() - t0\n"
"\n"
"                test_res = evaluate_ppr_lcilp(\n"
"                    classifier, data, split_edge, ppr_ext,\n"
"                    split='test', batch_size=SUB_BATCH_SIZE, device=DEVICE,\n"
"                    max_edges=None, cache_dir=cache_dir,\n"
"                    drnl_max_dist=DRNL_MAX_DIST)\n"
"\n"
"                save_full_results(\n"
"                    f'results/benchmark-ppr/{ds_name}/{enc_type}_a{ppr_alpha}_e{ppr_eps}',\n"
"                    {\n"
"                        'dataset': ds_name, 'encoder': enc_type,\n"
"                        'ppr_alpha': ppr_alpha, 'ppr_epsilon': ppr_eps,\n"
"                        'ppr_window': PPR_WINDOW,\n"
"                        'test_results': {k: float(v) for k, v in test_res.items()},\n"
"                        'train_time': train_time,\n"
"                        'best_epoch': hist.get('best_epoch', 0),\n"
"                        'stopped_early': hist.get('stopped_early', False),\n"
"                        'config': {\n"
"                            'ppr_alpha': ppr_alpha, 'ppr_epsilon': ppr_eps,\n"
"                            'ppr_window': PPR_WINDOW,\n"
"                            'drnl_max_dist': DRNL_MAX_DIST, 'margin': PPR_MARGIN,\n"
"                            'epochs': SUB_EPOCHS, 'batch_size': SUB_BATCH_SIZE,\n"
"                            'lr': enc_lr(enc_type), 'patience': SUB_PATIENCE,\n"
"                            'hidden_channels': HIDDEN_CHANNELS,\n"
"                            'num_layers': NUM_LAYERS, 'dropout': DROPOUT,\n"
"                        },\n"
"                        'seed': SEED,\n"
"                    })\n"
"\n"
"                del classifier, hist, test_res\n"
"                torch.cuda.empty_cache()\n"
"\n"
"            del ppr_ext"
)

code(
"import gc\n"
"gc.collect(); torch.cuda.empty_cache()\n"
"print('Static PPR done.')"
)

# ── Static k-hop ─────────────────────────────────────────────────────────────
md("## 4. Static k-hop Subgraph")

code(
"from subgrapher.benchmark_khop.run_khop_benchmark import load_or_create_khop_preprocessor\n"
"from subgrapher.benchmark_khop.khop_extractor import StaticKHopExtractor\n"
"from subgrapher.benchmark_khop.trainer_batched import train_model_khop_batched\n"
"from subgrapher.benchmark_khop.evaluator import evaluate_khop\n"
"from subgrapher.benchmark.evaluator import evaluate_link_prediction"
)

code(
"for ds_name in DATASETS:\n"
"    dd = datasets[ds_name]\n"
"    data = dd['data']; split_edge = dd['split_edge']\n"
"    ensure_train_only_edges(data, dd)\n"
"    in_ch = dd['feature_dim']\n"
"\n"
"    for k in K_VALUES:\n"
"        khop_pre = load_or_create_khop_preprocessor(ds_name, data, k)\n"
"        khop_ext = StaticKHopExtractor(data, k=k, preprocessor=khop_pre)\n"
"\n"
"        for enc_type in ENCODERS:\n"
"            print(f'\\n=== Static k-hop (k={k}): {ds_name} / {enc_type} ===')\n"
"            torch.manual_seed(SEED)\n"
"            encoder   = make_encoder(enc_type, in_ch=in_ch)\n"
"            predictor = make_predictor()\n"
"            encoder.reset_parameters()\n"
"            predictor.reset_parameters()\n"
"\n"
"            cache_dir = f'cache/benchmark-khop/{ds_name}/k{k}'\n"
"            t0 = time.time()\n"
"            hist = train_model_khop_batched(\n"
"                encoder, predictor, data, split_edge, khop_ext,\n"
"                epochs=SUB_EPOCHS, batch_size=SUB_BATCH_SIZE,\n"
"                lr=enc_lr(enc_type), eval_steps=EVAL_STEPS, device=DEVICE,\n"
"                verbose=True, patience=SUB_PATIENCE,\n"
"                weight_decay=WEIGHT_DECAY, grad_clip=enc_clip(enc_type),\n"
"                edges_per_epoch=EDGES_PER_EPOCH,\n"
"                cache_dir=cache_dir)\n"
"            train_time = time.time() - t0\n"
"\n"
"            test_res = evaluate_khop(\n"
"                encoder, predictor, data, split_edge, khop_ext,\n"
"                split='test', batch_size=SUB_BATCH_SIZE, device=DEVICE,\n"
"                max_edges=None, cache_dir=cache_dir)\n"
"            fg_test_res = evaluate_link_prediction(\n"
"                encoder, predictor, data, split_edge,\n"
"                split='test', batch_size=65536)\n"
"\n"
"            save_full_results(\n"
"                f'results/benchmark-khop/{ds_name}/{enc_type}_k{k}',\n"
"                {\n"
"                    'dataset': ds_name, 'encoder': enc_type, 'k': k,\n"
"                    'test_results': {k_: float(v) for k_, v in fg_test_res.items()},\n"
"                    'subgraph_test_results': {k_: float(v) for k_, v in test_res.items()},\n"
"                    'train_time': train_time,\n"
"                    'best_epoch': hist.get('best_epoch', 0),\n"
"                    'stopped_early': hist.get('stopped_early', False),\n"
"                    'config': {\n"
"                        'k': k, 'epochs': SUB_EPOCHS, 'batch_size': SUB_BATCH_SIZE,\n"
"                        'lr': enc_lr(enc_type), 'patience': SUB_PATIENCE,\n"
"                        'feature_dim': in_ch, 'hidden_channels': HIDDEN_CHANNELS,\n"
"                        'num_layers': NUM_LAYERS, 'dropout': DROPOUT,\n"
"                    },\n"
"                    'seed': SEED,\n"
"                })\n"
"\n"
"            del encoder, predictor, hist, test_res, fg_test_res\n"
"            torch.cuda.empty_cache()\n"
"\n"
"        del khop_ext, khop_pre"
)

code(
"import gc\n"
"gc.collect(); torch.cuda.empty_cache()\n"
"print('Static k-hop done.')"
)

# ── Summary ───────────────────────────────────────────────────────────────────
md("## 5. Done")

code(
"print('All experiments complete. Results saved to:')\n"
"for d in ['results/benchmark', 'results/benchmark-ppr', 'results/benchmark-khop']:\n"
"    if os.path.exists(d):\n"
"        n = sum(1 for root, _, files in os.walk(d) if 'full_results.json' in files)\n"
"        print(f'  {d}: {n} full_results.json files')\n"
"print('\\nRun learnable_ppr_planetoid.ipynb for Learnable PPR results.')\n"
"print('Then open benchmark_analysis.ipynb to compare all methods.')"
)

# ═══════════════════════════════════════════════════════════════════════════════
with open(r"d:\Work + Project\Thesis\MTCH\benchmark_runner_planetoid.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook written: benchmark_runner_planetoid.ipynb")
