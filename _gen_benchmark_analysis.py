"""Generates benchmark_analysis.ipynb — reads saved JSON, produces comparison visualizations.

Visualization plan by Lucy (data engineer):
 0. Experiment map (config + best-config rule)
 1. Summary leaderboard table
 2. Headline MRR by method (faceted by dataset, collapsed to best config)
 3. Encoder sensitivity (faceted by encoder)
 4. Static PPR: MRR by (alpha, epsilon) config
 5. k-hop: MRR vs k (line)
 6. Time vs MRR scatter
 7. Cross-dataset relative MRR
 8. Hits ladder
 9. Full metric table (supplement)
10. Conclusion
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
# Title
# ═══════════════════════════════════════════════════════════════════════════════
md(
"# Benchmark Analysis: Subgraph Strategies for Link Prediction\n"
"\n"
"This notebook reads pre-computed results from `benchmark_runner.ipynb` (and\n"
"optionally from an external `RESULT_DIR`) and produces a unified comparison.\n"
"\n"
"**No training happens here** — only data loading and visualization."
)

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Config + data loading
# ═══════════════════════════════════════════════════════════════════════════════
md(
"## 0. Setup & Data Loading\n"
"\n"
"**Best-config rule**: For methods with multiple configurations (Static PPR\n"
"has `alpha x epsilon` configs, k-hop has 3 `k` values), the **summary figures** show\n"
"only the **best config per (dataset, encoder)** selected by highest test MRR.\n"
"The sensitivity figures (Sections 4–5) show all configs."
)

code(
"import json, os, glob\n"
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"import matplotlib.ticker as mticker\n"
"import seaborn as sns\n"
"from collections import defaultdict\n"
"\n"
"# ── Style ──\n"
"sns.set_theme(style='whitegrid', font_scale=1.05)\n"
"ENCODER_COLORS = {'GCN': '#1b9e77', 'SAGE': '#d95f02', 'GAT': '#7570b3'}\n"
"METHOD_MARKERS = {\n"
"    'Full Graph': 'o', 'Static PPR': 's',\n"
"    'Static k-hop': '^', 'Learnable PPR': 'D',\n"
"}\n"
"FIGW = 7\n"
"\n"
"# ── Directories to scan for full_results.json ──\n"
"# Each maps a directory prefix to a (method_name, classify_fn).\n"
"# classify_fn(r) -> (config_label, config_value) from the loaded JSON dict.\n"
"RESULT_DIRS = {\n"
"    'results/benchmark':               'Full Graph',\n"
"    'results/benchmark-ppr':           'Static PPR',\n"
"    'results/benchmark-khop':          'Static k-hop',\n"
"    'results/benchmark-learnable-ppr': 'Learnable PPR',\n"
"}\n"
"\n"
"# Optional: extra directory with full_results.json from other runs.\n"
"EXTRA_RESULT_DIR = '<result_dir>'  # <-- FILL THIS IN (or set to None)\n"
"\n"
"def _classify(r, method):\n"
"    \"\"\"Return (config_label, config_value) from a full_results.json dict.\"\"\"\n"
"    if method == 'Static PPR':\n"
"        a = r.get('ppr_alpha', r.get('config', {}).get('ppr_alpha', '?'))\n"
"        e = r.get('ppr_epsilon', r.get('config', {}).get('ppr_epsilon',\n"
"            r.get('top_k', r.get('config', {}).get('top_k', '?'))))\n"
"        return f'a={a},e={e}', f'{a}_{e}'\n"
"    if method == 'Static k-hop':\n"
"        k = r.get('k', r.get('config', {}).get('k', '?'))\n"
"        return f'k={k}', k\n"
"    if method == 'Learnable PPR':\n"
"        e = r.get('ppr_epsilon', r.get('top_k', '?'))\n"
"        return f'eps={e}', e\n"
"    return '-', None\n"
"\n"
"def _detect_method(r):\n"
"    \"\"\"Guess method from a JSON dict when loaded from EXTRA_RESULT_DIR.\"\"\"\n"
"    if 'teleport_values' in r:\n"
"        return 'Learnable PPR'\n"
"    if 'k' in r or r.get('config', {}).get('k'):\n"
"        return 'Static k-hop'\n"
"    if 'ppr_alpha' in r or 'ppr_epsilon' in r or 'top_k' in r or 'ppr_alpha' in r.get('config', {}):\n"
"        return 'Static PPR'\n"
"    return 'Full Graph'\n"
"\n"
"def _parse_result(r, method):\n"
"    \"\"\"Turn a full_results.json dict into a flat row.\"\"\"\n"
"    tr = r.get('test_results', {})\n"
"    cfg, cfg_val = _classify(r, method)\n"
"    return {\n"
"        'method': method,\n"
"        'dataset': r.get('dataset', '?'),\n"
"        'encoder': r.get('encoder', '?'),\n"
"        'config': cfg,\n"
"        'config_value': cfg_val,\n"
"        'mrr':      float(tr.get('mrr', 0)),\n"
"        'auc':      float(tr.get('auc', 0)),\n"
"        'ap':       float(tr.get('ap', 0)),\n"
"        'hits@1':   float(tr.get('hits@1', 0)),\n"
"        'hits@3':   float(tr.get('hits@3', 0)),\n"
"        'hits@10':  float(tr.get('hits@10', 0)),\n"
"        'hits@50':  float(tr.get('hits@50', 0)),\n"
"        'hits@100': float(tr.get('hits@100', 0)),\n"
"        'train_time': float(r.get('train_time', r.get('search_time', 0))),\n"
"        'seed': r.get('seed', '?'),\n"
"        'run_id': r.get('run_id', ''),\n"
"        'timestamp': r.get('timestamp', ''),\n"
"    }\n"
"\n"
"# ── Scan standard directories ──\n"
"# Each experiment dir has: full_results.json (latest) + runs/{timestamp}/full_results.json\n"
"# We only read the top-level full_results.json to avoid duplicates.\n"
"raw = []\n"
"for base_dir, method in RESULT_DIRS.items():\n"
"    if not os.path.isdir(base_dir):\n"
"        print(f'  {base_dir}: not found, skipping')\n"
"        continue\n"
"    count = 0\n"
"    for root, dirs, files in os.walk(base_dir):\n"
"        if 'runs' in dirs:\n"
"            dirs.remove('runs')  # don't descend into timestamped copies\n"
"        if 'full_results.json' in files:\n"
"            fpath = os.path.join(root, 'full_results.json')\n"
"            try:\n"
"                with open(fpath) as f:\n"
"                    r = json.load(f)\n"
"                raw.append(_parse_result(r, method))\n"
"                count += 1\n"
"            except Exception as e:\n"
"                print(f'  Error: {fpath}: {e}')\n"
"    print(f'  {base_dir}: {count} experiments')\n"
"\n"
"# ── Scan extra directory ──\n"
"if EXTRA_RESULT_DIR and EXTRA_RESULT_DIR != '<result_dir>' and os.path.isdir(EXTRA_RESULT_DIR):\n"
"    count = 0\n"
"    for root, _, files in os.walk(EXTRA_RESULT_DIR):\n"
"        if 'full_results.json' in files:\n"
"            fpath = os.path.join(root, 'full_results.json')\n"
"            try:\n"
"                with open(fpath) as f:\n"
"                    r = json.load(f)\n"
"                method = _detect_method(r)\n"
"                raw.append(_parse_result(r, method))\n"
"                count += 1\n"
"            except Exception as e:\n"
"                print(f'  Error: {fpath}: {e}')\n"
"    print(f'  {EXTRA_RESULT_DIR}: {count} extra experiments')\n"
"\n"
"df = pd.DataFrame(raw)\n"
"print(f'\\nTotal rows: {len(df)}')\n"
"if not df.empty:\n"
"    print(f'Datasets:  {sorted(df[\"dataset\"].unique())}')\n"
"    print(f'Encoders:  {sorted(df[\"encoder\"].unique())}')\n"
"    print(f'Methods:   {sorted(df[\"method\"].unique())}')"
)

# ── Build collapsed summary (best config per method-dataset-encoder) ──────────
code(
"# For each (method, dataset, encoder), keep only the row with highest MRR\n"
"# This collapses PPR (alpha, epsilon) and k-hop k to the single best.\n"
"\n"
"if not df.empty:\n"
"    idx_best = df.groupby(['method', 'dataset', 'encoder'])['mrr'].idxmax()\n"
"    df_best = df.loc[idx_best].reset_index(drop=True)\n"
"\n"
"    print('Best config mapping (used in summary figures):')\n"
"    for _, row in df_best.iterrows():\n"
"        if row['config'] != '-':\n"
"            print(f'  {row[\"dataset\"]:10s} | {row[\"encoder\"]:5s} | '\n"
"                  f'{row[\"method\"]:16s} -> {row[\"config\"]}')\n"
"else:\n"
"    df_best = df"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Summary leaderboard
# ═══════════════════════════════════════════════════════════════════════════════
md("## 1. Summary Leaderboard (All Metrics)")

code(
"if not df_best.empty:\n"
"    metric_cols = ['mrr', 'auc', 'ap', 'hits@1', 'hits@10', 'hits@50']\n"
"    avail_metrics = [c for c in metric_cols if c in df_best.columns]\n"
"    display_cols = ['dataset', 'encoder', 'method', 'config'] + avail_metrics + ['train_time']\n"
"    display_cols = [c for c in display_cols if c in df_best.columns]\n"
"\n"
"    tbl = df_best[display_cols].sort_values(['dataset', 'encoder', 'method'])\n"
"\n"
"    fmt = {m: '{:.4f}' for m in avail_metrics}\n"
"    fmt['train_time'] = '{:.0f}s'\n"
"\n"
"    styled = (tbl.style\n"
"              .format(fmt)\n"
"              .highlight_max(subset=['mrr'], color='#b6d7a8', axis=0))\n"
"    display(styled)\n"
"else:\n"
"    print('No results loaded.')"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Headline MRR (faceted by dataset, best config)
# ═══════════════════════════════════════════════════════════════════════════════
md(
"## 2. Headline: Best MRR by Method\n"
"\n"
"Each bar is the **best config** for that method (e.g., PPR with optimal alpha/epsilon).\n"
"Faceted by dataset."
)

code(
"if not df_best.empty:\n"
"    ds_list = sorted(df_best['dataset'].unique())\n"
"    n_ds = len(ds_list)\n"
"    fig, axes = plt.subplots(1, n_ds, figsize=(FIGW * n_ds, 5), squeeze=False)\n"
"\n"
"    for i, ds in enumerate(ds_list):\n"
"        ax = axes[0, i]\n"
"        sub = df_best[df_best['dataset'] == ds].sort_values('mrr', ascending=True)\n"
"        labels = sub['method'] + '\\n' + sub['encoder']\n"
"        colors = [ENCODER_COLORS.get(e, '#999') for e in sub['encoder']]\n"
"        ax.barh(range(len(sub)), sub['mrr'], color=colors)\n"
"        ax.set_yticks(range(len(sub)))\n"
"        ax.set_yticklabels(labels, fontsize=9)\n"
"        ax.set_xlabel('MRR')\n"
"        ax.set_title(ds, fontsize=13, fontweight='bold')\n"
"        ax.set_xlim(left=max(0, sub['mrr'].min() - 0.02))\n"
"        for j, v in enumerate(sub['mrr']):\n"
"            ax.text(v + 0.002, j, f'{v:.4f}', va='center', fontsize=8)\n"
"\n"
"    fig.suptitle('Test MRR — Best Config per Method',\n"
"                 fontsize=15, fontweight='bold', y=1.02)\n"
"    plt.tight_layout()\n"
"    plt.show()"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Encoder sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
md(
"## 3. Encoder Sensitivity\n"
"\n"
"Same data as Section 2 but pivoted: **faceted by encoder** so you can see\n"
"whether conclusions are encoder-specific."
)

code(
"if not df_best.empty:\n"
"    enc_list = sorted(df_best['encoder'].unique())\n"
"    n_enc = len(enc_list)\n"
"    fig, axes = plt.subplots(1, n_enc, figsize=(FIGW * n_enc, 4.5), squeeze=False)\n"
"\n"
"    for i, enc in enumerate(enc_list):\n"
"        ax = axes[0, i]\n"
"        sub = df_best[df_best['encoder'] == enc].copy()\n"
"        sub['label'] = sub['dataset'] + ' / ' + sub['method']\n"
"        sub = sub.sort_values('mrr', ascending=True)\n"
"        ax.barh(range(len(sub)), sub['mrr'],\n"
"                color=ENCODER_COLORS.get(enc, '#999'), alpha=0.85)\n"
"        ax.set_yticks(range(len(sub)))\n"
"        ax.set_yticklabels(sub['label'], fontsize=9)\n"
"        ax.set_xlabel('MRR')\n"
"        ax.set_title(enc, fontsize=13, fontweight='bold')\n"
"        ax.set_xlim(left=max(0, sub['mrr'].min() - 0.02))\n"
"\n"
"    fig.suptitle('MRR by Dataset & Method — per Encoder',\n"
"                 fontsize=15, fontweight='bold', y=1.02)\n"
"    plt.tight_layout()\n"
"    plt.show()"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PPR: MRR by (alpha, epsilon) config
# ═══════════════════════════════════════════════════════════════════════════════
md(
"## 4. Static PPR: MRR by Configuration\n"
"\n"
"How sensitive is PPR performance to `(alpha, epsilon)`? Grouped bar chart\n"
"per encoder, faceted by dataset."
)

code(
"ppr_df = df[df['method'] == 'Static PPR'].copy()\n"
"if not ppr_df.empty and 'config_label' in ppr_df.columns:\n"
"    ds_list = sorted(ppr_df['dataset'].unique())\n"
"    n_ds = len(ds_list)\n"
"\n"
"    fig, axes = plt.subplots(1, max(n_ds, 1), figsize=(6 * max(n_ds, 1), 4.5), squeeze=False)\n"
"    for i, ds in enumerate(ds_list):\n"
"        ax = axes[0, i]\n"
"        sub = ppr_df[ppr_df['dataset'] == ds].copy()\n"
"        configs = sorted(sub['config_label'].unique())\n"
"        encs = sorted(sub['encoder'].unique())\n"
"        x = np.arange(len(configs))\n"
"        w = 0.8 / max(len(encs), 1)\n"
"        for j, enc in enumerate(encs):\n"
"            vals = []\n"
"            for cfg in configs:\n"
"                row = sub[(sub['encoder'] == enc) & (sub['config_label'] == cfg)]\n"
"                vals.append(row['mrr'].values[0] if len(row) > 0 else 0)\n"
"            ax.bar(x + j * w, vals, w, label=enc,\n"
"                   color=ENCODER_COLORS.get(enc, '#999'))\n"
"        ax.set_xticks(x + w * (len(encs) - 1) / 2)\n"
"        ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)\n"
"        ax.set_ylabel('MRR')\n"
"        ax.set_title(ds, fontsize=12, fontweight='bold')\n"
"        ax.legend(fontsize=9)\n"
"\n"
"    fig.suptitle('Static PPR — MRR by (alpha, epsilon)',\n"
"                 fontsize=14, fontweight='bold', y=1.02)\n"
"    plt.tight_layout()\n"
"    plt.show()\n"
"else:\n"
"    print('No Static PPR results found.')"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. k-hop: MRR vs k
# ═══════════════════════════════════════════════════════════════════════════════
md("## 5. Static k-hop: MRR vs `k`")

code(
"khop_df = df[df['method'] == 'Static k-hop'].copy()\n"
"if not khop_df.empty and 'config_value' in khop_df.columns:\n"
"    khop_df['k'] = pd.to_numeric(khop_df['config_value'], errors='coerce')\n"
"    khop_df = khop_df.dropna(subset=['k'])\n"
"    ds_list = sorted(khop_df['dataset'].unique())\n"
"    n_ds = len(ds_list)\n"
"\n"
"    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 4), squeeze=False)\n"
"    for i, ds in enumerate(ds_list):\n"
"        ax = axes[0, i]\n"
"        for enc in sorted(khop_df['encoder'].unique()):\n"
"            sub = khop_df[(khop_df['dataset'] == ds) & (khop_df['encoder'] == enc)]\n"
"            sub = sub.sort_values('k')\n"
"            ax.plot(sub['k'], sub['mrr'], marker='s', markersize=6,\n"
"                    color=ENCODER_COLORS.get(enc, '#999'), label=enc)\n"
"        ax.set_xlabel('k'); ax.set_ylabel('MRR')\n"
"        ax.set_title(ds, fontsize=12, fontweight='bold')\n"
"        ax.legend(fontsize=9)\n"
"        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))\n"
"\n"
"    fig.suptitle('Static k-hop — MRR vs k',\n"
"                 fontsize=14, fontweight='bold', y=1.02)\n"
"    plt.tight_layout()\n"
"    plt.show()\n"
"else:\n"
"    print('No k-hop results found.')"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Time vs MRR scatter
# ═══════════════════════════════════════════════════════════════════════════════
md(
"## 6. Training Time vs MRR (Pareto View)\n"
"\n"
"Marker **shape** = method, **color** = encoder. Log-scale x-axis."
)

code(
"if not df_best.empty:\n"
"    ds_list = sorted(df_best['dataset'].unique())\n"
"    n_ds = len(ds_list)\n"
"\n"
"    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), squeeze=False)\n"
"    for i, ds in enumerate(ds_list):\n"
"        ax = axes[0, i]\n"
"        sub = df_best[df_best['dataset'] == ds]\n"
"\n"
"        for _, row in sub.iterrows():\n"
"            m = METHOD_MARKERS.get(row['method'], 'x')\n"
"            c = ENCODER_COLORS.get(row['encoder'], '#999')\n"
"            ax.scatter(row['train_time'], row['mrr'],\n"
"                       marker=m, color=c, s=100, zorder=5,\n"
"                       edgecolors='black', linewidths=0.5)\n"
"            ax.annotate(f'{row[\"method\"][:8]}\\n{row[\"encoder\"]}',\n"
"                        (row['train_time'], row['mrr']),\n"
"                        fontsize=7, alpha=0.75, ha='left',\n"
"                        xytext=(6, 2), textcoords='offset points')\n"
"\n"
"        ax.set_xscale('log')\n"
"        ax.set_xlabel('Training Time (s, log scale)')\n"
"        ax.set_ylabel('MRR')\n"
"        ax.set_title(ds, fontsize=12, fontweight='bold')\n"
"        ax.grid(True, alpha=0.3)\n"
"\n"
"    # Build combined legend\n"
"    from matplotlib.lines import Line2D\n"
"    legend_enc = [Line2D([0], [0], marker='o', color='w',\n"
"                         markerfacecolor=c, markersize=8, label=e)\n"
"                  for e, c in ENCODER_COLORS.items()]\n"
"    legend_meth = [Line2D([0], [0], marker=m, color='w',\n"
"                          markerfacecolor='gray', markersize=8, label=meth)\n"
"                   for meth, m in METHOD_MARKERS.items()]\n"
"    axes[0, -1].legend(handles=legend_enc + legend_meth,\n"
"                       loc='lower right', fontsize=8, ncol=1)\n"
"\n"
"    fig.suptitle('MRR vs Training Time (best config)',\n"
"                 fontsize=14, fontweight='bold', y=1.02)\n"
"    plt.tight_layout()\n"
"    plt.show()"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Cross-dataset relative MRR
# ═══════════════════════════════════════════════════════════════════════════════
md(
"## 7. Cross-Dataset: Relative MRR (vs Full Graph)\n"
"\n"
"Each bar = MRR of method / MRR of Full Graph for the same (dataset, encoder).\n"
"Values > 1 mean the method **outperforms** the full-graph baseline."
)

code(
"if not df_best.empty and 'Full Graph' in df_best['method'].values:\n"
"    fg = df_best[df_best['method'] == 'Full Graph'][['dataset', 'encoder', 'mrr']]\n"
"    fg = fg.rename(columns={'mrr': 'fg_mrr'})\n"
"    merged = df_best.merge(fg, on=['dataset', 'encoder'], how='left')\n"
"    merged['rel_mrr'] = merged['mrr'] / merged['fg_mrr'].clip(lower=1e-8)\n"
"    merged = merged[merged['method'] != 'Full Graph']\n"
"\n"
"    ds_list = sorted(merged['dataset'].unique())\n"
"    n_ds = len(ds_list)\n"
"    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 4.5), squeeze=False,\n"
"                             sharey=True)\n"
"\n"
"    for i, ds in enumerate(ds_list):\n"
"        ax = axes[0, i]\n"
"        sub = merged[merged['dataset'] == ds].sort_values('rel_mrr', ascending=True)\n"
"        labels = sub['method'] + '\\n' + sub['encoder']\n"
"        colors = [ENCODER_COLORS.get(e, '#999') for e in sub['encoder']]\n"
"        ax.barh(range(len(sub)), sub['rel_mrr'], color=colors, alpha=0.85)\n"
"        ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)\n"
"        ax.set_yticks(range(len(sub)))\n"
"        ax.set_yticklabels(labels, fontsize=9)\n"
"        ax.set_xlabel('MRR / Full Graph MRR')\n"
"        ax.set_title(ds, fontsize=12, fontweight='bold')\n"
"\n"
"    fig.suptitle('Relative MRR vs Full Graph Baseline',\n"
"                 fontsize=14, fontweight='bold', y=1.02)\n"
"    plt.tight_layout()\n"
"    plt.show()\n"
"else:\n"
"    print('Need Full Graph results to compute relative MRR.')"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Hits ladder
# ═══════════════════════════════════════════════════════════════════════════════
md("## 8. Hits@k Ladder (Best Config)")

code(
"hits_cols = ['hits@1', 'hits@3', 'hits@10', 'hits@50']\n"
"avail_hits = [c for c in hits_cols if c in df_best.columns and df_best[c].sum() > 0]\n"
"\n"
"if not df_best.empty and avail_hits:\n"
"    ds_list = sorted(df_best['dataset'].unique())\n"
"    for ds in ds_list:\n"
"        sub = df_best[df_best['dataset'] == ds].copy()\n"
"        sub['label'] = sub['encoder'] + ' / ' + sub['method']\n"
"        sub = sub.sort_values('mrr', ascending=False)\n"
"\n"
"        fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(sub)), 5))\n"
"        x = np.arange(len(sub))\n"
"        width = 0.8 / len(avail_hits)\n"
"\n"
"        for j, hk in enumerate(avail_hits):\n"
"            ax.bar(x + j * width, sub[hk], width, label=hk)\n"
"\n"
"        ax.set_xticks(x + width * (len(avail_hits) - 1) / 2)\n"
"        ax.set_xticklabels(sub['label'], rotation=35, ha='right', fontsize=9)\n"
"        ax.set_ylabel('Score')\n"
"        ax.set_title(f'{ds} — Hits@k Comparison (best config)',\n"
"                     fontsize=13, fontweight='bold')\n"
"        ax.legend(fontsize=9)\n"
"        ax.grid(True, alpha=0.3, axis='y')\n"
"        plt.tight_layout()\n"
"        plt.show()\n"
"else:\n"
"    print('No Hits@k data available.')"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 8e. Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
md("## 8e. MRR Heatmap (Encoder x Method)")

code(
"if not df_best.empty:\n"
"    ds_list = sorted(df_best['dataset'].unique())\n"
"    for ds in ds_list:\n"
"        sub = df_best[df_best['dataset'] == ds]\n"
"        pivot = sub.pivot_table(\n"
"            index='encoder', columns='method', values='mrr', aggfunc='max')\n"
"        if pivot.empty: continue\n"
"\n"
"        fig, ax = plt.subplots(figsize=(max(6, 1.6 * pivot.shape[1]), 3.5))\n"
"        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu',\n"
"                    cbar_kws={'label': 'MRR'}, ax=ax, linewidths=0.5)\n"
"        ax.set_title(f'{ds} — MRR Heatmap',\n"
"                     fontsize=13, fontweight='bold')\n"
"        plt.tight_layout()\n"
"        plt.show()"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Full metric table (supplement / audit)
# ═══════════════════════════════════════════════════════════════════════════════
md(
"## 9. Full Metric Table (All Configs)\n"
"\n"
"Every experiment row including sub-configs — for audit and appendix."
)

code(
"if not df.empty:\n"
"    metric_cols = ['mrr', 'auc', 'ap', 'hits@1', 'hits@3', 'hits@10',\n"
"                   'hits@50', 'hits@100']\n"
"    avail = [c for c in metric_cols if c in df.columns]\n"
"    show = ['dataset', 'encoder', 'method', 'config'] + avail + ['train_time']\n"
"    show = [c for c in show if c in df.columns]\n"
"    tbl = df[show].sort_values(['dataset', 'method', 'config', 'encoder'])\n"
"\n"
"    fmt = {m: '{:.4f}' for m in avail}\n"
"    fmt['train_time'] = '{:.0f}s'\n"
"\n"
"    styled = tbl.style.format(fmt)\n"
"    display(styled)\n"
"else:\n"
"    print('No data.')"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Save analysis outputs
# ═══════════════════════════════════════════════════════════════════════════════
md("## 10. Save Analysis Outputs")

code(
"ANALYSIS_DIR = 'results/benchmark-analysis'\n"
"os.makedirs(ANALYSIS_DIR, exist_ok=True)\n"
"\n"
"if not df.empty:\n"
"    df.to_csv(os.path.join(ANALYSIS_DIR, 'full_table.csv'), index=False)\n"
"    print(f'Saved full_table.csv ({len(df)} rows)')\n"
"\n"
"if not df_best.empty:\n"
"    df_best.to_csv(os.path.join(ANALYSIS_DIR, 'best_config_table.csv'), index=False)\n"
"    print(f'Saved best_config_table.csv ({len(df_best)} rows)')\n"
"\n"
"    winners = []\n"
"    for (ds, enc), grp in df_best.groupby(['dataset', 'encoder']):\n"
"        best = grp.loc[grp['mrr'].idxmax()]\n"
"        winners.append({\n"
"            'dataset': ds, 'encoder': enc,\n"
"            'best_method': best['method'],\n"
"            'best_config': best['config'],\n"
"            'mrr': best['mrr'],\n"
"        })\n"
"    pd.DataFrame(winners).to_csv(\n"
"        os.path.join(ANALYSIS_DIR, 'winners.csv'), index=False)\n"
"    print(f'Saved winners.csv')\n"
"\n"
"print(f'\\nAll analysis outputs saved to {ANALYSIS_DIR}/')"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 11. Conclusion
# ═══════════════════════════════════════════════════════════════════════════════
md(
"## 11. Conclusion"
)

code(
"if not df_best.empty:\n"
"    print('=' * 80)\n"
"    print('BENCHMARK COMPARISON — SUMMARY')\n"
"    print('=' * 80)\n"
"\n"
"    methods = sorted(df_best['method'].unique())\n"
"    ds_list = sorted(df_best['dataset'].unique())\n"
"    enc_list = sorted(df_best['encoder'].unique())\n"
"\n"
"    # Win rate\n"
"    wins = defaultdict(int)\n"
"    total = 0\n"
"    for (ds, enc), grp in df_best.groupby(['dataset', 'encoder']):\n"
"        wins[grp.loc[grp['mrr'].idxmax()]['method']] += 1\n"
"        total += 1\n"
"    print(f'\\nWin rate by MRR ({total} cells):')\n"
"    for m in sorted(wins, key=wins.get, reverse=True):\n"
"        print(f'  {m:20s}: {wins[m]}/{total} ({100*wins[m]/total:.0f}%)')\n"
"\n"
"    # Per-dataset summary\n"
"    for ds in ds_list:\n"
"        print(f'\\n--- {ds} ---')\n"
"        for enc in enc_list:\n"
"            sub = df_best[(df_best['dataset'] == ds) & (df_best['encoder'] == enc)]\n"
"            if sub.empty: continue\n"
"            best = sub.loc[sub['mrr'].idxmax()]\n"
"            print(f'  {enc:5s} | {best[\"method\"]:16s} ({best[\"config\"]:>12s}) | '\n"
"                  f'MRR={best[\"mrr\"]:.4f} | Time={best[\"train_time\"]:.0f}s')\n"
"\n"
"    # Subgraph benefit\n"
"    if 'Full Graph' in df_best['method'].values:\n"
"        fg = df_best[df_best['method'] == 'Full Graph']\n"
"        others = df_best[df_best['method'] != 'Full Graph']\n"
"        if not fg.empty and not others.empty:\n"
"            fg_avg = fg['mrr'].mean()\n"
"            others_best_avg = others.groupby(['dataset', 'encoder'])['mrr'].max().mean()\n"
"            delta = others_best_avg - fg_avg\n"
"            print(f'\\nAvg MRR: Full Graph = {fg_avg:.4f}, '\n"
"                  f'Best Subgraph = {others_best_avg:.4f} '\n"
"                  f'(delta = {delta:+.4f})')\n"
"            if delta > 0.001:\n"
"                print('  -> Subgraph extraction provides a measurable benefit.')\n"
"            elif delta > -0.001:\n"
"                print('  -> Subgraph and full-graph are comparable.')\n"
"            else:\n"
"                print('  -> Full graph outperforms subgraph methods on average.')\n"
"\n"
"    print('\\n' + '=' * 80)\n"
"else:\n"
"    print('No results to summarize.')"
)

md(
"### Interpretation Guide\n"
"\n"
"**Q1: Does subgraph extraction help?**\n"
"Check the relative MRR chart (Section 7). Bars > 1.0 mean the subgraph\n"
"method beats the full-graph baseline for that (dataset, encoder).\n"
"\n"
"**Q2: Which subgraph strategy is best?**\n"
"The heatmap (Section 8e) and leaderboard (Section 1) give a direct answer.\n"
"If Learnable PPR consistently wins, the architecture search overhead is\n"
"justified.\n"
"\n"
"**Q3: Is the extra training time worth it?**\n"
"The Pareto scatter (Section 6) shows cost vs benefit. Points in the\n"
"upper-left are ideal (high MRR, low time).\n"
"\n"
"**Q4: How sensitive are static methods to their hyperparameter?**\n"
"PPR sensitivity (Section 4) and k-hop sensitivity (Section 5) show this.\n"
"Flat curves = robust; steep curves = careful tuning required.\n"
"\n"
"### Reproducibility\n"
"\n"
"All experiments were run with `SEED = 42` in `benchmark_runner.ipynb`.\n"
"For publication, re-run with 3-5 seeds and report mean +/- std.\n"
"\n"
"### Limitations\n"
"\n"
"- Single seed (extend for significance testing)\n"
"- No learnable k-hop baseline (not implemented in this codebase)\n"
"- Learnable PPR uses first-order DARTS approximation"
)

# ═══════════════════════════════════════════════════════════════════════════════
with open(r"d:\Work + Project\Thesis\MTCH\benchmark_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook written: benchmark_analysis.ipynb")
