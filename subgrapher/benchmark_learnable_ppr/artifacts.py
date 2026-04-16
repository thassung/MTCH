"""
Artifact saving for learnable PPR experiments.

Directory layout per run:
    <run_dir>/
        manifest.json          — top-level metadata (config, timing, metrics summary)
        test_metrics.json      — full test evaluation results
        search_history.json    — Phase 1 loss curves & config distributions
        finetune_history.json  — Phase 2 loss curves & validation MRR
        config_distributions.json  — per-split config counts + labels
        checkpoints/
            encoder.pt         — fine-tuned encoder state_dict
            predictor.pt       — fine-tuned predictor state_dict
            arch_net.pt        — architecture search net state_dict
            search_model.pt    — Phase 1 AutoLinkPPR state_dict
"""

import json
import os
from datetime import datetime
from pathlib import Path

import torch


def _to_serializable(obj):
    """Recursively convert numpy/tensor types to Python builtins for JSON."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _write_json(path, data):
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_to_serializable(data), f, indent=2)


def save_learnable_ppr_experiment(
    run_dir,
    dataset_name,
    encoder_type,
    run_id,
    teleport_values,
    alpha,
    epsilon,
    exp,
    multi_scale_ppr,
    extra_config=None,
):
    """
    Persist a complete learnable-PPR experiment to disk.

    Args:
        run_dir: Path (or str) for this run's output directory.
        dataset_name: e.g. 'FB15K237'.
        encoder_type: e.g. 'SAGE'.
        run_id: Unique run timestamp / identifier string.
        teleport_values: List of PPR teleport probabilities used.
        alpha: Combination weight list (e.g. [0.5]).
        epsilon: Approximate PPR precision threshold.
        exp: The experiment dict stored in all_results[dataset][encoder],
             expected keys: search_history, ft_history, test_results,
             ft_encoder, ft_predictor, model (AutoLinkPPR), arch_net,
             train_configs, val_configs, test_configs,
             train_counts, val_counts, test_counts.
        multi_scale_ppr: MultiScalePPR instance (for config labels).
        extra_config: Optional dict of additional hyperparameters.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    search_hist = exp.get('search_history', {})
    ft_hist = exp.get('ft_history', {})
    test_results = exp.get('test_results', {})

    # --- manifest.json ---
    manifest = {
        'run_id': run_id,
        'dataset': dataset_name,
        'encoder': encoder_type,
        'timestamp': datetime.now().isoformat(),
        'teleport_values': teleport_values,
        'alpha': alpha,
        'ppr_epsilon': epsilon,
        'search': {
            'best_epoch': search_hist.get('best_epoch'),
            'best_val_loss': search_hist.get('best_val_loss'),
            'total_time': search_hist.get('total_time'),
        },
        'finetune': {
            'best_epoch': ft_hist.get('best_epoch'),
            'best_val_mrr': ft_hist.get('best_val_mrr'),
            'stopped_early': ft_hist.get('stopped_early'),
            'total_time': ft_hist.get('total_time'),
        },
        'test_summary': {
            k: float(v) for k, v in test_results.items()
        } if test_results else {},
    }
    if extra_config:
        manifest['hyperparameters'] = extra_config
    _write_json(str(run_dir / 'manifest.json'), manifest)

    # --- test_metrics.json ---
    if test_results:
        _write_json(str(run_dir / 'test_metrics.json'), test_results)

    # --- search_history.json (loss curves only, skip heavy state dicts) ---
    search_save = {
        k: v for k, v in search_hist.items()
        if k not in ('best_model_state', 'best_arch_state')
    }
    _write_json(str(run_dir / 'search_history.json'), search_save)

    # --- finetune_history.json ---
    ft_save = {
        k: v for k, v in ft_hist.items()
    }
    _write_json(str(run_dir / 'finetune_history.json'), ft_save)

    # --- config_distributions.json ---
    config_labels = ([f"({tu},{tv})" for tu, tv in multi_scale_ppr.config_labels]
                     if multi_scale_ppr else [])
    config_dist = {
        'labels': config_labels,
        'train': _to_serializable(exp.get('train_counts')),
        'val': _to_serializable(exp.get('val_counts')),
        'test': _to_serializable(exp.get('test_counts')),
    }
    _write_json(str(run_dir / 'config_distributions.json'), config_dist)

    # --- model checkpoints ---
    if 'ft_encoder' in exp:
        torch.save(exp['ft_encoder'].state_dict(), str(ckpt_dir / 'encoder.pt'))
    if 'ft_predictor' in exp:
        torch.save(exp['ft_predictor'].state_dict(), str(ckpt_dir / 'predictor.pt'))
    if 'arch_net' in exp:
        torch.save(exp['arch_net'].state_dict(), str(ckpt_dir / 'arch_net.pt'))
    if 'model' in exp:
        torch.save(exp['model'].state_dict(), str(ckpt_dir / 'search_model.pt'))

    # --- per-edge config indices (for reproducibility) ---
    if 'train_configs' in exp:
        torch.save(exp['train_configs'], str(run_dir / 'train_config_indices.pt'))
    if 'val_configs' in exp:
        torch.save(exp['val_configs'], str(run_dir / 'val_config_indices.pt'))
    if 'test_configs' in exp:
        torch.save(exp['test_configs'], str(run_dir / 'test_config_indices.pt'))


def load_learnable_ppr_manifest(run_dir):
    """Load the manifest.json for a saved run. Returns dict or None."""
    path = Path(run_dir) / 'manifest.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
