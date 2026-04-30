"""
Learnable PPR-based subgraph benchmark.
Learns per-edge (teleport_u, teleport_v) configuration via bi-level optimization,
then fine-tunes on extracted subgraphs with learned configurations.
"""

from .multi_scale_ppr import MultiScalePPR
from .autolink_ppr import AutoLinkPPR
from .search_net import PPRSearchNet
from .run_learnable_ppr_benchmark import run_learnable_ppr_benchmark
from .option_a_model import LPPRGNN, PPRScaleSelector
from .option_a_extractor import LPPRSubgraphExtractor
from .run_option_a_benchmark import run_lppr_benchmark, run_lppr_experiment, run_one


def resolve_alpha_weights(alpha):
    """
    Convert an alpha list to (w_u, w_v) combination weights.

    - length 1: w_u = alpha[0], w_v = 1 - alpha[0]
    - length 2: normalize so w_u + w_v = 1
    """
    if isinstance(alpha, (int, float)):
        return float(alpha), 1.0 - float(alpha)
    if len(alpha) == 1:
        return float(alpha[0]), 1.0 - float(alpha[0])
    if len(alpha) == 2:
        s = float(alpha[0]) + float(alpha[1])
        return float(alpha[0]) / s, float(alpha[1]) / s
    raise ValueError(f"alpha must have 1 or 2 elements, got {len(alpha)}")


__all__ = [
    'MultiScalePPR',
    'AutoLinkPPR',
    'PPRSearchNet',
    'run_learnable_ppr_benchmark',
    'resolve_alpha_weights',
    'evaluate_learnable_ppr_fullgraph',
    # LPPR
    'LPPRGNN',
    'PPRScaleSelector',
    'LPPRSubgraphExtractor',
    'run_lppr_benchmark',
    'run_lppr_experiment',
    'run_one',
]
