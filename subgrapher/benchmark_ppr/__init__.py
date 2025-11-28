"""
Static PPR Subgraph Baseline Benchmark Module

Implements non-learnable PPR-based subgraph selection with fixed alpha=0.5.
Grid search over top-k values: [50, 100, 200, 300, 500].
"""

from .ppr_extractor import StaticPPRExtractor
from .run_ppr_benchmark import run_full_benchmark

__all__ = [
    'StaticPPRExtractor',
    'run_full_benchmark'
]

