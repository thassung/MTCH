"""
Static PPR-based subgraph benchmark.
Compares different top-k values for PPR-based subgraph selection.
"""

from .ppr_extractor import StaticPPRExtractor
from .run_ppr_benchmark import run_ppr_benchmark

__all__ = ['StaticPPRExtractor', 'run_ppr_benchmark']

