"""
Static PPR-based subgraph benchmark.
Uses LCILP-style approximate PPR + conductance sweep cut.
"""

from .ppr_extractor import StaticPPRExtractor
from .run_ppr_benchmark import run_ppr_benchmark

__all__ = ['StaticPPRExtractor', 'run_ppr_benchmark']

