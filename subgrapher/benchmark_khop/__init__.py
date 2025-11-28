"""
Static k-hop Subgraph Baseline Benchmark Module

Implements non-learnable k-hop neighborhood extraction.
Grid search over k values: [2, 3] (PS2 standard).
"""

from .khop_extractor import StaticKHopExtractor
from .run_khop_benchmark import run_full_benchmark

__all__ = [
    'StaticKHopExtractor',
    'run_full_benchmark'
]

