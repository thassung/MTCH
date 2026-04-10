"""
Static k-hop subgraph benchmark.
Compares k-hop neighborhood extraction for k=2 and k=3.
"""

from .khop_extractor import StaticKHopExtractor
from .run_khop_benchmark import run_khop_benchmark

__all__ = ['StaticKHopExtractor', 'run_khop_benchmark']

