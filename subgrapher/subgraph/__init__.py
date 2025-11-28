"""
Learnable PPR-based subgraph selection for link prediction.

High-level API:
    from subgrapher.subgraph import SubgraphLinkPrediction
    
    model = SubgraphLinkPrediction(encoder_type='SAGE')
    history = model.fit(train_data, val_data, epochs=100)
    predictions = model.predict(test_edges, test_data)

Low-level components:
    - DifferentiablePPRSelector: Learnable node selector
    - PPRComputer: Efficient PPR computation with caching
    - MetaLearningTrainer: Bi-level optimization trainer
    - extract_subgraph_soft/hard: Subgraph extraction utilities
"""

from .pipeline import SubgraphLinkPrediction
from .differentiable_ppr_selector import DifferentiablePPRSelector
from .ppr_computer import PPRComputer
from .meta_trainer import MetaLearningTrainer
from .results_manager import SubgraphResultsManager
from .extractor import (
    extract_subgraph_soft,
    extract_subgraph_hard,
    extract_k_hop_subgraph,
    extract_subgraph_for_visualization
)

__all__ = [
    # High-level API
    'SubgraphLinkPrediction',
    
    # Core components
    'DifferentiablePPRSelector',
    'PPRComputer',
    'MetaLearningTrainer',
    'SubgraphResultsManager',
    
    # Utilities
    'extract_subgraph_soft',
    'extract_subgraph_hard',
    'extract_k_hop_subgraph',
    'extract_subgraph_for_visualization'
]

__version__ = '0.1.0'

