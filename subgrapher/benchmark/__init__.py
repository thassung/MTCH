"""
Benchmark module for link prediction models.
separate GNN encoders and LinkPredictor.
"""

from .models import GCN, SAGE, GAT, LinkPredictor, LinkPredictionModel
from .data_prep import prepare_link_prediction_data, add_node_features
from .evaluator import evaluate_link_prediction
from .trainer import train_epoch, train_model

__all__ = [
    'GCN',
    'SAGE',
    'GAT',
    'LinkPredictor',
    'LinkPredictionModel',
    'prepare_link_prediction_data',
    'add_node_features',
    'evaluate_link_prediction',
    'train_epoch',
    'train_model',
]

