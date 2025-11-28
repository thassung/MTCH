"""
Plug-and-play pipeline for learnable subgraph-based link prediction.
High-level API hiding implementation complexity.
"""

import torch
import torch.nn as nn
import time
from ..utils.models import GCN, SAGE, GAT, LinkPredictor
from .differentiable_ppr_selector import DifferentiablePPRSelector
from .ppr_computer import PPRComputer
from .meta_trainer import MetaLearningTrainer
from .extractor import extract_subgraph_hard, extract_subgraph_for_visualization
from .results_manager import SubgraphResultsManager


class SubgraphLinkPrediction:
    """
    Complete pipeline for learnable PPR-based subgraph link prediction.
    
    Simple usage:
        model = SubgraphLinkPrediction(encoder_type='SAGE')
        history = model.fit(train_data, val_data, epochs=100)
        predictions = model.predict(test_edges, test_data)
    
    Args:
        encoder_type: GNN type - 'GCN', 'SAGE', or 'GAT'
        hidden_dim: Hidden dimension for GNN and predictor
        num_layers: Number of GNN layers
        dropout: Dropout rate
        ppr_alpha: PPR teleport probability (default: 0.85)
        adaptive_threshold: Use adaptive percentile-based threshold
        init_alpha: Initial alpha for combining PPR_u and PPR_v
        init_threshold: Initial threshold value or percentile
        sharpness: Sigmoid sharpness for soft thresholding
        device: 'cpu' or 'cuda'
    """
    
    def __init__(self,
                 encoder_type='SAGE',
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.3,
                 ppr_alpha=0.85,
                 adaptive_threshold=True,
                 init_alpha=0.5,
                 init_threshold=0.3,
                 sharpness=1000.0,  
                 device='cuda',
                 save_results=False,
                 dataset_name=None):
        
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.ppr_alpha = ppr_alpha
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        
        # These will be initialized in fit()
        self.encoder = None
        self.predictor = None
        self.selector = None
        self.ppr_computer = None
        self.trainer = None
        self.history = None
        
        # Store init parameters
        self.adaptive_threshold = adaptive_threshold
        self.init_alpha = init_alpha
        self.init_threshold = init_threshold
        self.sharpness = sharpness
        
        # Results management
        self.save_results = save_results
        self.dataset_name = dataset_name
        self.results_manager = SubgraphResultsManager() if save_results else None
        self.exp_dir = None
        self.train_time = None
    
    def _initialize_models(self, in_channels, num_nodes):
        """Initialize encoder, predictor, and selector."""
        # Create encoder
        if self.encoder_type == 'GCN':
            self.encoder = GCN(in_channels, self.hidden_dim, self.hidden_dim,
                              self.num_layers, self.dropout)
        elif self.encoder_type == 'SAGE':
            self.encoder = SAGE(in_channels, self.hidden_dim, self.hidden_dim,
                               self.num_layers, self.dropout)
        elif self.encoder_type == 'GAT':
            self.encoder = GAT(in_channels, self.hidden_dim, self.hidden_dim,
                              self.num_layers, self.dropout, heads=4)
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
        
        # Create predictor
        self.predictor = LinkPredictor(self.hidden_dim, self.hidden_dim, 1,
                                       num_layers=3, dropout=self.dropout)
        
        # Create selector
        self.selector = DifferentiablePPRSelector(
            adaptive_threshold=self.adaptive_threshold,
            init_alpha=self.init_alpha,
            init_threshold=self.init_threshold,
            sharpness=self.sharpness
        )
        
        # Move to device
        self.encoder = self.encoder.to(self.device)
        self.predictor = self.predictor.to(self.device)
        self.selector = self.selector.to(self.device)
    
    def fit(self, 
            data,
            train_edges,
            val_edges,
            epochs=100,
            inner_steps=2,
            inner_lr=0.01,
            outer_lr=0.001,
            eval_steps=5,
            patience=20,
            use_full_graph=False,
            meta_learning_order='first',
            verbose=True):
        """
        Train the model with meta-learning.
        
        Args:
            data: PyG Data object with node features and edges
            train_edges: List of (u, v) tuples for training
            val_edges: List of (u, v) tuples for validation
            epochs: Number of meta-training epochs
            inner_steps: Steps per inner loop (GNN training, default: 2)
            inner_lr: Learning rate for GNN (inner loop)
            outer_lr: Learning rate for selector (outer loop)
            eval_steps: Evaluate every N epochs
            patience: Early stopping patience
            use_full_graph: If True, skip subgraph extraction (baseline)
            meta_learning_order: 'first' (FOMAML, memory efficient) or 'second' (full MAML)
            verbose: Print progress
        
        Returns:
            history: Training history dictionary
        """
        # Initialize models if not done
        if self.encoder is None:
            in_channels = data.x.size(1)
            num_nodes = data.num_nodes
            self._initialize_models(in_channels, num_nodes)
            
            if verbose:
                print(f"Initialized {self.encoder_type} encoder with {self.hidden_dim}D")
                print(f"Total parameters: {self.count_parameters():,}")
        
        # Initialize PPR computer
        if self.ppr_computer is None:
            self.ppr_computer = PPRComputer(data, ppr_alpha=self.ppr_alpha)
            if verbose:
                print(f"Initialized PPR computer ({data.num_nodes} nodes)")
        
        # Create trainer
        self.trainer = MetaLearningTrainer(
            selector=self.selector,
            encoder=self.encoder,
            predictor=self.predictor,
            ppr_computer=self.ppr_computer,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_steps=inner_steps,
            device=self.device,
            meta_learning_order=meta_learning_order
        )
        
        # Train
        if verbose:
            print(f"\nStarting meta-training...")
            print(f"  Training edges: {len(train_edges)}")
            print(f"  Validation edges: {len(val_edges)}")
            print(f"  Inner steps: {inner_steps}, Inner LR: {inner_lr}")
            print(f"  Outer LR: {outer_lr}")
            print(f"  Use full graph: {use_full_graph}")
        
        # Setup results saving
        if self.save_results and self.dataset_name:
            model_name = 'LearnablePPR' if not use_full_graph else 'FullGraph'
            self.exp_dir = self.results_manager.create_experiment_dir(
                self.dataset_name, model_name, self.encoder_type
            )
        
        # Track training time
        start_time = time.time()
        
        self.history = self.trainer.train(
            data=data,
            train_edges=train_edges,
            val_edges=val_edges,
            epochs=epochs,
            eval_steps=eval_steps,
            patience=patience,
            use_full_graph=use_full_graph,
            verbose=verbose
        )
        
        self.train_time = time.time() - start_time
        
        # Save results if enabled
        if self.save_results and self.exp_dir:
            self._save_training_results(
                data, train_edges, val_edges, epochs, inner_steps, 
                inner_lr, outer_lr, use_full_graph
            )
        
        return self.history
    
    def predict(self, test_edges, data, use_hard_selection=True, batch_size=32):
        """
        Predict links on test edges.
        
        Args:
            test_edges: List of (u, v) tuples to predict
            data: PyG Data object
            use_hard_selection: Use hard (discrete) subgraph selection
            batch_size: Batch size for prediction
        
        Returns:
            predictions: Tensor of prediction scores
            metadata: Dictionary with prediction statistics
        """
        from tqdm import tqdm
        
        self.encoder.eval()
        self.predictor.eval()
        self.selector.eval()
        
        data = data.to(self.device)
        predictions = []
        subgraph_sizes = []
        
        # Calculate number of batches for progress bar
        num_batches = (len(test_edges) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            # Add progress bar
            pbar = tqdm(range(0, len(test_edges), batch_size), 
                       desc='Evaluating test set', 
                       total=num_batches,
                       unit='batch')
            
            for i in pbar:
                batch_edges = test_edges[i:i+batch_size]
                
                for u, v in batch_edges:
                    # Compute PPR
                    ppr_u, ppr_v = self.ppr_computer.compute_ppr_pair(u, v)
                    ppr_u = ppr_u.to(self.device)
                    ppr_v = ppr_v.to(self.device)
                    
                    # Get selection
                    if use_hard_selection:
                        hard_mask, _, meta = self.selector.get_hard_mask(ppr_u, ppr_v)
                        weighted_x = data.x * hard_mask.unsqueeze(1)
                        subgraph_sizes.append(meta['num_selected'])
                    else:
                        soft_mask, _, meta = self.selector(ppr_u, ppr_v)
                        weighted_x = data.x * soft_mask.unsqueeze(1)
                        subgraph_sizes.append(meta['num_selected_hard'])
                    
                    # Encode and predict
                    h = self.encoder(weighted_x, data.edge_index)
                    pred = self.predictor(h[u].unsqueeze(0), h[v].unsqueeze(0))
                    predictions.append(pred)
                
                # Update progress bar with statistics
                if subgraph_sizes:
                    avg_size = sum(subgraph_sizes) / len(subgraph_sizes)
                    pbar.set_postfix({
                        'edges': f'{len(predictions)}/{len(test_edges)}',
                        'avg_subgraph': f'{avg_size:.0f}'
                    })
        
        predictions = torch.cat(predictions).cpu()
        
        metadata = {
            'num_predictions': len(predictions),
            'avg_subgraph_size': sum(subgraph_sizes) / len(subgraph_sizes) if subgraph_sizes else 0,
            'min_subgraph_size': min(subgraph_sizes) if subgraph_sizes else 0,
            'max_subgraph_size': max(subgraph_sizes) if subgraph_sizes else 0,
            'pred_mean': predictions.mean().item(),
            'pred_std': predictions.std().item()
        }
        
        # Save test results and subgraph statistics if enabled
        if self.save_results and self.exp_dir and subgraph_sizes:
            self._save_test_results(predictions, metadata, subgraph_sizes, data.num_nodes)
        
        return predictions, metadata
    
    def get_selector_params(self):
        """
        Get learned selector parameters.
        
        Returns:
            params: Dictionary with alpha, threshold, etc.
        """
        if self.selector is None:
            return None
        
        params = {
            'alpha': self.selector.get_alpha().item(),
            'threshold_percentile': (self.selector.get_threshold_percentile().item() 
                                    if self.adaptive_threshold else None),
            'sharpness': self.selector.sharpness,
            'adaptive_threshold': self.adaptive_threshold
        }
        
        return params
    
    def visualize_subgraph(self, data, u, v, save_path=None):
        """
        Visualize selected subgraph for an edge pair.
        
        Args:
            data: PyG Data object
            u: Source node
            v: Target node
            save_path: Path to save visualization (optional)
        
        Returns:
            fig: matplotlib figure object
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Get selected nodes
        ppr_u, ppr_v = self.ppr_computer.compute_ppr_pair(u, v)
        ppr_u = ppr_u.to(self.device)
        ppr_v = ppr_v.to(self.device)
        
        hard_mask, selected_indices, meta = self.selector.get_hard_mask(ppr_u, ppr_v)
        
        # Extract subgraph for visualization
        G, pos, node_colors, edge_colors = extract_subgraph_for_visualization(
            data, selected_indices, u, v
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                              width=2, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label=f'Source (u={u})'),
            Patch(facecolor='blue', label=f'Target (v={v})'),
            Patch(facecolor='lightgray', label='Selected neighbors')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        # Add title with selector info
        selector_params = self.get_selector_params()
        title = (f"Learned PPR Subgraph (α={selector_params['alpha']:.3f})\n"
                f"Nodes: {meta['num_selected']}/{data.num_nodes}")
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        return fig
    
    def count_parameters(self):
        """Count total trainable parameters."""
        total = 0
        if self.encoder is not None:
            total += sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        if self.predictor is not None:
            total += sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        if self.selector is not None:
            total += sum(p.numel() for p in self.selector.parameters() if p.requires_grad)
        return total
    
    def _save_training_results(self, data, train_edges, val_edges, epochs, 
                               inner_steps, inner_lr, outer_lr, use_full_graph):
        """Save training results and visualizations."""
        if not self.results_manager or not self.exp_dir:
            return
        
        # Save configuration
        config = {
            'model_name': 'LearnablePPR' if not use_full_graph else 'FullGraph',
            'encoder_type': self.encoder_type,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'ppr_alpha': self.ppr_alpha,
            'adaptive_threshold': self.adaptive_threshold,
            'init_alpha': self.init_alpha,
            'init_threshold': self.init_threshold,
            'sharpness': self.sharpness,
            'training': {
                'epochs': epochs,
                'inner_steps': inner_steps,
                'inner_lr': inner_lr,
                'outer_lr': outer_lr,
                'use_full_graph': use_full_graph,
                'train_edges': len(train_edges),
                'val_edges': len(val_edges)
            },
            'device': str(self.device),
            'total_parameters': self.count_parameters()
        }
        self.results_manager.save_config(self.exp_dir, config)
        
        # Save training history
        self.results_manager.save_training_history(self.exp_dir, self.history)
        
        # Save selector parameters
        if not use_full_graph:
            selector_params = self.get_selector_params()
            self.results_manager.save_selector_params(
                self.exp_dir, selector_params, self.history
            )
        
        # Plot training curves
        model_name = f"LearnablePPR_{self.encoder_type}" if not use_full_graph else f"FullGraph_{self.encoder_type}"
        self.results_manager.plot_training_curve(
            self.exp_dir, self.history, model_name, self.dataset_name
        )
        
        # Plot selector evolution
        if not use_full_graph and 'selector_alpha' in self.history:
            self.results_manager.plot_selector_evolution(
                self.exp_dir, self.history, model_name
            )
        
        # Save summary
        summary_data = {
            'dataset': self.dataset_name,
            'model_name': 'LearnablePPR' if not use_full_graph else 'FullGraph',
            'encoder_type': self.encoder_type,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_params': self.count_parameters(),
            'train_time': self.train_time,
            'epochs_trained': len(self.history['train_loss']),
            'best_epoch': self.history.get('best_epoch', 0),
            'stopped_early': self.history.get('best_epoch', 0) < epochs,
            'best_val_loss': self.history.get('best_val_loss', 0)
        }
        
        if not use_full_graph:
            summary_data['selector_params'] = self.get_selector_params()
        
        self.results_manager.save_summary(self.exp_dir, summary_data)
        
        print(f"\n✓ Results saved to {self.exp_dir}")
    
    def _save_test_results(self, predictions, metadata, subgraph_sizes, total_nodes):
        """Save test results and subgraph statistics."""
        if not self.results_manager or not self.exp_dir:
            return
        
        # Update summary with test results
        summary_data = {
            'dataset': self.dataset_name,
            'model_name': 'LearnablePPR',
            'encoder_type': self.encoder_type,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_params': self.count_parameters(),
            'train_time': self.train_time,
            'best_val_loss': self.history.get('best_val_loss', 0),
            'selector_params': self.get_selector_params(),
            'test_results': {
                'pred_mean': metadata['pred_mean'],
                'pred_std': metadata['pred_std'],
                'num_predictions': metadata['num_predictions']
            },
            'subgraph_stats': {
                'avg_size': metadata['avg_subgraph_size'],
                'min_size': metadata['min_subgraph_size'],
                'max_size': metadata['max_subgraph_size'],
                'std_size': sum((x - metadata['avg_subgraph_size'])**2 for x in subgraph_sizes) / len(subgraph_sizes)**0.5 if len(subgraph_sizes) > 1 else 0
            }
        }
        
        # Re-save summary with test results
        self.results_manager.save_summary(self.exp_dir, summary_data)
        
        # Plot subgraph statistics
        model_name = f"LearnablePPR_{self.encoder_type}"
        self.results_manager.plot_subgraph_statistics(
            self.exp_dir, subgraph_sizes, total_nodes, model_name
        )
        
        print(f"✓ Test results and visualizations saved to {self.exp_dir}")
    
    def save(self, path):
        """Save model to file."""
        torch.save({
            'encoder_state': self.encoder.state_dict() if self.encoder else None,
            'predictor_state': self.predictor.state_dict() if self.predictor else None,
            'selector_state': self.selector.state_dict() if self.selector else None,
            'config': {
                'encoder_type': self.encoder_type,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'ppr_alpha': self.ppr_alpha,
                'adaptive_threshold': self.adaptive_threshold
            },
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore config
        config = checkpoint['config']
        self.encoder_type = config['encoder_type']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.ppr_alpha = config['ppr_alpha']
        self.adaptive_threshold = config['adaptive_threshold']
        
        # Load states
        if checkpoint['encoder_state']:
            # Need to determine in_channels from loaded state
            # This is a simplified approach; may need refinement
            self.encoder.load_state_dict(checkpoint['encoder_state'])
        if checkpoint['predictor_state']:
            self.predictor.load_state_dict(checkpoint['predictor_state'])
        if checkpoint['selector_state']:
            self.selector.load_state_dict(checkpoint['selector_state'])
        
        self.history = checkpoint.get('history')
        
        print(f"Model loaded from {path}")

