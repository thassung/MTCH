"""
Meta-learning trainer for learnable subgraph selection.
Implements bi-level optimization: selector learns from validation performance.
"""

import torch
import torch.nn as nn
import time
import copy
from tqdm import tqdm


class MetaLearningTrainer:
    """
    Meta-learning trainer for bi-level optimization.
    
    Inner loop: Train GNN on training set with current subgraph selection
    Outer loop: Update selector based on validation loss (gradients flow through inner loop)
    
    Args:
        selector: DifferentiablePPRSelector instance
        encoder: GNN encoder (GCN/SAGE/GAT)
        predictor: LinkPredictor instance
        ppr_computer: PPRComputer instance
        inner_lr: Learning rate for inner loop (GNN)
        outer_lr: Learning rate for outer loop (selector)
        inner_steps: Number of inner loop steps per meta-step
        device: 'cpu' or 'cuda'
    """
    
    def __init__(self,
                 selector,
                 encoder,
                 predictor,
                 ppr_computer,
                 inner_lr=0.01,
                 outer_lr=0.001,
                 inner_steps=5,
                 device='cpu',
                 meta_learning_order='first'):
        
        self.selector = selector.to(device)
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.ppr_computer = ppr_computer
        
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.device = device
        self.meta_learning_order = meta_learning_order  # 'first'=FOMAML (memory efficient), 'second'=full MAML 
        
        # Optimizer for selector (outer loop)
        self.selector_optimizer = torch.optim.Adam(
            self.selector.parameters(), 
            lr=outer_lr
        )
    
    def compute_loss_on_edges(self, encoder, predictor, data, edge_pairs, 
                               use_subgraph=False, selector=None):
        """
        Compute link prediction loss on given edge pairs.
        
        Args:
            encoder: GNN encoder
            predictor: Link predictor
            data: PyG Data object
            edge_pairs: Tensor [2, num_edges] or list of (u, v) tuples
            use_subgraph: Whether to use subgraph extraction
            selector: DifferentiablePPRSelector (required if use_subgraph=True)
        
        Returns:
            loss: Average loss over edges
            predictions: Model predictions
        """
        if not use_subgraph:
            # Full graph encoding (faster, but no subgraph selection)
            h = encoder(data.x, data.edge_index)
            
            if isinstance(edge_pairs, torch.Tensor) and edge_pairs.dim() == 2:
                src, dst = edge_pairs[0], edge_pairs[1]
            else:
                src = torch.tensor([u for u, v in edge_pairs], device=self.device)
                dst = torch.tensor([v for u, v in edge_pairs], device=self.device)
            
            pos_pred = predictor(h[src], h[dst])
            
            # Negative sampling
            neg_dst = torch.randint(0, data.num_nodes, (len(src),), 
                                   device=self.device, dtype=torch.long)
            neg_pred = predictor(h[src], h[neg_dst])
            
            # Binary cross-entropy loss
            pos_loss = -torch.log(pos_pred + 1e-15).mean()
            neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
            loss = pos_loss + neg_loss
            
            return loss, pos_pred, {}
        
        else:
            # Per-edge subgraph extraction (slower, but enables learning)
            if selector is None:
                raise ValueError("selector must be provided when use_subgraph=True")
            
            losses = []
            predictions = []
            mask_diagnostics = {}  # Track diagnostics for first edge
            
            for u, v in edge_pairs:
                # Compute PPR for this edge
                ppr_u, ppr_v = self.ppr_computer.compute_ppr_pair(u, v)
                ppr_u = ppr_u.to(self.device)
                ppr_v = ppr_v.to(self.device)
                
                # Get soft mask from selector
                soft_mask, _, mask_meta = selector(ppr_u, ppr_v)
                
                # Track mask statistics (for first edge only, to avoid spam)
                if len(losses) == 0:
                    mask_diagnostics = {
                        'mask_mean': soft_mask.mean().item(),
                        'mask_min': soft_mask.min().item(),
                        'mask_max': soft_mask.max().item(),
                        'num_high': (soft_mask > 0.9).sum().item(),
                        'num_low': (soft_mask < 0.1).sum().item(),
                        'num_middle': ((soft_mask >= 0.1) & (soft_mask <= 0.9)).sum().item(),
                        'alpha': mask_meta['alpha'],
                        'threshold': mask_meta['threshold']
                    }
                
                # Weight features by soft mask (differentiable)
                weighted_x = data.x * soft_mask.unsqueeze(1)
                
                # Encode with weighted features
                h = encoder(weighted_x, data.edge_index)
                
                # Predict on this edge
                pos_pred = predictor(h[u].unsqueeze(0), h[v].unsqueeze(0))
                
                # Negative sample
                neg_v = torch.randint(0, data.num_nodes, (1,), 
                                     device=self.device, dtype=torch.long)
                neg_pred = predictor(h[u].unsqueeze(0), h[neg_v])
                
                # Loss for this edge
                edge_loss = -torch.log(pos_pred + 1e-15) - torch.log(1 - neg_pred + 1e-15)
                losses.append(edge_loss)
                predictions.append(pos_pred)
            
            loss = torch.stack(losses).mean()
            predictions = torch.cat(predictions)
            
            return loss, predictions, mask_diagnostics
    
    def meta_train_step(self, data, train_edges, val_edges, 
                       use_full_graph=False):
        """
        Single meta-training step.
        
        Args:
            data: PyG Data object
            train_edges: List of (u, v) tuples for training
            val_edges: List of (u, v) tuples for validation
            use_full_graph: If True, use full graph (no subgraph extraction)
        
        Returns:
            metrics: Dictionary with losses and metadata
        """
        data = data.to(self.device)
        
        # Save initial GNN parameters
        initial_encoder_state = copy.deepcopy(self.encoder.state_dict())
        initial_predictor_state = copy.deepcopy(self.predictor.state_dict())
        
        # Inner loop: Train GNN on training set
        inner_optimizer = torch.optim.SGD(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=self.inner_lr
        )
        
        train_loss_history = []
        selector_diagnostics = []  # Track selector behavior
        
        for inner_step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            # Sample batch of training edges
            batch_size = min(64, len(train_edges))  # Increased from 32 for more stable gradients
            batch_indices = torch.randperm(len(train_edges))[:batch_size]
            batch_edges = [train_edges[i] for i in batch_indices]
            
            # Compute loss with current selector
            train_loss, _, mask_diag = self.compute_loss_on_edges(
                self.encoder, self.predictor, data, batch_edges,
                use_subgraph=not use_full_graph,
                selector=self.selector if not use_full_graph else None
            )
            
            # Store diagnostics for first step
            if inner_step == 0 and mask_diag:
                selector_diagnostics.append(mask_diag)
            
            # FOMAML: Only keep graph if using second-order meta-learning
            # NOTE: create_graph=True is expensive (5-10x memory), FOMAML uses first-order approx
            create_graph = (self.meta_learning_order == 'second')
            train_loss.backward(create_graph=create_graph)
            
            # Check gradient norms before step
            total_grad_norm = 0.0
            for p in list(self.encoder.parameters()) + list(self.predictor.parameters()):
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            inner_optimizer.step()
            
            train_loss_history.append(train_loss.item())
            
            # Verbose logging for first meta-step to debug
            if len(train_loss_history) == 1 and inner_step == 0:
                # Store for later comparison
                self._first_inner_loss = train_loss.item()
                self._first_grad_norm = total_grad_norm
        
        # Outer loop: Evaluate on validation and update selector
        self.selector_optimizer.zero_grad()
        
        # Sample validation batch
        val_batch_size = min(16, len(val_edges))
        val_batch_indices = torch.randperm(len(val_edges))[:val_batch_size]
        val_batch_edges = [val_edges[i] for i in val_batch_indices]
        
        # Compute validation loss (gradients flow through selector)
        val_loss, val_pred, _ = self.compute_loss_on_edges(
            self.encoder, self.predictor, data, val_batch_edges,
            use_subgraph=not use_full_graph,
            selector=self.selector if not use_full_graph else None
        )
        
        # Backprop through validation loss to update selector
        val_loss.backward()
        self.selector_optimizer.step()
        
        # Restore GNN parameters for next meta-step
        self.encoder.load_state_dict(initial_encoder_state)
        self.predictor.load_state_dict(initial_predictor_state)
        
        # Collect metrics
        metrics = {
            'train_loss': sum(train_loss_history) / len(train_loss_history),
            'train_loss_initial': train_loss_history[0] if train_loss_history else 0,
            'train_loss_final': train_loss_history[-1] if train_loss_history else 0,
            'train_loss_decrease': (train_loss_history[0] - train_loss_history[-1]) if len(train_loss_history) > 1 else 0,
            'val_loss': val_loss.item(),
            'val_pred_mean': val_pred.mean().item(),
            'inner_steps': self.inner_steps,
            'selector_diagnostics': selector_diagnostics[0] if selector_diagnostics else {}
        }
        
        # Add selector metadata
        if not use_full_graph:
            with torch.no_grad():
                # Get selector parameters for first val edge
                u, v = val_batch_edges[0]
                ppr_u, ppr_v = self.ppr_computer.compute_ppr_pair(u, v)
                ppr_u, ppr_v = ppr_u.to(self.device), ppr_v.to(self.device)
                _, _, selector_meta = self.selector(ppr_u, ppr_v)
                metrics['selector'] = selector_meta
        
        return metrics
    
    def train(self, data, train_edges, val_edges, 
              epochs=100, eval_steps=5, patience=20,
              use_full_graph=False, verbose=True):
        """
        Full meta-training loop.
        
        Args:
            data: PyG Data object
            train_edges: List of (u, v) tuples for training
            val_edges: List of (u, v) tuples for validation
            epochs: Number of meta-epochs
            eval_steps: Evaluate every N epochs
            patience: Early stopping patience
            use_full_graph: If True, train without subgraph extraction
            verbose: Print progress
        
        Returns:
            history: Training history dictionary
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'selector_alpha': [],
            'selector_threshold': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        iterator = tqdm(range(epochs), desc='Meta-Training') if verbose else range(epochs)
        
        for epoch in iterator:
            # Meta-training step
            metrics = self.meta_train_step(
                data, train_edges, val_edges,
                use_full_graph=use_full_graph
            )
            
            # Detailed logging for first epoch
            if epoch == 0 and verbose:
                print(f"\n[Epoch 1 Diagnostic]")
                print(f"  Inner loop (GNN training):")
                print(f"    Initial loss: {metrics['train_loss_initial']:.6f}")
                print(f"    Final loss: {metrics['train_loss_final']:.6f}")
                print(f"    Decrease: {metrics['train_loss_decrease']:.6f}")
                if metrics['train_loss_decrease'] < 0.0001:
                    print(f"    ⚠️  WARNING: Loss barely decreased in inner loop!")
                    print(f"    This suggests:")
                    print(f"      - Gradients may be too small")
                    print(f"      - Learning rate may be too low")
                    print(f"      - Soft mask may be saturated")
                    
                    # Show mask diagnostics if available
                    if metrics.get('selector_diagnostics'):
                        diag = metrics['selector_diagnostics']
                        print(f"\n    Soft Mask Diagnostics:")
                        print(f"      Mean: {diag.get('mask_mean', 0):.4f}")
                        print(f"      Range: [{diag.get('mask_min', 0):.4f}, {diag.get('mask_max', 0):.4f}]")
                        print(f"      High (>0.9): {diag.get('num_high', 0)}/{data.num_nodes}")
                        print(f"      Low (<0.1): {diag.get('num_low', 0)}/{data.num_nodes}")
                        print(f"      Middle (0.1-0.9): {diag.get('num_middle', 0)}/{data.num_nodes}")
                        
                        # Diagnose the problem
                        if diag.get('num_high', 0) == 0 and diag.get('num_low', 0) > data.num_nodes * 0.9:
                            print(f"      ⚠️  PROBLEM: Almost all nodes have mask < 0.1 (excluded)")
                            print(f"      → Threshold too high or scores too low")
                            print(f"      → Threshold: {diag.get('threshold', 0):.6f}")
                        elif diag.get('num_low', 0) == 0 and diag.get('num_high', 0) > data.num_nodes * 0.9:
                            print(f"      ⚠️  PROBLEM: Almost all nodes have mask > 0.9 (included)")
                            print(f"      → Threshold too low")
                        elif diag.get('num_middle', 0) > data.num_nodes * 0.8:
                            print(f"      ⚠️  PROBLEM: Most nodes in middle range (0.1-0.9)")
                            print(f"      → Sigmoid is saturated in linear region")
                            print(f"      → Need higher sharpness")
                else:
                    print(f"    ✓ Loss is decreasing normally")
                print(f"  Outer loop (selector update):")
                print(f"    Val loss: {metrics['val_loss']:.6f}")
                if not use_full_graph and 'selector' in metrics:
                    print(f"    Alpha: {metrics['selector']['alpha']:.6f}")
                    print(f"    Threshold: {metrics['selector']['threshold']:.6f}")
                print()
            
            history['train_loss'].append(metrics['train_loss'])
            history['val_loss'].append(metrics['val_loss'])
            
            if not use_full_graph and 'selector' in metrics:
                history['selector_alpha'].append(metrics['selector']['alpha'])
                history['selector_threshold'].append(metrics['selector']['threshold'])
            
            # Periodic evaluation
            if (epoch + 1) % eval_steps == 0:
                current_val_loss = metrics['val_loss']
                
                # Check for improvement
                if current_val_loss < best_val_loss - 0.0001:
                    best_val_loss = current_val_loss
                    history['best_val_loss'] = best_val_loss
                    history['best_epoch'] = epoch + 1
                    epochs_without_improvement = 0
                    
                    # Save best model
                    history['best_selector_state'] = copy.deepcopy(
                        self.selector.state_dict()
                    )
                else:
                    epochs_without_improvement += eval_steps
                
                # Progress update
                if verbose:
                    info = {
                        'train_loss': f"{metrics['train_loss']:.4f}",
                        'inner_decrease': f"{metrics['train_loss_decrease']:.4f}",  # Show inner loop decrease
                        'val_loss': f"{metrics['val_loss']:.4f}",
                        'patience': f"{epochs_without_improvement}/{patience}"
                    }
                    if not use_full_graph and 'selector' in metrics:
                        info['alpha'] = f"{metrics['selector']['alpha']:.3f}"
                    iterator.set_postfix(info)
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"\n[Early Stop] No improvement for {patience} epochs")
                        print(f"Best val loss: {best_val_loss:.4f} at epoch {history['best_epoch']}")
                    break
        
        # Restore best selector
        if 'best_selector_state' in history:
            self.selector.load_state_dict(history['best_selector_state'])
            if verbose:
                print(f"[Restored] Best selector from epoch {history['best_epoch']}")
        
        return history

