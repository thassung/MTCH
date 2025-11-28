"""
Differentiable PPR-based subgraph selector with learnable parameters.
Implements soft thresholding for differentiable node selection.
"""

import torch
import torch.nn as nn


class DifferentiablePPRSelector(nn.Module):
    """
    Learnable PPR-based node selector for subgraph extraction.
    
    Uses sigmoid thresholding with learnable alpha (for combining PPR_u and PPR_v)
    and adaptive threshold (percentile-based).
    
    Args:
        adaptive_threshold: If True, uses percentile-based adaptive threshold
        init_alpha: Initial value for alpha (default: 0.5)
        init_threshold: Initial threshold value or percentile (default: 0.3)
        sharpness: Steepness of sigmoid (higher = closer to hard threshold)
    """
    
    def __init__(self, 
                 adaptive_threshold=True, 
                 init_alpha=0.5,
                 init_threshold=0.3,
                 sharpness=10.0):
        super(DifferentiablePPRSelector, self).__init__()
        
        # Learnable alpha: combines PPR_u and PPR_v
        # Will be passed through sigmoid to ensure it's in (0, 1)
        alpha_init = self._inverse_sigmoid(init_alpha)
        self.alpha_raw = nn.Parameter(alpha_init.clone().detach().requires_grad_(True))
        
        # Learnable threshold
        self.adaptive_threshold = adaptive_threshold
        if adaptive_threshold:
            # Percentile-based threshold (will be passed through sigmoid)
            threshold_init = self._inverse_sigmoid(init_threshold)
            self.threshold_percentile_raw = nn.Parameter(
                threshold_init.clone().detach().requires_grad_(True)
            )
        else:
            # Fixed threshold value
            self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float32))
        
        # Sharpness controls how close to hard threshold (can be made learnable)
        self.sharpness = sharpness
    
    @staticmethod
    def _inverse_sigmoid(x):
        """Compute inverse sigmoid (logit) for initialization."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = torch.clamp(x, 0.01, 0.99)
        return torch.log(x / (1 - x))
    
    def get_alpha(self):
        """Get current alpha value (in [0, 1])."""
        return torch.sigmoid(self.alpha_raw)
    
    def get_threshold_percentile(self):
        """Get current threshold percentile (in [0, 1])."""
        if self.adaptive_threshold:
            return torch.sigmoid(self.threshold_percentile_raw)
        return None
    
    def forward(self, ppr_u, ppr_v):
        """
        Compute soft node selection mask.
        
        Args:
            ppr_u: PPR scores from seed u [num_nodes]
            ppr_v: PPR scores from seed v [num_nodes]
        
        Returns:
            soft_mask: Differentiable node weights [num_nodes]
            metadata: Dictionary with alpha, threshold, scores
        """
        # Step 1: Combine PPR scores with learnable alpha
        alpha = self.get_alpha()
        combined_scores = alpha * ppr_u + (1 - alpha) * ppr_v
        
        # Step 2: Normalize combined scores for sigmoid numerical stability
        score_min = combined_scores.min()
        score_max = combined_scores.max()
        combined_scores_norm = (combined_scores - score_min) / (score_max - score_min + 1e-8)
        
        # Step 3: Compute threshold
        if self.adaptive_threshold:
            # Adaptive: use percentile of scores
            percentile = self.get_threshold_percentile()
            # Use quantile to find threshold based on current score distribution
            threshold = torch.quantile(combined_scores_norm, percentile)
        else:
            # Fixed threshold
            threshold = self.threshold
        
        # Step 4: Soft thresholding via sigmoid
        # Nodes with score > threshold get high weight, others get low weight
        soft_mask = torch.sigmoid(self.sharpness * (combined_scores_norm - threshold))
        
        # Metadata for monitoring
        metadata = {
            'alpha': alpha.item(),
            'threshold': threshold.item(),
            'threshold_percentile': percentile.item() if self.adaptive_threshold else None,
            'num_selected_soft': soft_mask.sum().item(),
            'num_selected_hard': (soft_mask > 0.5).sum().item(),
            'score_mean': combined_scores.mean().item(),
            'score_std': combined_scores.std().item(),
            'mask_mean': soft_mask.mean().item()
        }
        
        return soft_mask, combined_scores, metadata
    
    def get_hard_mask(self, ppr_u, ppr_v, use_threshold=True, top_k=None):
        """
        Get hard (binary) mask for inference/visualization.
        
        Args:
            ppr_u: PPR scores from seed u [num_nodes]
            ppr_v: PPR scores from seed v [num_nodes]
            use_threshold: If True, use learned threshold; if False, use top_k
            top_k: Number of top nodes to select (only if use_threshold=False)
        
        Returns:
            hard_mask: Binary mask [num_nodes]
            selected_indices: Indices of selected nodes
            metadata: Dictionary with selection info
        """
        with torch.no_grad():
            # Get soft mask first
            soft_mask, combined_scores, metadata = self.forward(ppr_u, ppr_v)
            
            if use_threshold:
                # Threshold-based selection
                hard_mask = (soft_mask > 0.5).float()
            else:
                # Top-k selection
                if top_k is None:
                    raise ValueError("top_k must be specified when use_threshold=False")
                hard_mask = torch.zeros_like(soft_mask)
                topk_indices = torch.topk(combined_scores, min(top_k, len(combined_scores))).indices
                hard_mask[topk_indices] = 1.0
            
            selected_indices = torch.where(hard_mask > 0.5)[0]
            
            metadata['num_selected'] = len(selected_indices)
            metadata['selection_method'] = 'threshold' if use_threshold else f'top_{top_k}'
            
            return hard_mask, selected_indices, metadata
    
    def anneal_sharpness(self, epoch, total_epochs, init_sharpness=1.0, final_sharpness=100.0):
        """
        Anneal sharpness from low to high over training.
        Low sharpness early on provides smoother gradients.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            init_sharpness: Starting sharpness
            final_sharpness: Ending sharpness
        """
        progress = epoch / total_epochs
        self.sharpness = init_sharpness + (final_sharpness - init_sharpness) * progress
    
    def extra_repr(self):
        """String representation for printing."""
        return (f"adaptive_threshold={self.adaptive_threshold}, "
                f"sharpness={self.sharpness:.2f}, "
                f"alpha={self.get_alpha().item():.4f}")

