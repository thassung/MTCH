"""
Architecture search network for PPR configuration selection.
Ported from PS2's SearchGraph_l31 (model.py lines 1150-1193).

Takes cross-pair representations [batch, num_configs, D] and outputs
softmax attention weights over configurations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPRSearchNet(nn.Module):
    """
    MLP that scores PPR configurations per-edge using cross-pair representations.

    During training: outputs softmax(scores / temperature) for differentiable selection.
    During inference: outputs one-hot argmax for hard selection.

    Args:
        in_channels: Dimension of cross-pair vectors (= GNN hidden_channels)
        hidden_channels: Hidden dimension of the scoring MLP
        num_layers: Number of MLP layers
        cat_type: 'multi' or 'concat' (matches AutoLinkPPR)
        temperature: Softmax temperature (lower = sharper)
    """

    def __init__(self, in_channels, hidden_channels=256, num_layers=3,
                 cat_type='multi', temperature=0.07):
        super().__init__()

        self.temperature = temperature
        self.num_layers = num_layers
        self.cat_type = cat_type

        if cat_type == 'multi':
            input_dim = in_channels
        else:
            input_dim = in_channels * 2

        self.trans = nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.trans.append(nn.Linear(input_dim, hidden_channels,
                                            bias=False))
            else:
                self.trans.append(nn.Linear(hidden_channels, hidden_channels,
                                            bias=False))
        self.trans.append(nn.Linear(hidden_channels, 1, bias=False))

    def reset_parameters(self):
        for layer in self.trans:
            layer.reset_parameters()

    def forward(self, x, grad=False):
        """
        Score configurations and return attention weights.

        Args:
            x: Cross-pair representations [batch, num_configs, D]
            grad: If True during eval, return soft weights (for Hessian computation)

        Returns:
            arch_set: [batch, num_configs] attention weights
        """
        for layer in self.trans[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.trans[-1](x)
        x = x.squeeze(-1)  # [batch, num_configs]

        arch_set = torch.softmax(x / self.temperature, dim=1)

        if not self.training:
            if grad:
                return arch_set.detach()
            else:
                device = arch_set.device
                B, C = arch_set.shape
                eyes = torch.eye(C, device=device)
                _, indices = torch.max(arch_set, dim=1)
                arch_set = eyes[indices]

        return arch_set

    def get_config_indices(self, x):
        """
        Get hard config index for each edge (for subgraph extraction).

        Args:
            x: Cross-pair representations [batch, num_configs, D]

        Returns:
            indices: [batch] index of selected config per edge
        """
        self.eval()
        with torch.no_grad():
            for layer in self.trans[:-1]:
                x = layer(x)
                x = F.relu(x)
            x = self.trans[-1](x).squeeze(-1)
            return torch.argmax(x, dim=1)
