"""
Multi-scale PPR: loads precomputed PPR vectors at multiple teleport probabilities
and computes batched cross-pair representations for architecture search.
"""

import torch
import os
from ..utils.ppr_preprocessor import PPRPreprocessor


class MultiScalePPR:
    """
    Stores PPR vectors at multiple teleport scales per node.
    Computes PPR-weighted cross-pair representations for edge pairs.

    Args:
        dataset_name: Name of dataset (e.g., 'FB15K237')
        data: PyG Data object (needed if creating new preprocessors)
        teleport_values: List of teleport probabilities to use
        preprocessed_dir: Directory containing preprocessed PPR files
    """

    def __init__(self, dataset_name, data=None, teleport_values=None,
                 preprocessed_dir='preprocessed'):
        if teleport_values is None:
            teleport_values = [0.50, 0.85, 0.95]
        self.teleport_values = sorted(teleport_values)
        self.num_scales = len(self.teleport_values)
        self.num_configs = self.num_scales ** 2
        self.dataset_name = dataset_name

        self.preprocessors = {}
        self._load_all(dataset_name, data, preprocessed_dir)

        self.num_nodes = next(iter(self.preprocessors.values())).num_nodes

        self.config_labels = []
        for si in self.teleport_values:
            for sj in self.teleport_values:
                self.config_labels.append((si, sj))

    def _load_all(self, dataset_name, data, preprocessed_dir):
        """Load preprocessors for each teleport value."""
        for alpha in self.teleport_values:
            path = os.path.join(preprocessed_dir, dataset_name,
                                f'ppr_alpha{alpha}.pt')
            if os.path.exists(path):
                print(f"  Loading PPR alpha={alpha} from {path}")
                self.preprocessors[alpha] = PPRPreprocessor.load(path, data)
            elif data is not None:
                print(f"  Computing PPR alpha={alpha} (not found at {path})")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                log_path = path.replace('.pt', '_log.txt')
                preprocessor = PPRPreprocessor(data, ppr_alpha=alpha,
                                               log_file=log_path)
                preprocessor.save(path)
                self.preprocessors[alpha] = preprocessor
            else:
                raise FileNotFoundError(
                    f"PPR file not found: {path}. "
                    f"Run preprocessing first or pass data= to compute on the fly."
                )

    def get_ppr(self, node, teleport):
        """Get PPR vector for a node at a specific teleport value."""
        return self.preprocessors[teleport].get_ppr(node)

    def get_ppr_cross_pair(self, u, v, H):
        """
        Compute cross-pair representations for a single edge (u, v).

        For each (teleport_u, teleport_v) combination:
            r_u = PPR_{teleport_u}[u] @ H   (PPR-weighted embedding)
            r_v = PPR_{teleport_v}[v] @ H
            cross = r_u * r_v               (element-wise product)

        Args:
            u: Source node index
            v: Target node index
            H: Node embeddings [num_nodes, D]

        Returns:
            cross_pairs: [num_configs, D]
        """
        reps = []
        for si in self.teleport_values:
            ppr_u = self.get_ppr(u, si)  # [N]
            r_u = ppr_u @ H  # [D]
            for sj in self.teleport_values:
                ppr_v = self.get_ppr(v, sj)  # [N]
                r_v = ppr_v @ H  # [D]
                reps.append(r_u * r_v)
        return torch.stack(reps)  # [num_configs, D]

    def get_ppr_cross_pair_batch(self, sources, targets, H):
        """
        Batched cross-pair computation for multiple edges.

        Args:
            sources: Tensor of source node indices [B]
            targets: Tensor of target node indices [B]
            H: Node embeddings [num_nodes, D]

        Returns:
            cross_pairs: [B, num_configs, D]
        """
        B = len(sources)
        D = H.size(1)
        device = H.device

        ppr_u_all = {}
        ppr_v_all = {}
        for si in self.teleport_values:
            ppr_u_stack = torch.stack([
                self.get_ppr(s.item(), si) for s in sources
            ]).to(device)  # [B, N]
            ppr_u_all[si] = ppr_u_stack @ H  # [B, D]

        for sj in self.teleport_values:
            ppr_v_stack = torch.stack([
                self.get_ppr(t.item(), sj) for t in targets
            ]).to(device)  # [B, N]
            ppr_v_all[sj] = ppr_v_stack @ H  # [B, D]

        cross_pairs = torch.zeros(B, self.num_configs, D, device=device)
        idx = 0
        for si in self.teleport_values:
            for sj in self.teleport_values:
                cross_pairs[:, idx, :] = ppr_u_all[si] * ppr_v_all[sj]
                idx += 1

        return cross_pairs

    def get_config_for_index(self, config_idx):
        """Map config index to (teleport_u, teleport_v) pair."""
        return self.config_labels[config_idx]

    def get_stats(self):
        """Get memory and cache statistics."""
        total_cached = sum(p.get_stats()['num_cached']
                           for p in self.preprocessors.values())
        total_mb = sum(p.get_stats()['memory_mb']
                       for p in self.preprocessors.values())
        return {
            'num_scales': self.num_scales,
            'num_configs': self.num_configs,
            'teleport_values': self.teleport_values,
            'total_cached_vectors': total_cached,
            'total_memory_mb': total_mb,
        }

    def __repr__(self):
        stats = self.get_stats()
        return (f"MultiScalePPR(scales={self.teleport_values}, "
                f"configs={self.num_configs}, "
                f"memory={stats['total_memory_mb']:.1f}MB)")
