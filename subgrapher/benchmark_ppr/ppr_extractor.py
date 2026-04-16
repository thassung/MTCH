"""
Static PPR-based subgraph extractor using LCILP-style
approximate PPR (Andersen et al. 2006) + conductance sweep cut.

No global per-node precomputation needed — PPR is computed locally
per edge from the joint seed set {u, v}.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from ..utils.local_ppr import build_sparse_adj, extract_local_community


class StaticPPRExtractor:
    """
    Joint-seed approximate PPR + conductance sweep cut extractor.

    For each edge (u, v):
      1. Run approximate PPR with seeds = {u, v}
      2. Conductance sweep cut to select the local community
      3. Force-include u and v
      4. Induce the subgraph on those nodes

    Args:
        data: PyG Data object containing the full graph
        alpha: Teleportation / damping for approximate PPR (default: 0.85)
        epsilon: PPR precision threshold (default: 1e-3)
        window: Sweep cut early-stop window (default: 10)
    """

    def __init__(self, data, alpha=0.85, epsilon=1e-3, window=10):
        self.data = data
        self.alpha = alpha
        self.epsilon = epsilon
        self.window = window

        self.adj_csr = build_sparse_adj(data.edge_index, data.num_nodes)

        print(f"StaticPPRExtractor initialized: alpha={alpha}, "
              f"epsilon={epsilon}, window={window}")
        print(f"  Graph: {data.num_nodes} nodes, "
              f"{data.edge_index.size(1)} edges")

    def extract_subgraph(self, u, v):
        """
        Extract subgraph around edge (u, v) using joint-seed PPR + sweep cut.

        Returns:
            subgraph_data: PyG Data object with remapped indices
            selected_nodes: Tensor of original node indices in the subgraph
            metadata: Dictionary with extraction metadata
        """
        community = extract_local_community(
            self.adj_csr, {u, v},
            alpha=self.alpha, epsilon=self.epsilon, window=self.window)

        selected_nodes = torch.tensor(sorted(community), dtype=torch.long)

        edge_index_sub, edge_attr_sub = subgraph(
            selected_nodes,
            self.data.edge_index,
            edge_attr=(self.data.edge_attr
                       if hasattr(self.data, 'edge_attr')
                       and self.data.edge_attr is not None else None),
            relabel_nodes=True,
            num_nodes=self.data.num_nodes)

        node_mapping = {old_idx.item(): new_idx
                        for new_idx, old_idx in enumerate(selected_nodes)}
        u_sub = node_mapping.get(u, -1)
        v_sub = node_mapping.get(v, -1)

        x_sub = self.data.x[selected_nodes]
        subgraph_data = Data(
            x=x_sub,
            edge_index=edge_index_sub,
            edge_attr=edge_attr_sub,
            num_nodes=len(selected_nodes))

        metadata = {
            'u_original': u,
            'v_original': v,
            'u_subgraph': u_sub,
            'v_subgraph': v_sub,
            'num_nodes_selected': len(selected_nodes),
            'num_edges_subgraph': edge_index_sub.size(1),
            'alpha': self.alpha,
            'epsilon': self.epsilon,
        }

        return subgraph_data, selected_nodes, metadata

    def __repr__(self):
        return (f"StaticPPRExtractor(alpha={self.alpha}, "
                f"epsilon={self.epsilon}, window={self.window})")
