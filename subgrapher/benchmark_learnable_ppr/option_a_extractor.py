"""
Score-threshold PPR subgraph extraction for LPPR.

Extracts subgraphs around query pairs (u, v) by:
  1. Running approximate PPR from seed set {u, v} with small epsilon (1e-5)
  2. Keeping all nodes where absorbed_score > tau (plus seeds)

Unlike the conductance-sweep approach used in Phase 2, this gives subgraphs
whose SIZE varies with alpha (large alpha = small subgraph), because the PPR
mass stays local rather than spreading uniformly.

The extraction alpha is fixed to the WIDEST scale (smallest alpha = most
spread) so the subgraph is large enough to contain useful structural context
for ALL selector choices.  Within this fixed envelope, the soft PPR adjacency
uses multiple alphas to shape GNN message passing.
"""

import os
import torch
import numpy as np
from torch_geometric.utils import negative_sampling, add_self_loops

from ..utils.local_ppr import approximate_ppr, build_sparse_adj


class LPPRSubgraphExtractor:
    """
    Extracts score-threshold PPR subgraphs for LPPR.

    Args:
        data: PyG Data object (full graph)
        push_epsilon: PPR precision threshold (smaller = more nodes explored)
        score_tau: Keep nodes with absorbed_score > tau (plus seeds)
        extraction_alpha: Classic restart probability used for envelope subgraph.
                          Should match the widest (smallest) alpha in teleport_values.
    """

    def __init__(self, data, push_epsilon=1e-5, score_tau=1e-3,
                 extraction_alpha=0.25):
        self.data = data
        self.push_epsilon = push_epsilon
        self.score_tau = score_tau
        # local_ppr.approximate_ppr uses continuation prob (1-restart)
        self.continuation_alpha = 1.0 - extraction_alpha
        self.adj_csr = build_sparse_adj(data.edge_index, data.num_nodes)

    def extract(self, u, v):
        """
        Extract score-threshold subgraph for pair (u, v).

        Returns:
            selected_nodes: LongTensor of sorted global node indices
            u_local: int, index of u within selected_nodes (-1 if absent)
            v_local: int, index of v within selected_nodes (-1 if absent)
        """
        scores = approximate_ppr(
            self.adj_csr, {u, v},
            alpha=self.continuation_alpha,
            epsilon=self.push_epsilon)

        above = set(int(n) for n in np.where(scores > self.score_tau)[0])
        community = above | {u, v}

        selected_nodes = torch.tensor(sorted(community), dtype=torch.long)
        node_to_local = {n.item(): i for i, n in enumerate(selected_nodes)}

        u_local = node_to_local.get(u, -1)
        v_local = node_to_local.get(v, -1)
        return selected_nodes, u_local, v_local


# ---------------------------------------------------------------------------
# Subgraph cache: extract once, reuse every epoch
# ---------------------------------------------------------------------------

class LPPRSubgraphCache:
    """
    Pre-extracted LPPR subgraphs for one data split.

    Stores only (selected_nodes, u_local, v_local) — no edge_index needed
    since LPPR's GNN uses PPR matrices, not structural edges.

    Usage::

        cache = LPPRSubgraphCache.build(extractor, split_edge, 'train')
        cache.save('cache/train_opt_a.pt')
        cache = LPPRSubgraphCache.load('cache/train_opt_a.pt')
    """

    def __init__(self, selected_nodes_list, u_local_list, v_local_list):
        self.selected_nodes = selected_nodes_list  # list[LongTensor]
        self.u_local = u_local_list                # list[int]
        self.v_local = v_local_list                # list[int]

    def __len__(self):
        return len(self.selected_nodes)

    def __getitem__(self, idx):
        return self.selected_nodes[idx], self.u_local[idx], self.v_local[idx]

    @classmethod
    def build(cls, extractor, split_edge, split='train', verbose=True):
        from tqdm import tqdm
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        n = source.size(0)

        sel_list, u_list, v_list = [], [], []
        skipped = 0

        it = tqdm(range(n), desc=f'Caching {split} subgraphs (LPPR)',
                  mininterval=5) if verbose else range(n)
        for i in it:
            u = source[i].item()
            v = target[i].item()
            nodes_S, u_loc, v_loc = extractor.extract(u, v)

            if u_loc == -1 or v_loc == -1:
                # Fallback: just the two seeds
                nodes_S = torch.tensor(sorted({u, v}), dtype=torch.long)
                nm = {n.item(): i for i, n in enumerate(nodes_S)}
                u_loc = nm.get(u, 0)
                v_loc = nm.get(v, 0)
                skipped += 1

            sel_list.append(nodes_S)
            u_list.append(u_loc)
            v_list.append(v_loc)

        if verbose and skipped:
            print(f'  ({skipped}/{n} edges fell back to seed-only subgraph)')
        return cls(sel_list, u_list, v_list)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'selected_nodes': self.selected_nodes,
            'u_local': self.u_local,
            'v_local': self.v_local,
        }, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location='cpu', weights_only=False)
        return cls(d['selected_nodes'], d['u_local'], d['v_local'])


def build_or_load_cache(extractor, split_edge, split, cache_dir,
                        verbose=True):
    """Build an LPPRSubgraphCache or load it from disk."""
    if cache_dir:
        path = os.path.join(cache_dir, f'{split}_opt_a_subgraphs.pt')
        if os.path.isfile(path):
            if verbose:
                print(f'[Cache] Loading {split} subgraphs from {path}')
            return LPPRSubgraphCache.load(path)

    if verbose:
        print(f'[Cache] Extracting {split} subgraphs (one-time cost)...')
    cache = LPPRSubgraphCache.build(extractor, split_edge, split, verbose)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f'{split}_opt_a_subgraphs.pt')
        cache.save(path)
        mb = os.path.getsize(path) / 1e6
        if verbose:
            print(f'[Cache] Saved {split} cache: {path} ({mb:.0f} MB)')
    return cache


# ---------------------------------------------------------------------------
# Negative subgraph generation (inline, for training batches)
# ---------------------------------------------------------------------------

def sample_neg_subgraphs(extractor, neg_edges, max_cache=None):
    """Extract subgraphs for a batch of negative edges (not cached)."""
    result = []
    for i in range(neg_edges.size(1)):
        u = neg_edges[0, i].item()
        v = neg_edges[1, i].item()
        nodes_S, u_loc, v_loc = extractor.extract(u, v)
        if u_loc == -1 or v_loc == -1:
            nodes_S = torch.tensor(sorted({u, v}), dtype=torch.long)
            nm = {n.item(): i for i, n in enumerate(nodes_S)}
            u_loc = nm.get(u, 0)
            v_loc = nm.get(v, 0)
        result.append((nodes_S, u_loc, v_loc))
    return result
