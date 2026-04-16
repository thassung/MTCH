"""
Local approximate PPR + conductance sweep cut.

Ported from LCILP (Andersen et al. 2006 push algorithm + sweep cut).
Shared by Static PPR and Learnable PPR Phase 2.
"""

import numpy as np
import torch
from collections import deque
from scipy import sparse


def build_sparse_adj(edge_index, num_nodes):
    """Convert PyG edge_index [2, E] to a scipy CSR adjacency matrix.

    The result is symmetric (undirected).  Duplicate edges are collapsed
    to weight 1.  The matrix is cached by the caller (build once per graph).
    """
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones(len(row), dtype=np.float64)
    adj = sparse.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = adj.tocsr()
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)
    return adj


def approximate_ppr(adj_csr, seed_set, alpha=0.85, epsilon=1e-3):
    """Approximate Personalized PageRank via the push method.

    Implements the algorithm from Andersen, Chung & Lang,
    "Local graph partitioning using pagerank vectors", FOCS 2006.

    Parameters
    ----------
    adj_csr : scipy.sparse.csr_matrix
        Symmetric adjacency matrix.
    seed_set : set or list of int
        Seed node(s).  Initial residual is uniform over seeds.
    alpha : float
        Damping / random-walk continuation probability.
        Higher alpha means the walk travels farther from seeds.
    epsilon : float
        Precision threshold.  Smaller = more nodes explored.

    Returns
    -------
    prob : np.ndarray [num_nodes]
        Approximate PPR scores.
    """
    degree = np.asarray(adj_csr.sum(axis=1)).ravel()
    n_nodes = adj_csr.shape[0]

    prob = np.zeros(n_nodes)
    res = np.zeros(n_nodes)

    seed_list = list(seed_set)
    for s in seed_list:
        res[s] = 1.0 / len(seed_list)

    next_nodes = deque(seed_list)

    while next_nodes:
        node = next_nodes.pop()
        d = degree[node]
        if d == 0:
            continue
        push_val = res[node] - 0.5 * epsilon * d
        if push_val <= 0:
            continue
        res[node] = 0.5 * epsilon * d
        prob[node] += (1.0 - alpha) * push_val
        put_val = alpha * push_val

        start, end = adj_csr.indptr[node], adj_csr.indptr[node + 1]
        neighbors = adj_csr.indices[start:end]
        weights = adj_csr.data[start:end]

        for j in range(len(neighbors)):
            neighbor = neighbors[j]
            w = weights[j]
            old_res = res[neighbor]
            res[neighbor] += put_val * w / d
            threshold = epsilon * degree[neighbor]
            if res[neighbor] >= threshold > old_res:
                next_nodes.appendleft(neighbor)

    return prob


def conductance_sweep_cut(adj_csr, score, window=10):
    """Sweep cut minimizing conductance, following Andersen et al. 2006.

    Parameters
    ----------
    adj_csr : scipy.sparse.csr_matrix
        Symmetric adjacency matrix.
    score : np.ndarray [num_nodes]
        Node scores (e.g. from approximate_ppr).
    window : int
        Stop after this many consecutive non-improvements.

    Returns
    -------
    best_sweep_set : set of int
        Nodes in the best conductance community found.
    """
    degree = np.asarray(adj_csr.sum(axis=1)).ravel()
    total_volume = degree.sum()

    sorted_nodes = [n for n in range(len(score)) if score[n] > 0]
    if not sorted_nodes:
        return set()
    sorted_nodes.sort(key=lambda n: score[n], reverse=True)

    sweep_set = set()
    volume = 0.0
    cut = 0.0
    best_conductance = 1.0
    best_sweep_set = {sorted_nodes[0]}
    inc_count = 0

    for node in sorted_nodes:
        volume += degree[node]

        start, end = adj_csr.indptr[node], adj_csr.indptr[node + 1]
        neighbors = adj_csr.indices[start:end]
        for neighbor in neighbors:
            if neighbor in sweep_set:
                cut -= 1
            else:
                cut += 1
        sweep_set.add(node)

        if volume >= total_volume:
            break

        conductance = cut / min(volume, total_volume - volume)
        if conductance < best_conductance:
            best_conductance = conductance
            best_sweep_set = set(sweep_set)
            inc_count = 0
        else:
            inc_count += 1
            if inc_count >= window:
                break

    return best_sweep_set


def extract_local_community(adj_csr, seeds, alpha=0.85, epsilon=1e-3,
                            window=10):
    """PPR + conductance sweep cut in one call.

    Parameters
    ----------
    adj_csr : scipy.sparse.csr_matrix
    seeds : set or list of int
    alpha, epsilon, window : PPR / sweep parameters

    Returns
    -------
    community : set of int
        Node indices in the local community (always includes the seeds).
    """
    scores = approximate_ppr(adj_csr, seeds, alpha=alpha, epsilon=epsilon)
    community = conductance_sweep_cut(adj_csr, scores, window=window)
    community.update(seeds)
    return community


def combine_ppr_and_sweep(adj_csr, u, v, alpha_u=0.85, alpha_v=0.85,
                          epsilon=1e-3, blend=0.5, window=10):
    """Two single-seed approximate PPRs, combine, then sweep cut.

    Used by Learnable PPR Phase 2 where each endpoint has its own
    learned teleportation (alpha) value.

    Parameters
    ----------
    adj_csr : scipy.sparse.csr_matrix
    u, v : int
        Source and target node indices.
    alpha_u, alpha_v : float
        Per-endpoint teleportation / damping values (from Phase 1 search).
    epsilon : float
        PPR precision.
    blend : float
        Combination weight: combined = blend * ppr_u + (1 - blend) * ppr_v.
    window : int
        Sweep cut window.

    Returns
    -------
    community : set of int
        Node indices (always includes u and v).
    """
    ppr_u = approximate_ppr(adj_csr, {u}, alpha=alpha_u, epsilon=epsilon)
    ppr_v = approximate_ppr(adj_csr, {v}, alpha=alpha_v, epsilon=epsilon)

    combined = blend * ppr_u + (1.0 - blend) * ppr_v
    community = conductance_sweep_cut(adj_csr, combined, window=window)
    community.add(u)
    community.add(v)
    return community
