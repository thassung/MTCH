from __future__ import annotations

"""DRNL (Double Radius Node Labeling) features for subgraph link prediction.

Mirrors the labeling scheme from LCILP / SEAL:
  - For each node w in the subgraph of (u, v):
      label = (dist(u, w), dist(v, w))  (BFS distances within the subgraph)
  - One-hot encode each distance and concatenate → 2*(max_dist+1) features
  - Unreachable nodes get distance clamped to max_dist (the final bin acts as
    an "infinity" indicator, consistent with LCILP's clip at max_distance).
"""

from collections import deque

import torch


def _bfs_dist(adj: list[list], source: int, n: int, max_dist: int) -> list[int]:
    """BFS distances from `source` in an adjacency list, clamped to max_dist."""
    dist = [max_dist] * n
    dist[source] = 0
    q = deque([source])
    while q:
        node = q.popleft()
        d = dist[node]
        if d >= max_dist:
            continue
        for nb in adj[node]:
            if dist[nb] == max_dist and nb != source:
                dist[nb] = d + 1
                q.append(nb)
    return dist


def compute_drnl_for_csr(csr, max_dist: int = 6, verbose: bool = True) -> torch.Tensor:
    """Pre-compute DRNL for every subgraph in a SubgraphCSR. One-time cost.

    Returns FloatTensor [total_nodes, 2*(max_dist+1)] aligned with csr.node_ids.
    Store in csr.drnl_feats and save the cache — training then does a single
    tensor index per batch instead of Python BFS on every forward pass.
    """
    from tqdm import tqdm

    total_nodes = int(csr.node_ids.size(0))
    dim = max_dist + 1
    x = torch.zeros(total_nodes, 2 * dim)

    N = int(csr.u_sub.size(0))
    it = tqdm(range(N), desc='Pre-computing DRNL', leave=False,
              mininterval=10) if verbose else range(N)

    for i in it:
        if not bool(csr.valid_mask[i].item()):
            continue
        n_start = int(csr.node_offs[i].item())
        n_end   = int(csr.node_offs[i + 1].item())
        e_start = int(csr.edge_offs[i].item())
        e_end   = int(csr.edge_offs[i + 1].item())
        n_i = n_end - n_start
        if n_i == 0:
            continue

        adj: list[list] = [[] for _ in range(n_i)]
        for e in range(e_start, e_end):
            s = int(csr.edge_src[e].item())
            d = int(csr.edge_dst[e].item())
            if 0 <= s < n_i and 0 <= d < n_i:
                adj[s].append(d)

        u_loc = int(csr.u_sub[i].item())
        v_loc = int(csr.v_sub[i].item())

        d_u = _bfs_dist(adj, u_loc, n_i, max_dist)
        d_v = _bfs_dist(adj, v_loc, n_i, max_dist)

        for j in range(n_i):
            x[n_start + j, d_u[j]] = 1.0
            x[n_start + j, dim + d_v[j]] = 1.0

    return x


def compute_ppr_feats_for_csr(csr, ppr_cache: dict, verbose: bool = True) -> torch.Tensor:
    """Pre-compute PPR-score structural features for every subgraph in a SubgraphCSR.

    For subgraph i of pair (u_i, v_i), each node w gets (π_{u_i}(w), π_{v_i}(w)).
    u_i and v_i are recovered from csr.node_ids using csr.u_sub / csr.v_sub.

    Parameters
    ----------
    ppr_cache : dict[int, Tensor[num_nodes]]
        ppr_cache[s] is the full PPR score vector from source node s.
        Typically PPRPreprocessor.ppr_cache.

    Returns
    -------
    FloatTensor [total_nodes, 2]  — col 0: π_u, col 1: π_v for each node.
    """
    from tqdm import tqdm

    total_nodes = int(csr.node_ids.size(0))
    x = torch.zeros(total_nodes, 2)
    N = int(csr.u_sub.size(0))
    it = (tqdm(range(N), desc='Pre-computing PPR features', leave=False,
               mininterval=10)
          if verbose else range(N))

    for i in it:
        if not bool(csr.valid_mask[i].item()):
            continue
        n_start = int(csr.node_offs[i].item())
        n_end   = int(csr.node_offs[i + 1].item())
        if n_end == n_start:
            continue

        node_ids_i = csr.node_ids[n_start:n_end]
        u = int(node_ids_i[int(csr.u_sub[i].item())].item())
        v = int(node_ids_i[int(csr.v_sub[i].item())].item())

        ppr_u = ppr_cache.get(u)
        ppr_v = ppr_cache.get(v)
        if ppr_u is not None:
            x[n_start:n_end, 0] = ppr_u[node_ids_i]
        if ppr_v is not None:
            x[n_start:n_end, 1] = ppr_v[node_ids_i]

    return x


def compute_drnl_for_batch(batch: dict, max_dist: int = 6) -> torch.Tensor:
    """Compute DRNL features for every node in a SubgraphCSR mega-batch.

    Parameters
    ----------
    batch : dict
        Output of SubgraphCSR.make_batch (must contain 'edge_index', 'u_idx',
        'v_idx', 'batch_node_offsets', 'num_nodes_vec', 'total_nodes').
    max_dist : int
        Maximum BFS distance (distances >= max_dist share the last one-hot bin).

    Returns
    -------
    FloatTensor of shape [total_nodes, 2*(max_dist+1)]
        DRNL features ready to be used as GNN node features.
    """
    total_nodes: int = batch["total_nodes"]
    device = batch["edge_index"].device

    dim = max_dist + 1
    if total_nodes == 0:
        return torch.zeros(0, 2 * dim, device=device)

    B = int(batch["u_idx"].size(0))
    offsets = batch["batch_node_offsets"]   # [B+1], on device
    nn_b = batch["num_nodes_vec"]            # [B], on device
    u_global = batch["u_idx"]               # [B]
    v_global = batch["v_idx"]               # [B]
    ei = batch["edge_index"].cpu()          # [2, E]

    x_cpu = torch.zeros(total_nodes, 2 * dim)

    for i in range(B):
        n_i = int(nn_b[i].item())
        if n_i == 0:
            continue
        off = int(offsets[i].item())

        # Local edges for subgraph i (src in [off, off+n_i))
        mask = (ei[0] >= off) & (ei[0] < off + n_i)
        ei_local = ei[:, mask] - off  # local indices in [0, n_i)

        # Build undirected adjacency list (edge_index may already be undirected)
        adj: list[list] = [[] for _ in range(n_i)]
        for e in range(ei_local.shape[1]):
            s = int(ei_local[0, e])
            d = int(ei_local[1, e])
            if 0 <= s < n_i and 0 <= d < n_i:
                adj[s].append(d)

        u_loc = int(u_global[i].item()) - off
        v_loc = int(v_global[i].item()) - off

        d_u = _bfs_dist(adj, u_loc, n_i, max_dist)
        d_v = _bfs_dist(adj, v_loc, n_i, max_dist)

        for j in range(n_i):
            x_cpu[off + j, d_u[j]] = 1.0
            x_cpu[off + j, dim + d_v[j]] = 1.0

    return x_cpu.to(device)
