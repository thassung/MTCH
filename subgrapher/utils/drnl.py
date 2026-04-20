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
