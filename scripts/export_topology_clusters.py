"""
Topology-only community labels for a graph (Louvain via NetworkX).

Use for thesis figures: color embedding visualizations or correlate subgraph
methods with structural position. Does not require trained models.

Outputs in --output_dir:
  - node_cluster_labels.pt   LongTensor [num_nodes] with community id per node
  - clustering_meta.json     method, seed, counts, source path

Example:
  python scripts/export_topology_clusters.py \\
    --dataset_path data/FB15K237/train.txt \\
    --output_dir results/topology-clusters/FB15K237
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from subgrapher.utils.loader import load_txt_to_pyg


def louvain_labels(edge_index, num_nodes, seed=42, resolution=1.0):
    """Return LongTensor [num_nodes] and number of communities."""
    G = nx.Graph()
    G.add_nodes_from(range(int(num_nodes)))
    G.add_edges_from(edge_index.t().tolist())
    try:
        communities = nx.community.louvain_communities(
            G, seed=seed, resolution=resolution)
    except TypeError:
        communities = nx.community.louvain_communities(G, seed=seed)
    ordered = sorted(communities, key=lambda c: min(c))
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for cid, comm in enumerate(ordered):
        for u in comm:
            labels[int(u)] = cid
    return labels, len(ordered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Edge list, e.g. data/FB15K237/train.txt')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for node_cluster_labels.pt and meta JSON')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Louvain resolution (default 1.0)')
    args = parser.parse_args()

    data, _, _ = load_txt_to_pyg(args.dataset_path)
    labels, n_comm = louvain_labels(
        data.edge_index, data.num_nodes,
        seed=args.seed, resolution=args.resolution)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(labels, out / 'node_cluster_labels.pt')

    meta = {
        'algorithm': 'networkx.community.louvain_communities',
        'resolution': args.resolution,
        'seed': args.seed,
        'num_nodes': int(data.num_nodes),
        'num_edges': int(data.edge_index.size(1)),
        'num_communities': int(n_comm),
        'dataset_path': str(Path(args.dataset_path).resolve()),
        'created': datetime.now().isoformat(),
    }
    with open(out / 'clustering_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Wrote {out / "node_cluster_labels.pt"} ({n_comm} communities)')
    print(f'Wrote {out / "clustering_meta.json"}')


if __name__ == '__main__':
    main()
