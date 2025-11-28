
import torch
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
from subgrapher.utils.loader import load_txt_to_pyg

def personalized_pagerank(G, seeds, alpha=0.85, weight=None):
    """
    Computes personalized PageRank for a given graph and seeds.
    Seeds: dict {node: initial_weight}
    Returns: dict {node: ppr_score}
    """
    return nx.pagerank(G, alpha=alpha, personalization=seeds, weight=weight)

def extract_topk_subgraph_pyg(data, u, v, k=100, alpha=0.5, ppr_alpha=0.85):
    """
    Extracts top-k subgraph from data, with u and v as seeds.
    data: PyG Data object (can be created using load_txt_to_pyg)
    u, v: node indices (as used in data)
    Returns: PyG Data object
    """
    G = to_networkx(data, to_undirected=True)
    nodes = list(G.nodes)
    
    # Personalized PageRank from u and v
    seeds_u = {n: 1.0 if n == u else 0.0 for n in nodes}
    seeds_v = {n: 1.0 if n == v else 0.0 for n in nodes}
    ppr_u = personalized_pagerank(G, seeds_u, alpha=ppr_alpha)
    ppr_v = personalized_pagerank(G, seeds_v, alpha=ppr_alpha)
    
    # Score nodes
    scores = {n: alpha * ppr_u[n] + (1 - alpha) * ppr_v[n] for n in nodes}
    
    # Always include u and v
    selected = set([u, v])
    # Select top-k nodes by score (excluding u and v)
    candidates = [n for n in nodes if n not in selected]
    topk = sorted(candidates, key=lambda n: scores[n], reverse=True)[:k]
    selected.update(topk)
    
    # Induced subgraph
    subG = G.subgraph(selected)
    
    # Convert back to PyG Data
    sub_data = from_networkx(subG)
    # Optionally, copy node features if present
    if hasattr(data, 'x') and data.x is not None:
        # Map old node indices to new ones
        old_to_new = {old: i for i, old in enumerate(subG.nodes())}
        sub_data.x = data.x[torch.tensor(list(subG.nodes()))]
    return sub_data

# from loader import load_txt_to_pyg
# from ppr_scorer import extract_topk_subgraph_pyg
# data_path = 'data/FB15K237/test.txt'
# data, node2idx, idx2node = load_txt_to_pyg(data_path)
# u = node2idx['/m/08966']
# v = node2idx['/m/05lf_']
# sub_data = extract_topk_subgraph_pyg(data, u, v, k=100)


