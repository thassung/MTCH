import torch
from torch_geometric.data import Data
import pandas as pd

def load_txt_to_pyg(path):
    """
    Loads a .txt file with lines: node1\trelation\tnode2
    Ignores relation, treats as undirected edge list.
    Returns:
        data: PyG Data object
        node2idx: dict mapping node string IDs to integer indices
        idx2node: dict mapping integer indices to node string IDs
    """
    df = pd.read_csv(path, sep='\t', header=None, names=['src', 'rel', 'dst'])
    
    # Get list of nodes
    nodes = pd.unique(df[['src', 'dst']].values.ravel())

    # Create mapping between node and index
    node2idx = {n: i for i, n in enumerate(nodes)}
    idx2node = {i: n for n, i in node2idx.items()}

    # Build edge index
    src = df['src'].map(node2idx).values
    dst = df['dst'].map(node2idx).values

    # Undirected: add both directions
    edge_index = torch.tensor(
        [list(src) + list(dst), list(dst) + list(src)], dtype=torch.long
    )
    data = Data(edge_index=edge_index)
    return data, node2idx, idx2node
