"""
Subgraph extraction utilities for link prediction.
Supports both soft (differentiable) and hard (discrete) extraction.
"""

import torch
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx


def extract_subgraph_soft(data, node_mask, u, v, min_weight=0.01):
    """
    Extract subgraph using soft (continuous) node weights.
    For differentiable training.
    
    Args:
        data: PyG Data object
        node_mask: Soft weights for each node [num_nodes]
        u: Source node index
        v: Target node index
        min_weight: Minimum weight threshold for inclusion (default: 0.01)
    
    Returns:
        sub_data: PyG Data object for subgraph with weighted features
        node_map: Mapping from subgraph indices to original indices
        metadata: Dictionary with subgraph statistics
    """
    # Always include u and v
    node_mask = node_mask.clone()
    node_mask[u] = 1.0
    node_mask[v] = 1.0
    
    # Select nodes with weight above threshold
    selected = node_mask > min_weight
    selected_indices = torch.where(selected)[0]
    
    if len(selected_indices) == 0:
        # Edge case: no nodes selected, include just u and v
        selected_indices = torch.tensor([u, v], device=node_mask.device)
    
    # Extract subgraph edges
    sub_edge_index, edge_mask = subgraph(
        selected_indices, 
        data.edge_index, 
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )
    
    # Create mapping from new to old indices
    node_map = selected_indices
    
    # Extract and weight node features
    if hasattr(data, 'x') and data.x is not None:
        sub_x = data.x[selected_indices].clone()
        # Apply soft weights to features (differentiable)
        weights = node_mask[selected_indices].unsqueeze(1)
        sub_x = sub_x * weights
    else:
        # Create identity features if none exist
        sub_x = torch.eye(len(selected_indices))
    
    # Create subgraph data object
    sub_data = Data(x=sub_x, edge_index=sub_edge_index)
    
    # Metadata
    metadata = {
        'num_nodes': len(selected_indices),
        'num_edges': sub_edge_index.size(1),
        'avg_weight': node_mask[selected_indices].mean().item(),
        'min_weight': node_mask[selected_indices].min().item(),
        'max_weight': node_mask[selected_indices].max().item(),
        'includes_seeds': (u in selected_indices and v in selected_indices)
    }
    
    return sub_data, node_map, metadata


def extract_subgraph_hard(data, node_indices, u, v):
    """
    Extract subgraph using hard (discrete) node selection.
    For inference, evaluation, and visualization.
    
    Args:
        data: PyG Data object
        node_indices: Tensor of node indices to include
        u: Source node index
        v: Target node index
    
    Returns:
        sub_data: PyG Data object for subgraph
        node_map: Mapping from subgraph indices to original indices
        metadata: Dictionary with subgraph statistics
    """
    # Ensure u and v are included
    node_indices = torch.unique(torch.cat([node_indices, torch.tensor([u, v])]))
    
    # Extract subgraph
    sub_edge_index, edge_mask = subgraph(
        node_indices,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )
    
    # Extract node features
    if hasattr(data, 'x') and data.x is not None:
        sub_x = data.x[node_indices]
    else:
        sub_x = torch.eye(len(node_indices))
    
    # Create subgraph data
    sub_data = Data(x=sub_x, edge_index=sub_edge_index)
    
    # Node mapping
    node_map = node_indices
    
    # Metadata
    metadata = {
        'num_nodes': len(node_indices),
        'num_edges': sub_edge_index.size(1),
        'density': sub_edge_index.size(1) / (len(node_indices) * (len(node_indices) - 1) + 1e-10),
        'includes_seeds': (u in node_indices and v in node_indices)
    }
    
    return sub_data, node_map, metadata


def extract_k_hop_subgraph(data, u, v, k=2):
    """
    Extract k-hop neighborhood subgraph around u and v.
    Baseline method for comparison.
    
    Args:
        data: PyG Data object
        u: Source node index
        v: Target node index
        k: Number of hops (default: 2)
    
    Returns:
        sub_data: PyG Data object for subgraph
        node_map: Mapping from subgraph indices to original indices
        metadata: Dictionary with subgraph statistics
    """
    # Get k-hop neighborhood for both seeds
    nodes_u, edge_index_u, _, _ = k_hop_subgraph(
        [u], k, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes
    )
    nodes_v, edge_index_v, _, _ = k_hop_subgraph(
        [v], k, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes
    )
    
    # Union of both neighborhoods
    node_indices = torch.unique(torch.cat([nodes_u, nodes_v]))
    
    # Extract subgraph with relabeling
    sub_edge_index, _ = subgraph(
        node_indices,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )
    
    # Extract features
    if hasattr(data, 'x') and data.x is not None:
        sub_x = data.x[node_indices]
    else:
        sub_x = torch.eye(len(node_indices))
    
    sub_data = Data(x=sub_x, edge_index=sub_edge_index)
    node_map = node_indices
    
    metadata = {
        'num_nodes': len(node_indices),
        'num_edges': sub_edge_index.size(1),
        'k_hops': k,
        'nodes_from_u': len(nodes_u),
        'nodes_from_v': len(nodes_v)
    }
    
    return sub_data, node_map, metadata


def extract_subgraph_for_visualization(data, node_indices, u, v, include_labels=True):
    """
    Extract subgraph with additional information for visualization.
    Creates NetworkX graph with node attributes for plotting.
    
    Args:
        data: PyG Data object
        node_indices: Tensor of selected node indices
        u: Source node index
        v: Target node index
        include_labels: Whether to include node labels
    
    Returns:
        G: NetworkX graph with visualization attributes
        pos: Node positions (if available)
        node_colors: List of colors for nodes
        edge_colors: List of colors for edges
    """
    # Extract subgraph
    sub_data, node_map, metadata = extract_subgraph_hard(data, node_indices, u, v)
    
    # Convert to NetworkX
    G = to_networkx(sub_data, to_undirected=True)
    
    # Map original indices to subgraph indices
    old_to_new = {old.item(): new for new, old in enumerate(node_map)}
    u_sub = old_to_new.get(u, None)
    v_sub = old_to_new.get(v, None)
    
    # Assign node attributes
    node_colors = []
    for node in G.nodes():
        if node == u_sub:
            G.nodes[node]['type'] = 'source'
            node_colors.append('red')
        elif node == v_sub:
            G.nodes[node]['type'] = 'target'
            node_colors.append('blue')
        else:
            G.nodes[node]['type'] = 'neighbor'
            node_colors.append('lightgray')
        
        # Add original index as label
        if include_labels:
            G.nodes[node]['label'] = str(node_map[node].item())
    
    # Edge colors (highlight edges connected to u or v)
    edge_colors = []
    for edge in G.edges():
        if u_sub in edge or v_sub in edge:
            edge_colors.append('orange')
        else:
            edge_colors.append('gray')
    
    # Compute layout
    try:
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    except:
        pos = None
    
    return G, pos, node_colors, edge_colors


def get_subgraph_node_indices(u_sub, v_sub, node_map):
    """
    Helper to find subgraph indices of u and v.
    
    Args:
        u_sub: Original index of u
        v_sub: Original index of v
        node_map: Mapping from subgraph to original indices
    
    Returns:
        u_idx: Index of u in subgraph
        v_idx: Index of v in subgraph
    """
    u_idx = (node_map == u_sub).nonzero(as_tuple=True)[0]
    v_idx = (node_map == v_sub).nonzero(as_tuple=True)[0]
    
    if len(u_idx) == 0 or len(v_idx) == 0:
        raise ValueError(f"Nodes u={u_sub} or v={v_sub} not found in subgraph")
    
    return u_idx[0].item(), v_idx[0].item()

