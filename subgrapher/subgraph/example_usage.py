"""
Example usage of learnable PPR-based subgraph link prediction.
Demonstrates plug-and-play API and basic training workflow.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from subgrapher.subgraph import SubgraphLinkPrediction
from subgrapher.utils.loader import load_txt_to_pyg
from subgrapher.benchmark.data_prep import add_node_features, split_edges


def prepare_data(dataset_path, feature_method='random', feature_dim=128):
    """Load and prepare dataset."""
    print(f"Loading dataset from {dataset_path}...")
    data, node2idx, idx2node = load_txt_to_pyg(dataset_path)
    
    # Add node features
    data = add_node_features(data, method=feature_method, feature_dim=feature_dim)
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.x.size(1)}D ({feature_method})")
    
    # Split edges
    split_edge = split_edges(data.edge_index, data.num_nodes,
                            val_ratio=0.1, test_ratio=0.1)
    
    # Convert to edge lists
    train_edges = list(zip(
        split_edge['train']['source_node'].tolist(),
        split_edge['train']['target_node'].tolist()
    ))
    val_edges = list(zip(
        split_edge['valid']['source_node'].tolist(),
        split_edge['valid']['target_node'].tolist()
    ))
    test_edges = list(zip(
        split_edge['test']['source_node'].tolist(),
        split_edge['test']['target_node'].tolist()
    ))
    
    # Use only training edges for graph structure
    train_edge_index = torch.stack([
        split_edge['train']['source_node'],
        split_edge['train']['target_node']
    ], dim=0)
    data.edge_index = train_edge_index
    
    print(f"  Train edges: {len(train_edges)}")
    print(f"  Val edges: {len(val_edges)}")
    print(f"  Test edges: {len(test_edges)}")
    
    return data, train_edges, val_edges, test_edges


def example_basic_training():
    """Basic training example with default parameters."""
    print("="*80)
    print("EXAMPLE 1: Basic Training with Learnable PPR Subgraphs")
    print("="*80)
    
    # Load data (use smaller subset for quick demo)
    data, train_edges, val_edges, test_edges = prepare_data(
        'data/FB15K237/train.txt',
        feature_method='random',
        feature_dim=128
    )
    
    # Limit dataset size for quick demo
    train_edges = train_edges[:1000]
    val_edges = val_edges[:200]
    test_edges = test_edges[:200]
    
    print(f"\n(Using subset: {len(train_edges)} train, {len(val_edges)} val, {len(test_edges)} test)")
    
    # Create model
    model = SubgraphLinkPrediction(
        encoder_type='SAGE',
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        ppr_alpha=0.85,
        adaptive_threshold=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train with meta-learning
    print("\nTraining with meta-learning...")
    history = model.fit(
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        epochs=50,
        inner_steps=3,
        inner_lr=0.01,
        outer_lr=0.001,
        patience=10,
        verbose=True
    )
    
    # Inspect learned parameters
    print("\nLearned selector parameters:")
    params = model.get_selector_params()
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Predict on test set
    print("\nEvaluating on test set...")
    predictions, pred_meta = model.predict(test_edges, data)
    print(f"  Predictions: {len(predictions)}")
    print(f"  Mean: {pred_meta['pred_mean']:.4f}")
    print(f"  Avg subgraph size: {pred_meta['avg_subgraph_size']:.1f} nodes")
    
    # Visualize one subgraph
    print("\nVisualizing example subgraph...")
    u, v = test_edges[0]
    fig = model.visualize_subgraph(data, u, v, save_path='subgraph_example.png')
    
    return model, history


def example_comparison():
    """Compare full graph vs. learnable subgraph."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Comparison - Full Graph vs. Learnable Subgraph")
    print("="*80)
    
    # Load data
    data, train_edges, val_edges, test_edges = prepare_data(
        'data/FB15K237/train.txt',
        feature_method='random',
        feature_dim=128
    )
    
    # Subset for demo
    train_edges = train_edges[:1000]
    val_edges = val_edges[:200]
    
    # Train baseline (full graph, no subgraph selection)
    print("\n[1/2] Training baseline (full graph)...")
    model_full = SubgraphLinkPrediction(
        encoder_type='SAGE',
        hidden_dim=128,
        num_layers=2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    history_full = model_full.fit(
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        epochs=30,
        use_full_graph=True,  # No subgraph extraction
        verbose=True
    )
    
    # Train with learnable subgraphs
    print("\n[2/2] Training with learnable PPR subgraphs...")
    model_subgraph = SubgraphLinkPrediction(
        encoder_type='SAGE',
        hidden_dim=128,
        num_layers=2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    history_subgraph = model_subgraph.fit(
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        epochs=30,
        use_full_graph=False,  # Use learnable subgraphs
        verbose=True
    )
    
    # Compare results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"Full Graph:")
    print(f"  Best val loss: {history_full['best_val_loss']:.4f}")
    print(f"  Final val loss: {history_full['val_loss'][-1]:.4f}")
    
    print(f"\nLearnable Subgraph:")
    print(f"  Best val loss: {history_subgraph['best_val_loss']:.4f}")
    print(f"  Final val loss: {history_subgraph['val_loss'][-1]:.4f}")
    print(f"  Learned alpha: {model_subgraph.get_selector_params()['alpha']:.3f}")
    
    return model_full, model_subgraph


def example_custom_parameters():
    """Example with custom selector parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Selector Parameters")
    print("="*80)
    
    # Load data
    data, train_edges, val_edges, test_edges = prepare_data(
        'data/FB15K237/train.txt',
        feature_method='random',
        feature_dim=128
    )
    
    # Subset
    train_edges = train_edges[:1000]
    val_edges = val_edges[:200]
    
    # Create model with custom parameters
    model = SubgraphLinkPrediction(
        encoder_type='GAT',  # Use GAT instead of SAGE
        hidden_dim=256,
        num_layers=3,
        dropout=0.5,
        ppr_alpha=0.9,  # Higher PPR alpha (less exploration)
        adaptive_threshold=True,
        init_alpha=0.7,  # Bias towards first seed
        init_threshold=0.4,  # Higher percentile threshold
        sharpness=20.0,  # Sharper sigmoid
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nModel parameters:")
    print(f"  Encoder: {model.encoder_type}")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  PPR alpha: {model.ppr_alpha}")
    print(f"  Init selector alpha: {model.init_alpha}")
    print(f"  Init threshold: {model.init_threshold}")
    
    # Train
    history = model.fit(
        data=data,
        train_edges=train_edges,
        val_edges=val_edges,
        epochs=50,
        inner_steps=5,
        inner_lr=0.005,
        outer_lr=0.0005,
        patience=15,
        verbose=True
    )
    
    # Show parameter evolution
    if 'selector_alpha' in history:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Alpha evolution
        ax1.plot(history['selector_alpha'])
        ax1.axhline(y=model.init_alpha, color='r', linestyle='--', 
                   label=f'Init: {model.init_alpha}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Alpha')
        ax1.set_title('Selector Alpha Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Threshold evolution
        ax2.plot(history['selector_threshold'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Threshold')
        ax2.set_title('Selector Threshold Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('selector_evolution.png', dpi=150)
        print(f"\nSaved selector evolution plot to selector_evolution.png")
    
    return model, history


def main():
    """Run all examples."""
    print("Learnable PPR Subgraph Selection - Examples\n")
    
    # Example 1: Basic training
    try:
        model1, history1 = example_basic_training()
        print("\n✓ Example 1 completed successfully")
    except Exception as e:
        print(f"\n✗ Example 1 failed: {e}")
    
    # Example 2: Comparison
    try:
        model_full, model_subgraph = example_comparison()
        print("\n✓ Example 2 completed successfully")
    except Exception as e:
        print(f"\n✗ Example 2 failed: {e}")
    
    # Example 3: Custom parameters
    try:
        model3, history3 = example_custom_parameters()
        print("\n✓ Example 3 completed successfully")
    except Exception as e:
        print(f"\n✗ Example 3 failed: {e}")
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == '__main__':
    main()

