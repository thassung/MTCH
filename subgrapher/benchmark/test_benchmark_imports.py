"""
Quick test to verify all benchmark imports work correctly.
"""

import warnings

def test_imports():
    """Test that all benchmark modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test main module
        import subgrapher.benchmark as benchmark
        print("✓ subgrapher.benchmark")
        
        # Test models
        from subgrapher.benchmark.models import GCN, SAGE, GAT, LinkPredictor, LinkPredictionModel
        print("✓ models: GCN, SAGE, GAT, LinkPredictor, LinkPredictionModel")
        
        # Test data prep
        from subgrapher.benchmark.data_prep import prepare_link_prediction_data, add_node_features
        print("✓ data_prep: prepare_link_prediction_data, add_node_features")
        
        # Test evaluator
        from subgrapher.benchmark.evaluator import evaluate_link_prediction
        print("✓ evaluator: evaluate_link_prediction")
        
        # Test trainer
        from subgrapher.benchmark.trainer import train_epoch, train_model, benchmark_model
        print("✓ trainer: train_epoch, train_model, benchmark_model")
        
        # Test run_benchmark
        from subgrapher.benchmark.run_benchmark import run_benchmark
        print("✓ run_benchmark: run_benchmark")
        
        # Test run_full_benchmark
        from subgrapher.benchmark.run_full_benchmark import run_full_benchmark
        print("✓ run_full_benchmark: run_full_benchmark")
        
        print("\n✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        warnings.warn(f">>> IMPORT ERROR <<<\n{e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test that models can be instantiated."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from subgrapher.benchmark.models import GCN, SAGE, GAT, LinkPredictor
        
        in_channels = 128
        hidden_channels = 256
        num_layers = 3
        dropout = 0.3
        
        # Create GCN
        gcn = GCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        print(f"✓ GCN created: {sum(p.numel() for p in gcn.parameters()):,} parameters")
        
        # Create SAGE
        sage = SAGE(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        print(f"✓ SAGE created: {sum(p.numel() for p in sage.parameters()):,} parameters")
        
        # Create GAT
        gat = GAT(in_channels, hidden_channels, hidden_channels, num_layers, dropout, heads=4)
        print(f"✓ GAT created: {sum(p.numel() for p in gat.parameters()):,} parameters")
        
        # Create LinkPredictor
        predictor = LinkPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)
        print(f"✓ LinkPredictor created: {sum(p.numel() for p in predictor.parameters()):,} parameters")
        
        print("\n✓ All models created successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Model creation error: {e}")
        warnings.warn(f">>> MODEL CREATION ERROR <<<\n{e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass through models."""
    print("\nTesting forward pass...")
    
    try:
        import torch
        from subgrapher.benchmark.models import GCN, LinkPredictor
        
        # Create dummy data
        num_nodes = 100
        num_edges = 500
        in_channels = 128
        hidden_channels = 256
        
        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_label_index = torch.randint(0, num_nodes, (2, 10))
        
        # Create model
        encoder = GCN(in_channels, hidden_channels, hidden_channels, 3, 0.3)
        predictor = LinkPredictor(hidden_channels, hidden_channels, 1, 3, 0.3)
        
        # Forward pass
        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            h = encoder(x, edge_index)
            src, dst = edge_label_index[0], edge_label_index[1]
            pred = predictor(h[src], h[dst])
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Embedding shape: {h.shape}")
        print(f"  Prediction shape: {pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Forward pass error: {e}")
        warnings.warn(f">>> FORWARD PASS ERROR <<<\n{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Benchmark Module Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("Model Creation Test", test_model_creation()))
    results.append(("Forward Pass Test", test_forward_pass()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! Benchmark module is ready to use.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        warnings.warn(f">>> TEST SUMMARY <<<\n{results}")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

