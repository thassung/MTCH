"""
Example: Using the improved training system with early stopping.

This example demonstrates:
1. Default early stopping configuration (recommended)
2. Custom configuration for specific needs
3. How to interpret training outputs
"""

from subgrapher.benchmark import run_benchmark


def example_1_default_training():
    """Example 1: Use defaults (recommended for most cases)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Default Training with Early Stopping")
    print("="*80)
    
    results = run_benchmark(
        'data/FB15K237/train.txt',
        models_to_run=['GCN']  # Just run GCN for quick demo
    )
    
    # Check if early stopping worked
    result = results[0]
    print(f"\n📊 Training Results:")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Stopped early: {result['stopped_early']}")
    print(f"  Training time: {result['train_time']:.1f}s")
    print(f"  Test MRR: {result['test_results']['mrr']:.4f}")
    
    return results


def example_2_patient_training():
    """Example 2: More patient training (for complex datasets)"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Patient Training (for harder datasets)")
    print("="*80)
    
    results = run_benchmark(
        'data/WN18RR/train.txt',
        models_to_run=['GAT'],
        # Give model more time to converge
        patience=40,           # Wait longer before stopping
        eval_steps=5,          # Frequent evaluation
        lr=0.0005,            # Lower learning rate for stability
        epochs=800,           # Higher max (but will likely stop earlier)
    )
    
    return results


def example_3_aggressive_training():
    """Example 3: Fast training for quick experiments"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Fast Training (for quick experiments)")
    print("="*80)
    
    results = run_benchmark(
        'data/NELL-995/train.txt',
        models_to_run=['SAGE'],
        # Quick convergence settings
        patience=10,           # Stop sooner
        eval_steps=5,          # Check often
        lr=0.005,             # Higher learning rate
        lr_scheduler='cosine', # Smooth decay
        epochs=200,           # Lower max
    )
    
    return results


def example_4_detailed_monitoring():
    """Example 4: Access detailed training history"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Detailed Training Monitoring")
    print("="*80)
    
    results = run_benchmark(
        'data/FB15K237/train.txt',
        models_to_run=['GCN']
    )
    
    result = results[0]
    history = result['history']
    
    # Analyze training progression
    print(f"\n📈 Training Analysis:")
    print(f"  Total epochs: {len(history['train_loss'])}")
    print(f"  Evaluations: {len(history['val_results'])}")
    print(f"  Initial loss: {history['train_loss'][0]:.4f}")
    print(f"  Final loss: {history['train_loss'][-1]:.4f}")
    
    # Check validation progression
    val_mrrs = [res['mrr'] for res in history['val_results']]
    print(f"\n  Initial val MRR: {val_mrrs[0]:.4f}")
    print(f"  Best val MRR: {max(val_mrrs):.4f}")
    print(f"  Final val MRR: {val_mrrs[-1]:.4f}")
    
    # Check learning rate changes
    print(f"\n  Initial LR: {history['learning_rates'][0]:.6f}")
    print(f"  Final LR: {history['learning_rates'][-1]:.6f}")
    print(f"  LR reduced: {history['learning_rates'][0] != history['learning_rates'][-1]}")
    
    # Early stopping info
    if history['stopped_early']:
        print(f"\n  ⚠️ Early stopping triggered!")
        print(f"  Reason: {history['stop_reason']}")
        print(f"  Saved time by stopping {500 - len(history['train_loss'])} epochs early")
    
    return results


def example_5_compare_configurations():
    """Example 5: Compare with vs without early stopping"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Compare Early Stopping Impact")
    print("="*80)
    
    # With early stopping (default)
    print("\n🚀 Training WITH early stopping:")
    results_with = run_benchmark(
        'data/FB15K237/train.txt',
        models_to_run=['GCN'],
        patience=20,
        epochs=500
    )
    
    # Without early stopping (set very high patience)
    print("\n🐌 Training WITHOUT early stopping (fixed 100 epochs):")
    results_without = run_benchmark(
        'data/FB15K237/train.txt',
        models_to_run=['GCN'],
        patience=1000,  # Effectively disables early stopping
        epochs=100,     # Fixed epochs
        lr_scheduler=None  # Disable LR scheduling too
    )
    
    # Compare results
    with_es = results_with[0]
    without_es = results_without[0]
    
    print(f"\n📊 Comparison:")
    print(f"{'Metric':<20} | {'With Early Stop':<15} | {'Without (100 ep)':<15} | {'Improvement'}")
    print(f"{'-'*75}")
    print(f"{'Training time (s)':<20} | {with_es['train_time']:<15.1f} | {without_es['train_time']:<15.1f} | {(1 - with_es['train_time']/without_es['train_time'])*100:+.1f}%")
    print(f"{'Epochs run':<20} | {with_es['best_epoch']:<15} | {100:<15} | -")
    print(f"{'Test MRR':<20} | {with_es['test_results']['mrr']:<15.4f} | {without_es['test_results']['mrr']:<15.4f} | {(with_es['test_results']['mrr'] - without_es['test_results']['mrr'])*100:+.2f}%")
    print(f"{'Val MRR (best)':<20} | {with_es['best_val_mrr']:<15.4f} | {without_es['best_val_mrr']:<15.4f} | {(with_es['best_val_mrr'] - without_es['best_val_mrr'])*100:+.2f}%")
    
    return results_with, results_without


def example_6_custom_all_params():
    """Example 6: Full customization of all training parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Fully Customized Training")
    print("="*80)
    
    results = run_benchmark(
        'data/FB15K237/train.txt',
        models_to_run=['GAT'],
        
        # Architecture
        hidden_channels=512,
        num_layers=4,
        dropout=0.2,
        
        # Training
        epochs=400,
        batch_size=32768,
        lr=0.002,
        
        # Early stopping
        patience=25,
        eval_steps=5,
        
        # Regularization
        weight_decay=1e-4,
        grad_clip=2.0,
        
        # Learning rate
        lr_scheduler='cosine',
        
        # Device
        device='cuda'
    )
    
    return results


if __name__ == '__main__':
    # Run examples (comment out ones you don't want to run)
    
    print("\n" + "🎯 IMPROVED TRAINING EXAMPLES" + "\n")
    print("These examples demonstrate the new training system with:")
    print("  ✅ Early stopping (prevents overfitting)")
    print("  ✅ Learning rate scheduling (better convergence)")
    print("  ✅ Gradient clipping (stability)")
    print("  ✅ Weight decay (regularization)")
    print("  ✅ Model checkpointing (best model saved)")
    
    # Example 1: Simple usage (recommended)
    example_1_default_training()
    
    # Example 2: Patient training
    # example_2_patient_training()
    
    # Example 3: Fast training
    # example_3_aggressive_training()
    
    # Example 4: Detailed monitoring
    # example_4_detailed_monitoring()
    
    # Example 5: Compare configurations
    # example_5_compare_configurations()
    
    # Example 6: Full customization
    # example_6_custom_all_params()
    
    print("\n" + "="*80)
    print("✅ Examples completed!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Default settings work well for most cases")
    print("2. Early stopping typically saves 30-40% training time")
    print("3. Models automatically converge to best validation performance")
    print("4. Customize patience/lr for specific needs")
    print("\nSee TRAINING_IMPROVEMENTS.md for full documentation.")

