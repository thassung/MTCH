"""
Diagnostic script to check if the selector is producing meaningful masks.
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from subgrapher.subgraph.differentiable_ppr_selector import DifferentiablePPRSelector
from subgrapher.utils.loader import load_txt_to_pyg
from subgrapher.benchmark.data_prep import add_node_features
from subgrapher.subgraph.ppr_computer import PPRComputer

def diagnose_selector():
    """Check what values the selector produces."""
    
    print("="*80)
    print("SELECTOR DIAGNOSTIC")
    print("="*80)
    
    # Load a small dataset
    print("\nLoading FB15K237...")
    data, _, _ = load_txt_to_pyg('data/FB15K237/train.txt')
    data = add_node_features(data, method='random', feature_dim=128)
    
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    
    # Create selector
    print("\nCreating selector with default parameters...")
    selector = DifferentiablePPRSelector(
        adaptive_threshold=True,
        init_alpha=0.5,
        init_threshold=0.3,
        sharpness=100.0  # Updated to match new default
    )
    
    print(f"  Alpha: {selector.get_alpha().item():.4f}")
    print(f"  Threshold percentile: {selector.get_threshold_percentile().item():.4f}")
    print(f"  Sharpness: {selector.sharpness}")
    
    # Create PPR computer
    print("\nComputing PPR for sample edge...")
    ppr_computer = PPRComputer(data, ppr_alpha=0.85)
    
    # Pick a random edge
    u, v = 0, 100
    ppr_u, ppr_v = ppr_computer.compute_ppr_pair(u, v)
    
    print(f"\nPPR Statistics:")
    print(f"  PPR_u range: [{ppr_u.min():.6f}, {ppr_u.max():.6f}]")
    print(f"  PPR_u mean: {ppr_u.mean():.6f}")
    print(f"  PPR_v range: [{ppr_v.min():.6f}, {ppr_v.max():.6f}]")
    print(f"  PPR_v mean: {ppr_v.mean():.6f}")
    
    # Get selector output
    print("\nComputing selector output...")
    with torch.no_grad():
        soft_mask, combined_scores, metadata = selector(ppr_u, ppr_v)
    
    print(f"\nCombined Scores (raw):")
    print(f"  Range: [{combined_scores.min():.6f}, {combined_scores.max():.6f}]")
    print(f"  Mean: {combined_scores.mean():.6f}")
    print(f"  Std: {combined_scores.std():.6f}")
    
    # Show normalized scores (what actually goes into sigmoid)
    alpha_val = 0.5  # Default alpha
    combined_raw = alpha_val * ppr_u + (1 - alpha_val) * ppr_v
    
    # Then normalize for sigmoid
    combined_norm = (combined_raw - combined_raw.min()) / (combined_raw.max() - combined_raw.min() + 1e-8)
    
    print(f"\nRaw PPR scores (preserving absolute magnitudes):")
    print(f"  PPR_u range: [{ppr_u.min():.8f}, {ppr_u.max():.6f}]")
    print(f"  PPR_v range: [{ppr_v.min():.8f}, {ppr_v.max():.6f}]")
    print(f"\nCombined (raw, alpha={alpha_val}):")
    print(f"  Range: [{combined_raw.min():.8f}, {combined_raw.max():.6f}]")
    print(f"  Mean: {combined_raw.mean():.8f}")
    print(f"  Std: {combined_raw.std():.8f}")
    print(f"\nCombined (normalized for sigmoid):")
    print(f"  Range: [{combined_norm.min():.6f}, {combined_norm.max():.6f}]")
    print(f"  Mean: {combined_norm.mean():.6f}")
    print(f"  Std: {combined_norm.std():.6f}")
    
    print(f"\nThreshold:")
    print(f"  Value: {metadata['threshold']:.6f}")
    print(f"  Percentile: {metadata['threshold_percentile']:.4f}")
    
    print(f"\nSoft Mask:")
    print(f"  Range: [{soft_mask.min():.6f}, {soft_mask.max():.6f}]")
    print(f"  Mean: {soft_mask.mean():.6f}")
    print(f"  Std: {soft_mask.std():.6f}")
    print(f"  Num > 0.5: {(soft_mask > 0.5).sum().item()}")
    print(f"  Num > 0.9: {(soft_mask > 0.9).sum().item()}")
    print(f"  Num < 0.1: {(soft_mask < 0.1).sum().item()}")
    
    # Check if mask is saturated
    saturated_high = (soft_mask > 0.9).float().mean().item()
    saturated_low = (soft_mask < 0.1).float().mean().item()
    middle = ((soft_mask >= 0.1) & (soft_mask <= 0.9)).float().mean().item()
    
    print(f"\nSaturation Analysis:")
    print(f"  High (>0.9): {saturated_high*100:.1f}%")
    print(f"  Low (<0.1): {saturated_low*100:.1f}%")
    print(f"  Middle (0.1-0.9): {middle*100:.1f}%")
    
    # Diagnosis
    print(f"\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    if middle > 0.8:
        print("❌ PROBLEM: Soft mask is mostly in middle range (0.1-0.9)")
        print("   This means sigmoid is in its linear region for most nodes")
        print("   → Weak selection signal")
        print("   → Weak gradients")
        print("   → Poor learning")
        print()
        print("   CAUSE: PPR scores are too small relative to sharpness")
        print(f"   Normalized scores: ~{combined_norm.mean():.6f}")
        print(f"   Threshold: {metadata['threshold']:.6f}")
        print(f"   Difference: ~{abs(combined_norm.mean() - metadata['threshold']):.6f}")
        print(f"   Scaled by sharpness: ~{abs(combined_norm.mean() - metadata['threshold']) * selector.sharpness:.3f}")
        print()
        print("   FIX: Increase sharpness or normalize PPR scores")
        
    elif saturated_high > 0.4 and saturated_low > 0.4:
        print("✅ GOOD: Soft mask is well-separated")
        print("   Strong selection signal")
        print("   Good gradients")
        
    elif saturated_high > 0.9 or saturated_low > 0.9:
        print("⚠️  WARNING: Mask is almost entirely saturated")
        print("   Either selecting almost everything or almost nothing")
        print("   → Gradients near zero")
        print("   → Cannot learn")
        
    # Test gradient flow
    print(f"\n" + "="*80)
    print("GRADIENT FLOW TEST")
    print("="*80)
    
    # Enable gradients
    selector.train()
    ppr_u_grad = ppr_u.clone().requires_grad_(False)
    ppr_v_grad = ppr_v.clone().requires_grad_(False)
    
    soft_mask_grad, _, _ = selector(ppr_u_grad, ppr_v_grad)
    
    # Compute a simple loss
    loss = soft_mask_grad.mean()
    loss.backward()
    
    alpha_grad = selector.alpha_raw.grad
    threshold_grad = selector.threshold_percentile_raw.grad if selector.adaptive_threshold else None
    
    print(f"\nGradients:")
    print(f"  Alpha gradient: {alpha_grad.item() if alpha_grad is not None else 'None'}")
    print(f"  Threshold gradient: {threshold_grad.item() if threshold_grad is not None else 'None'}")
    
    if alpha_grad is not None and abs(alpha_grad.item()) < 1e-6:
        print("  ❌ Alpha gradient is near zero - cannot learn!")
    elif alpha_grad is not None:
        print("  ✅ Alpha gradient is non-zero - can learn")
    
    if threshold_grad is not None and abs(threshold_grad.item()) < 1e-6:
        print("  ❌ Threshold gradient is near zero - cannot learn!")
    elif threshold_grad is not None:
        print("  ✅ Threshold gradient is non-zero - can learn")
    
    # Final verdict based on gradients (most important signal)
    print(f"\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    has_good_gradients = (alpha_grad is not None and abs(alpha_grad.item()) > 1e-5 and
                          threshold_grad is not None and abs(threshold_grad.item()) > 1e-5)
    
    if has_good_gradients:
        print("✅ GRADIENTS ARE GOOD - Training should work!")
        print("   Even if soft mask appears saturated in raw scores,")
        print("   the internal normalization is working properly.")
        print()
        print("   Next step: Run 1 epoch test:")
        print("   python -m subgrapher.subgraph.run_subgraph_benchmark \\")
        print("       --datasets FB15K237 --encoders SAGE --epochs 1 \\")
        print("       --inner_steps 5 --device cuda")
    else:
        print("❌ GRADIENTS ARE TOO SMALL - Training will not work!")
        print("   Apply the normalization fix in differentiable_ppr_selector.py")


if __name__ == '__main__':
    diagnose_selector()

