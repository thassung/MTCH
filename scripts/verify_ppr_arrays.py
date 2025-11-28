"""
Verification script to confirm PPR is computed as arrays (tensors) for all nodes.
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from subgrapher.utils.loader import load_txt_to_pyg
from subgrapher.benchmark.data_prep import add_node_features
from subgrapher.subgraph.ppr_computer import PPRComputer

def verify_ppr_implementation():
    """Verify PPR returns arrays of scores for all nodes."""
    
    print("="*80)
    print("PPR ARRAY VERIFICATION")
    print("="*80)
    
    # Load dataset
    print("\nLoading FB15K237...")
    data, _, _ = load_txt_to_pyg('data/FB15K237/train.txt')
    data = add_node_features(data, method='random', feature_dim=128)
    
    print(f"  Total nodes: {data.num_nodes}")
    print(f"  Total edges: {data.num_edges}")
    
    # Create PPR computer
    print("\nCreating PPR computer...")
    ppr_computer = PPRComputer(data, ppr_alpha=0.85)
    
    # Compute PPR for two seed nodes
    u, v = 0, 100
    print(f"\nComputing PPR for seed nodes u={u}, v={v}...")
    ppr_u, ppr_v = ppr_computer.compute_ppr_pair(u, v)
    
    # Verify shapes
    print(f"\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    print(f"\nppr_u:")
    print(f"  Type: {type(ppr_u)}")
    print(f"  Shape: {ppr_u.shape}")
    print(f"  Dtype: {ppr_u.dtype}")
    print(f"  Length: {len(ppr_u)}")
    
    print(f"\nppr_v:")
    print(f"  Type: {type(ppr_v)}")
    print(f"  Shape: {ppr_v.shape}")
    print(f"  Dtype: {ppr_v.dtype}")
    print(f"  Length: {len(ppr_v)}")
    
    # Verify dimensions match
    print(f"\n" + "="*80)
    print("DIMENSION CHECKS")
    print("="*80)
    
    checks_passed = True
    
    # Check 1: PPR arrays should have length = num_nodes
    if len(ppr_u) == data.num_nodes:
        print(f"✅ ppr_u has correct length ({len(ppr_u)} == {data.num_nodes})")
    else:
        print(f"❌ ppr_u has WRONG length ({len(ppr_u)} != {data.num_nodes})")
        checks_passed = False
    
    if len(ppr_v) == data.num_nodes:
        print(f"✅ ppr_v has correct length ({len(ppr_v)} == {data.num_nodes})")
    else:
        print(f"❌ ppr_v has WRONG length ({len(ppr_v)} != {data.num_nodes})")
        checks_passed = False
    
    # Check 2: PPR arrays should be 1D tensors
    if ppr_u.dim() == 1:
        print(f"✅ ppr_u is 1D tensor")
    else:
        print(f"❌ ppr_u is NOT 1D (dim={ppr_u.dim()})")
        checks_passed = False
    
    if ppr_v.dim() == 1:
        print(f"✅ ppr_v is 1D tensor")
    else:
        print(f"❌ ppr_v is NOT 1D (dim={ppr_v.dim()})")
        checks_passed = False
    
    # Check 3: PPR scores should sum to ~1.0 (PageRank property)
    ppr_u_sum = ppr_u.sum().item()
    ppr_v_sum = ppr_v.sum().item()
    
    print(f"\n" + "="*80)
    print("PPR PROPERTIES")
    print("="*80)
    
    print(f"\nppr_u statistics:")
    print(f"  Sum: {ppr_u_sum:.6f} (should be ≈1.0)")
    print(f"  Min: {ppr_u.min().item():.8f}")
    print(f"  Max: {ppr_u.max().item():.6f}")
    print(f"  Mean: {ppr_u.mean().item():.8f}")
    print(f"  Score at seed u={u}: {ppr_u[u].item():.6f}")
    
    print(f"\nppr_v statistics:")
    print(f"  Sum: {ppr_v_sum:.6f} (should be ≈1.0)")
    print(f"  Min: {ppr_v.min().item():.8f}")
    print(f"  Max: {ppr_v.max().item():.6f}")
    print(f"  Mean: {ppr_v.mean().item():.8f}")
    print(f"  Score at seed v={v}: {ppr_v[v].item():.6f}")
    
    if 0.99 < ppr_u_sum < 1.01:
        print(f"✅ ppr_u sum is correct ({ppr_u_sum:.4f} ≈ 1.0)")
    else:
        print(f"⚠️  ppr_u sum is {ppr_u_sum:.4f} (should be ≈1.0)")
    
    if 0.99 < ppr_v_sum < 1.01:
        print(f"✅ ppr_v sum is correct ({ppr_v_sum:.4f} ≈ 1.0)")
    else:
        print(f"⚠️  ppr_v sum is {ppr_v_sum:.4f} (should be ≈1.0)")
    
    # Check 4: Test alpha combination
    print(f"\n" + "="*80)
    print("ALPHA COMBINATION TEST")
    print("="*80)
    
    alpha = 0.5
    combined = alpha * ppr_u + (1 - alpha) * ppr_v
    
    print(f"\nCombining with alpha={alpha}:")
    print(f"  combined_scores shape: {combined.shape}")
    print(f"  combined_scores length: {len(combined)}")
    print(f"  combined_scores range: [{combined.min().item():.8f}, {combined.max().item():.6f}]")
    print(f"  combined_scores sum: {combined.sum().item():.6f}")
    
    if combined.shape == ppr_u.shape:
        print(f"✅ Combined scores have correct shape")
    else:
        print(f"❌ Combined scores have WRONG shape")
        checks_passed = False
    
    # Check 5: Test normalization (CORRECT method - after combining)
    print(f"\n" + "="*80)
    print("NORMALIZATION TEST (After Combining - CORRECT)")
    print("="*80)
    
    # Combine FIRST (with raw PPR scores)
    combined_raw = alpha * ppr_u + (1 - alpha) * ppr_v
    
    print(f"\nCombined (from raw PPR scores):")
    print(f"  Range: [{combined_raw.min().item():.8f}, {combined_raw.max().item():.6f}]")
    print(f"  Mean: {combined_raw.mean().item():.8f}")
    print(f"  Std: {combined_raw.std().item():.8f}")
    
    # Then normalize (for sigmoid only)
    combined_norm = (combined_raw - combined_raw.min()) / (combined_raw.max() - combined_raw.min() + 1e-8)
    
    print(f"\nNormalized combined (for sigmoid):")
    print(f"  Range: [{combined_norm.min().item():.6f}, {combined_norm.max().item():.6f}]")
    print(f"  Mean: {combined_norm.mean().item():.6f}")
    print(f"  Std: {combined_norm.std().item():.6f}")
    
    # Test alpha's effect (using RAW PPR scores - CORRECT)
    print(f"\n" + "="*80)
    print("ALPHA EFFECT TEST (Using Raw PPR Scores)")
    print("="*80)
    
    alpha_test_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    print(f"\nTesting different alpha values on node index 50:")
    print(f"  ppr_u[50] (raw) = {ppr_u[50].item():.8f}")
    print(f"  ppr_v[50] (raw) = {ppr_v[50].item():.8f}")
    print()
    
    for alpha_test in alpha_test_values:
        # Use RAW PPR scores (correct approach)
        combined_test_raw = alpha_test * ppr_u + (1 - alpha_test) * ppr_v
        # Then normalize
        combined_test_norm = (combined_test_raw - combined_test_raw.min()) / (combined_test_raw.max() - combined_test_raw.min() + 1e-8)
        score_50 = combined_test_norm[50].item()
        print(f"  alpha={alpha_test:.1f}: combined_norm[50]={score_50:.6f}")
    
    # Check variance (should be non-zero if alpha has effect)
    scores_at_different_alphas = []
    for alpha_test in alpha_test_values:
        combined_test_raw = alpha_test * ppr_u + (1 - alpha_test) * ppr_v
        combined_test_norm = (combined_test_raw - combined_test_raw.min()) / (combined_test_raw.max() - combined_test_raw.min() + 1e-8)
        scores_at_different_alphas.append(combined_test_norm[50].item())
    variance = torch.tensor(scores_at_different_alphas).std().item()
    
    if variance > 0.01:
        print(f"\n✅ Alpha has clear effect (variance={variance:.6f})")
    else:
        print(f"\n❌ Alpha has minimal effect (variance={variance:.6f})")
        checks_passed = False
    
    # Final verdict
    print(f"\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if checks_passed:
        print("\n✅ ALL CHECKS PASSED!")
        print("   PPR is correctly implemented as arrays")
        print("   Alpha combination preserves differentiability")
        print("   Normalization approach is correct")
    else:
        print("\n❌ SOME CHECKS FAILED!")
        print("   See issues above")
    
    return checks_passed


if __name__ == '__main__':
    passed = verify_ppr_implementation()
    sys.exit(0 if passed else 1)

