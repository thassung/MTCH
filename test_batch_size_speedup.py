"""
Quick test to verify batch size optimization speedup.
Compares training speed with different batch sizes on a single dataset.
"""

import torch
import time
from subgrapher.utils.loader import load_txt_to_pyg
from subgrapher.benchmark.data_prep import prepare_link_prediction_data
from subgrapher.utils.models import GCN, LinkPredictor
from subgrapher.benchmark_ppr.ppr_extractor import StaticPPRExtractor
from subgrapher.benchmark_ppr.trainer import train_epoch_ppr
from subgrapher.utils.ppr_preprocessor import PPRPreprocessor

def test_batch_size_speed():
    """Test training speed with different batch sizes."""
    print("=" * 60)
    print("Batch Size Speedup Test")
    print("=" * 60)
    
    # Configuration
    dataset_name = 'FB15K237'
    dataset_path = f'data/{dataset_name}/train.txt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDataset: {dataset_name}")
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading dataset...")
    dataset_dict = prepare_link_prediction_data(
        dataset_path,
        feature_method='random',
        feature_dim=128
    )
    
    data = dataset_dict['data']
    split_edge = dataset_dict['split_edge']
    num_train_edges = split_edge['train']['source_node'].size(0)
    
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Train edges: {num_train_edges}")
    
    # Load or create PPR preprocessor
    print(f"\nLoading PPR preprocessor...")
    preprocessed_path = f"preprocessed/{dataset_name}/ppr_alpha0.85.pt"
    
    import os
    if os.path.exists(preprocessed_path):
        print(f"  Loading from: {preprocessed_path}")
        ppr_preprocessor = PPRPreprocessor.load(preprocessed_path, data)
        print(f"  ✓ Loaded")
    else:
        print(f"  Creating new preprocessor...")
        ppr_preprocessor = PPRPreprocessor(data, ppr_alpha=0.85)
    
    # Create PPR extractor
    ppr_extractor = StaticPPRExtractor(
        data,
        alpha=0.5,
        top_k=100,
        ppr_alpha=0.85,
        preprocessor=ppr_preprocessor
    )
    
    # Create models
    print(f"\nCreating models...")
    encoder = GCN(128, 256, 256, 3, 0.3)
    predictor = LinkPredictor(256, 256, 1, 3, 0.3)
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    results = []
    
    print(f"\n{'='*60}")
    print(f"Testing Batch Sizes")
    print(f"{'='*60}")
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Reset models
        encoder = GCN(128, 256, 256, 3, 0.3)
        predictor = LinkPredictor(256, 256, 1, 3, 0.3)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(predictor.parameters()),
            lr=0.005
        )
        
        # Time one epoch
        print(f"  Running 1 epoch...")
        start = time.time()
        
        try:
            loss = train_epoch_ppr(
                encoder, predictor, data, split_edge, ppr_extractor,
                optimizer, batch_size, device, grad_clip=1.0, verbose=False
            )
            
            epoch_time = time.time() - start
            batches_per_epoch = (num_train_edges + batch_size - 1) // batch_size
            time_per_batch = epoch_time / batches_per_epoch
            subgraphs_per_second = num_train_edges / epoch_time
            
            print(f"  ✓ Epoch time: {epoch_time:.2f}s")
            print(f"    Batches: {batches_per_epoch}")
            print(f"    Time/batch: {time_per_batch:.3f}s")
            print(f"    Subgraphs/sec: {subgraphs_per_second:.1f}")
            
            results.append({
                'batch_size': batch_size,
                'epoch_time': epoch_time,
                'batches': batches_per_epoch,
                'time_per_batch': time_per_batch,
                'throughput': subgraphs_per_second,
                'success': True
            })
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({
                'batch_size': batch_size,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    
    print(f"\n{'Batch Size':>12} | {'Epoch Time':>12} | {'Batches':>10} | {'Throughput':>15}")
    print("-" * 60)
    
    for r in results:
        if r['success']:
            print(f"{r['batch_size']:>12} | {r['epoch_time']:>11.2f}s | {r['batches']:>10} | {r['throughput']:>12.1f} sub/s")
        else:
            print(f"{r['batch_size']:>12} | {'FAILED':>12} | {'-':>10} | {'-':>15}")
    
    # Find optimal
    successful = [r for r in results if r['success']]
    if successful:
        fastest = min(successful, key=lambda x: x['epoch_time'])
        best_throughput = max(successful, key=lambda x: x['throughput'])
        
        print(f"\n{'='*60}")
        print(f"Recommendations")
        print(f"{'='*60}")
        print(f"\nFastest epoch: batch_size={fastest['batch_size']} ({fastest['epoch_time']:.2f}s/epoch)")
        print(f"Best throughput: batch_size={best_throughput['batch_size']} ({best_throughput['throughput']:.1f} subgraphs/sec)")
        
        # Calculate training time estimates
        epochs = 150  # Typical early stop point
        print(f"\nEstimated training time for {epochs} epochs:")
        for r in successful:
            if r['batch_size'] in [32, 64, 128, 256]:
                total_time = r['epoch_time'] * epochs / 60
                print(f"  batch_size={r['batch_size']:>4}: {total_time:>6.1f} minutes")
        
        # Speedup comparison
        if len(successful) > 1:
            baseline = next((r for r in successful if r['batch_size'] == 1024), None)
            optimized = next((r for r in successful if r['batch_size'] == 64), None)
            
            if baseline and optimized:
                speedup = baseline['epoch_time'] / optimized['epoch_time']
                print(f"\nSpeedup (1024→64): {speedup:.1f}x faster!")
                print(f"  Old: {baseline['epoch_time']:.2f}s/epoch = {baseline['epoch_time']*epochs/3600:.1f} hours for {epochs} epochs")
                print(f"  New: {optimized['epoch_time']:.2f}s/epoch = {optimized['epoch_time']*epochs/60:.1f} minutes for {epochs} epochs")
    
    print(f"\n{'='*60}")
    print(f"Test Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_batch_size_speed()

