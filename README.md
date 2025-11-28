# MTCH - Graph Subgraphing for Link Prediction

A Python package for graph subgraphing designed for link prediction.

## Features

- **Link Prediction Benchmark**: Compare GCN, GraphSAGE, and GAT models
- **Two-part Architecture**: Separate GNN encoder + MLP predictor
- **Comprehensive Metrics**: MRR, Hit@K, AUC-ROC, Average Precision
- **Production-Ready Training**: Early stopping, LR scheduling, gradient clipping
- **Plug-and-Play Design**: Ready for future subgrapher integration
- **GPU Acceleration**: Full CUDA support for fast training

## Installation

This project uses Poetry for dependency management. To install the dependencies:

```bash
poetry install
```

## Quick Start

### Run Single Dataset Benchmark

```bash
# Default: Early stopping enabled (recommended)
python -m subgrapher.benchmark.run_benchmark --dataset data/FB15K237/train.txt

# Custom configuration
python -m subgrapher.benchmark.run_benchmark \
    --dataset data/FB15K237/train.txt \
    --patience 30 \
    --lr 0.001 \
    --lr_scheduler reduce_on_plateau
```

### Run Full Benchmark (All Models × All Datasets)

```bash
# Run all 3 models on all 3 datasets (9 experiments total)
python -m subgrapher.benchmark.run_full_benchmark

# With custom settings
python -m subgrapher.benchmark.run_full_benchmark --epochs 50 --device cuda
```

See `subgrapher/benchmark/FULL_BENCHMARK_GUIDE.md` for detailed instructions.

### Programmatic Usage

```python
from subgrapher.benchmark.run_benchmark import run_benchmark

# Default: Early stopping enabled (recommended)
results = run_benchmark(
    dataset_path='data/FB15K237/train.txt',
    device='cuda'
)

# Custom: Adjust training parameters
results = run_benchmark(
    dataset_path='data/FB15K237/train.txt',
    patience=30,          # Early stopping patience
    lr=0.001,            # Initial learning rate
    weight_decay=1e-5,   # L2 regularization
    lr_scheduler='reduce_on_plateau',  # LR scheduling
    device='cuda'
)
```

## 🚀 New Training Features

The benchmark now includes **production-ready training** with modern best practices:

- ✅ **Early Stopping**: Automatically stops when validation plateaus (saves 30-40% time)
- ✅ **Learning Rate Scheduling**: Adaptive LR reduction for better convergence
- ✅ **Gradient Clipping**: Prevents exploding gradients and training crashes
- ✅ **Weight Decay**: L2 regularization for better generalization
- ✅ **Model Checkpointing**: Best model automatically saved and restored

**Key Benefits:**
- 🎯 Better test performance (less overfitting)
- ⚡ Faster training (stops when converged)
- 🛡️ More robust across datasets
- 📊 Rich progress monitoring

See **[TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md)** for detailed documentation and examples.

## Project Structure

```
MTCH/
├── data/                      # Knowledge graph datasets
│   ├── FB15K237/             # Freebase subset
│   ├── WN18RR/               # WordNet
│   └── NELL-995/             # NELL
├── subgrapher/
│   ├── benchmark/            # Link prediction benchmark
│   │   ├── models.py                 # GCN, SAGE, GAT implementations
│   │   ├── trainer.py                # Training pipeline
│   │   ├── evaluator.py              # Evaluation metrics
│   │   ├── run_benchmark.py          # Single dataset runner
│   │   ├── run_full_benchmark.py     # Full benchmark (all datasets)
│   │   └── FULL_BENCHMARK_GUIDE.md   # Full benchmark guide
│   └── utils/
│       ├── loader.py        # Data loading utilities
│       └── ppr_scorer.py    # PPR-based subgraph extraction
├── results/benchmark/        # Benchmark results (generated)
├── example_benchmark.py      # Example usage script
└── README.md                 # This file
```

## Benchmark Results

The benchmark compares three GNN architectures on link prediction:

| Model | Architecture | Expected Performance |
|-------|-------------|---------------------|
| GCN | Spectral convolution | Fastest training, baseline accuracy |
| GraphSAGE | Sampling-based | Best balance of speed/accuracy |
| GAT | Multi-head attention | Highest accuracy, slower training |

See `subgrapher/benchmark/README.md` for detailed benchmark documentation.

