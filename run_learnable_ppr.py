#!/usr/bin/env python3
"""
Learnable PPR end-to-end runner (CLI parity with `learnable_ppr.ipynb` / `_gen_notebook.py`).

Run from the repo root, for example:

  python run_learnable_ppr.py
  python run_learnable_ppr.py --datasets FB15K237 WN18RR --encoders SAGE GCN

Progress: outer tqdm over dataset×encoder runs; Phase 1 uses `Arch Search` tqdm;
Phase 2 uses `Fine-tuning` tqdm (and short-lived batch bars on early epochs).
"""
from subgrapher.benchmark_learnable_ppr.run_learnable_ppr_benchmark import main

if __name__ == '__main__':
    main()
