"""
Report which benchmark result files exist for cross-method comparison.

Checks:
  - results/benchmark/ (full graph)
  - results/benchmark-ppr/          (static PPR)
  - results/benchmark-khop/         (static k-hop)
  - results/benchmark-learnable-ppr/ (full_results.json + runs/*/manifest.json)

Run from repo root: python scripts/verify_benchmark_artifacts.py
"""

import argparse
import os
from pathlib import Path


def _scan_dir(label, base, want_csv=True):
    base = Path(base)
    rows = []
    if not base.is_dir():
        rows.append((label, str(base), 'missing_dir', ''))
        return rows

    csv_path = base / 'comparison_table.csv'
    if want_csv and csv_path.is_file():
        rows.append((label, str(csv_path), 'ok', 'comparison_table.csv'))
    elif want_csv:
        rows.append((label, str(csv_path), 'missing', 'comparison_table.csv'))

    for p in base.rglob('full_results.json'):
        rows.append((label, str(p), 'ok', 'full_results.json'))
    return rows


def _scan_learnable(base='results/benchmark-learnable-ppr'):
    base = Path(base)
    rows = []
    if not base.is_dir():
        rows.append(('learnable', str(base), 'missing_dir', ''))
        return rows

    for p in base.rglob('full_results.json'):
        rows.append(('learnable', str(p), 'ok', 'full_results.json'))

    for p in base.rglob('manifest.json'):
        if p.parent.parent.name == 'runs':
            rows.append(
                ('learnable_run', str(p), 'ok',
                 f"run/{p.parent.name}"))

    for p in base.glob('*/*'):
        if not p.is_dir() or p.name == 'runs':
            continue
        if (p / 'full_results.json').exists():
            continue
        if (p / 'runs').is_dir() and any((p / 'runs').iterdir()):
            continue
        rows.append(
            ('learnable', str(p), 'partial',
             'no full_results.json or runs/'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--root', type=str, default='.',
        help='Repository root (default: current directory)')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    os.chdir(root)

    print('Benchmark artifact check (paths relative to repo root)\n')
    all_rows = []
    all_rows += _scan_dir('full_graph', 'results/benchmark')
    all_rows += _scan_dir('static_ppr', 'results/benchmark-ppr')
    all_rows += _scan_dir('static_khop', 'results/benchmark-khop')
    all_rows += _scan_learnable()

    by_status = {}
    for label, path, status, kind in all_rows:
        by_status.setdefault(status, 0)
        by_status[status] += 1
        st = 'OK ' if status == 'ok' else status.upper()
        print(f'  [{st:6}] {label:14} {kind:30} {path}')

    print()
    print('Summary counts by status:', dict(by_status))
    print()
    print('To generate comparison_table.csv where missing, run the corresponding '
          'benchmark drivers (e.g. run_full_benchmark.py, run_ppr_benchmark.py, '
          'run_khop_benchmark.py).')


if __name__ == '__main__':
    main()
