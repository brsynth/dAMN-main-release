#!/usr/bin/env python
"""
dFBA growth-curve R2 on the M28 media splits of damn_splits.py.

Runs the M28 dFBA benchmark cell of ../dFBA.ipynb (cell 3: dFBA.predict_dFBA
on every medium of M28_conc_20.csv, uptake scaler 0.2, 0.02 for glucose,
xylose and succinate; R2 per growth curve on log(OD) as dAMN) and averages
the R2 over the test media of each split. dFBA has no training, so the split
only selects which media are averaged.

Reads results/damn_splits/damn_splits.csv (damn_splits.py).
Writes results/damn_splits/dfba_r2_per_medium.csv and dfba_r2_by_split.csv.

Usage (about 20 min for the 280 media):
    python dfba_m28_splits.py
    python dfba_m28_splits.py --n-media 5      # quick check, no output files
"""

import argparse
import json
import os
import sys

import pandas as pd

OUT_DIR = 'results/damn_splits'


def run_cell(repo, n_media=None):
    """Execute dFBA.ipynb cell 3 up to the plots; return media IDs and R2."""
    repo = os.path.abspath(repo)
    src = ''.join(json.load(open(os.path.join(repo, 'dFBA.ipynb')))['cells'][3]['source'])
    src = src.split('# -----------------------------\n# 3) Build Pred')[0]
    if n_media:
        src = src.replace('for z in range(Z):', f'for z in range({n_media}):')
    cwd = os.getcwd()
    os.chdir(repo)
    sys.path.insert(0, repo)
    g = {}
    try:
        exec(compile(src, 'dFBA.ipynb cell 3', 'exec'), g)
    finally:
        os.chdir(cwd)
    n = n_media or len(g['exp_ids'])
    return [int(e) for e in g['exp_ids'][:n]], g['R2'][:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='..')
    p.add_argument('--n-media', type=int, default=None)
    args = p.parse_args()
    ids, r2 = run_cell(args.repo, args.n_media)
    per = pd.DataFrame({'ID': ids, 'R2_dFBA': r2})
    print(f'dFBA R2 over {len(per)} media: mean {per.R2_dFBA.mean():.3f}, '
          f'median {per.R2_dFBA.median():.3f}')
    if args.n_media:
        return
    per.to_csv(f'{OUT_DIR}/dfba_r2_per_medium.csv', index=False)
    splits = pd.read_csv(f'{OUT_DIR}/damn_splits.csv').merge(per, on='ID')
    rows = []
    for col in [c for c in splits.columns if c.startswith('seed')]:
        seed, x_fold = col[4:].split('_xfold')
        test = splits[splits[col] == 'test'].R2_dFBA
        rows.append(dict(seed=int(seed), x_fold=x_fold, n_train=len(splits) - len(test),
                         n_test=len(test), R2_mean=test.mean(), R2_sd=test.std(),
                         R2_median=test.median()))
    out = pd.DataFrame(rows).round(3)
    out.to_csv(f'{OUT_DIR}/dfba_r2_by_split.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
