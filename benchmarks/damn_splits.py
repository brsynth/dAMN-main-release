#!/usr/bin/env python
"""
Train/test media splits of the existing dAMN on E. coli M28 (SI Table S2 of
the dAMN paper): x_fold 1.5 (93 train / 187 test), 2 (140 / 140) and
3 (186 / 94; the paper reports 187 / 93).

Reproduces dAMN's create_model_train_val (model.py, 'medium' split):
np.random.seed(seed), shuffle the 280 media IDs (1..280, in the order of
M28_media.csv), train = first int(280 * (1 - 1/x_fold)), test = the rest.
Seed 30 is the seed of dAMN.ipynb and reproduces
model/M28_CONC_20_medium_val_ids.txt; seed 10 reproduces
model/M28_OD_20_medium_val_ids.txt of release v1.0.0.

Writes results/damn_splits/damn_splits.csv, one row per medium, with
a column per (seed, x_fold) holding 'train' or 'test'.

Usage:
    python damn_splits.py
"""
import os

import numpy as np
import pandas as pd

N_MEDIA = 280
SEEDS = [30, 10]
X_FOLDS = [1.5, 2, 3]
OUT_DIR = 'results/damn_splits'


def damn_split(seed, x_fold, n_media=N_MEDIA):
    np.random.seed(seed)
    exp_ids = list(range(1, n_media + 1))
    np.random.shuffle(exp_ids)
    split = int(len(exp_ids) * (1 - 1 / x_fold))
    return exp_ids[:split], exp_ids[split:]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    table = pd.DataFrame({'ID': range(1, N_MEDIA + 1)})
    for seed in SEEDS:
        for x_fold in X_FOLDS:
            train_ids, test_ids = damn_split(seed, x_fold)
            col = f'seed{seed}_xfold{x_fold:g}'
            table[col] = np.where(table['ID'].isin(test_ids), 'test', 'train')
            print(f'{col}: {len(train_ids)} train / {len(test_ids)} test')
    path = os.path.join(OUT_DIR, 'damn_splits.csv')
    table.to_csv(path, index=False)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
