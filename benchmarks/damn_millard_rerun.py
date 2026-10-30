#!/usr/bin/env python
"""
Re-run dAMN on the Millard acetate-overflow experiment of SI Fig. S9b
(trained on glucose and biomass, acetate held out, UB_in = 0, UB_out = 0.3,
seed 1, 3 models) and save the ensemble mean and sd of every variable.

Runs a cell of ../dAMN.ipynb on CPU, up to its plotting section:
  this release: cell 16 with train_test_split = 'measure-ac_e' (default)
  release v1.0.0 (the code of the published figure), from a checkout of the
  v1.0.0 tag:   --repo <checkout> --cell 10 --replace (none)

Writes results/millard/damn_rerun_<tag>.csv (time, <met>_mean, <met>_sd,
<met>_ref) and prints R2 against the measurements and the distance to the
published curve (digitised in millard_curves.csv, millard_acetate.py).

Usage:
    python damn_millard_rerun.py
    git worktree add /tmp/damn-v1 v1.0.0
    python damn_millard_rerun.py --repo /tmp/damn-v1 --cell 10 --replace --tag v1.0.0
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

OUT_DIR = os.path.abspath('results/millard')
DEFAULT_REPLACE = ["train_test_split = 'measure-glc__D_e'=>train_test_split = 'measure-ac_e'"]


def run_cell(repo, idx, replacements):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    repo = os.path.abspath(repo)
    src = ''.join(json.load(open(os.path.join(repo, 'dAMN.ipynb')))['cells'][idx]['source'])
    for rep in replacements:
        a, b = rep.split('=>')
        assert src.count(a) == 1, f'{a!r} not found once in cell {idx}'
        src = src.replace(a, b)
    cut = src.find('# 4)')
    if cut > 0:
        src = src[:src.rfind('\n# ---', 0, cut)]
    cwd = os.getcwd()
    os.chdir(repo)
    sys.path.insert(0, repo)
    g = {}
    try:
        exec(compile(src, f'dAMN.ipynb cell {idx}', 'exec'), g)
    finally:
        os.chdir(cwd)
    mdl = g.get('template_mdl') or g.get('mdl')
    mean, sd, ref = g['Pred_mean'][0], g['Pred_std'][0], g['Ref'][0]
    tab = pd.DataFrame({'time': np.asarray(mdl.times)[:mean.shape[0]]})
    for j, m in enumerate(g['metabolite_ids']):
        tab[f'{m}_mean'], tab[f'{m}_sd'], tab[f'{m}_ref'] = mean[:, j], sd[:, j], ref[:, j]
    return tab


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='..')
    p.add_argument('--cell', type=int, default=16)
    p.add_argument('--replace', nargs='*', default=DEFAULT_REPLACE)
    p.add_argument('--tag', default='this_release')
    args = p.parse_args()
    tab = run_cell(args.repo, args.cell, args.replace)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f'{OUT_DIR}/damn_rerun_{args.tag}.csv'
    tab.to_csv(out, index=False)
    print('wrote', out)
    pub = os.path.join(OUT_DIR, 'millard_curves.csv')
    pub = pd.read_csv(pub) if os.path.exists(pub) else None
    for v in ('glc__D_e', 'ac_e', 'BIOMASS'):
        y, m = tab[f'{v}_ref'].to_numpy(), tab[f'{v}_mean'].to_numpy()
        r2 = 1 - np.sum((y - m) ** 2) / np.sum((y - y.mean()) ** 2)
        line = f'{v:9s} R2 {r2:.3f}'
        if pub is not None:
            P, Psd = pub[f'dAMN_{v}'].to_numpy(), (pub[f'dAMN_{v}_high'] - pub[f'dAMN_{v}_low']).to_numpy() / 2
            line += (f'   vs published: RMSE {np.sqrt(np.mean((m - P) ** 2)):.3f}, '
                     f'{np.sum(np.abs(m - P) <= Psd + 0.02)}/{len(m)} points within published +-sd')
        print(line)


if __name__ == '__main__':
    main()
