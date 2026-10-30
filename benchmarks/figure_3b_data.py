#!/usr/bin/env python
"""
Curves for Figure 3b (Millard data, acetate overflow): dFBA, MLP and dAMN on
glucose, biomass (training fits) and acetate (prediction, never trained on).

dFBA  textbook dFBA of millard_acetate.py with Vmax_glc = 8 and O2 max = 15
      mmol/gDW/h, the best glucose + biomass fit on that script's first, coarse
      grid (Vmax_glc 4-10.5 x O2 6-25). On the finer grid (millard_dfba_grid.csv)
      the best fit is Vmax_glc 8.75, O2 14, and the 9 sets with glucose and
      biomass R2 >= 0.98 predict acetate R2 from -5.7 to 0.73; this set is the
      best case among them (acetate R2 0.73).
MLP, dAMN  read from results/millard/millard_curves.csv (millard_acetate.py).

Writes results/millard/figure_3b_curves.csv and figure_3b_scores.csv.

Usage:
    python millard_acetate.py      # once, for millard_curves.csv
    python figure_3b_data.py
"""

import os
import warnings

import pandas as pd

import millard_acetate as m

VMAX_GLC, O2_MAX = 8.0, 15.0
DATA_DIR = '../data'


def main():
    warnings.filterwarnings('ignore')
    import cobra
    t, Y = m.load_millard(DATA_DIR)
    model = cobra.io.read_sbml_model(
        os.path.join(DATA_DIR, 'iML1515_duplicated.xml'))
    P = m.simulate_dfba(model, t, Y, VMAX_GLC, O2_MAX, km_glc=0.01, vmax_ac=10.0,
                        km_ac=0.2, dt=0.05)
    prev = pd.read_csv(f'{m.OUT_DIR}/millard_curves.csv')
    tab = pd.DataFrame({'time': t})
    rows = []
    for j, v in enumerate(m.VARS):
        tab[f'measured_{v}'] = Y[:, j]
        tab[f'dFBA_{v}'] = P[:, j]
        for name in ('MLP', 'dAMN'):
            tab[f'{name}_{v}'] = prev[f'{name}_{v}']
            tab[f'{name}_{v}_sd'] = (prev[f'{name}_{v}_high'] - prev[f'{name}_{v}_low']) / 2
    for name in ('dFBA', 'MLP', 'dAMN'):
        pred = tab[[f'{name}_{v}' for v in m.VARS]].to_numpy()
        rows.append(dict(model=name, **{k: round(x, 3) for k, x in m.scores(pred, Y).items()}))
    tab.to_csv(f'{m.OUT_DIR}/figure_3b_curves.csv', index=False)
    sc = pd.DataFrame(rows)
    sc.to_csv(f'{m.OUT_DIR}/figure_3b_scores.csv', index=False)
    print(sc.to_string(index=False))


if __name__ == '__main__':
    main()
