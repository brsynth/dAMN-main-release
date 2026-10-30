#!/usr/bin/env python
"""
Acetate overflow on the Millard data (dAMN paper, SI Fig. S9b): models see
glucose and biomass only, acetate is predicted.

E. coli MG1655, one batch culture, 13 time points (0-4.58 h), C(0) =
glucose 12.66 mM, acetate 0.96 mM, biomass 0.063 gDW/L.

dFBA  textbook dFBA, static optimisation approach (Varma & Palsson 1994;
      Mahadevan et al. 2002) on iML1515 (iML1515_duplicated, default ATP
      maintenance): at each step of dt = 0.05 h, Michaelis-Menten uptake
      bounds v_glc <= Vmax_glc G/(Km_glc + G), v_ac <= Vmax_ac A/(Km_ac + A)
      (and <= what is left in the medium), O2 uptake <= O2 max, growth
      maximised, fluxes made unique by pFBA, then X and C integrated exactly
      at constant fluxes; secretion (e.g. acetate overflow) is free.
        literature : Vmax_glc 10.5, O2 max 15 mmol/gDW/h (Varma & Palsson 1994),
                     Km_glc 0.01 mM, Km_ac 0.2 mM (reported half-saturation
                     constants), Vmax_ac 10 mmol/gDW/h (assumed)
        fitted     : Vmax_glc and O2 max by grid search on glucose and biomass
                     only (acetate never used): best fit (lowest loss) and the
                     envelope (min-max) of every grid point whose glucose and
                     biomass R2 are both >= --envelope-r2 (0.98). Glucose and
                     biomass fix a ridge in (Vmax_glc, O2 max) along which the
                     acetate prediction changes widely, so the envelope shows
                     how much of acetate the data actually determine.
      No cap on acetate secretion (dAMN's UB_out is not part of dFBA).
MLP   t -> Dense(500, ReLU) -> Dropout(0.2) -> Dense(3) = (glucose, acetate,
      biomass), pred(t) = C(0) + s * (f(t) - f(0)) (s = per-variable scale),
      so every output starts at the measured C(0). MSE on glucose and biomass
      only: the acetate output gets no training signal. 3 seeds.
dAMN  not rerun: the published Fig. S9b run (ensemble of 3 models, trained on
      glucose and biomass, UB_out = 0.3) is digitised from its plots in
      published_figures/Millard_12_1_concentration-ACE_* (plots of the v1.0.0
      release, the run published as SI Fig. S9b),
      calibrated on the measured data dots drawn in each plot.

Writes results/millard/millard_curves.csv, millard_scores.csv,
millard_dfba_grid.csv and millard_acetate.png.

Usage:
    python millard_acetate.py
    python millard_acetate.py --data-dir <path to data/>
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

OUT_DIR = 'results/millard'
BIOMASS_RXN = 'BIOMASS_Ec_iML1515_core_75p37M'
VARS = ['glc__D_e', 'ac_e', 'BIOMASS']
LABELS = {'glc__D_e': 'Glucose (mM)', 'ac_e': 'Acetate (mM)', 'BIOMASS': 'Biomass (gDW/L)'}
PNG = 'Millard_12_1_concentration-ACE_exp1_{}_ensemble.png'


def load_millard(data_dir):
    d = pd.read_csv(os.path.join(data_dir, 'Millard_conc_12.csv'))
    t = d['time'].to_numpy(float)
    Y = np.column_stack([d[f'{v}_1'].to_numpy(float) for v in VARS])
    return t, Y


# ---------------------------------------------------------------- dAMN plots
def digitise_damn(png, t, y_meas):
    """Mean line and +-1 sd band of a dAMN ensemble plot, at the times t.
    Pixel -> data maps are fitted on the measured data dots of the plot."""
    import matplotlib.image as mpimg
    from scipy import ndimage
    img = mpimg.imread(png)[..., :3] * 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    # data dots: filled black discs (about 29 px); the mean line is drawn
    # over some of them and cuts them, so x comes from every dot and y only
    # from uncut ones (full height)
    dark = (r < 60) & (g < 60) & (b < 60)
    lab, n = ndimage.label(dark)
    comps = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = (lab[sl] == i).sum()
        if 15 <= h <= 35 and 24 <= w <= 35 and h <= w + 2 and area > 0.45 * h * w:
            comps.append(((sl[1].start + sl[1].stop - 1) / 2,
                          (sl[0].start + sl[0].stop - 1) / 2, h, w))
    comps = sorted(comps)
    if len(comps) != len(t):
        raise RuntimeError(f'{png}: found {len(comps)} data dots, expected {len(t)}')
    cx, cy, h, w = np.array(comps).T
    full = h >= w - 3
    ax_, bx_ = np.polyfit(t, cx, 1)                    # x_pix = ax_ * t + bx_
    ay_, by_ = np.polyfit(y_meas[full], cy[full], 1)   # y_pix = ay_ * y + by_
    to_y = lambda py: (py - by_) / ay_
    # mean line: tab:blue (31, 119, 180); band: light blue fill
    line = (np.abs(r - 31) < 45) & (np.abs(g - 119) < 45) & (np.abs(b - 180) < 45)
    band = (b > r + 12) & (b > 150) & ~line
    def column(mask, x):
        rows = np.where(mask[:, int(round(x))])[0]
        return rows if rows.size else None
    def at(mask, k, side, fn):
        # value at t[k], extrapolated linearly from two columns on one side
        # (dots hide the line at t[k]; the curves are linear between points)
        x0 = ax_ * t[k] + bx_
        xs = [x0 + side * 22, x0 + side * 40]
        vs = []
        for x in xs:
            rows = column(mask | line, x) if mask is band else column(mask, x)
            vs.append(np.nan if rows is None else fn(rows))
        return vs[0] + (vs[0] - vs[1]) * 22 / 18
    mean, lo, hi = [], [], []
    for k in range(len(t)):
        side = 1 if k == 0 else -1
        mean.append(to_y(at(line, k, side, np.mean)))
        top, bot = to_y(at(band, k, side, np.min)), to_y(at(band, k, side, np.max))
        lo.append(min(top, bot)), hi.append(max(top, bot))
    mean, lo, hi = map(np.array, (mean, lo, hi))
    sd = np.maximum((hi - lo) / 2, 0.0)
    return mean, sd


# ---------------------------------------------------------------------- dFBA
def simulate_dfba(model, t, Y, vmax_glc, o2_max, km_glc, vmax_ac, km_ac, dt,
                  return_t_out=False):
    """Textbook dFBA (static optimisation approach) on iML1515, reported at t
    (and, with return_t_out, the time glucose is exhausted, inf if never)."""
    import cobra
    with model as md:
        md.reactions.get_by_id('EX_o2_e_i').upper_bound = o2_max
        G, A, X = Y[0]
        tt = np.arange(0.0, t[-1] + dt / 2, dt)
        traj = [(G, A, X)]
        for _ in tt[1:]:
            # Michaelis-Menten uptake bounds (mmol/gDW/h), never more than
            # what is left in the medium during the step
            md.reactions.get_by_id('EX_glc__D_e_i').upper_bound = \
                min(vmax_glc * G / (km_glc + G), G / (X * dt))
            md.reactions.get_by_id('EX_ac_e_i').upper_bound = \
                min(vmax_ac * A / (km_ac + A), A / (X * dt))
            sol = md.optimize()
            if sol.status == 'optimal':
                sol = cobra.flux_analysis.pfba(md)
                v = sol.fluxes
                mu = max(v[BIOMASS_RXN], 0.0)
                net_g = v['EX_glc__D_e_o'] - v['EX_glc__D_e_i']
                net_a = v['EX_ac_e_o'] - v['EX_ac_e_i']
            else:
                mu, net_g, net_a = 0.0, 0.0, 0.0
            # exact integration over the step at constant fluxes
            # (Varma & Palsson 1994): X(t+dt) = X e^(mu dt), dC = v X (e^(mu dt) - 1) / mu
            gain = X * np.expm1(mu * dt) / mu if mu > 1e-12 else X * dt
            X = X * np.exp(mu * dt)
            G = max(G + net_g * gain, 0.0)
            A = max(A + net_a * gain, 0.0)
            traj.append((G, A, X))
    traj = np.array(traj)
    P = np.column_stack([np.interp(t, tt, traj[:, j]) for j in range(3)])
    if return_t_out:
        out = np.where(traj[:, 0] <= 1e-6)[0]
        return P, (tt[out[0]] if out.size else np.inf)
    return P


def fit_loss(P, Y):
    """Glucose + biomass SSE, each normalised by the variance of the data."""
    return sum(np.sum((P[:, j] - Y[:, j]) ** 2) / np.var(Y[:, j]) for j in (0, 2))


def fit_dfba(data_dir, t, Y, args):
    import cobra
    model = cobra.io.read_sbml_model(
        os.path.join(data_dir, 'iML1515_duplicated.xml'))
    kw = dict(km_glc=args.km_glc, vmax_ac=args.vmax_ac, km_ac=args.km_ac, dt=args.dt)
    lit = simulate_dfba(model, t, Y, args.vmax_glc, args.o2_max, **kw)
    print(f'dFBA literature (Vmax_glc {args.vmax_glc}, O2 max {args.o2_max}): '
          f'glc+bio loss {fit_loss(lit, Y):.2f}')
    rows, curves = [], []
    for o2 in args.o2_grid:
        for vg in args.vmax_glc_grid:
            P, t_out = simulate_dfba(model, t, Y, vg, o2, return_t_out=True, **kw)
            sc = scores(P, Y)
            rows.append(dict(vmax_glc=vg, o2_max=o2, t_glc_out=t_out,
                             loss_glc_bio=fit_loss(P, Y), ac_max=P[:, 1].max(),
                             ac_end=P[-1, 1], **sc))
            curves.append(P)
    grid = pd.DataFrame(rows)
    curves = np.array(curves)
    b = int(grid['loss_glc_bio'].idxmin())
    ok = ((grid['R2_glc__D_e'] >= args.envelope_r2) &
          (grid['R2_BIOMASS'] >= args.envelope_r2)).to_numpy()
    grid['in_envelope'] = ok
    env = (curves[ok].min(0), curves[ok].max(0))
    print(f'dFBA fitted: Vmax_glc {grid.vmax_glc[b]}, O2 max {grid.o2_max[b]}, '
          f'glc+bio loss {grid.loss_glc_bio[b]:.2f}; envelope: {ok.sum()} sets, '
          f'acetate R2 {grid.R2_ac_e[ok].min():.2f} to {grid.R2_ac_e[ok].max():.2f}')
    return grid, lit, curves[b], env, (grid.vmax_glc[b], grid.o2_max[b])


# ----------------------------------------------------------------------- MLP
def fit_mlp(t, Y, seeds, epochs):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    scale = np.array([Y[:, 0].std(), 1.0, Y[:, 2].std()])  # acetate: 1 mM (no data)
    tt = (t / t.max())[:, None].astype('float32')
    preds = []
    for seed in seeds:
        tf.keras.utils.set_random_seed(seed)
        f = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation='linear'),
        ])
        C0 = tf.constant(Y[0], tf.float32)
        sc = tf.constant(scale, tf.float32)
        t0 = tf.zeros((1, 1))
        y = tf.constant(Y, tf.float32)
        w = tf.constant([1.0, 0.0, 1.0])  # acetate is not in the loss
        opt = tf.keras.optimizers.Adam(1e-3)
        def pred(training):
            return C0 + sc * (f(tt, training=training) - f(t0, training=training))
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(w * tf.square((pred(True) - y) / sc))
            opt.apply_gradients(zip(tape.gradient(loss, f.trainable_variables),
                                    f.trainable_variables))
        preds.append(pred(False).numpy())
    return np.array(preds)


# ------------------------------------------------------------------- scoring
def scores(P, Y):
    out = {}
    for j, v in enumerate(VARS):
        res = Y[:, j] - P[:, j]
        out[f'R2_{v}'] = 1 - np.sum(res ** 2) / np.sum((Y[:, j] - Y[:, j].mean()) ** 2)
        out[f'RMSE_{v}'] = np.sqrt(np.mean(res ** 2))
    return out


def plot(t, Y, curves, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = {
        'dFBA, fitted': dict(color='#d95f02', ls='-'),
        'dFBA, literature': dict(color='#d95f02', ls='--', alpha=0.7),
        'MLP':         dict(color='#7570b3', ls='-'),
        'dAMN':        dict(color='#1b9e77', ls='-'),
    }
    titles = ['Glucose (training fit)', 'Biomass (training fit)', 'Acetate (prediction, not trained)']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, j, title in zip(axes, [0, 2, 1], titles):
        for name, (mean, lo, hi) in curves.items():
            ax.plot(t, mean[:, j], lw=2, label=name, **style[name])
            if lo is not None:
                ax.fill_between(t, lo[:, j], hi[:, j],
                                color=style[name]['color'], alpha=0.18, lw=0)
        ax.plot(t, Y[:, j], 'ko', ms=5, label='measured')
        ax.set_title(title)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel(LABELS[VARS[j]])
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle('E. coli, Millard data: models trained on glucose and biomass, '
                 'acetate predicted (cf. dAMN SI Fig. S9b)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='../data')
    p.add_argument('--published-dir', default='published_figures')
    p.add_argument('--vmax-glc', type=float, default=10.5)
    p.add_argument('--o2-max', type=float, default=15.0)
    p.add_argument('--km-glc', type=float, default=0.01)
    p.add_argument('--vmax-ac', type=float, default=10.0)
    p.add_argument('--km-ac', type=float, default=0.2)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--vmax-glc-grid', type=float, nargs='+',
                   default=list(np.arange(7.0, 9.01, 0.25)))
    p.add_argument('--o2-grid', type=float, nargs='+',
                   default=[12, 13, 14, 15, 16, 17, 18])
    p.add_argument('--envelope-r2', type=float, default=0.98)
    p.add_argument('--mlp-seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--mlp-epochs', type=int, default=3000)
    args = p.parse_args()
    warnings.filterwarnings('ignore')
    os.makedirs(OUT_DIR, exist_ok=True)

    t, Y = load_millard(args.data_dir)

    fig_dir = args.published_dir
    damn = [digitise_damn(os.path.join(fig_dir, PNG.format(v)), t, Y[:, j])
            for j, v in enumerate(VARS)]
    damn_mean = np.column_stack([m for m, _ in damn])
    damn_sd = np.column_stack([s for _, s in damn])

    grid, dfba_lit, dfba_fit, dfba_env, dfba_par = fit_dfba(args.data_dir, t, Y, args)
    grid.to_csv(f'{OUT_DIR}/millard_dfba_grid.csv', index=False)

    mlp = fit_mlp(t, Y, args.mlp_seeds, args.mlp_epochs)
    mlp_mean, mlp_sd = mlp.mean(0), mlp.std(0)

    curves = {
        'dFBA, literature': (dfba_lit, None, None),
        'dFBA, fitted': (dfba_fit, *dfba_env),
        'MLP': (mlp_mean, mlp_mean - mlp_sd, mlp_mean + mlp_sd),
        'dAMN': (damn_mean, damn_mean - damn_sd, damn_mean + damn_sd),
    }
    rows = []
    for name, (mean, _, _) in curves.items():
        rows.append(dict(model=name, **{k: round(v, 3) for k, v in scores(mean, Y).items()}))
    rows[0].update(vmax_glc=args.vmax_glc, o2_max=args.o2_max)
    rows[1].update(vmax_glc=dfba_par[0], o2_max=dfba_par[1])
    env = grid[grid.in_envelope]
    rows.append(dict(model=f'dFBA, envelope ({len(env)} sets, min-max)',
                     **{f'{k}_{v}': f'{env[f"{k}_{v}"].min():.3f} to {env[f"{k}_{v}"].max():.3f}'
                        for v in VARS for k in ('R2', 'RMSE')}))
    sc = pd.DataFrame(rows)
    sc.to_csv(f'{OUT_DIR}/millard_scores.csv', index=False)
    print(sc.round(3).to_string(index=False))

    tab = pd.DataFrame({'time': t})
    for j, v in enumerate(VARS):
        tab[f'measured_{v}'] = Y[:, j]
        for name, (mean, lo, hi) in curves.items():
            key = name.replace(', ', '_').replace(' ', '_')
            tab[f'{key}_{v}'] = mean[:, j]
            if lo is not None:
                tab[f'{key}_{v}_low'] = lo[:, j]
                tab[f'{key}_{v}_high'] = hi[:, j]
    tab.to_csv(f'{OUT_DIR}/millard_curves.csv', index=False)

    plot(t, Y, curves, f'{OUT_DIR}/millard_acetate.png')


if __name__ == '__main__':
    main()
