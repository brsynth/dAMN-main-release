#!/usr/bin/env python
"""
Pure MLP (no dynamics, no mechanism) on the dAMN M28 media splits
(SI Table S2 of the dAMN paper; splits from damn_splits.py).

MLP: x -> Dense(500, ReLU) -> Dropout(0.2) -> Dense(T-1, linear)
  x = C(0) as in dAMN: the 28 nutrient concentrations of the medium
      (M28_media.csv) + the measured biomass at the first time point,
      standardised on the training media
  output = biomass at the T-1 = 19 later time points, in one pass:
      no Euler step, no time-stepping, no Sv = 0, v >= 0 or lag phase.
  The hidden layer and dropout are those of dAMN's flux network; the first
  time point is the measured biomass, as in dAMN.
--target logod : the output is log(OD), the space R2 is scored in (default)
--target conc  : the output is biomass in gDW/L, the unit of dAMN's loss;
                 MSE in gDW/L barely weighs the low early values, and with
                 a linear output some test curves get biomass <= 0, which
                 ruins log(OD) (R2 0.54-0.81 over seeds, vs 0.84-0.90 with
                 logod)
Training (dAMN settings): MSE on the output at all later time points of the
training media, Adam lr 1e-3 decayed x0.9 every 280 steps, batch 10,
up to 1000 epochs, early stopping with patience 100 (best weights restored).
--stop inner : early stopping on 20 % of the training media (held out,
               not trained on), the clean protocol (default)
--stop test  : early stopping on the test media, as dAMN's train_model does
               (the reported test R2 is then optimistic, like dAMN's)

R2 per test medium as dAMN (utils.r2_growth_curve_with_std): r2_score on
log(OD) = log(biomass / 0.37) over the 20 time points, clipped at 0.
Reported for each model seed, and for the ensemble (mean log-OD prediction
of the seeds), which is how the dAMN notebook scores its 3 models.

Reads results/damn_splits/damn_splits.csv and the dAMN data files.
Writes results/damn_splits/mlp_r2_per_medium_<target>_<stop>.csv and
mlp_r2_by_split_<target>_<stop>.csv.

Usage:
    python mlp_damn_splits.py
    python mlp_damn_splits.py --stop test
    python mlp_damn_splits.py --target conc
    python mlp_damn_splits.py --damn-data <path to data/>
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score

tf.config.set_visible_devices([], 'GPU')  # small nets; the Metal plugin aborts at random

OD_TO_CONC = 0.37  # dAMN utils.OD_TO_CONC (gDW/L per OD unit)
SPLIT_FILE = 'results/damn_splits/damn_splits.csv'
OUT_DIR = 'results/damn_splits'


def load_m28(damn_data):
    """Return media IDs, X (Z, 29) = [medium, biomass(t0)] and Y (Z, T) biomass."""
    media = pd.read_csv(os.path.join(damn_data, 'M28_media.csv'))
    meas = pd.read_csv(os.path.join(damn_data, 'M28_conc_20.csv'))
    ids = media['ID'].to_numpy(dtype=int)
    Y = np.stack([meas[f'BIOMASS_{i}'].to_numpy(dtype=float) for i in ids])
    X = np.column_stack([media.drop(columns='ID').to_numpy(dtype=float), Y[:, 0]])
    return ids, X, Y


def build_mlp(n_in, n_out, hidden=500, dropout=0.2):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_in,)),
        tf.keras.layers.Dense(hidden, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_out, activation='linear'),
    ])


def fit_predict(X_tr, Y_tr, X_val, Y_val, X_te, seed, args):
    """Train on (X_tr, Y_tr[:, 1:]), early-stop on the validation set,
    return biomass curves (incl. the measured first point) for X_te."""
    tf.keras.utils.set_random_seed(seed)
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1.0
    scale = lambda X: (X - mu) / sd
    mdl = build_mlp(X_tr.shape[1], Y_tr.shape[1] - 1)
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-3, decay_steps=280, decay_rate=0.9, staircase=True)
    mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=args.patience, restore_best_weights=True)
    to_out = log_od if args.target == 'logod' else (lambda y: y)
    mdl.fit(scale(X_tr), to_out(Y_tr[:, 1:]),
            validation_data=(scale(X_val), to_out(Y_val[:, 1:])),
            epochs=args.epochs, batch_size=args.batch_size, callbacks=[stop],
            shuffle=True, verbose=0)
    pred = mdl.predict(scale(X_te), verbose=0)
    if args.target == 'logod':
        pred = OD_TO_CONC * np.exp(pred)
    return np.column_stack([X_te[:, -1], pred]), stop.stopped_epoch


def log_od(x):
    return np.log(np.maximum(np.asarray(x) / OD_TO_CONC, 1e-8))


def r2_curves(log_pred, log_ref):
    return np.array([max(0.0, r2_score(r, p)) for p, r in zip(log_pred, log_ref)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--damn-data', default='../data')
    p.add_argument('--target', choices=['logod', 'conc'], default='logod')
    p.add_argument('--stop', choices=['inner', 'test'], default='inner')
    p.add_argument('--split-seeds', type=int, nargs='+', default=[30, 10])
    p.add_argument('--x-folds', nargs='+', default=['1.5', '2', '3'])
    p.add_argument('--model-seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--val-frac', type=float, default=0.2)
    args = p.parse_args()

    ids, X, Y = load_m28(args.damn_data)
    splits = pd.read_csv(SPLIT_FILE).set_index('ID').loc[ids]
    per_medium, summary = [], []
    for split_seed in args.split_seeds:
        for x_fold in args.x_folds:
            col = f'seed{split_seed}_xfold{x_fold}'
            te = np.where(splits[col].to_numpy() == 'test')[0]
            tr_all = np.where(splits[col].to_numpy() == 'train')[0]
            log_preds, r2_seeds = [], []
            for seed in args.model_seeds:
                if args.stop == 'inner':
                    rng = np.random.default_rng(seed)
                    perm = rng.permutation(tr_all)
                    n_val = int(round(args.val_frac * len(tr_all)))
                    val, tr = perm[:n_val], perm[n_val:]
                else:
                    val, tr = te, tr_all
                pred, ep = fit_predict(X[tr], Y[tr], X[val], Y[val], X[te], seed, args)
                lp = log_od(pred)
                r2 = r2_curves(lp, log_od(Y[te]))
                log_preds.append(lp)
                r2_seeds.append(r2.mean())
                for i, r in zip(ids[te], r2):
                    per_medium.append(dict(split=col, model_seed=seed, ID=i, R2=r))
                print(f'{col} seed {seed}: train {len(tr)} val {len(val)} test {len(te)} '
                      f'stopped epoch {ep}  R2 mean {r2.mean():.3f}  median {np.median(r2):.3f}')
            r2_ens = r2_curves(np.mean(log_preds, axis=0), log_od(Y[te]))
            for i, r in zip(ids[te], r2_ens):
                per_medium.append(dict(split=col, model_seed='ensemble', ID=i, R2=r))
            summary.append(dict(
                split_seed=split_seed, x_fold=x_fold, n_train=len(tr_all), n_test=len(te),
                R2_mean_over_seeds=np.mean(r2_seeds), R2_sd_over_seeds=np.std(r2_seeds),
                R2_ensemble_mean=r2_ens.mean(), R2_ensemble_median=np.median(r2_ens)))
            print(f'{col} ensemble: R2 mean {r2_ens.mean():.3f}  median {np.median(r2_ens):.3f}')

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(per_medium).to_csv(f'{OUT_DIR}/mlp_r2_per_medium_{args.target}_{args.stop}.csv', index=False)
    out = pd.DataFrame(summary).round(3)
    out.to_csv(f'{OUT_DIR}/mlp_r2_by_split_{args.target}_{args.stop}.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
