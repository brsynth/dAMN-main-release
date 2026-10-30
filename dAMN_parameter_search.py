import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from itertools import product
import random
import utils
import data
import model

# -----------------------------
# Basic config
# -----------------------------
train_test_split = 'medium'
folder = './'
file_name = 'VIO_conc_17'
media_file      = folder + 'data/' + 'VIO_media.csv'
od_file         = folder + 'data/' + file_name + '.csv'
cobra_model_file = folder + 'data/' + 'iML1515_VIO_duplicated.xml'
biomass_rxn_id  = 'BIOMASS_Ec_iML1515_core_75p37M'
hidden_layers_lag  = [50]
hidden_layers_flux = [500]

run_name = f'{file_name}_{train_test_split}'

# -----------------------------
# Parameter search spaces
# order: SV, negV, C, dropC, KO
# Search over SV(l1), negV(l2), dropC(l4), KO(l5),
# and their decays SV(k1), negV(k2), dropC(k4), KO(k5).
# C weight (l3) and C decay (k3) are fixed at 1 and 0 (fit all time points equally).
# -----------------------------
l = [0.001, 0.01, 0.1, 1.0] #  1. SV, 2. negV, 3. C, 4. dropC, 5. KO
k = [0.0, 0.25, 0.50, 0.75, 1.0]
N_search   = 100
random.seed(100)

l_fixed = 1.0
k_fixed = 0.0

# (l1, l2, l4, l5, k1, k2, k4, k5)
param_grid = list(product(l, l, l, l, k, k, k, k))
param_samples = random.sample(param_grid, min(N_search, len(param_grid)))

short_num_epochs = 500
batch_size       = 10
patience         = 100
x_fold           = 6

results = []

# -----------------------------
# Best model tracking — ranked by mean(BIOMASS R2, TARGET R2)
# -----------------------------
best_R2_mean = -np.inf
best_config  = None

# ── Build model + data ───────────────────────────────────────────────────
template_mdl, train_array, train_dev, val_array, val_dev, val_ids = \
     model.create_model_train_val(
            media_file, od_file, cobra_model_file, biomass_rxn_id,
            x_fold=x_fold,
            hidden_layers_lag=hidden_layers_lag,
            hidden_layers_flux=hidden_layers_flux,
            lag_function='hill',
            dropout_rate=0.2,
            loss_weight=[1, 1, 1, 1, 1], # dummy values
            loss_decay=[0, 0, 0, 0, 0],
            verbose=False,
            train_test_split=train_test_split
    )


# ── Loop over parameters ─────────────────────────────────────────────────────
for idx, (l1_, l2_, l4_, l5_, k1_, k2_, k4_, k5_) in enumerate(param_samples):
    
    mdl = model.reset_model(template_mdl, verbose=False)
    model.loss_weight = loss_weight = [l1_, l2_, l_fixed, l4_, l5_]
    model.loss_decay  = loss_decay  = [k1_, k2_, k_fixed, k4_, k5_]

    # ── Train (2 passes, same schedule as VIO notebook) ──────────────────────
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=280,
        decay_rate=0.9, staircase=True
    )
    for _ in range(2):
        model.train_model(
            mdl, train_array, val_array=val_array,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            num_epochs=short_num_epochs, batch_size=batch_size,
            patience=patience, verbose=False,
            train_test_split=train_test_split, x_fold=x_fold
        )

    # ── Predict ───────────────────────────────────────────────────────────────
    pred, ref = model.predict_on_val_data(mdl, val_array, verbose=False)
    pred = np.expand_dims(np.asarray(pred), axis=0)   # (1, N_exp, T, k)
    ref  = np.expand_dims(np.asarray(ref),  axis=0)

    # ── R2 for BIOMASS (primary metric, with OD conversion) ──────────────────
    R2_bio = utils.r2_growth_curve(pred, ref, OD=True)
    R2_bio_mean   = np.mean(R2_bio)
    R2_bio_std    = np.std(R2_bio)
    R2_bio_median = np.median(R2_bio)

    # ── R2 for TARGET (secondary metric, no OD conversion, NaN-safe) ─────────
    metabolite_ids = list(mdl.metabolite_ids)
    target_idx = (metabolite_ids.index('TARGET')
                  if 'TARGET' in metabolite_ids else None)

    if target_idx is not None:
        R2_tgt = utils.r2_growth_curve(pred, ref, OD=False, met_idx=target_idx)
        R2_tgt_mean   = float(np.mean(R2_tgt))
        R2_tgt_std   = float(np.std(R2_tgt))
        R2_tgt_median = float(np.median(R2_tgt))
    else:
        R2_tgt_mean = R2_tgt_sdt = R2_tgt_median = float('nan')

    # ── Save best model (ranked by mean of BIOMASS + TARGET R2) ─────────────
    R2_combined = np.nanmean([R2_bio_mean, R2_tgt_mean])
    if R2_combined > best_R2_mean:
        best_R2_mean = R2_combined
        best_config  = (loss_weight, loss_decay)
        model_name = f'{folder}model/{run_name}_0'
        mdl.save_model(model_name=model_name, verbose=False)
        np.savetxt(f'{folder}model/{run_name}_train_array.txt', train_array, fmt='%f')
        np.savetxt(f'{folder}model/{run_name}_train_dev.txt',   train_dev,   fmt='%f')
        np.savetxt(f'{folder}model/{run_name}_val_array.txt',   val_array,   fmt='%f')
        np.savetxt(f'{folder}model/{run_name}_val_dev.txt',     val_dev,     fmt='%f')
        np.savetxt(f'{folder}model/{run_name}_val_ids.txt',
                   np.asarray(val_ids), fmt='%d')
        tag = ' *** BEST ***'
    else:
        tag = ''

    print(
        f'[{idx+1:3d}/{len(param_samples)}] '
        f'λ={loss_weight}  k={loss_decay} | '
        f'BIOMASS R2={R2_bio_mean:.3f}±{R2_bio_std:.3f} (med {R2_bio_median:.3f}) | '
        f'TARGET R2={R2_tgt_mean:.3f}±{R2_tgt_std:.3f} (med {R2_tgt_median:.3f})'
        f'{tag}'
    )

    results.append({
        'l1': l1_, 'l2': l2_, 'l3': l_fixed, 'l4': l4_, 'l5': l5_,
        'k1': k1_, 'k2': k2_, 'k3': k_fixed, 'k4': k4_, 'k5': k5_,
        'R2_bio_mean':   R2_bio_mean,
        'R2_bio_std':    R2_bio_std,
        'R2_bio_median': R2_bio_median,
        'R2_tgt_mean':   R2_tgt_mean,
        'R2_tgt_std':   R2_tgt_std,
        'R2_tgt_median': R2_tgt_median,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
results = sorted(results, key=lambda x: np.nanmean([x['R2_bio_median'], x['R2_tgt_median']]), reverse=True)
print("\nHyperparameter Search Results  order: SV, negV, C, dropC, KO")
header = ('l1\tl2\tl3\tl4\tl5\tk1\tk2\tk3\tk4\tk5\t'
          'R2bio-mean\tR2bio-std\tR2bio-median\tR2tgt-mean\tR2tgt-std\tR2tgt-median')
print(header)
for r in results:
    print(
        f"{r['l1']}\t{r['l2']}\t{r['l3']}\t{r['l4']}\t{r['l5']}\t"
        f"{r['k1']}\t{r['k2']}\t{r['k3']}\t{r['k4']}\t{r['k5']}\t"
        f"{r['R2_bio_mean']:.3f}\t{r['R2_bio_std']:.3f}\t"
        f"{r['R2_bio_median']:.3f}\t"
        f"{r['R2_tgt_mean']:.3f}\t{r['R2_tgt_std']:.3f}\t"
        f"{r['R2_tgt_median']:.3f}"
    )

print(f"\nBest config: λ={best_config[0]}  k={best_config[1]}")
print(f"Best combined R2 mean (BIOMASS + TARGET): {best_R2_mean:.3f}")
