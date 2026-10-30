import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf
import pandas as pd
import cobra
import json
import matplotlib.pyplot as plt
import sklearn

###############################################################################
# GENERAL UTILITIES
###############################################################################

OD_TO_CONC = 0.37
def concentration_to_OD(x):
    return x / OD_TO_CONC
    
def OD_to_concentration(x):
    return OD_TO_CONC * x  # resulting in gDW/L

def concentration_to_logOD(x):
    return np.log(np.maximum(np.array(x) / OD_TO_CONC, 1e-8))
    
def logOD_to_concentration(x):
    return OD_TO_CONC * np.exp(x)  # resulting in gDW/L


def ReLU(x):
    return x * (x > 0)
    
def r2_growth_curve(
    Pred, Ref,
    OD=True,
    met_idx=-1,  # column index to evaluate; -1 = last (BIOMASS)
):
    """
    Compute R² between predicted and reference growth curves for all experiments.
    Pred, Ref: (N_iter, N_exp, n_times, n_met)
    met_idx: which metabolite column to use (-1 = BIOMASS, or explicit index)
    Returns: np.array of R² values (length N_exp)
    """
    from sklearn.metrics import r2_score
   
    N_iter, N_exp, n_times, n_met = Pred.shape
    assert Ref.shape == Pred.shape

    Pred_bio = Pred[..., met_idx]
    Ref_bio = Ref[..., met_idx]

    if OD:
        Pred_bio, Ref_bio = concentration_to_logOD(Pred_bio), concentration_to_logOD(Ref_bio)

    pred_bio_mean = np.mean(Pred_bio, axis=0)
    ref_bio_mean  = np.mean(Ref_bio, axis=0)

    R2 = []
    for i in range(N_exp):
        valid = ~np.isnan(ref_bio_mean[i])
        if np.sum(valid) < 2:
            R2.append(0.0)
            continue
        r2 = max(0, r2_score(ref_bio_mean[i][valid], pred_bio_mean[i][valid]))
        R2.append(r2)
    return np.array(R2)

def r2_growth_curve_single(
    Pred, Ref,
    OD=True,
):
    from sklearn.metrics import r2_score
    if OD:
        Pred, Ref = concentration_to_logOD(Pred), concentration_to_logOD(Ref)
    return max(0, r2_score(Ref, Pred))

def r2_growth_curve_with_std(
    Pred, Ref,
    OD=True,
    met_idx=-1,  # column index to evaluate; -1 = last (BIOMASS)
):
    """
    Compute R² between predicted and reference growth curves for all experiments.
    Pred, Ref: (N_iter, N_exp, n_times, n_met)
    met_idx: which metabolite column to use (-1 = BIOMASS, or explicit index)
    Returns: np.array of R² values (length N_exp)
    """
    from sklearn.metrics import r2_score
   
    N_iter, N_exp, n_times, n_met = Pred.shape
    assert Ref.shape == Pred.shape

    Pred_bio = Pred[..., met_idx]
    Ref_bio = Ref[..., met_idx]

    if OD:
        Pred_bio, Ref_bio = concentration_to_logOD(Pred_bio), concentration_to_logOD(Ref_bio)

    pred_bio_mean = np.mean(Pred_bio, axis=0)
    ref_bio_mean  = np.mean(Ref_bio, axis=0)

    R2, R2dev = [], []
    for i in range(N_exp):
        # Mask NaN reference values (e.g. TARGET is only sparsely measured)
        valid = ~np.isnan(ref_bio_mean[i])
        if np.sum(valid) < 2:
            R2.append(0.0)
            R2dev.append(0.0)
            continue
        r2 = max(0, r2_score(ref_bio_mean[i][valid], pred_bio_mean[i][valid]))
        R2.append(r2)
        r2dev = []
        for j in range(N_iter): # compute R2 for each model
            r2j = r2_score(ref_bio_mean[i][valid], Pred_bio[j, i][valid])
            r2dev.append(r2j)
        R2dev.append(np.std(r2dev))
        
    return np.array(R2), np.array(R2dev)

def r2_concentrations_all(
    Pred, Ref
):
    """
    Compute R² for ALL metabolites (not just BIOMASS) for all experiments.
    Uses only non-NaN reference values.

    Parameters
    ----------
    Pred, Ref : np.ndarray
        Shape (N_iter, N_exp, n_times, n_met)

    Returns
    -------
    R2_all : np.ndarray
        Shape (N_exp, n_met)
        R² values for each experiment × each metabolite.
        (Values clipped to >= 0)
    """
    from sklearn.metrics import r2_score

    # Shape check
    N_iter, N_exp, n_times, n_met = Pred.shape
    assert Ref.shape == Pred.shape

    # Mean over iterations (same approach as r2_growth_curve)
    Pred_mean = np.mean(Pred, axis=0)   # (N_exp, n_times, n_met)
    Ref_mean  = np.mean(Ref,  axis=0)   # (N_exp, n_times, n_met)

    R2_all = np.zeros((N_exp, n_met), dtype=float)

    # Loop over experiments and metabolites
    for i in range(N_exp):
        for j in range(n_met):
            ref_series  = Ref_mean[i, :, j]
            pred_series = Pred_mean[i, :, j]

            # Mask missing reference values
            valid = ~np.isnan(ref_series)

            if np.sum(valid) < 2:
                # Not enough data → undefined → return 0 by convention
                R2_all[i, j] = 0.0
                continue

            try:
                r2 = r2_score(ref_series[valid], pred_series[valid])
                R2_all[i, j] = max(r2, 0)  # clip negative values
            except Exception:
                R2_all[i, j] = 0.0

    return R2_all



    
