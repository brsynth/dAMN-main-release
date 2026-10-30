from typing import List

import cobra
import numpy as np

from data import prepare_experiment_array, process_data


###############################################################################
# dFBA functions
###############################################################################

def build_dfba_inputs(
    media_file,
    od_file,
    cobra_model_file,
    biomass_rxn_id,
    verbose=False,
):
    """
    Build full time-series and concentration arrays for all experiments,
    without any train/validation split and without creating a MetabolicModel.

    Returns
    -------
    times : np.ndarray, shape (T,)
    metabolite_ids : list of str, length k
    experiment_array : np.ndarray, shape (Z, T*k)
    dev_array : np.ndarray or None, shape (Z, T)
        BIOMASS deviation array if available, aligned with exp_ids.
    exp_ids : list
    Stoichiometry : np.ndarray
    reaction_KO : np.ndarray, shape (Z, n_rxn)
    Transport : np.ndarray
    rxn_ids : list of str
    """
    (
        times,
        metabolite_ids,
        experiment_data,
        dev_data,
        Stoichiometry,
        reaction_KO,
        Transport,
        rxn_ids,
    ) = process_data(
        media_file,
        od_file,
        cobra_model_file,
        biomass_rxn_id,
        verbose=verbose,
    )

    exp_ids = sorted(experiment_data.keys())
    total_time_steps = len(times)

    experiment_array = prepare_experiment_array(
        total_time_steps,
        metabolite_ids,
        {eid: experiment_data[eid] for eid in exp_ids},
    )

    # Align reaction_KO with experiments actually present in experiment_data
    # (process_data returns None when the media file has no KO columns)
    if reaction_KO is not None:
        exp_id_to_media_row = {int(eid): i for i, eid in enumerate(exp_ids)}
        reaction_KO_exp = np.stack(
            [reaction_KO[exp_id_to_media_row[eid]] for eid in exp_ids],
            axis=0
        )
    else:
        reaction_KO_exp = None

    # Keep only BIOMASS deviation for dFBA benchmarking
    if len(dev_data) > 0 and "BIOMASS" in metabolite_ids:
        bio_idx = metabolite_ids.index("BIOMASS")
        dev_array = np.asarray(
            [dev_data[eid][:, bio_idx] for eid in exp_ids],
            dtype=np.float32,
        )
    else:
        dev_array = None

    if verbose:
        print(f"[build_dfba_inputs] #experiments Z = {len(exp_ids)}")
        print(f"[build_dfba_inputs] #time points T = {len(times)}")
        print(f"[build_dfba_inputs] #metabolites k = {len(metabolite_ids)}")
        print(f"[build_dfba_inputs] experiment_array shape = {experiment_array.shape}")
        if reaction_KO_exp is not None:
            print(f"[build_dfba_inputs] reaction_KO shape = {reaction_KO_exp.shape}")
        if dev_array is not None:
            print(f"[build_dfba_inputs] dev_array shape = {dev_array.shape}")

    return (
        times,
        metabolite_ids,
        experiment_array,
        dev_array,
        exp_ids,
        Stoichiometry,
        reaction_KO_exp,
        Transport,
        rxn_ids,
    )


def predict_dFBA(
    cobra_model_file,
    biomass_rxn_id,
    media_conc,
    ex_ids,
    time_points,
    volume=1.0,
    uptake_scaler=None,
    solver="glpk",
    mu_max=2.0,
    X0=0.01,
    X_max=10.0,
    X_init=None,
    reaction_ko_vec=None,
    rxn_ids=None,
    verbose=False,
):
    """
    Classical dynamic FBA predictor (single medium condition).

    Parameters
    ----------
    cobra_model_file : str
        Path to SBML/XML metabolic model.
    biomass_rxn_id : str
        ID of the biomass reaction to maximize.
    media_conc : np.array shape (k_ex,) or (1, k_ex)
        Initial extracellular metabolite concentrations for the subset of
        metabolites corresponding to ex_ids.
        IMPORTANT: should NOT include BIOMASS.
    ex_ids : list of str
        Exchange reaction IDs corresponding to media_conc.
    time_points : np.array shape (T,)
        Simulation time grid (hours).
    volume : float
        Culture volume (L).
    uptake_scaler : array-like or None
        Per-metabolite scaling factor for uptake.
    solver : str
        COBRA solver.
    mu_max : float
        Maximum allowed growth rate (1/hr).
    X0 : float
        Default initial biomass (gDW/L) if X_init is None.
    X_max : float
        Maximum allowed biomass (gDW/L).
    X_init : float or None
        Experimental initial biomass (gDW/L).
    reaction_ko_vec : np.ndarray or None
        Optional binary reaction KO vector for one experiment.
    rxn_ids : list of str or None
        Reaction IDs corresponding to reaction_ko_vec.

    Returns
    -------
    pred_biomass : np.array shape (T,)
        Biomass concentration predicted by dFBA at each time point.
    """
    media_conc = np.asarray(media_conc, dtype=float)
    if media_conc.ndim == 2 and media_conc.shape[0] == 1:
        media_conc = media_conc[0]

    if media_conc.ndim != 1:
        raise ValueError(
            f"media_conc must be 1D (k_ex,) or (1,k_ex), got shape {media_conc.shape}"
        )

    if len(ex_ids) != media_conc.shape[0]:
        raise ValueError(
            f"len(ex_ids) = {len(ex_ids)} but media_conc has length {media_conc.shape[0]}.\n"
            "They must match and correspond to the same metabolites."
        )

    if uptake_scaler is None:
        uptake_scaler = np.ones_like(media_conc, dtype=float)
    else:
        uptake_scaler = np.asarray(uptake_scaler, dtype=float)
        if uptake_scaler.shape != media_conc.shape:
            raise ValueError(
                f"uptake_scaler shape {uptake_scaler.shape} does not match media_conc {media_conc.shape}"
            )

    model = cobra.io.read_sbml_model(cobra_model_file)
    model.solver = solver

    if biomass_rxn_id not in model.reactions:
        raise ValueError(f"Biomass reaction {biomass_rxn_id} not found in the model.")

    for ex in ex_ids:
        if ex not in model.reactions:
            raise ValueError(f"Exchange reaction {ex} not found in the model.")

    if reaction_ko_vec is not None:
        if rxn_ids is None:
            raise ValueError("If reaction_ko_vec is provided, rxn_ids must also be provided.")

        reaction_ko_vec = np.asarray(reaction_ko_vec).astype(int)
        if len(reaction_ko_vec) != len(rxn_ids):
            raise ValueError(
                f"reaction_ko_vec has length {len(reaction_ko_vec)} but rxn_ids has length {len(rxn_ids)}"
            )

        for flag, rid in zip(reaction_ko_vec, rxn_ids):
            if flag == 1:
                rxn = model.reactions.get_by_id(rid)
                rxn.lower_bound = 0.0
                rxn.upper_bound = 0.0

    time_points = np.asarray(time_points, dtype=float)
    T = len(time_points)
    if T < 2:
        raise ValueError("time_points must contain at least 2 values.")

    dt = np.maximum(np.diff(time_points), 1e-6)

    X = np.zeros(T, dtype=float)
    X[0] = float(X_init) if X_init is not None else float(X0)

    C = media_conc.copy()

    for t in range(T - 1):
        dt_t = float(dt[t])

        for i, ex in enumerate(ex_ids):
            rxn = model.reactions.get_by_id(ex)
            C_i = max(C[i], 0.0)
            v_max_i = uptake_scaler[i] * C_i / dt_t / volume
            rxn.upper_bound = v_max_i

        solution = model.optimize()

        if (solution.status != "optimal") or (solution.objective_value is None):
            mu = 0.0
            uptake_fluxes = np.zeros(len(ex_ids), dtype=float)
        else:
            mu_raw = float(solution.objective_value)
            mu = 0.0 if not np.isfinite(mu_raw) else max(0.0, min(mu_raw, mu_max))
            uptake_fluxes = np.array(
                [solution.fluxes.get(ex_id, 0.0) for ex_id in ex_ids],
                dtype=float,
            )

        v_up = np.maximum(uptake_fluxes, 0.0)
        dCdt = -v_up * X[t] / volume
        C_prev = C
        C = np.maximum(C + dCdt * dt_t, 0.0)

        X_next = X[t] * np.exp(mu * dt_t)
        if not np.isfinite(X_next):
            X_next = X[t]
        X[t + 1] = min(X_next, X_max)

        if verbose:
            print(
                f"t={t} mu={mu} C={C_prev} dt={dt_t} "
                f"X[t]={X[t]} -> X[t+1]={X[t + 1]} C[t+1]={C}"
            )

    return X
    