from typing import List, Tuple, Dict, Set
import numpy as np
import pandas as pd
import cobra
import utils

###############################################################################
# Split media and KO columns
###############################################################################

def validate_measured_ids(
    cobra_model_file: str,
    measured_ids: List[str],
    verbose: bool = False
):
    """
    Ensure measured variables are present in the SBML model, except BIOMASS.

    BIOMASS is allowed even though it is not a COBRA metabolite ID.
    """
    model = cobra.io.read_sbml_model(cobra_model_file)
    model_met_ids = {met.id for met in model.metabolites}

    missing = [
        mid for mid in measured_ids
        if mid != "BIOMASS" and mid not in model_met_ids
    ]

    if verbose:
        print(f"[validate_measured_ids] measured_ids = {measured_ids}")
        print(f"[validate_measured_ids] missing in model = {missing}")

    if missing:
        raise ValueError(
            "The following measured variables are not present as metabolite IDs "
            f"in the SBML model: {missing}\n"
            "BIOMASS is the only allowed exception."
        )

def split_media_and_ko_columns(media_df: pd.DataFrame):
    """
    Split a media+KO dataframe into:
      - media-only dataframe (ID + media columns)
      - list of media metabolite column names
      - list of KO column names

    KO columns start at the first header that begins with 'KO_'.
    """
    cols = list(media_df.columns)

    if "ID" not in cols:
        raise ValueError("media_file must contain an 'ID' column.")

    ko_start_idx = None
    for i, c in enumerate(cols):
        if str(c).startswith("KO_"):
            ko_start_idx = i
            break

    if ko_start_idx is None:
        media_cols = cols
        ko_cols = []
    else:
        media_cols = cols[:ko_start_idx]
        ko_cols = cols[ko_start_idx:]

    media_only_df = media_df[media_cols].copy()
    media_metabolite_ids = media_cols[1:]  # skip ID

    return media_only_df, media_metabolite_ids, ko_cols
    
################################################################################
#  Get Stoichiometry, Transport and KO matrices
###############################################################################

def build_Stoichiometry_and_Transport_from_cobra(
    cobra_model_file: str,
    medium_met_ids: List[str],
    biomass_rxn_id: str,
    verbose=False
) -> Tuple[np.ndarray, np.ndarray, int, int, List[str]]:
    """
    Build stoichiometric and Transport matrices from a COBRA model.

    medium_met_ids:
        List of external/metabolite IDs used as state variables
        (e.g. ['glc__D_e', 'ac_e', 'BIOMASS']).
    """

    model = cobra.io.read_sbml_model(cobra_model_file)

    rxn_ids = [rxn.id for rxn in model.reactions]
    full_met_ids = [met.id for met in model.metabolites]

    m = len(full_met_ids)
    n = len(rxn_ids)

    met_id_to_idx = {met_id: i for i, met_id in enumerate(full_met_ids)}

    Stoichiometry = np.zeros((m, n), dtype=np.float32)
    for j, rxn in enumerate(model.reactions):
        for met, coeff in rxn.metabolites.items():
            i = met_id_to_idx[met.id]
            Stoichiometry[i, j] = coeff

    Transport = np.zeros((len(medium_met_ids), n), dtype=np.float32)
    for col_idx, rxn_obj in enumerate(model.reactions):
        rxn_id = rxn_obj.id

        if rxn_id == biomass_rxn_id:
            if "BIOMASS" in medium_met_ids:
                row_idx = medium_met_ids.index("BIOMASS")
                Transport[row_idx, col_idx] = 1.0
                if verbose:
                    print(f"Transport[{row_idx},{col_idx}] = 1.0  BIOMASS")
            continue

        if rxn_id.startswith("EX_"):
            io = rxn_id[-1:]
            met_id_candidate = rxn_id[3:-2]
            if met_id_candidate in medium_met_ids:
                row_idx = medium_met_ids.index(met_id_candidate)
                Transport[row_idx, col_idx] = -1.0 if io == "i" else 0.0

                if met_id_candidate == "ac_e" or met_id_candidate == "TARGET": 
                    Transport[row_idx, col_idx] = -1.0 if io == "i" else 1.0

                if verbose and Transport[row_idx, col_idx] != 0:
                    print(
                        f"Transport[{row_idx},{col_idx}] = "
                        f"{Transport[row_idx, col_idx]}  {rxn_id} {met_id_candidate}"
                    )

    if verbose:
        print(f"Transport shape: {Transport.shape}, Stoichiometry shape: {Stoichiometry.shape}")
        print(f"Number of metabolites (k): {len(medium_met_ids)}, Number of fluxes (n): {n}")

    return Stoichiometry, Transport, m, n, rxn_ids

def build_reaction_ko_matrix(
    media_df: pd.DataFrame,
    ko_cols: List[str],
    model: cobra.Model,
    rxn_ids: List[str],
    verbose: bool = False
) -> np.ndarray:
    """
    Build a binary reaction_KO matrix of shape (num_rows, num_reactions).

    For each row:
      - find KO genes from columns KO_<gene_name> with value == 1
      - map gene names to COBRA gene ids
      - evaluate each reaction's GPR with those knockouts
      - reaction_KO[row, j] = 1 if reaction j becomes non-functional
    """
    
    if len(ko_cols) < 1:
        return None
        
    n_rows = len(media_df)
    n_rxns = len(rxn_ids)
    reaction_KO = np.zeros((n_rows, n_rxns), dtype=np.float32)

    rxn_objects = [model.reactions.get_by_id(rid) for rid in rxn_ids]
    model_gene_ids = {g.id for g in model.genes}

    # Map gene names -> gene ids
    gene_name_to_id = {}
    for g in model.genes:
        if g.name is not None and str(g.name).strip() != "":
            gene_name_to_id[str(g.name).strip()] = g.id

    for row_idx, (_, row) in enumerate(media_df.iterrows()):
        knocked_out_gene_ids: Set[str] = set()
        knocked_out_gene_names: Set[str] = set()

        for ko_col in ko_cols:
            val = row[ko_col]
            if pd.notna(val) and float(val) == 1.0:
                gene_name = str(ko_col)[3:]   # remove "KO_"
                knocked_out_gene_names.add(gene_name)

                if gene_name in gene_name_to_id:
                    knocked_out_gene_ids.add(gene_name_to_id[gene_name])

        missing_names = knocked_out_gene_names - set(gene_name_to_id.keys())
        if verbose and missing_names:
            print(
                f"[build_reaction_ko_matrix] warning: gene names not found in model: "
                f"{sorted(missing_names)}"
            )

        missing_ids = knocked_out_gene_ids - model_gene_ids
        if verbose and missing_ids:
            print(
                f"[build_reaction_ko_matrix] warning: mapped gene ids not found in model: "
                f"{sorted(missing_ids)}"
            )

        for j, rxn in enumerate(rxn_objects):
            if not rxn.gene_reaction_rule:
                continue

            try:
                is_functional = rxn.gpr.eval(knockouts=knocked_out_gene_ids)
                reaction_KO[row_idx, j] = 0.0 if is_functional else 1.0
            except Exception:
                rxn_gene_ids = {g.id for g in rxn.genes}
                reaction_KO[row_idx, j] = 1.0 if len(rxn_gene_ids & knocked_out_gene_ids) > 0 else 0.0
    if verbose:
        print(f"Reaction KO shape: {reaction_KO.shape}")

    return reaction_KO
    
###############################################################################
# Parse measurement 
###############################################################################

def parse_measurement_layout(meas_df: pd.DataFrame):
    """
    Parse a measurement file with columns like:
        time, X_1, DEV_1, X_2, DEV_2, ...

    Returns
    -------
    layout : dict
        layout[exp_id] = {
            "measurements": [
                {"name": "BIOMASS", "value_col": "BIOMASS_1", "dev_col": "DEV_1"},
                {"name": "VIO",     "value_col": "VIO_1",     "dev_col": "DEV_1.1"},
                ...
            ]
        }
    measurement_names : list of str
        Unique measurement names found in the file.
    """
    cols = list(meas_df.columns)

    if "time" not in cols:
        raise ValueError("Measurement file must contain a single 'time' column.")

    layout = {}
    measurement_names = []

    i = 1  # skip 'time'
    while i < len(cols):
        value_col = cols[i]

        # Skip stray DEV columns if any
        if str(value_col).startswith("DEV_"):
            i += 1
            continue

        if "_" not in str(value_col):
            raise ValueError(
                f"Measurement column '{value_col}' does not follow the expected NAME_ID format."
            )

        value_name, exp_id_str = str(value_col).rsplit("_", 1)
        exp_id = int(exp_id_str)

        dev_col = None
        if i + 1 < len(cols) and str(cols[i + 1]).startswith("DEV_"):
            dev_col = cols[i + 1]
            i += 2
        else:
            i += 1

        if exp_id not in layout:
            layout[exp_id] = {"measurements": []}

        layout[exp_id]["measurements"].append(
            {"name": value_name, "value_col": value_col, "dev_col": dev_col}
        )

        if value_name not in measurement_names:
            measurement_names.append(value_name)

    return layout, measurement_names

###############################################################################
# Process data 
###############################################################################

def process_measurement(
    media_only_df: pd.DataFrame,
    meas_df: pd.DataFrame,
    metabolite_ids: List[str],
    times: np.ndarray,
    verbose: bool = False
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Process a measurement file with a single shared 'time' column and one or more
    measurements per experiment.

    Expected columns:
      time, X_1, DEV_1, X_2, DEV_2, ...

    Rules
    -----
    - all measured values are assumed to already be concentrations
    - BIOMASS is treated as a special state variable, but no conversion is applied
    - media is inserted at time 0
    - missing deviations default to zeros
    - the shared time column may extend beyond available measurements
    """
    experiment_data: Dict[int, np.ndarray] = {}
    dev_data: Dict[int, np.ndarray] = {}

    layout, _ = parse_measurement_layout(meas_df)
    present_ids = set(layout.keys())
    local_times = meas_df["time"].to_numpy(dtype=float)

    for exp_id in media_only_df["ID"]:
        exp_id = int(exp_id)
        if exp_id not in present_ids:
            continue

        conc_matrix = np.full((len(times), len(metabolite_ids)), np.nan, dtype=np.float32)
        dev_matrix = np.full((len(times), len(metabolite_ids)), 0.0, dtype=np.float32)

        # Insert media at time 0
        row_in_media = media_only_df.loc[
            media_only_df["ID"] == exp_id
        ].iloc[0, 1:].to_numpy(dtype=np.float32)

        if np.any(np.isclose(times, 0.0)):
            t0_idx = np.where(np.isclose(times, 0.0))[0][0]
        else:
            t0_idx = 0

        conc_matrix[t0_idx, :len(row_in_media)] = row_in_media

        for meas in layout[exp_id]["measurements"]:
            met_name = meas["name"]
            if met_name not in metabolite_ids:
                continue

            met_idx = metabolite_ids.index(met_name)
            values = meas_df[meas["value_col"]].to_numpy(dtype=np.float32)

            if meas["dev_col"] is not None and meas["dev_col"] in meas_df.columns:
                dev_values = meas_df[meas["dev_col"]].to_numpy(dtype=np.float32)
            else:
                dev_values = np.zeros(len(local_times), dtype=np.float32)

            for i_time, t_val in enumerate(times):
                if t_val in local_times:
                    idx_t = np.where(local_times == t_val)[0][0]
                    if not np.isnan(values[idx_t]):          # never overwrite with NaN
                        conc_matrix[i_time, met_idx] = values[idx_t]
                        dev_matrix[i_time, met_idx] = dev_values[idx_t]

        experiment_data[exp_id] = conc_matrix
        dev_data[exp_id] = dev_matrix

    if verbose:
        print("[process_data] detected generic measurement mode (single shared time column)")

    return experiment_data, dev_data
    
def process_data(
    media_file: str,
    measurement_file: str,
    cobra_model_file: str,
    biomass_rxn_id: str,
    verbose=False
) -> Tuple[
    np.ndarray,
    List[str],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str]
]:
    """
    Process media(+KO), measurements, and COBRA model to generate structured data.

    Parameters
    ----------
    media_file : str
        File containing ID, media columns, and optionally KO_<gene> columns.
    measurement_file : str
        Measurement file containing:
            time, X_1, DEV_1, X_2, DEV_2, ...
    cobra_model_file : str
        SBML file for the strain model.
    biomass_rxn_id : str
        Biomass reaction ID.
    """
    media_df = pd.read_csv(media_file, encoding="latin1")
    media_df.columns = [
        c.encode("ascii", errors="ignore").decode("ascii").strip()
        for c in media_df.columns
    ]

    meas_df = pd.read_csv(measurement_file)
    media_only_df, media_metabolite_ids, ko_cols = split_media_and_ko_columns(media_df)

    if "time" not in meas_df.columns:
        raise ValueError("Measurement file must contain a single 'time' column.")
    _, measurement_names = parse_measurement_layout(meas_df)

    # Build metabolite_ids = media metabolites + measured variables (no duplicates)
    metabolite_ids = list(media_metabolite_ids)
    for meas_name in measurement_names:
        if meas_name not in metabolite_ids:
            metabolite_ids.append(meas_name)

    # Check measured variables against SBML model
    validate_measured_ids(
        cobra_model_file,
        measurement_names,
        verbose=verbose
    )

    times = meas_df["time"].to_numpy(dtype=float)
    experiment_data, dev_data = process_measurement(
        media_only_df=media_only_df,
        meas_df=meas_df,
        metabolite_ids=metabolite_ids,
        times=times,
        verbose=verbose,
    )

    model = cobra.io.read_sbml_model(cobra_model_file)
    Stoichiometry, Transport, _, _, rxn_ids = build_Stoichiometry_and_Transport_from_cobra(
        cobra_model_file, metabolite_ids, biomass_rxn_id, verbose=verbose
    )

    reaction_KO = build_reaction_ko_matrix(
        media_df=media_df,
        ko_cols=ko_cols,
        model=model,
        rxn_ids=rxn_ids,
        verbose=verbose,
    )
    
    return (
        np.array(times, dtype=np.float32),
        metabolite_ids,
        experiment_data,
        dev_data,
        Stoichiometry,
        reaction_KO,
        Transport,
        rxn_ids,
    )
    
###############################################################################
# Prepare array
###############################################################################

def prepare_experiment_array(
    time_ids: np.ndarray,
    metabolite_ids: List[str],
    experiment_data: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Flatten experimental data into a 2D array with shape:
        (num_experiments, T * k)

    time_ids:
        - either an integer T (number of time steps),
        - or a 1D array of times of length T.
    """
    if isinstance(time_ids, (int, np.integer)):
        T = int(time_ids)
    else:
        T = len(time_ids)

    all_flat = []
    for exp_id in experiment_data.keys():
        conc_matrix = experiment_data[exp_id]
        conc_slice = conc_matrix[:T, :]
        row_flat = conc_slice.reshape(-1)
        all_flat.append(row_flat)

    return np.stack(all_flat, axis=0)
