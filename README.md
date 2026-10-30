[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17908126.svg)](https://doi.org/10.5281/zenodo.17908126)

# dAMN — Dynamic Artificial Metabolic Network  
Hybrid Neural Network Framework for Dynamic Flux Balance Analysis (dFBA)

This repository contains the reference implementation of dAMN, a hybrid modelling framework that combines:
- a neural network predicting effective intracellular flux distributions from initial media composition, and  
- a dynamic FBA (dFBA) engine simulating extracellular metabolite and biomass trajectories.

Release v1.0.0 is the implementation used for the dAMN publication. This
version adds:
- **gene knockouts**: media files can list knocked-out genes, which dAMN maps
  to reactions through the gene-reaction rules of the metabolic model and
  uses in training (section 4);
- a single, generic time-series format for biomass and any measured
  metabolite (section 3), replacing the separate OD and concentration formats;
- held-out measured variables (`measure-<id>`), e.g. training on glucose and
  biomass and predicting acetate;
- a classical dFBA baseline (`dFBA.py`, `dFBA.ipynb`), new datasets
  (violacein-producing strains with knockouts, `VIO_*`), and benchmarks of
  dAMN against dFBA and a pure MLP (`benchmarks/`).

---

## 1. Repository Structure

The project is organised as follows:

    data/                     Experimental datasets (media tables, time-series files, SBML models)
    figure/                   Training curves, prediction plots, diagnostics (auto-generated)
    model/                    Trained model weights and validation arrays
    benchmarks/               dAMN vs dFBA vs MLP (see benchmarks/README.md)

    dAMN.ipynb                Main notebook: data loading, training, prediction, visualisation
    dAMN_parameter_search.py  Loss-weight and decay parameter search
    data.py                   Data loading, knockouts, preprocessing, structure building
    model.py                  Neural network definition and dFBA integration routines
    plot.py                   Plotting utilities
    utils.py                  Auxiliary functions (OD conversion, R2, etc.)
    dFBA.py                   Classical dFBA baseline (inputs and per-medium simulation)
    dFBA.ipynb                dFBA benchmark on the M28 dataset
    RP2SBML.ipynb             Prepares the violacein model (iML1515_VIO_duplicated.xml): product reaction as exchange
    environment.yml           Conda environment specification
    LICENSE                   License information
    README.md                 Project documentation
    README.rtf                Legacy documentation (superseded by this file)

---

## 2. Media File Format

Initial extracellular conditions for each experiment are specified in a CSV
file (`data/*_media*.csv`):

    ID, <met1>, <met2>, ..., <metN> [, KO_<gene1>, KO_<gene2>, ...]

Rules:
- `ID` uniquely identifies an experiment (one row per experiment).
- Metabolite columns are extracellular metabolite IDs of the COBRA model;
  values are initial concentrations.
- `BIOMASS` must not appear in this table; it is appended internally as the last species.
- Optional knockout columns start at the first header beginning with `KO_`
  (section 4).

---

## 3. Time-Series Format

Measurements are given in a CSV file with a single shared time column:

    time, <var>_<ID>, DEV_..., <var>_<ID>, DEV_..., ...

Rules:
- `<var>` is `BIOMASS` (gDW/L) or an extracellular metabolite ID of the COBRA
  model (e.g. `glc__D_e`, `ac_e`), `<ID>` the experiment ID of the media file.
- Each value column may be followed by a deviation column whose name starts
  with `DEV_` (e.g. `DEV_1` or `DEV_BIOMASS_1`); missing deviations are zero.
- Several variables can be measured per experiment; values may be missing
  (empty cells).
- The media composition is inserted as the first time point.

Examples: `M28_conc_20.csv` (biomass of 280 media),
`Millard_conc_12.csv` (glucose, acetate and biomass of one culture),
`VIO_conc_17.csv` (biomass and violacein, `TARGET`).

The OD and concentration files of release v1.0.0 (`M28_OD_20.csv`,
`Millard_12_1.csv`, ...) are kept in `data/` for reference; they are read by
v1.0.0 only.

---

## 4. Gene Knockouts

A column `KO_<gene>` of the media file set to 1 knocks out `<gene>` (gene name
as in the COBRA model) in that experiment. For each experiment, `data.py`
evaluates the gene-reaction rules of the model with these genes removed and
builds a binary matrix `reaction_KO` (experiments × reactions), 1 for every
reaction that loses all its enzymes (`build_reaction_ko_matrix`). In the
model:
- a KO loss term penalises flux through knocked-out reactions; it is the
  fifth entry of `loss_weight` / `loss_decay`, ordered
  `[SV, negV, C, dropC, KO]`;
- with `ko_embed_dim > 0`, the knockout vector is also embedded and given as
  input to the flux and lag networks.

Media files without `KO_` columns behave as before (no KO term).

---

## 5. Critical Internal Convention

Regardless of dataset:

    BIOMASS MUST BE THE LAST METABOLITE.

This requirement is essential because:
- Multiple loss terms operate specifically on the last index.
- The biomass reaction is identified based on this position.
- Monotonicity checks and state-transition constraints rely on BIOMASS being last.

Violating this ordering will break training and simulation.

---

## 6. Using dAMN

### 6.1 Environment Setup

    conda env create -f environment.yml
    conda activate dAMN_env

### 6.2 Running the Model

Place all dataset files in the `data/` directory.

Open the notebook `dAMN.ipynb` to:

- load and validate datasets  
- train the neural network + dFBA model  
- run dynamic predictions  
- visualise trajectories and diagnostics  
- export figures to the `figure/` directory  

Training/test splits (`train_test_split`):
- `medium`: split by media (`x_fold` sets the test fraction 1/x_fold);
- `forecast`: train on the first time points of every medium, predict the rest;
- `measure-<id>`: hold out one measured variable from the concentration loss
  (e.g. `measure-ac_e`), with optional bounds `UB_in` / `UB_out` on its
  consumption and production per time step.

### 6.3 Parameter Search

To run a search over the loss weights and decays:

    python dAMN_parameter_search.py

Outputs are saved in `model/` and `figure/`.

### 6.4 dFBA Baseline and Benchmarks

`dFBA.ipynb` runs a classical dFBA (`dFBA.py`) on the M28 dataset.
`benchmarks/` compares dAMN with dFBA and a pure MLP on the M28 media splits
and on acetate overflow (Millard data); see `benchmarks/README.md`.

---

## 7. Contributors

- Jean-Loup Faulon (INRAE) — Conceptualization, methodology, core model design, implementation, dataset integration, training pipeline  
- Danilo Dursoniah (INRAE) — Testing and maintenance

---

## 8. License

This project is distributed under the terms of the `LICENSE` file.
