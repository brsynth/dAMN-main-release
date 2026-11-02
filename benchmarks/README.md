# Benchmarks: dAMN against dFBA and a pure MLP

Scripts comparing dAMN with a mechanistic baseline (dynamic FBA) and a
data-driven baseline (multilayer perceptron, MLP) on the two *E. coli*
datasets of the dAMN paper. Run every script from this folder; data are read
from `../data` and outputs written to `results/`.

## 1. Acetate overflow (Millard data, SI Fig. S9b)

One batch culture of *E. coli* MG1655 in M9 (12.7 mM glucose, 0.96 mM acetate
at t = 0, 13 time points over 4.6 h, `../data/Millard_conc_12.csv`). All
models are trained or fitted on glucose and biomass only; acetate is
predicted (only its initial concentration is given).

| Script | What it does |
|---|---|
| `millard_acetate.py` | Textbook dFBA, MLP control and dAMN; three-panel figure and scores |
| `acetate_panel_data.py` | Curves and scores of the acetate panel (dFBA with Vmax,glc = 8, O2max = 15) |
| `acetate_panel_pptx.py` | The acetate panel as an editable PowerPoint slide, methods in the notes |
| `damn_millard_rerun.py` | Re-runs dAMN on this experiment and compares with the published curve |

**dFBA**: standard dynamic FBA, static optimisation approach (Varma & Palsson
1994; Mahadevan et al. 2002), with iML1515 in COBRApy: Michaelis-Menten uptake
bounds (Km,glc = 0.01 mM, Km,ac = 0.2 mM, Vmax,ac = 10 mmol/gDW/h), O2 uptake
bound, growth maximised and fluxes made unique by parsimonious FBA at each
0.05 h step, acetate secretion unconstrained. Vmax,glc and O2max are set
either from the literature (10.5 and 15) or by grid search on glucose and
biomass only. Glucose and biomass do not determine the acetate prediction:
parameter sets that fit them equally well (R2 >= 0.98) give acetate R2 from
-5.7 to 0.73 (`results/millard/millard_dfba_grid.csv`).

**MLP**: t -> Dense(500, ReLU) -> Dropout(0.2) -> Dense(3), trained on glucose
and biomass; the acetate output gets no training signal (negative control).

**dAMN**: the published Fig. S9b run (release v1.0.0), digitised from its plots
in `published_figures/` (the digitised curves reproduce the R2 printed on the
plots). `damn_millard_rerun.py` re-runs it with this release
(`measure-ac_e`, `UB_out = 0.3`, seed 1, 3 models) or with v1.0.0. Reruns are
not bit-identical (TensorFlow on CPU): acetate R2 was 0.94-0.97 over our runs,
against 0.944 for the published curve, mostly within its ± 1 s.d.

| Acetate R2 | dFBA (Vmax,glc 8, O2 15) | dFBA, literature | MLP | dAMN (published) |
|---|---|---|---|---|
| | 0.73 | -11.1 | < 0 | 0.94 |

## 2. Growth curves on M28 media (SI Table S2)

280 media, growth curves (`../data/M28_conc_20.csv`); train/test splits by
medium with x_fold 1.5, 2 and 3 (93/187, 140/140 and 186/94 media).

| Script | What it does |
|---|---|
| `damn_splits.py` | Rebuilds dAMN's media splits (seeds 30 and 10; they reproduce the saved test IDs of `model/`) |
| `dfba_m28_splits.py` | dFBA (`../dFBA.ipynb`, cell 3) R2 on the test media of each split |
| `mlp_damn_splits.py` | Pure MLP, medium + initial biomass -> log(OD) at the 19 later time points, no dynamics and no mechanism |

| Training / test media | dAMN (paper, Table S2) | dFBA | MLP (clean early stopping) |
|---|---|---|---|
| 93 / 187 | 0.85 ± 0.03 | 0.35 | 0.87 ± 0.01 |
| 140 / 140 | 0.87 ± 0.01 | 0.35 | 0.90 ± 0.00 |
| 186 / 94 | 0.90 ± 0.01 | 0.36 | 0.89 ± 0.01 |

(seed-30 splits; R2 per growth curve on log(OD), as in dAMN.) On random media
splits the MLP matches dAMN; the dAMN numbers are those of the paper, whose
early stopping used the test media (`--stop test` reproduces that protocol
for the MLP). Note that `mlp_damn_splits.py` must run on CPU (the TensorFlow
Metal plugin stops training early).

## 3. Requirements

The dAMN environment (`../environment.yml`); `acetate_panel_pptx.py` also needs
`python-pptx`.
