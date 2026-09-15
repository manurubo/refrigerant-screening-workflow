# Leave-one-descriptor-out ranking ablation — summary

Keep all **21** profile fluids. Drop one of the fixed **15** descriptors,
re-rank each strategy pool with the remaining **14**, and compare to the
full 15-D ranking (both recomputed Euclidean to the mean of the 21).

- Descriptor folds: **15**
- Top-k: **100**
- Strategies: I, IIa, IIb, III

## Spearman ρ (15-D vs each 14-D drop)

Two paired correlations are reported:

- **Full universe** — all molecules in the strategy ranking pool.
- **Top-100 baseline** — only molecules that were in the baseline
  top-100 **rows** (their ranks under 15-D vs 14-D).

Set overlap (retention) is separate from Spearman.

| Strategy | Full mean ρ | Full min ρ | Top-100 mean ρ | Top-100 min ρ | Worst full drop | Worst top-100 drop | Top-100 retention | Filtered retention |
|---|---:|---:|---:|---:|---|---|---:|---:|
| I | 0.9761 | 0.8754 | 0.9650 | 0.9075 | `NumRotatableBonds` (0.8754) | `fr_alkyl_halide` (0.9075) | 0.944 | 0.937 |
| III | 0.9960 | 0.9772 | 0.9960 | 0.9772 | `FpDensityMorgan3` (0.9772) | `FpDensityMorgan3` (0.9772) | 1.000 | 1.000 |
| IIa | 0.9762 | 0.9224 | 0.9655 | 0.8281 | `TPSA` (0.9224) | `FpDensityMorgan3` (0.8281) | 0.946 | 0.933 |
| IIb | 0.9770 | 0.9219 | 0.9585 | 0.8803 | `TPSA` (0.9219) | `FpDensityMorgan3` (0.8803) | 0.918 | 0.918 |

## Descriptors whose removal hurts ranking most (mean across strategies)

| Left-out descriptor | Full mean ρ | Full min ρ | Top-100 mean ρ | Top-100 min ρ | Mean top-100 retention |
|---|---:|---:|---:|---:|---:|
| `NumRotatableBonds` | 0.9402 | 0.8754 | 0.9647 | 0.9398 | 0.898 |
| `TPSA` | 0.9477 | 0.9219 | 0.9702 | 0.9420 | 0.933 |
| `FpDensityMorgan3` | 0.9641 | 0.9526 | 0.9070 | 0.8281 | 0.921 |
| `NumHAcceptors` | 0.9674 | 0.9541 | 0.9577 | 0.9124 | 0.934 |
| `MinPartialCharge` | 0.9728 | 0.9493 | 0.9608 | 0.9442 | 0.946 |
| `MaxPartialCharge` | 0.9852 | 0.9738 | 0.9935 | 0.9902 | 0.987 |
| `MolMR` | 0.9878 | 0.9821 | 0.9622 | 0.9299 | 0.936 |
| `fr_halogen` | 0.9881 | 0.9824 | 0.9799 | 0.9609 | 0.956 |
| `MinEStateIndex` | 0.9887 | 0.9792 | 0.9929 | 0.9886 | 0.985 |
| `NumHeteroatoms` | 0.9924 | 0.9862 | 0.9787 | 0.9625 | 0.966 |
| `SMR_VSA1` | 0.9964 | 0.9939 | 0.9941 | 0.9903 | 1.000 |
| `fr_alkyl_halide` | 0.9965 | 0.9937 | 0.9518 | 0.9075 | 0.907 |
| `VSA_EState1` | 0.9968 | 0.9961 | 0.9779 | 0.9322 | 0.959 |
| `MolLogP` | 0.9978 | 0.9973 | 0.9934 | 0.9874 | 0.994 |
| `LabuteASA` | 0.9978 | 0.9974 | 0.9840 | 0.9773 | 0.960 |

## Compounds A–E

Same style as fluid LOO (`loo_mean_profile` ranks): universe rank /
distance under 15-D baseline, then stability across the 15 descriptor drops.

| Code | Strat | Base rank | Base dist | Base rank in filtered | % LODO top-100 | Mean LODO rank | Mean LODO dist | % LODO filter | Mean filt-rank when passed | Worst drop (max abs Δrank) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | III | 36 | 0.4415 | 17 | 100% | 35.1 | 0.4263 | 100% | 16.8 | `MinPartialCharge` (5) |
| B | I | 96 | 0.3023 | 62 | 60% | 95.3 | 0.2915 | 60% | 56.6 | `fr_alkyl_halide` (27) |
| C | III | 12 | 0.2685 | 5 | 100% | 12.7 | 0.2592 | 100% | 5.0 | `FpDensityMorgan3` (3) |
| D | IIa | 54 | 0.3283 | 26 | 100% | 55.5 | 0.3168 | 100% | 27.3 | `fr_alkyl_halide` (28) |
| E | IIb | 45 | 0.4787 | 35 | 100% | 47.2 | 0.4622 | 100% | 36.7 | `MolMR` (13) |

## Outputs

- `ranks/strategy_*/baseline_top100.csv`, `baseline_filtered.csv`, `fold_spearman_retention.csv`
- `ranks/aggregate/spearman_retention_by_fold.csv`
- `ranks/aggregate/spearman_summary_by_strategy.csv`
- `ranks/aggregate/descriptor_drop_impact.csv`
- `ranks/aggregate/matrix_spearman_rho.csv`, `matrix_spearman_rho_top100.csv`
- `ranks/aggregate/matrix_top100_retention.csv`, `matrix_filtered_retention.csv`
- `ranks/aggregate/AE_summary.csv`, `AE_distance_rank_by_fold.csv`
- `ranks/aggregate/headline_claims.json`
- `ranks/aggregate/figures/fig_spearman_heatmap*.pdf`, `fig_top100_retention_heatmap.pdf`, `fig_filtered_retention_heatmap.pdf`

## Note on baseline distances

Baseline ranks use **recomputed** 15-D Euclidean distances (not the
historical CSV `RefrigerantDistance`) so Spearman vs 14-D folds is
apples-to-apples. Filtered set sizes may therefore differ slightly
from the paper/Excel Thermo handoff counts.

