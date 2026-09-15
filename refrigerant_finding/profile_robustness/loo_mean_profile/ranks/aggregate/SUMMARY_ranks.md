# LOO mean-profile re-rank — summary

Baseline = historical CSV distances; LOO folds = recomputed Euclidean vs leave-one-out mean (scales frozen on refrigerants ∪ universe).

- Folds: **21**
- Top-k: **100**

## Baseline filtered sizes (Thermo handoff path)

| Strategy | Universe | Baseline filtered n | Code/Excel n | Paper n | Match code? |
|---|---|---:|---:|---:|---|
| I | 28236 | 66 | 65 | 65 | False |
| IIa | 24230 | 56 | 56 | 56 | True |
| IIb | 10632 | 84 | 84 | 64 | True |
| III | 69 | 23 | 23 | 23 | True |

_Baseline uses CSV RefrigerantDistance (historical). I/IIa/III: top-100 → drop known refrigerants + dups → 65/56/23. IIb (visualize_novel_molecules_grid.py): same novel CSV as IIa → keep PubChem-absent in rank order → top-100 → dedup → Excel 84 (paper cites 64). NOT IIa-handoff-then-PubChem (~17). LOO folds recompute distances; retention vs this baseline set._

## Retention + Spearman under LOO (mean / min over 21 folds)

| Strategy | Full mean ρ | Top-100 mean ρ | Top-100 rows (unique) | Top-100 retention | Filtered retention |
|---|---:|---:|---|---:|---:|
| I | 0.9993 | 0.9724 | 100 (79 unique) | 0.963 (min 0.924) | 0.959 (min 0.909) |
| III | 0.9961 | 0.9961 | 69 (57 unique) | 1.000 (min 1.000) | 1.000 (min 1.000) |
| IIa | 0.9984 | 0.9688 | 100 (69 unique) | 0.952 (min 0.913) | 0.941 (min 0.893) |
| IIb | 0.9875 | 0.9670 | 100 (84 unique) | 0.907 (min 0.833) | 0.907 (min 0.833) |

## Compounds A–E

| Code | Strat | Base rank | Base dist | Base rank in filtered | % LOO top-100 | Mean LOO rank | Mean LOO dist | % LOO filter | Mean filt-rank when passed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | III | 36 | 0.4415 | 17 | 100% | 35.0 | 0.4521 | 100% | 17.5 |
| B | I | 96 | 0.3023 | 62 | 71% | 94.0 | 0.3087 | 71% | 56.4 |
| C | III | 12 | 0.2685 | 5 | 100% | 10.5 | 0.2475 | 100% | 4.5 |
| D | IIa | 54 | 0.3283 | 26 | 100% | 51.3 | 0.3311 | 100% | 23.2 |
| E | IIb | 45 | 0.4787 | 35 | 100% | 53.8 | 0.5487 | 100% | 42.3 |

## Outputs

- `ranks/strategy_*/baseline_top100.csv`, `baseline_filtered.csv`, `fold_retention.csv`
- `ranks/aggregate/fold_retention_all_strategies.csv`
- `ranks/aggregate/retention_summary_by_strategy.csv`
- `ranks/aggregate/matrix_spearman_rho.csv`, `matrix_spearman_rho_top100.csv`
- `ranks/aggregate/matrix_top100_retention.csv`, `matrix_filtered_retention.csv`
- `ranks/aggregate/figures/fig_spearman_heatmap*.pdf`, `fig_top100_retention_heatmap.pdf`, `fig_filtered_retention_heatmap.pdf`
- `ranks/aggregate/AE_distance_rank_by_fold.csv`, `AE_summary.csv`
- `ranks/aggregate/headline_claims_ranks.json`

