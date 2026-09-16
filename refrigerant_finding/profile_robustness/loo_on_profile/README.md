# LOO descriptor-metrics robustness (profile fluids)

Leave-one-out analysis of Jensen–Shannon distances and Mann–Whitney U tests for
the working-fluid profile descriptors.

## What this is

- **Profile set:** 21 size-matched working fluids (heavy atoms 5–10)
- **Reference set:** 177 size-matched database molecules (fixed across folds)
- **Per fold:** drop one fluid → recompute JS + MW for all ~170 descriptors vs the 177
- **Fixed 15:** paper-curated descriptors; report pass/fail at `p < 0.05`, JS, rank, deltas vs full-21 baseline
- **Near-substitutes:** other descriptors that are significant and high-JS (see `run_config.json`)

This experiment does **not** recompute mean profiles or candidate rankings.

## How to run

From the `RefrigerantFinding` repo root:

```bash
python refrigerant_finding/profile_robustness/loo_descriptor_metrics.py
```

Useful flags:

```bash
python refrigerant_finding/profile_robustness/loo_descriptor_metrics.py \
  --combined-descriptors refrigerant_finding/statistical_comparison/combined_descriptors.csv \
  --refrigerants-csv refrigerant_finding/refrigerants.csv \
  --output-dir refrigerant_finding/profile_robustness/loo_on_profile \
  --p-threshold 0.05 \
  --near-substitute-top-n 30 \
  --near-substitute-js-mode both
```

## Methodology (matched to original pipeline)

Implements the same JS procedure as `refrigerant_finding/refrigerant_comparison.py`:

1. Per descriptor, drop NaNs independently in each group; require ≥2 values/side
2. If overall `nunique() <= 20` → discrete probability mass over shared unique values
3. Else → 30-bin histograms with `density=True`; **bin edges from the lexicographically
   first category** (`database_size_matched` before `refrigerant`)
4. JS distance = `sqrt(JS divergence)` via `scipy.stats.entropy`
5. Mann–Whitney U: two-sided (`scipy.stats.mannwhitneyu`), refrigerant vs reference

## Layout

```
loo_on_profile/
  README.md
  run_config.json
  molecule_manifest.json          # the exact 21 + 177 identifiers
  baseline/
    js_mannwhitney_all_descriptors.csv
    fixed15_summary.csv
    near_substitutes.csv
  folds/
    fold_00_<compound>/
      left_out.json
      js_mannwhitney_all_descriptors.csv
      fixed15_summary.csv
      near_substitutes.csv
  aggregate/
    fixed15_loo_summary.csv
    fixed15_pass_matrix.csv
    fixed15_js_matrix.csv
    fixed15_rank_matrix.csv
    fixed15_pvalue_matrix.csv
    near_substitutes_frequency.csv
```

## Inputs

- `statistical_comparison/combined_descriptors.csv` — must contain categories
  `refrigerant` (21) and `database_size_matched` (177)
- `refrigerants.csv` — used only to attach compound names to the 21 profile fluids
