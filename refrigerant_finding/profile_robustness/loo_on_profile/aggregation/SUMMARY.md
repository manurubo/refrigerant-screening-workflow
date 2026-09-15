# LOO descriptor-metrics aggregation — summary

Leave-one-out (LOO) of the **21** size-matched profile fluids, recomputing
Jensen–Shannon distances and Mann–Whitney tests vs the fixed **177**
size-matched references. Aggregation consumes existing fold outputs;
it does **not** re-run LOO or mean-profile / candidate-ranking analyses.

## Headline claims

- **All 15 curated descriptors pass** Mann–Whitney `p < 0.05` in **all 21 folds** (worst LOO *p* = `1.209e-03`).
- Mean absolute ΔJS across descriptors ≈ **0.0039**; largest single |ΔJS| = **0.0582** (`NumRotatableBonds`).
- Rank movement is modest on average (mean |Δrank| ≈ **1.17**); largest |Δrank| = **15** (`NumRotatableBonds`).
- Left-out fluid most often associated with per-descriptor extremes: **R134a**.
- **52** near-substitutes with frequency ≥ **90%** (50 appear in all folds).

## Magnitude of JS / rank movement

### What ΔJS means (not missing absolute value)

- **Baseline** = metrics with **all 21** profile fluids.
- **Signed** ΔJS = JS_LOO − JS_baseline.
- Values **below 0** mean that leaving one fluid out **weakened** separation (JS decreased). Values above 0 mean separation strengthened.
- `fig_js_delta_distribution`: **A** LOO JS with baseline mark; **B** relative % zoomed (± at least 1%); **C** full % range. `fig_js_abs_delta_distribution` shows **|ΔJS|** (magnitude only).

### Largest |ΔJS| (top 5 descriptors)

| Feature | max_abs_delta_JS | mean_abs_delta_JS | loo_JS_std |
| --- | --- | --- | --- |
| NumRotatableBonds | 0.05819 | 0.008936 | 0.0155 |
| VSA_EState1 | 0.04077 | 0.007035 | 0.01324 |
| fr_halogen | 0.03918 | 0.0071 | 0.01261 |
| fr_alkyl_halide | 0.03309 | 0.009175 | 0.0134 |
| MaxPartialCharge | 0.03231 | 0.002365 | 0.007077 |


### Largest |Δrank| span (top 5 descriptors)

| Feature | max_abs_delta_rank | mean_abs_delta_rank | loo_rank_min | loo_rank_max |
| --- | --- | --- | --- | --- |
| NumRotatableBonds | 15 | 1.429 | 71 | 87 |
| fr_alkyl_halide | 7 | 2.667 | 10 | 21 |
| fr_halogen | 6 | 1.19 | 9 | 17 |
| TPSA | 6 | 1.333 | 17 | 25 |
| NumHAcceptors | 6 | 2.19 | 10 | 19 |


### Fluids that most often drive extremes

| left_out | n_extreme_slots |
| --- | --- |
| R134a | 12 |
| R601 | 11 |
| R601a | 4 |
| R1345fz | 4 |
| R1123 | 3 |
| R13I1 | 3 |
| R1233zd(E) | 2 |
| R236fa | 2 |

### Near-invariant JS under LOO

These descriptors show essentially **zero JS variance** across folds (discrete / shared values): `TPSA, NumHAcceptors`.

## Near-substitutes (frequency ≥ 90%)

Full list: `aggregate/near_substitutes_frequency.csv`. Filtered (≥90%): `aggregate/near_substitutes_frequency_ge90.csv`.

These are **not** automatic paper replacements; they are candidates for later swap / complement ablation studies after correlation filtering.

| Feature | n_folds_as_near_substitute | frequency | JS_mean | JS_rank_mean |
| --- | --- | --- | --- | --- |
| SlogP_VSA10 | 21 | 1 | 0.7338 | 6.905 |
| FpDensityMorgan2 | 21 | 1 | 0.7292 | 8.286 |
| Chi0v | 21 | 1 | 0.7111 | 9.762 |
| PEOE_VSA14 | 21 | 1 | 0.7054 | 11.95 |
| Chi1 | 21 | 1 | 0.703 | 11.19 |
| Chi1v | 21 | 1 | 0.703 | 12.62 |
| NOCount | 21 | 1 | 0.6968 | 15.1 |
| EState_VSA10 | 21 | 1 | 0.6867 | 18.38 |
| AvgIpc | 21 | 1 | 0.6829 | 19.52 |
| Chi0n | 21 | 1 | 0.6812 | 20.29 |
| Chi1n | 21 | 1 | 0.6776 | 21.86 |
| VSA_EState7 | 21 | 1 | 0.6665 | 23.48 |


## Correlation vs curated fixed-15

Pearson *r* on `combined_descriptors.csv` (21+177 molecules). High: |r| ≥ 0.7; low: |r| < 0.4 (max abs corr vs any fixed-15).

- High-corr (within-group swap pool): **19**
- Low-corr (profile-complement pool): **5**
- Mid: **28**

### High-corr near-subs (closest fixed-15)

| near_substitute | closest_fixed15 | corr_with_closest | max_abs_corr_vs_fixed15 |
| --- | --- | --- | --- |
| NOCount | TPSA | 0.9484 | 0.9484 |
| FpDensityMorgan2 | FpDensityMorgan3 | 0.9219 | 0.9219 |
| Chi0v | MolMR | 0.9172 | 0.9172 |
| SlogP_VSA10 | VSA_EState1 | 0.8746 | 0.8746 |
| Chi1 | LabuteASA | 0.8576 | 0.8576 |
| HeavyAtomCount | LabuteASA | 0.8549 | 0.8549 |
| EState_VSA10 | MinEStateIndex | -0.8371 | 0.8371 |
| Kappa2 | NumRotatableBonds | 0.831 | 0.831 |
| NumValenceElectrons | LabuteASA | 0.8084 | 0.8084 |
| Chi0n | MolMR | 0.8048 | 0.8048 |
| PEOE_VSA14 | MaxPartialCharge | 0.8014 | 0.8014 |
| Chi0 | LabuteASA | 0.7975 | 0.7975 |
| Chi1v | MolMR | 0.7974 | 0.7974 |
| EState_VSA1 | SMR_VSA1 | 0.7853 | 0.7853 |
| Chi1n | MolMR | 0.7835 | 0.7835 |
| NumHDonors | TPSA | 0.7492 | 0.7492 |
| NHOHCount | TPSA | 0.7421 | 0.7421 |
| PEOE_VSA4 | VSA_EState1 | 0.7144 | 0.7144 |
| VSA_EState7 | MinEStateIndex | 0.7132 | 0.7132 |


### Low-corr complements (max |r| vs fixed-15)

| near_substitute | closest_fixed15 | corr_with_closest | max_abs_corr_vs_fixed15 |
| --- | --- | --- | --- |
| BCUT2D_MRHI | LabuteASA | 0.3532 | 0.3532 |
| SMR_VSA6 | NumRotatableBonds | 0.3444 | 0.3444 |
| BCUT2D_MWHI | MolLogP | 0.3319 | 0.3319 |
| EState_VSA9 | LabuteASA | 0.2995 | 0.2995 |
| EState_VSA6 | MolMR | 0.2277 | 0.2277 |


## Interpretation for reviewers

Statistical support (Mann–Whitney p and JS distance) for the curated fixed-15 descriptors is stable under leave-one-out of the 21 profile fluids. This tests metric stability, not automatic re-selection of the 15.

In short: the **statistical support** for the curated descriptor set (significant separation + high JS) is **stable** when any one of the 21 profile fluids is removed. Hydrocarbon outliers (e.g. R601 / R601a) can shift JS more than halogenated fluids for some descriptors, but none of the fixed-15 lose significance.

## Caveats

- Fixed-15 was curated multi-criteria (not auto top-15 by JS alone).
- LOO here does not recompute mean profiles or candidate rankings.
- Near-substitutes are significant high-JS alternatives, not replacements claimed in the paper.
- Some descriptors (e.g. TPSA, NumHAcceptors) can show near-zero JS change when values are discrete/shared.
- Signed ΔJS can be negative; use |ΔJS| figure when only magnitude matters.

## Artifacts

- Tables: `tables/` (filters/correlations also under `../aggregate/`)
- Figures: `figures/` (PNG + PDF)
- Machine-readable claims: `headline_claims.json`

_Generated 2026-07-29T13:59:22.885928+00:00_
