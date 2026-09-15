# LOO aggregation (descriptor metrics)

Publication-oriented products built from existing `loo_on_profile/` fold outputs.

## Re-run

From the `RefrigerantFinding` repo root:

```bash
python3 refrigerant_finding/profile_robustness/aggregate_loo_descriptor_metrics.py
```

Optional:

```bash
python3 refrigerant_finding/profile_robustness/aggregate_loo_descriptor_metrics.py   --loo-dir refrigerant_finding/profile_robustness/loo_on_profile   --output-dir refrigerant_finding/profile_robustness/loo_on_profile/aggregation   --near-sub-min-frequency 0.9   --high-corr-threshold 0.7   --low-corr-threshold 0.4   --top-near-substitutes 20
```

## Contents

| Path | Description |
| --- | --- |
| `tables/fixed15_stability.*` | Pass rate, JS/rank mean±std / min–max, Δs |
| `tables/per_fold_extremes.*` | Left-out fluid causing max \|ΔJS\| / max *p* / max \|Δrank\| |
| `tables/near_substitutes_frequency.*` | All near-substitutes (any fold frequency) |
| `tables/near_substitutes_frequency_ge90.*` | Near-subs with frequency ≥ 90% |
| `tables/near_vs_fixed15_correlation_*` | Pearson *r* matrix + summary vs curated 15 |
| `tables/near_substitutes_high_corr_groups.*` | High-\|r\| swap pool |
| `tables/near_substitutes_low_corr_complements.*` | Low max-\|r\| complement pool |
| `tables/descriptor_category_map.*` | Feature → chemical category (fixed-15 + near-subs) |
| `figures/fig_js_delta_heatmap.*` | 21×15 signed ΔJS heatmap |
| `figures/fig_pvalue_neglog10_heatmap.*` | 21×15 −log10(*p*) heatmap |
| `figures/fig_rank_stability.*` | Baseline rank vs LOO rank range |
| `figures/fig_near_substitutes_frequency.*` | Top near-substitute frequencies (≥90% filter) |
| `figures/fig_js_delta_distribution.*` | A: LOO JS vs baseline; B/C: relative % zoom + full |
| `figures/fig_js_delta_distribution_v2.*` | Preferred 2-panel: JS vs baseline + full % range |
| `figures/fig_js_abs_delta_distribution.*` | \|ΔJS\| magnitude boxplots |
| `figures/fig_near_vs_fixed15_correlation.*` | Near-sub × fixed-15 *r* heatmap (category-ordered) |

Mirrored under `../aggregate/`: `near_substitutes_frequency.csv`,
`near_substitutes_frequency_ge90.csv`, correlation CSVs.

### Descriptor taxonomy (correlation heatmap)

Axes of `fig_near_vs_fixed15_correlation` are ordered by chemical category,
reusing the paper Methods families (electronic, surface area, pharmacophore,
size/composition, topology, structural complexity) plus Halogen and
Lipophilicity for fixed-15 names not covered by that six-way list. Mapping
lives in `profile_robustness/descriptor_taxonomy.py` and
`tables/descriptor_category_map.*`.

### ΔJS definition

Baseline = full 21 fluids. Signed ΔJS = JS_LOO − JS_baseline.
Negative ⇒ weaker separation when one fluid is left out. Absolute figure is separate.

This step does **not** re-run LOO or mean-profile / candidate ranking.
