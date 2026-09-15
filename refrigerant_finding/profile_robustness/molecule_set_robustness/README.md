# Reference molecule-set robustness

Varies the **size-matched comparison molecules** used for Jensen–Shannon
distances and Mann–Whitney tests, while keeping the **21 profile fluids fixed**.

This is the sibling of `loo_on_profile/` (which leaves out profile fluids and
holds the 177 references fixed).

## Question

If we had drawn a different size-matched reference set of similar size (~177)
from the same filtering pipeline, would fixed-15 significance / JS / ranks
change?

## How the paper's ~177 set is produced

From `refrigerant_finding/refrigerant_comparison.py` (typical flags:
`--size_match --atom_range 5 10 --n_samples 10000 --size_buffer 1.0`):

1. Sample ~10k SMILES from `molecular_formulas.csv`
2. Keep molecules with heavy-atom count in **[5, 10]**
3. That filtered set (~177 for the published run) becomes
   `category=database_size_matched` in
   `statistical_comparison/combined_descriptors.csv`

## Modes

| Mode | What varies | Needs |
|------|-------------|-------|
| `bootstrap` (default) | Resample the published 177 **with replacement** | `combined_descriptors.csv` only |
| `pool_redraw` | Draw 177 **without replacement** from a larger precomputed pool | `--pool-descriptors` |
| `candidate_redraw` | Re-run sample→filter→descriptors with new seeds | RDKit + `molecular_formulas.csv` |

Prefer `bootstrap` for a fast sensitivity check. Prefer `pool_redraw` /
`candidate_redraw` to mimic a different draw from the large candidate DB.

## How to run

From the `RefrigerantFinding` repo root:

```bash
# Fast default (bootstrap the published 177)
python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py \
  --mode bootstrap --n-replicates 50

# Build a reusable size-matched pool, then redraw
python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py \
  --build-pool refrigerant_finding/profile_robustness/molecule_set_robustness/reference_pool.csv \
  --molecular-formulas /path/to/molecular_formulas.csv \
  --pool-target-size 2000 --n-candidates 50000

python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py \
  --mode pool_redraw --n-replicates 50 \
  --pool-descriptors refrigerant_finding/profile_robustness/molecule_set_robustness/reference_pool.csv

# Authentic redraw from formulas (slower; RDKit per replicate; use n_candidates≈10000)
python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py \
  --mode candidate_redraw --n-replicates 20 --n-candidates 10000 \
  --molecular-formulas /path/to/molecular_formulas.csv
```

## Outputs

```
molecule_set_robustness/
  README.md
  run_config.json
  molecule_manifest.json
  baseline/                 # paper 21 vs paper 177
  replicates/rep_XXX/       # each reference draw
    draw_manifest.json
    js_mannwhitney_all_descriptors.csv
    fixed15_summary.csv
    near_substitutes.csv
  aggregate/
    fixed15_resample_summary.csv
    fixed15_pass_matrix.csv
    fixed15_js_matrix.csv
    fixed15_rank_matrix.csv
    fixed15_pvalue_matrix.csv
    near_substitutes_frequency.csv
    pairwise_jaccard_significant_fixed15.csv
    headline_claims.json
    figures/                # publication PDFs/PNGs (see below)
```

## Figures

Publication figures (LOO-style analogs + redraw-specific panels) are generated
from existing aggregate CSVs — they do **not** re-run metrics.

```bash
# Default: authentic candidate_redraw run
python3 refrigerant_finding/profile_robustness/plot_molecule_set_robustness.py

# Explicit paths (e.g. bootstrap root run)
python3 refrigerant_finding/profile_robustness/plot_molecule_set_robustness.py \
  --run-dir refrigerant_finding/profile_robustness/molecule_set_robustness \
  --output-dir refrigerant_finding/profile_robustness/molecule_set_robustness/aggregate/figures
```

Primary figures live under `candidate_redraw/aggregate/figures/`.
`fig_near_vs_fixed15_correlation` uses the same chemical-category axis
ordering as LOO (`profile_robustness/descriptor_taxonomy.py`); a companion
`aggregate/descriptor_category_map.csv` is written next to the figures.

## What is measured

- Per fixed-15 descriptor: pass rate across replicates, JS/rank mean±spread,
  deltas vs the published baseline
- Pairwise Jaccard of which fixed-15 descriptors remain significant
- Near-substitute frequency under changing reference sets

Does **not** recompute mean profiles or candidate rankings (same scope as LOO).
