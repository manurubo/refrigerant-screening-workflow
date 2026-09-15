# LOO mean-profile + candidate-rank sensitivity

Leave-one-out over the **21** size-matched profile fluids.

## Status

| Stage | Status |
|-------|--------|
| **1. Mean-profile variation** | Done — `loo_mean_profile_ranks.py --stage profile` |
| **2. Full candidate re-rank** | Done — `loo_mean_profile_ranks.py --stage ranks` → `ranks/` |

## What stage 1 does

For each left-out fluid:

1. Recompute the **mean of the fixed 15** on the remaining 20
2. Compare to the full-21 baseline mean (raw + min–max–normalized with scales frozen on the 21)
3. Report L2 / max|Δ| / per-descriptor sensitivity

No candidate re-ranking, no Thermo pipeline, no descriptor re-selection.

## Run

```bash
python refrigerant_finding/profile_robustness/loo_mean_profile_ranks.py

# equivalent
python refrigerant_finding/profile_robustness/loo_mean_profile_ranks.py \
  --stage profile \
  --output-dir refrigerant_finding/profile_robustness/loo_mean_profile
```

Outputs under `loo_mean_profile/`:

```
run_config.json
molecule_manifest.json          # 21 profile fluids
baseline/
  mean_profile.csv
  minmax_scales.json
folds/fold_XX_<compound>/
  left_out.json
  mean_profile.csv
  profile_delta_summary.json
aggregate/
  SUMMARY.md
  headline_claims.json
  fold_profile_deltas.csv
  matrix_delta_mean_normalized.csv
  descriptor_mean_stability.csv
  ...
```

## Stage 2 (full re-rank)

```bash
python3 refrigerant_finding/profile_robustness/loo_mean_profile_ranks.py --stage ranks
```

Re-ranks full strategy `all_molecules_ranked.csv` pools against baseline + each
LOO mean (scales frozen per pool on refrigerants ∪ pool).

Outputs under `ranks/`:

- `strategy_*/baseline_top100.csv`, `baseline_filtered.csv`, `fold_retention.csv`
- `aggregate/SUMMARY_ranks.md`
- `aggregate/retention_summary_by_strategy.csv`
- `aggregate/AE_summary.csv`, `AE_distance_rank_by_fold.csv`

### Filter path (Thermo handoff)

**I / IIa / III** (reproduces paper 65 / 56 / 23 with CSV distances):

1. Full `all_molecules_ranked.csv`
2. Top-100 by `RefrigerantDistance`
3. Drop refrigerants + known (`refrigerants.csv`) + duplicate canons

**IIb** (`visualize_novel_molecules_grid.py` — *not* “IIa handoff then PubChem”):

1. Same novel ranked CSV as IIa
2. Keep **PubChem-absent** SMILES in that rank order  
   (`CID-SMILES-filtered-lt70_canonical.txt`)
3. Top-100 → dedup canons → `Excel_candidatos_novel_pubchem.xlsx` (**84**)  
   Paper text cites **64** (likely manual cull after Excel)

Filtering the IIa handoff (56) by PubChem only yields ~17 — that is **not** the IIb path.

### Strategy pools

| Strategy | Pool | Paper filtered n |
|----------|------|------------------|
| I | `candidates/` | 65 |
| IIa | `candidates_novel/` | 56 |
| IIb | `candidates_novel_seguridad/` | 64 |
| III | `candidates_refrigerantsmf/` | 23 |

A–E mapping: `focal_candidates.json`.

## Scope boundaries

| This experiment | Not this experiment |
|-----------------|---------------------|
| LOO → mean profile (+ later ranks) | Descriptor JS / MW LOO (`loo_on_profile/`) |
| Fixed curated 15 | Near-substitute re-selection |
| Profile-fluid leave-one-out | Reference-set bootstrap (`molecule_set_robustness/`) |
