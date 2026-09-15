# LOO mean-profile variation — summary

Leave-one-out over the 21 size-matched working fluids; fixed curated 15
descriptors. Each fold recomputes the **mean profile** on the remaining
20 fluids. Candidate re-ranking is **not** included in this stage.

- Folds: **21**
- Scale mode: `freeze_baseline_21` (min–max from all 21; applied to raw means for comparable L2)

## Headline (normalized space)

- Mean L2(Δ mean): **0.03958** (median 0.03215; max 0.09963 when leaving out **R601**)
- Mean of per-fold max|Δ|: **0.01986** (worst fold max|Δ| = 0.04524)
- Mean of per-fold mean|Δ|: **0.008148**

### Most sensitive descriptors (by mean |Δ| across folds)

- `FpDensityMorgan3`: mean|Δ|=0.01243, max|Δ|=0.0325
- `MolLogP`: mean|Δ|=0.01151, max|Δ|=0.02727
- `LabuteASA`: mean|Δ|=0.01139, max|Δ|=0.02619
- `MinPartialCharge`: mean|Δ|=0.009601, max|Δ|=0.034
- `MolMR`: mean|Δ|=0.009535, max|Δ|=0.02971

### Least sensitive descriptors

- `NumHAcceptors`: mean|Δ|=0
- `TPSA`: mean|Δ|=0
- `MaxPartialCharge`: mean|Δ|=0.007203
- `MinEStateIndex`: mean|Δ|=0.008123
- `NumRotatableBonds`: mean|Δ|=0.008163

## Per-fold L2 (normalized)

| Left out | L2(Δ) | max\|Δ\| | argmax |
|---|---:|---:|---|
| R601 | 0.09963 | 0.04524 | NumRotatableBonds |
| R601a | 0.09006 | 0.03442 | MaxPartialCharge |
| R227ea | 0.06747 | 0.02724 | FpDensityMorgan3 |
| R134a | 0.06216 | 0.0325 | FpDensityMorgan3 |
| R1345fz | 0.05595 | 0.02381 | LabuteASA |
| R1336yf | 0.05595 | 0.02381 | LabuteASA |
| R1123 | 0.0484 | 0.02381 | fr_alkyl_halide |
| R125 | 0.03566 | 0.01613 | MolMR |
| R1224yd(E) | 0.03507 | 0.01984 | MolLogP |
| R1224yd(Z) | 0.03507 | 0.01984 | MolLogP |
| R245fa | 0.03215 | 0.02024 | NumRotatableBonds |
| R1225ye(Z) | 0.02736 | 0.01141 | MinPartialCharge |
| R1336mzz(Z) | 0.02519 | 0.0164 | MolLogP |
| R1243zf | 0.02519 | 0.0164 | MolLogP |
| R1233zd(E) | 0.02316 | 0.01562 | FpDensityMorgan3 |
| R1336mzz(E) | 0.02239 | 0.01562 | FpDensityMorgan3 |
| R13I1 | 0.02076 | 0.01145 | FpDensityMorgan3 |
| R236fa | 0.01923 | 0.01017 | LabuteASA |
| R1234ze(E) | 0.01789 | 0.0124 | MinPartialCharge |
| R1234ze(Z) | 0.01789 | 0.0124 | MinPartialCharge |
| R1234yf | 0.01462 | 0.008368 | MinPartialCharge |

## Outputs

- `fold_profile_deltas.csv` — one row per fold
- `matrix_delta_mean_normalized.csv` — folds × descriptors
- `descriptor_mean_stability.csv` — per-descriptor sensitivity
- `headline_claims.json`

## Next stage (not run yet)

Full candidate re-rank against each LOO mean on strategy CSVs, then:

1. Of the baseline **top-100** (by distance), how many remain in top-100?
2. Of the **filtered Thermo handoff sets** (top-100 → drop dups + known
   refrigerants → ~65 / 56 / 64 / 23 for I / IIa / IIb / III), how many
   still survive that same filter path under the LOO mean?
3. A–E: Δdistance, Δrank, and whether they enter/leave top-100.

## Figures and SI tables

- Figures: `aggregate/figures/` (PNG + PDF; start with `fig_mean_profile_loo_summary`)
- Tables: `aggregate/tables/` (`SI_tables_draft.md`, per-fold and per-descriptor CSVs)
