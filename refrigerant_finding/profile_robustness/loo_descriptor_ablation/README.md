# Leave-one-descriptor-out (LODO) ranking ablation

Complementary to fluid LOO in `loo_mean_profile/`.

## What it does

Keep all **21** size-matched working fluids. For each of the fixed **15**
descriptors:

1. Drop that descriptor (use the other **14**)
2. Recompute Euclidean distance of each strategy pool to the mean of the 21
3. Compare the ranking to the full **15-D** ranking via Spearman ρ, top-100
   retention, Thermo-filtered retention, and A–E rank shifts

## Run

```bash
python refrigerant_finding/profile_robustness/loo_descriptor_ablation_ranks.py
```

Outputs under `loo_descriptor_ablation/ranks/`.

## Scope

| This experiment | Not this experiment |
|-----------------|---------------------|
| Leave out 1 of 15 descriptors | Leave out 1 of 21 fluids (`loo_mean_profile/`) |
| Ranking stability (Spearman) | Descriptor JS / Mann–Whitney re-selection |
| Strategy pools (`candidates*`) | Raw 8M generation dump without descriptors |
