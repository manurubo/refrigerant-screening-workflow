# Distance-metric ranking sensitivity — summary

Same **15** descriptors and mean of all **21** fluids. Re-rank each
strategy pool under alternative distances in min–max–normalized space,
compared to **Euclidean** (paper baseline).

- Alternatives: `manhattan`, `cosine`, `chebyshev`
- Strategies: I, IIa, IIb, III
- Top-k: **100**

## Cosine note

Cosine distance = `1 − cosine_similarity` of the normalized descriptor
vectors (identical direction → 0).

## Mean over strategies (vs Euclidean)

| Metric | Full mean ρ | Full min ρ | Top-100 mean ρ | Top-100 retention | Filtered retention |
|---|---:|---:|---:|---:|---:|
| `manhattan` | 0.9636 | 0.9511 | 0.9238 | 0.910 | 0.898 |
| `cosine` | 0.9574 | 0.9475 | 0.9437 | 0.916 | 0.909 |
| `chebyshev` | 0.9134 | 0.8750 | 0.8617 | 0.865 | 0.858 |

## Per strategy

| Strategy | Metric | Full ρ | Top-100 ρ | Top-100 retention | Filtered retention |
|---|---|---:|---:|---:|---:|
| I | `chebyshev` | 0.9417 | 0.8606 | 0.823 | 0.818 |
| I | `cosine` | 0.9669 | 0.9559 | 0.886 | 0.879 |
| I | `manhattan` | 0.9511 | 0.9206 | 0.911 | 0.894 |
| III | `chebyshev` | 0.9539 | 0.9539 | 1.000 | 1.000 |
| III | `cosine` | 0.9651 | 0.9651 | 1.000 | 1.000 |
| III | `manhattan` | 0.9910 | 0.9910 | 1.000 | 1.000 |
| IIa | `chebyshev` | 0.8830 | 0.8773 | 0.899 | 0.875 |
| IIa | `cosine` | 0.9502 | 0.9321 | 0.899 | 0.875 |
| IIa | `manhattan` | 0.9556 | 0.8925 | 0.870 | 0.839 |
| IIb | `chebyshev` | 0.8750 | 0.7548 | 0.738 | 0.738 |
| IIb | `cosine` | 0.9475 | 0.9216 | 0.881 | 0.881 |
| IIb | `manhattan` | 0.9567 | 0.8911 | 0.857 | 0.857 |

## Compounds A–E (ranks under each metric)

| Code | Strat | Euclidean rank | Euclidean dist | manhattan rank | cosine rank | chebyshev rank | Still in top-100 (all alts)? |
|---|---|---:|---:|---:|---:|---:|---|
| A | III | 36 | 0.4415 | 38 (Δ+2) | 34 (Δ-2) | 20 (Δ-16) | yes |
| B | I | 96 | 0.3023 | 80 (Δ-16) | 80 (Δ-16) | 115 (Δ+19) | no |
| C | III | 12 | 0.2685 | 10 (Δ-2) | 20 (Δ+8) | 11 (Δ-1) | yes |
| D | IIa | 54 | 0.3283 | 45 (Δ-9) | 32 (Δ-22) | 74 (Δ+20) | yes |
| E | IIb | 45 | 0.4787 | 57 (Δ+12) | 60 (Δ+15) | 25 (Δ-20) | yes |

## Outputs

- `ranks/strategy_*/baseline_euclidean_top100.csv`, `metric_vs_euclidean.csv`
- `ranks/aggregate/metric_vs_euclidean_all_strategies.csv`
- `ranks/aggregate/summary_by_metric.csv`, `summary_by_strategy_metric.csv`
- `ranks/aggregate/matrix_*.csv`
- `ranks/aggregate/AE_summary.csv`, `AE_distance_rank_by_metric.csv`
- `ranks/aggregate/figures/fig_*_heatmap.pdf`

## Note

Baseline distances are **recomputed Euclidean** (not historical CSV
`RefrigerantDistance`) so comparisons to Manhattan / cosine / Chebyshev
share the same normalization and mean profile.

