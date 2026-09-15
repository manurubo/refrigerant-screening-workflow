# Distance-metric ranking sensitivity

R2 minor #3: justify Euclidean; compare alternatives.

## Metrics

| Metric | Definition (normalized 15-D space) |
|--------|-------------------------------------|
| `euclidean` | L2 to mean of 21 (paper baseline) |
| `manhattan` | L1 |
| `cosine` | 1 − cosine similarity |
| `chebyshev` | L∞ (max abs component) |

## Run

```bash
python refrigerant_finding/profile_robustness/distance_metric_sensitivity_ranks.py
```

Outputs under `distance_metric_sensitivity/ranks/`.
