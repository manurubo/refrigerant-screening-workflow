#!/usr/bin/env python3
"""
Distance-metric sensitivity of candidate rankings (R2 minor #3).

Keep the fixed 15 descriptors and the paper's hardcoded mean profile
(``PAPER_HARDCODED_MEANS``, with min-max scales from the full unfiltered
per-strategy CSV -- reproducing the published screening pipeline exactly).
Re-rank each strategy pool under alternative distances in the same
min–max–normalized space, and compare each alternative to the Euclidean
baseline via Spearman, top-100 / filtered retention, and A–E rank shifts.

Metrics:
  - euclidean   (baseline; paper)
  - manhattan   (L1)
  - cosine      (1 − cosine similarity; identical direction → distance 0)
  - chebyshev   (L∞ / max abs difference)

Typical usage::

    python refrigerant_finding/profile_robustness/distance_metric_sensitivity_ranks.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from loo_descriptor_metrics import (  # noqa: E402
    FIXED_15,
    default_combined_descriptors_path,
    default_refrigerants_csv_path,
    load_combined_descriptors,
)
from loo_mean_profile_ranks import (  # noqa: E402
    CODE_FILTERED_N,
    IIB_ABSENT_CACHE,
    SHIPPED_IIB_CACHE,
    PAPER_FILTERED_N,
    STRATEGY_POOLS,
    TOP_K,
    _filtered_rank_lookup,
    _rank_lookup,
    apply_thermo_filter,
    canonical_smiles,
    copy_or_write_manifest,
    load_focal_candidates,
    load_iib_pubchem_absent_smiles,
    load_known_refrigerant_smiles,
    normalize_matrix,
    normalize_mean_vector,
    paper_hardcoded_mean,
    paper_pool_minmax_scales,
    prepare_strategy_pool,
    write_json,
)

logger = logging.getLogger(__name__)

BASELINE_METRIC = "euclidean"
ALT_METRICS = ("manhattan", "cosine", "chebyshev")
ALL_METRICS = (BASELINE_METRIC,) + ALT_METRICS


@dataclass(frozen=True)
class RunConfig:
    combined_descriptors: str
    refrigerants_csv: Optional[str]
    output_dir: str
    fixed_15: list[str]
    n_refrigerants: int
    strategies: list[str]
    metrics: list[str]
    top_k: int
    scale_mode: str
    timestamp_utc: str


def default_output_dir() -> Path:
    return _PKG / "distance_metric_sensitivity"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _rf_root() -> Path:
    return _PKG.parent


def ensure_iib_cache(output_dir: Path) -> Path:
    dest = output_dir / IIB_ABSENT_CACHE
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for src in (
        _PKG / "loo_mean_profile" / IIB_ABSENT_CACHE,
        _rf_root() / SHIPPED_IIB_CACHE,
    ):
        if src.is_file():
            shutil.copy2(src, dest)
            logger.info("Copied IIb PubChem-absent cache from %s", src)
            return dest
    return dest


def distances_to_mean(
    norm_X: np.ndarray, mean_norm: np.ndarray, metric: str
) -> np.ndarray:
    """Distances in normalized descriptor space."""
    diff = norm_X - mean_norm.reshape(1, -1)
    if metric == "euclidean":
        return np.linalg.norm(diff, axis=1)
    if metric == "manhattan":
        return np.sum(np.abs(diff), axis=1)
    if metric == "chebyshev":
        return np.max(np.abs(diff), axis=1)
    if metric == "cosine":
        # 1 − cosine similarity; identical vectors → 0
        x_norm = np.linalg.norm(norm_X, axis=1)
        m_norm = float(np.linalg.norm(mean_norm))
        denom = np.clip(x_norm * m_norm, 1e-12, None)
        cos_sim = (norm_X @ mean_norm) / denom
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return 1.0 - cos_sim
    raise ValueError(f"Unknown metric: {metric}")


def rank_pool_with_metric(
    pool_meta: pd.DataFrame,
    norm_X: np.ndarray,
    mean_norm: np.ndarray,
    metric: str,
) -> pd.DataFrame:
    out = pool_meta.copy()
    out["distance"] = distances_to_mean(norm_X, mean_norm, metric)
    out = out.sort_values("distance", kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def spearman_aligned(
    base_ranked: pd.DataFrame, other_ranked: pd.DataFrame
) -> dict[str, float]:
    a = base_ranked[["mol_id", "rank"]].rename(columns={"rank": "rank_base"})
    b = other_ranked[["mol_id", "rank"]].rename(columns={"rank": "rank_other"})
    m = a.merge(b, on="mol_id", how="inner")
    if len(m) < 3:
        return {
            "spearman_rho": float("nan"),
            "spearman_pvalue": float("nan"),
            "n_paired": int(len(m)),
        }
    rho, pval = spearmanr(m["rank_base"], m["rank_other"])
    return {
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
        "n_paired": int(len(m)),
    }


def spearman_on_subset(
    base_ranked: pd.DataFrame,
    other_ranked: pd.DataFrame,
    mol_ids: set[int],
) -> dict[str, float]:
    a = base_ranked[base_ranked["mol_id"].isin(mol_ids)][
        ["mol_id", "rank"]
    ].rename(columns={"rank": "rank_base"})
    b = other_ranked[other_ranked["mol_id"].isin(mol_ids)][
        ["mol_id", "rank"]
    ].rename(columns={"rank": "rank_other"})
    m = a.merge(b, on="mol_id", how="inner")
    if len(m) < 3:
        return {
            "spearman_rho_top100_baseline": float("nan"),
            "spearman_pvalue_top100_baseline": float("nan"),
            "n_paired_top100_baseline": int(len(m)),
        }
    rho, pval = spearmanr(m["rank_base"], m["rank_other"])
    return {
        "spearman_rho_top100_baseline": float(rho),
        "spearman_pvalue_top100_baseline": float(pval),
        "n_paired_top100_baseline": int(len(m)),
    }


def _plot_heatmap(
    pivot: pd.DataFrame,
    fig_dir: Path,
    *,
    title: str,
    stem: str,
    vmin: float,
    cbar_label: str,
) -> None:
    if plt is None or pivot.empty:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Distance metric (vs Euclidean)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label=cbar_label)
    for path in (fig_dir / f"{stem}.pdf", fig_dir / f"{stem}.png"):
        fig.savefig(path, dpi=150)
    plt.close(fig)


def run_distance_sensitivity(
    refrigerants: pd.DataFrame,
    properties: list[str],
    output_dir: Path,
    *,
    refrigerants_csv: Path,
    strategies: list[str],
    metrics: list[str],
    top_k: int = TOP_K,
) -> dict[str, Any]:
    ranks_root = output_dir / "ranks"
    agg_dir = ranks_root / "aggregate"
    ranks_root.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    if BASELINE_METRIC not in metrics:
        metrics = [BASELINE_METRIC] + [m for m in metrics if m != BASELINE_METRIC]
    alt_metrics = [m for m in metrics if m != BASELINE_METRIC]

    known = load_known_refrigerant_smiles(refrigerants_csv)
    focal_path = output_dir / "focal_candidates.json"
    if not focal_path.is_file():
        src_focal = _PKG / "loo_mean_profile" / "focal_candidates.json"
        if src_focal.is_file():
            shutil.copy2(src_focal, focal_path)
        else:
            raise FileNotFoundError(focal_path)
    focals = load_focal_candidates(focal_path)
    write_json(ranks_root / "known_refrigerants_canonical.json", sorted(known))
    write_json(
        ranks_root / "procedure_notes.json",
        {
            "experiment": (
                "Re-rank strategy pools under alternative distances to the "
                "paper's hardcoded mean profile (PAPER_HARDCODED_MEANS) in "
                "the same min–max–normalized 15-D space (scales from the "
                "full unfiltered per-strategy CSV, reproducing the "
                "published screening pipeline). Compare each alternative "
                "to Euclidean."
            ),
            "baseline_metric": BASELINE_METRIC,
            "alternative_metrics": alt_metrics,
            "scale_mode": "min–max on the full, unfiltered all_molecules_ranked.csv "
            "per strategy (paper-consistent)",
            "mean_profile": "PAPER_HARDCODED_MEANS (historical screening-pipeline "
            "snapshot, not recomputed from the 21 profile fluids)",
            "cosine_definition": "distance = 1 − cosine_similarity",
            "metrics_reported": [
                "Spearman ρ (full universe) vs Euclidean",
                "Spearman ρ among Euclidean baseline top-100 rows",
                "top-100 unique-SMILES retention vs Euclidean",
                "Thermo-filtered retention vs Euclidean",
                "A–E ranks under each metric",
            ],
        },
    )

    ensure_iib_cache(output_dir)
    iib_pool_path = _rf_root() / STRATEGY_POOLS["IIb"] / "all_molecules_ranked.csv"
    iib_absent = (
        load_iib_pubchem_absent_smiles(output_dir, iib_pool_path)
        if "IIb" in strategies
        else None
    )

    mean_raw = paper_hardcoded_mean(properties)

    compare_rows: list[dict[str, Any]] = []
    ae_rows: list[dict[str, Any]] = []
    strategy_summaries: dict[str, Any] = {}

    for strategy in strategies:
        rel = STRATEGY_POOLS.get(strategy)
        if rel is None:
            logger.warning("Unknown strategy %s — skip", strategy)
            continue
        pool_path = _rf_root() / rel / "all_molecules_ranked.csv"
        if not pool_path.is_file():
            logger.warning("Missing pool for %s: %s", strategy, pool_path)
            continue

        logger.info("Distance sensitivity — strategy %s", strategy)
        pool_full = pd.read_csv(pool_path)
        missing = [p for p in properties if p not in pool_full.columns]
        if missing:
            raise ValueError(f"{pool_path}: missing descriptors {missing}")

        pool = prepare_strategy_pool(
            strategy, pool_full, iib_absent=iib_absent if strategy == "IIb" else None
        ).copy()
        pool["canon_smiles"] = pool["smiles"].map(canonical_smiles)
        pool["mol_id"] = np.arange(len(pool), dtype=int)
        pool_meta = pool[
            ["mol_id", "smiles", "canon_smiles"]
            + (["molecule_type"] if "molecule_type" in pool.columns else [])
            + (["MolName"] if "MolName" in pool.columns else [])
        ].copy()

        scales = paper_pool_minmax_scales(pool_full, properties)
        raw_X = pool[properties].astype(float).to_numpy()
        norm_X = normalize_matrix(raw_X, scales, properties)
        mean_n = normalize_mean_vector(mean_raw, scales, properties)

        strat_dir = ranks_root / f"strategy_{strategy}"
        strat_dir.mkdir(parents=True, exist_ok=True)
        write_json(strat_dir / "minmax_scales.json", scales)

        ranked_by_metric: dict[str, pd.DataFrame] = {}
        for metric in metrics:
            ranked_by_metric[metric] = rank_pool_with_metric(
                pool_meta, norm_X, mean_n, metric
            )

        base_ranked = ranked_by_metric[BASELINE_METRIC]
        base_top = base_ranked.nsmallest(top_k, "distance")
        base_filtered = apply_thermo_filter(base_ranked, known, top_k=top_k)
        base_top_set = set(base_top["canon_smiles"].dropna())
        base_filt_set = set(base_filtered["canon_smiles"].dropna())
        base_top_mol_ids = set(base_top["mol_id"].astype(int))
        base_top_n_unique = int(len(base_top_set))

        base_top[
            ["mol_id", "smiles", "canon_smiles", "distance", "rank"]
        ].to_csv(strat_dir / "baseline_euclidean_top100.csv", index=False)
        base_filt_out = base_filtered.copy()
        base_filt_out["rank_in_filtered"] = np.arange(1, len(base_filt_out) + 1)
        base_filt_out[
            ["mol_id", "smiles", "canon_smiles", "distance", "rank", "rank_in_filtered"]
        ].to_csv(strat_dir / "baseline_euclidean_filtered.csv", index=False)

        strategy_summaries[strategy] = {
            "pool_path": str(pool_path),
            "n_ranking_universe": int(len(pool)),
            "baseline_metric": BASELINE_METRIC,
            "baseline_top_k_unique": base_top_n_unique,
            "baseline_filtered_n": int(len(base_filtered)),
            "paper_filtered_n": PAPER_FILTERED_N.get(strategy),
            "code_excel_filtered_n": CODE_FILTERED_N.get(strategy),
            "note": (
                "Baseline is recomputed Euclidean using the paper's hardcoded "
                "mean profile and full-CSV min-max scales; reproduces the "
                "historical CSV RefrigerantDistance (Tables S2-S5) to >=6 "
                "significant figures."
            ),
        }
        logger.info(
            "  %s universe=%d; Euclidean top100 unique=%d; filtered=%d",
            strategy,
            len(pool),
            base_top_n_unique,
            len(base_filtered),
        )

        focals_here = [f for f in focals if f["strategy"] == strategy]
        for metric in metrics:
            ranked = ranked_by_metric[metric]
            filtered = apply_thermo_filter(ranked, known, top_k=top_k)
            for f in focals_here:
                c = canonical_smiles(f["smiles"])
                rnk, dist = _rank_lookup(ranked, c)
                rnk_filt = _filtered_rank_lookup(filtered, c)
                ae_rows.append(
                    {
                        "code": f["code"],
                        "strategy": strategy,
                        "metric": metric,
                        "smiles": f["smiles"],
                        "canon_smiles": c,
                        "distance": dist,
                        "rank": rnk,
                        "rank_in_filtered": rnk_filt,
                        "in_top100": bool(rnk is not None and rnk <= top_k),
                        "in_filtered": bool(rnk_filt is not None),
                    }
                )

        strat_compare: list[dict[str, Any]] = []
        for metric in alt_metrics:
            ranked = ranked_by_metric[metric]
            top = ranked.nsmallest(top_k, "distance")
            filtered = apply_thermo_filter(ranked, known, top_k=top_k)
            top_set = set(top["canon_smiles"].dropna())
            filt_set = set(filtered["canon_smiles"].dropna())
            sp = spearman_aligned(base_ranked, ranked)
            sp100 = spearman_on_subset(base_ranked, ranked, base_top_mol_ids)
            n_top_kept = len(base_top_set & top_set)
            n_filt_kept = len(base_filt_set & filt_set)
            row = {
                "strategy": strategy,
                "metric": metric,
                "baseline_metric": BASELINE_METRIC,
                **sp,
                **sp100,
                "top100_baseline_unique": base_top_n_unique,
                "top100_kept": n_top_kept,
                "top100_retention": n_top_kept / max(base_top_n_unique, 1),
                "filtered_baseline": len(base_filt_set),
                "filtered_kept": n_filt_kept,
                "filtered_new_size": int(len(filtered)),
                "filtered_retention": n_filt_kept / max(len(base_filt_set), 1),
            }
            compare_rows.append(row)
            strat_compare.append(row)

            # A–E deltas vs Euclidean
            for f in focals_here:
                c = canonical_smiles(f["smiles"])
                base_ae = next(
                    r
                    for r in ae_rows
                    if r["code"] == f["code"]
                    and r["metric"] == BASELINE_METRIC
                    and r["strategy"] == strategy
                )
                other = next(
                    r
                    for r in ae_rows
                    if r["code"] == f["code"]
                    and r["metric"] == metric
                    and r["strategy"] == strategy
                )
                other["delta_rank_vs_euclidean"] = (
                    None
                    if other["rank"] is None or base_ae["rank"] is None
                    else int(other["rank"] - base_ae["rank"])
                )
                other["delta_distance_vs_euclidean"] = (
                    None
                    if other["distance"] is None or base_ae["distance"] is None
                    else float(other["distance"] - base_ae["distance"])
                )

        pd.DataFrame(strat_compare).to_csv(
            strat_dir / "metric_vs_euclidean.csv", index=False
        )

    compare_df = pd.DataFrame(compare_rows)
    ae_df = pd.DataFrame(ae_rows)
    # fill Euclidean AE deltas as 0
    for col in ("delta_rank_vs_euclidean", "delta_distance_vs_euclidean"):
        if col not in ae_df.columns:
            ae_df[col] = None
    ae_df.loc[ae_df["metric"] == BASELINE_METRIC, "delta_rank_vs_euclidean"] = 0
    ae_df.loc[ae_df["metric"] == BASELINE_METRIC, "delta_distance_vs_euclidean"] = 0.0

    compare_df.to_csv(agg_dir / "metric_vs_euclidean_all_strategies.csv", index=False)
    ae_df.to_csv(agg_dir / "AE_distance_rank_by_metric.csv", index=False)

    summary = (
        compare_df.groupby("metric")
        .agg(
            spearman_rho_mean=("spearman_rho", "mean"),
            spearman_rho_min=("spearman_rho", "min"),
            spearman_top100_mean=("spearman_rho_top100_baseline", "mean"),
            spearman_top100_min=("spearman_rho_top100_baseline", "min"),
            top100_retention_mean=("top100_retention", "mean"),
            top100_retention_min=("top100_retention", "min"),
            filtered_retention_mean=("filtered_retention", "mean"),
            filtered_retention_min=("filtered_retention", "min"),
        )
        .reset_index()
    )
    # keep metric order
    summary["metric"] = pd.Categorical(
        summary["metric"], categories=list(alt_metrics), ordered=True
    )
    summary = summary.sort_values("metric")
    summary.to_csv(agg_dir / "summary_by_metric.csv", index=False)

    by_strat = (
        compare_df.groupby(["strategy", "metric"])
        .agg(
            spearman_rho=("spearman_rho", "first"),
            spearman_top100=("spearman_rho_top100_baseline", "first"),
            top100_retention=("top100_retention", "first"),
            filtered_retention=("filtered_retention", "first"),
        )
        .reset_index()
    )
    by_strat.to_csv(agg_dir / "summary_by_strategy_metric.csv", index=False)

    def _pivot(col: str) -> pd.DataFrame:
        p = compare_df.pivot(index="metric", columns="strategy", values=col)
        return p.reindex(alt_metrics)

    pivot_sp = _pivot("spearman_rho")
    pivot_sp100 = _pivot("spearman_rho_top100_baseline")
    pivot_ret100 = _pivot("top100_retention")
    pivot_ret_filt = _pivot("filtered_retention")
    pivot_sp.to_csv(agg_dir / "matrix_spearman_rho.csv")
    pivot_sp100.to_csv(agg_dir / "matrix_spearman_rho_top100.csv")
    pivot_ret100.to_csv(agg_dir / "matrix_top100_retention.csv")
    pivot_ret_filt.to_csv(agg_dir / "matrix_filtered_retention.csv")

    fig_dir = agg_dir / "figures"
    for pivot, title, stem, label, floor, default in (
        (
            pivot_sp,
            "Spearman ρ vs Euclidean (full universe)",
            "fig_spearman_heatmap",
            "ρ",
            0.5,
            0.85,
        ),
        (
            pivot_sp100,
            "Spearman ρ vs Euclidean (baseline top-100)",
            "fig_spearman_heatmap_top100",
            "ρ",
            0.0,
            0.5,
        ),
        (
            pivot_ret100,
            "Top-100 retention vs Euclidean ranking",
            "fig_top100_retention_heatmap",
            "retention",
            0.3,
            0.7,
        ),
        (
            pivot_ret_filt,
            "Filtered-set retention vs Euclidean ranking",
            "fig_filtered_retention_heatmap",
            "retention",
            0.3,
            0.7,
        ),
    ):
        vmin = float(np.nanmin(pivot.to_numpy(dtype=float)))
        _plot_heatmap(
            pivot,
            fig_dir,
            title=title,
            stem=stem,
            vmin=max(floor, min(default, vmin - 0.05)),
            cbar_label=label,
        )

    # A–E wide summary
    ae_sum_rows = []
    for code, g in ae_df.groupby("code"):
        base = g[g["metric"] == BASELINE_METRIC].iloc[0]
        row: dict[str, Any] = {
            "code": code,
            "strategy": base["strategy"],
            "euclidean_rank": base["rank"],
            "euclidean_distance": base["distance"],
            "euclidean_rank_in_filtered": base["rank_in_filtered"],
            "euclidean_in_top100": base["in_top100"],
            "euclidean_in_filtered": base["in_filtered"],
        }
        for metric in alt_metrics:
            m = g[g["metric"] == metric].iloc[0]
            row[f"{metric}_rank"] = m["rank"]
            row[f"{metric}_distance"] = m["distance"]
            row[f"{metric}_in_top100"] = m["in_top100"]
            row[f"{metric}_in_filtered"] = m["in_filtered"]
            row[f"{metric}_delta_rank"] = m.get("delta_rank_vs_euclidean")
        ae_sum_rows.append(row)
    ae_summary = pd.DataFrame(ae_sum_rows).sort_values("code")
    ae_summary.to_csv(agg_dir / "AE_summary.csv", index=False)

    headline = {
        "baseline_metric": BASELINE_METRIC,
        "alternative_metrics": alt_metrics,
        "strategies": strategies,
        "top_k": top_k,
        "spearman_rho_mean_by_metric": {
            r["metric"]: float(r["spearman_rho_mean"]) for _, r in summary.iterrows()
        },
        "spearman_top100_mean_by_metric": {
            r["metric"]: float(r["spearman_top100_mean"]) for _, r in summary.iterrows()
        },
        "top100_retention_mean_by_metric": {
            r["metric"]: float(r["top100_retention_mean"]) for _, r in summary.iterrows()
        },
        "filtered_retention_mean_by_metric": {
            r["metric"]: float(r["filtered_retention_mean"])
            for _, r in summary.iterrows()
        },
        "AE_in_top100_by_metric": {
            metric: {
                r["code"]: bool(r[f"{metric}_in_top100"])
                for _, r in ae_summary.iterrows()
            }
            for metric in alt_metrics
        },
        "strategy_summaries": strategy_summaries,
    }
    write_json(agg_dir / "headline_claims.json", headline)
    write_json(ranks_root / "strategy_summaries.json", strategy_summaries)

    summary_md = _format_summary_md(summary, by_strat, ae_summary, headline, alt_metrics)
    (agg_dir / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    if fig_dir.is_dir():
        logger.info("Wrote distance-metric figures under %s", fig_dir)
    return headline


def _format_summary_md(
    summary: pd.DataFrame,
    by_strat: pd.DataFrame,
    ae_summary: pd.DataFrame,
    headline: dict[str, Any],
    alt_metrics: list[str],
) -> str:
    lines = [
        "# Distance-metric ranking sensitivity — summary",
        "",
        "Same **15** descriptors and mean of all **21** fluids. Re-rank each",
        "strategy pool under alternative distances in min–max–normalized space,",
        "compared to **Euclidean** (paper baseline).",
        "",
        f"- Alternatives: {', '.join(f'`{m}`' for m in alt_metrics)}",
        f"- Strategies: {', '.join(headline['strategies'])}",
        f"- Top-k: **{headline['top_k']}**",
        "",
        "## Cosine note",
        "",
        "Cosine distance = `1 − cosine_similarity` of the normalized descriptor",
        "vectors (identical direction → 0).",
        "",
        "## Mean over strategies (vs Euclidean)",
        "",
        "| Metric | Full mean ρ | Full min ρ | Top-100 mean ρ | Top-100 retention | Filtered retention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| `{r['metric']}` | {r['spearman_rho_mean']:.4f} | "
            f"{r['spearman_rho_min']:.4f} | {r['spearman_top100_mean']:.4f} | "
            f"{r['top100_retention_mean']:.3f} | "
            f"{r['filtered_retention_mean']:.3f} |"
        )
    lines += [
        "",
        "## Per strategy",
        "",
        "| Strategy | Metric | Full ρ | Top-100 ρ | Top-100 retention | Filtered retention |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in by_strat.sort_values(["strategy", "metric"]).iterrows():
        lines.append(
            f"| {r['strategy']} | `{r['metric']}` | {r['spearman_rho']:.4f} | "
            f"{r['spearman_top100']:.4f} | {r['top100_retention']:.3f} | "
            f"{r['filtered_retention']:.3f} |"
        )

    # A–E table
    rank_cols = " | ".join(f"{m} rank" for m in alt_metrics)
    lines += [
        "",
        "## Compounds A–E (ranks under each metric)",
        "",
        f"| Code | Strat | Euclidean rank | Euclidean dist | {rank_cols} | Still in top-100 (all alts)? |",
        f"|---|---|---:|---:|{'---:|' * len(alt_metrics)}---|",
    ]
    for _, r in ae_summary.iterrows():
        alt_ranks = " | ".join(
            f"{int(r[f'{m}_rank'])}"
            + (f" (Δ{int(r[f'{m}_delta_rank']):+d})" if pd.notna(r[f'{m}_delta_rank']) else "")
            for m in alt_metrics
        )
        all_top = all(bool(r[f"{m}_in_top100"]) for m in alt_metrics)
        lines.append(
            f"| {r['code']} | {r['strategy']} | {int(r['euclidean_rank'])} | "
            f"{r['euclidean_distance']:.4f} | {alt_ranks} | "
            f"{'yes' if all_top else 'no'} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `ranks/strategy_*/baseline_euclidean_top100.csv`, `metric_vs_euclidean.csv`",
        "- `ranks/aggregate/metric_vs_euclidean_all_strategies.csv`",
        "- `ranks/aggregate/summary_by_metric.csv`, `summary_by_strategy_metric.csv`",
        "- `ranks/aggregate/matrix_*.csv`",
        "- `ranks/aggregate/AE_summary.csv`, `AE_distance_rank_by_metric.csv`",
        "- `ranks/aggregate/figures/fig_*_heatmap.pdf`",
        "",
        "## Note",
        "",
        "Baseline distances are **recomputed Euclidean** (not historical CSV",
        "`RefrigerantDistance`) so comparisons to Manhattan / cosine / Chebyshev",
        "share the same normalization and mean profile.",
        "",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distance-metric sensitivity of candidate rankings."
    )
    p.add_argument("--combined-descriptors", type=Path, default=None)
    p.add_argument("--refrigerants-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--strategies",
        type=str,
        default="I,IIa,IIb,III",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default=",".join(ALL_METRICS),
        help=f"Comma-separated metrics (default: {','.join(ALL_METRICS)})",
    )
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    combined = args.combined_descriptors or default_combined_descriptors_path()
    refs_csv = args.refrigerants_csv or default_refrigerants_csv_path()
    output_dir = (args.output_dir or default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in ALL_METRICS]
    if unknown:
        logger.error("Unknown metrics %s; choose from %s", unknown, ALL_METRICS)
        return 1

    if not Path(combined).is_file():
        logger.error("combined_descriptors not found: %s", combined)
        return 1

    logger.info("Loading descriptors from %s", combined)
    _, refrigerants, _ = load_combined_descriptors(
        Path(combined), Path(refs_csv) if refs_csv else None
    )
    logger.info(
        "Loaded %d fluids; metrics=%s; strategies=%s",
        len(refrigerants),
        metrics,
        strategies,
    )
    copy_or_write_manifest(output_dir, refrigerants)

    cfg = RunConfig(
        combined_descriptors=str(Path(combined).resolve()),
        refrigerants_csv=str(Path(refs_csv).resolve()) if refs_csv else None,
        output_dir=str(output_dir),
        fixed_15=list(FIXED_15),
        n_refrigerants=int(len(refrigerants)),
        strategies=strategies,
        metrics=metrics,
        top_k=int(args.top_k),
        scale_mode="freeze_refrigerants_union_pool",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output_dir / "run_config.json", asdict(cfg))

    readme = """# Distance-metric ranking sensitivity

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
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    headline = run_distance_sensitivity(
        refrigerants,
        list(FIXED_15),
        output_dir,
        refrigerants_csv=Path(refs_csv),
        strategies=strategies,
        metrics=metrics,
        top_k=args.top_k,
    )
    logger.info(
        "Done. Mean Spearman vs Euclidean: %s. See %s",
        headline["spearman_rho_mean_by_metric"],
        output_dir / "ranks" / "aggregate" / "SUMMARY.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
