#!/usr/bin/env python3
"""
Leave-one-descriptor-out (LODO) ablation of the 15-property working-fluid profile.

For each of the 15 descriptors, drop that one property and re-rank each
strategy pool with the remaining 14 (Euclidean distance to the paper's
hardcoded mean profile, ``PAPER_HARDCODED_MEANS``, with min-max scales taken
from the full unfiltered per-strategy CSV -- i.e. reproducing exactly the
screening pipeline that produced the published candidate tables), and compare
to the full 15-D ranking via:

  - Spearman rank correlation (full universe)
  - Top-100 / Thermo-filtered set retention
  - A–E rank shifts

This is the descriptor-ablation / alternative-subset check requested by R2
(and by supervisors), complementary to fluid LOO in ``loo_mean_profile/``.

Typical usage::

    python refrigerant_finding/profile_robustness/loo_descriptor_ablation_ranks.py

    python refrigerant_finding/profile_robustness/loo_descriptor_ablation_ranks.py \\
        --strategies I,IIa,IIb,III \\
        --output-dir refrigerant_finding/profile_robustness/loo_descriptor_ablation
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

# Ensure sibling imports work when launched as a script from any cwd.
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
    rank_pool_against_mean,
    write_json,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    combined_descriptors: str
    refrigerants_csv: Optional[str]
    output_dir: str
    fixed_15: list[str]
    n_refrigerants: int
    strategies: list[str]
    top_k: int
    baseline_distance_source: str
    scale_mode: str
    timestamp_utc: str


def default_output_dir() -> Path:
    return _PKG / "loo_descriptor_ablation"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _safe_slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _rf_root() -> Path:
    return _PKG.parent


def ensure_iib_cache(output_dir: Path) -> Path:
    """
    Prefer the already-built IIb absent-SMILES cache from loo_mean_profile
    to avoid rescanning the PubChem dump.
    """
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


def spearman_aligned(
    base_ranked: pd.DataFrame, fold_ranked: pd.DataFrame
) -> dict[str, float]:
    """Spearman ρ between two rankings aligned on stable mol_id."""
    a = base_ranked[["mol_id", "rank"]].rename(columns={"rank": "rank_base"})
    b = fold_ranked[["mol_id", "rank"]].rename(columns={"rank": "rank_fold"})
    m = a.merge(b, on="mol_id", how="inner")
    if len(m) < 3:
        return {
            "spearman_rho": float("nan"),
            "spearman_pvalue": float("nan"),
            "n_paired": int(len(m)),
        }
    rho, pval = spearmanr(m["rank_base"], m["rank_fold"])
    return {
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
        "n_paired": int(len(m)),
    }


def spearman_on_subset(
    base_ranked: pd.DataFrame,
    fold_ranked: pd.DataFrame,
    mol_ids: set[int],
) -> dict[str, float]:
    """Spearman among a subset of mol_ids (e.g. baseline top-100 rows)."""
    a = base_ranked[base_ranked["mol_id"].isin(mol_ids)][
        ["mol_id", "rank"]
    ].rename(columns={"rank": "rank_base"})
    b = fold_ranked[fold_ranked["mol_id"].isin(mol_ids)][
        ["mol_id", "rank"]
    ].rename(columns={"rank": "rank_fold"})
    m = a.merge(b, on="mol_id", how="inner")
    if len(m) < 3:
        return {
            "spearman_rho_top100_baseline": float("nan"),
            "spearman_pvalue_top100_baseline": float("nan"),
            "n_paired_top100_baseline": int(len(m)),
        }
    rho, pval = spearmanr(m["rank_base"], m["rank_fold"])
    return {
        "spearman_rho_top100_baseline": float(rho),
        "spearman_pvalue_top100_baseline": float(pval),
        "n_paired_top100_baseline": int(len(m)),
    }


def run_ablation(
    refrigerants: pd.DataFrame,
    properties: list[str],
    output_dir: Path,
    *,
    refrigerants_csv: Path,
    strategies: list[str],
    top_k: int = TOP_K,
) -> dict[str, Any]:
    ranks_root = output_dir / "ranks"
    agg_dir = ranks_root / "aggregate"
    ranks_root.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    known = load_known_refrigerant_smiles(refrigerants_csv)
    focal_path = output_dir / "focal_candidates.json"
    if not focal_path.is_file():
        src_focal = _PKG / "loo_mean_profile" / "focal_candidates.json"
        if src_focal.is_file():
            shutil.copy2(src_focal, focal_path)
        else:
            raise FileNotFoundError(
                f"focal_candidates.json not found at {focal_path} or {src_focal}"
            )
    focals = load_focal_candidates(focal_path)
    write_json(ranks_root / "known_refrigerants_canonical.json", sorted(known))
    write_json(
        ranks_root / "procedure_notes.json",
        {
            "experiment": (
                "Leave-one-descriptor-out: drop one of the fixed 15 "
                "descriptors; re-rank with remaining 14 vs full 15-D "
                "baseline. Both use the paper's hardcoded mean profile "
                "(PAPER_HARDCODED_MEANS) and min-max scales taken from the "
                "full, unfiltered all_molecules_ranked.csv per strategy — "
                "i.e. exactly reproducing the historical screening pipeline "
                "(find_refrigerant_candidates.py), not the mean dynamically "
                "recomputed from the 21 profile fluids."
            ),
            "baseline_distance_source": (
                "recomputed Euclidean vs PAPER_HARDCODED_MEANS on all 15 "
                "descriptors, using min-max scales from the full unfiltered "
                "pool CSV; reproduces historical CSV RefrigerantDistance "
                "(Tables S2-S5) to >=6 significant figures"
            ),
            "fold_distance_source": (
                "recomputed Euclidean vs PAPER_HARDCODED_MEANS on the 14 "
                "kept descriptors, same scale convention"
            ),
            "scale_mode": "min–max on the full, unfiltered all_molecules_ranked.csv "
            "per strategy (paper-consistent; active descriptors only)",
            "I_IIa_III_universe": "full all_molecules_ranked.csv",
            "IIb_universe": (
                "PubChem-absent candidates (same cache/procedure as "
                "loo_mean_profile ranks / visualize_novel_molecules_grid.py)"
            ),
            "metrics": [
                "Spearman ρ (full ranking universe)",
                "Spearman ρ among baseline top-100 rows",
                "top-100 unique-SMILES retention",
                "Thermo-filtered retention",
                "A–E rank / distance",
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

    # Mean profile: the paper's hardcoded PAPER_HARDCODED_MEANS snapshot,
    # not the mean dynamically recomputed from the 21 profile fluids.
    mean_raw_all = paper_hardcoded_mean(properties)

    fold_specs: list[dict[str, Any]] = [
        {
            "fold_name": "baseline_15",
            "left_out_descriptor": None,
            "properties": list(properties),
        }
    ]
    for prop in properties:
        kept = [p for p in properties if p != prop]
        fold_specs.append(
            {
                "fold_name": f"drop_{_safe_slug(prop)}",
                "left_out_descriptor": prop,
                "properties": kept,
            }
        )

    spearman_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
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

        logger.info("Descriptor ablation — strategy %s from %s", strategy, pool_path)
        pool_full = pd.read_csv(pool_path)
        missing = [p for p in properties if p not in pool_full.columns]
        if missing:
            raise ValueError(f"{pool_path}: missing raw descriptors {missing}")

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

        # Full 15-D scales (also used to document baseline). Paper-consistent:
        # min-max over the full, unfiltered pool CSV (already contains both
        # refrigerant- and candidate-tagged rows), not refrigerants(21) ∪
        # ranking-universe.
        scales_all = paper_pool_minmax_scales(pool_full, properties)

        strat_dir = ranks_root / f"strategy_{strategy}"
        strat_dir.mkdir(parents=True, exist_ok=True)
        write_json(strat_dir / "minmax_scales_15.json", scales_all)

        ranked_by_fold: dict[str, pd.DataFrame] = {}
        for spec in fold_specs:
            props = spec["properties"]
            scales = paper_pool_minmax_scales(pool_full, props)
            raw_X = pool[props].astype(float).to_numpy()
            norm_X = normalize_matrix(raw_X, scales, props)
            mean_n = normalize_mean_vector(mean_raw_all, scales, props)
            ranked = rank_pool_against_mean(pool_meta, norm_X, mean_n)
            ranked_by_fold[spec["fold_name"]] = ranked

            if spec["left_out_descriptor"] is None:
                write_json(strat_dir / "minmax_scales_baseline.json", scales)

        base_ranked = ranked_by_fold["baseline_15"]
        base_top = base_ranked.nsmallest(top_k, "distance")
        base_filtered = apply_thermo_filter(base_ranked, known, top_k=top_k)
        base_top_set = set(base_top["canon_smiles"].dropna())
        base_filt_set = set(base_filtered["canon_smiles"].dropna())
        base_top_mol_ids = set(base_top["mol_id"].astype(int))
        base_top_n_rows = int(len(base_top))
        base_top_n_unique = int(len(base_top_set))

        base_top[
            ["mol_id", "smiles", "canon_smiles", "distance", "rank"]
        ].to_csv(strat_dir / "baseline_top100.csv", index=False)
        base_filt_out = base_filtered.copy()
        base_filt_out["rank_in_filtered"] = np.arange(1, len(base_filt_out) + 1)
        base_filt_out[
            ["mol_id", "smiles", "canon_smiles", "distance", "rank", "rank_in_filtered"]
        ].to_csv(strat_dir / "baseline_filtered.csv", index=False)

        strategy_summaries[strategy] = {
            "pool_path": str(pool_path),
            "n_pool_full_csv": int(len(pool_full)),
            "n_ranking_universe": int(len(pool)),
            "baseline_top_k_rows": base_top_n_rows,
            "baseline_top_k_unique": base_top_n_unique,
            "baseline_filtered_n": int(len(base_filtered)),
            "paper_filtered_n": PAPER_FILTERED_N.get(strategy),
            "code_excel_filtered_n": CODE_FILTERED_N.get(strategy),
            "note": (
                "Baseline is recomputed 15-D Euclidean using the paper's "
                "hardcoded mean profile and full-CSV min-max scales; "
                "reproduces the historical CSV RefrigerantDistance "
                "(Tables S2-S5) to >=6 significant figures."
            ),
        }
        logger.info(
            "  %s universe=%d; baseline top100 unique=%d; filtered=%d",
            strategy,
            len(pool),
            base_top_n_unique,
            len(base_filtered),
        )

        focals_here = [f for f in focals if f["strategy"] == strategy]
        for f in focals_here:
            c = canonical_smiles(f["smiles"])
            rnk, dist = _rank_lookup(base_ranked, c)
            rnk_filt = _filtered_rank_lookup(base_filtered, c)
            ae_rows.append(
                {
                    "code": f["code"],
                    "strategy": strategy,
                    "fold": "baseline_15",
                    "left_out_descriptor": None,
                    "smiles": f["smiles"],
                    "canon_smiles": c,
                    "distance": dist,
                    "rank": rnk,
                    "rank_in_filtered": rnk_filt,
                    "in_top100": bool(rnk is not None and rnk <= top_k),
                    "in_filtered": bool(rnk_filt is not None),
                    "delta_rank": 0 if rnk is not None else None,
                    "delta_distance": 0.0 if dist is not None else None,
                }
            )

        fold_ret_rows: list[dict[str, Any]] = []
        for spec in fold_specs:
            if spec["left_out_descriptor"] is None:
                continue
            left = spec["left_out_descriptor"]
            ranked = ranked_by_fold[spec["fold_name"]]
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
                "fold_name": spec["fold_name"],
                "left_out_descriptor": left,
                "n_descriptors_used": len(spec["properties"]),
                **sp,
                **sp100,
                "top100_baseline_rows": base_top_n_rows,
                "top100_baseline_unique": base_top_n_unique,
                "top100_kept": n_top_kept,
                "top100_retention": n_top_kept / max(base_top_n_unique, 1),
                "filtered_baseline": len(base_filt_set),
                "filtered_kept": n_filt_kept,
                "filtered_new_size": int(len(filtered)),
                "filtered_retention": n_filt_kept / max(len(base_filt_set), 1),
            }
            spearman_rows.append(row)
            retention_rows.append(row)
            fold_ret_rows.append(row)

            for f in focals_here:
                c = canonical_smiles(f["smiles"])
                base_ae = next(
                    r
                    for r in ae_rows
                    if r["code"] == f["code"] and r["fold"] == "baseline_15"
                )
                rnk, dist = _rank_lookup(ranked, c)
                rnk_filt = _filtered_rank_lookup(filtered, c)
                ae_rows.append(
                    {
                        "code": f["code"],
                        "strategy": strategy,
                        "fold": spec["fold_name"],
                        "left_out_descriptor": left,
                        "smiles": f["smiles"],
                        "canon_smiles": c,
                        "distance": dist,
                        "rank": rnk,
                        "rank_in_filtered": rnk_filt,
                        "in_top100": bool(rnk is not None and rnk <= top_k),
                        "in_filtered": bool(rnk_filt is not None),
                        "delta_rank": (
                            None
                            if rnk is None or base_ae["rank"] is None
                            else int(rnk - base_ae["rank"])
                        ),
                        "delta_distance": (
                            None
                            if dist is None or base_ae["distance"] is None
                            else float(dist - base_ae["distance"])
                        ),
                    }
                )

        pd.DataFrame(fold_ret_rows).to_csv(
            strat_dir / "fold_spearman_retention.csv", index=False
        )

    spearman_df = pd.DataFrame(spearman_rows)
    ae_df = pd.DataFrame(ae_rows)
    spearman_df.to_csv(agg_dir / "spearman_retention_by_fold.csv", index=False)
    ae_df.to_csv(agg_dir / "AE_distance_rank_by_fold.csv", index=False)

    # Per-strategy summary (mean / min Spearman over 15 drops)
    strat_summary = (
        spearman_df.groupby("strategy")
        .agg(
            n_folds=("left_out_descriptor", "count"),
            spearman_rho_mean=("spearman_rho", "mean"),
            spearman_rho_min=("spearman_rho", "min"),
            spearman_rho_max=("spearman_rho", "max"),
            spearman_top100_mean=("spearman_rho_top100_baseline", "mean"),
            spearman_top100_min=("spearman_rho_top100_baseline", "min"),
            spearman_top100_max=("spearman_rho_top100_baseline", "max"),
            top100_retention_mean=("top100_retention", "mean"),
            top100_retention_min=("top100_retention", "min"),
            filtered_retention_mean=("filtered_retention", "mean"),
            filtered_retention_min=("filtered_retention", "min"),
        )
        .reset_index()
    )
    # which descriptor hurts full / top-100 Spearman most
    worst = (
        spearman_df.loc[
            spearman_df.groupby("strategy")["spearman_rho"].idxmin()
        ][["strategy", "left_out_descriptor", "spearman_rho"]]
        .rename(
            columns={
                "left_out_descriptor": "worst_drop_descriptor",
                "spearman_rho": "worst_drop_spearman",
            }
        )
    )
    worst100 = (
        spearman_df.loc[
            spearman_df.groupby("strategy")["spearman_rho_top100_baseline"].idxmin()
        ][
            [
                "strategy",
                "left_out_descriptor",
                "spearman_rho_top100_baseline",
            ]
        ].rename(
            columns={
                "left_out_descriptor": "worst_drop_descriptor_top100",
                "spearman_rho_top100_baseline": "worst_drop_spearman_top100",
            }
        )
    )
    strat_summary = strat_summary.merge(worst, on="strategy", how="left")
    strat_summary = strat_summary.merge(worst100, on="strategy", how="left")
    strat_summary.to_csv(agg_dir / "spearman_summary_by_strategy.csv", index=False)

    # Per-descriptor impact (averaged across strategies)
    desc_impact = (
        spearman_df.groupby("left_out_descriptor")
        .agg(
            spearman_rho_mean=("spearman_rho", "mean"),
            spearman_rho_min=("spearman_rho", "min"),
            spearman_top100_mean=("spearman_rho_top100_baseline", "mean"),
            spearman_top100_min=("spearman_rho_top100_baseline", "min"),
            top100_retention_mean=("top100_retention", "mean"),
            top100_retention_min=("top100_retention", "min"),
            filtered_retention_mean=("filtered_retention", "mean"),
        )
        .reset_index()
        .sort_values("spearman_rho_mean")
    )
    desc_impact.to_csv(agg_dir / "descriptor_drop_impact.csv", index=False)

    # Wide Spearman matrices: descriptors × strategies
    pivot = spearman_df.pivot(
        index="left_out_descriptor", columns="strategy", values="spearman_rho"
    ).reindex(properties)
    pivot.to_csv(agg_dir / "matrix_spearman_rho.csv")
    pivot100 = spearman_df.pivot(
        index="left_out_descriptor",
        columns="strategy",
        values="spearman_rho_top100_baseline",
    ).reindex(properties)
    pivot100.to_csv(agg_dir / "matrix_spearman_rho_top100.csv")
    pivot_ret100 = spearman_df.pivot(
        index="left_out_descriptor",
        columns="strategy",
        values="top100_retention",
    ).reindex(properties)
    pivot_ret100.to_csv(agg_dir / "matrix_top100_retention.csv")
    pivot_ret_filt = spearman_df.pivot(
        index="left_out_descriptor",
        columns="strategy",
        values="filtered_retention",
    ).reindex(properties)
    pivot_ret_filt.to_csv(agg_dir / "matrix_filtered_retention.csv")

    # A–E summary (same style as loo_mean_profile ranks)
    ae_base = ae_df[ae_df["fold"] == "baseline_15"]
    ae_loo = ae_df[ae_df["fold"] != "baseline_15"]
    ae_sum_rows = []
    for code, g in ae_loo.groupby("code"):
        base = ae_base[ae_base["code"] == code].iloc[0]
        n_folds = len(g)
        n_top = int(g["in_top100"].sum())
        n_filt = int(g["in_filtered"].sum())
        g_top = g.loc[g["in_top100"]]
        g_filt = g.loc[g["in_filtered"]]
        ae_sum_rows.append(
            {
                "code": code,
                "strategy": base["strategy"],
                "baseline_rank": base["rank"],
                "baseline_distance": base["distance"],
                "baseline_rank_in_filtered": base["rank_in_filtered"],
                "pct_folds_in_top100": 100.0 * n_top / max(n_folds, 1),
                "pct_folds_in_filtered": 100.0 * n_filt / max(n_folds, 1),
                "mean_rank_lodo": g["rank"].mean(skipna=True),
                "mean_distance_lodo": g["distance"].mean(skipna=True),
                "mean_rank_lodo_when_in_top100": (
                    g_top["rank"].mean() if n_top else None
                ),
                "mean_rank_in_filtered_when_passed": (
                    g_filt["rank_in_filtered"].mean() if n_filt else None
                ),
                "max_abs_delta_rank": g["delta_rank"].abs().max(skipna=True),
                "mean_delta_rank": g["delta_rank"].mean(skipna=True),
                "worst_drop_descriptor": (
                    g.loc[g["delta_rank"].abs().idxmax(), "left_out_descriptor"]
                    if n_folds and g["delta_rank"].notna().any()
                    else None
                ),
                "n_folds_total": n_folds,
                "n_folds_in_top100": n_top,
                "n_folds_in_filtered": n_filt,
            }
        )
    ae_summary = pd.DataFrame(ae_sum_rows).sort_values("code")
    ae_summary.to_csv(agg_dir / "AE_summary.csv", index=False)

    headline = {
        "n_descriptors": len(properties),
        "n_folds_per_strategy": len(properties),
        "strategies": strategies,
        "top_k": top_k,
        "spearman_rho_mean_by_strategy": {
            r["strategy"]: float(r["spearman_rho_mean"])
            for _, r in strat_summary.iterrows()
        },
        "spearman_rho_min_by_strategy": {
            r["strategy"]: float(r["spearman_rho_min"])
            for _, r in strat_summary.iterrows()
        },
        "spearman_top100_mean_by_strategy": {
            r["strategy"]: float(r["spearman_top100_mean"])
            for _, r in strat_summary.iterrows()
        },
        "spearman_top100_min_by_strategy": {
            r["strategy"]: float(r["spearman_top100_min"])
            for _, r in strat_summary.iterrows()
        },
        "worst_drop_by_strategy": {
            r["strategy"]: {
                "descriptor": r["worst_drop_descriptor"],
                "spearman_rho": float(r["worst_drop_spearman"]),
            }
            for _, r in strat_summary.iterrows()
        },
        "worst_drop_top100_by_strategy": {
            r["strategy"]: {
                "descriptor": r["worst_drop_descriptor_top100"],
                "spearman_rho_top100": float(r["worst_drop_spearman_top100"]),
            }
            for _, r in strat_summary.iterrows()
        },
        "top100_retention_mean_by_strategy": {
            r["strategy"]: float(r["top100_retention_mean"])
            for _, r in strat_summary.iterrows()
        },
        "AE_pct_in_top100": {
            r["code"]: float(r["pct_folds_in_top100"])
            for _, r in ae_summary.iterrows()
        },
        "strategy_summaries": strategy_summaries,
    }
    write_json(agg_dir / "headline_claims.json", headline)
    write_json(ranks_root / "strategy_summaries.json", strategy_summaries)

    summary_md = _format_summary_md(strat_summary, desc_impact, ae_summary, headline)
    (agg_dir / "SUMMARY.md").write_text(summary_md, encoding="utf-8")

    _maybe_plot_spearman(
        pivot, pivot100, pivot_ret100, pivot_ret_filt, agg_dir / "figures"
    )

    return headline


def _format_summary_md(
    strat_summary: pd.DataFrame,
    desc_impact: pd.DataFrame,
    ae_summary: pd.DataFrame,
    headline: dict[str, Any],
) -> str:
    lines = [
        "# Leave-one-descriptor-out ranking ablation — summary",
        "",
        "Keep all **21** profile fluids. Drop one of the fixed **15** descriptors,",
        "re-rank each strategy pool with the remaining **14**, and compare to the",
        "full 15-D ranking (both recomputed Euclidean to the mean of the 21).",
        "",
        f"- Descriptor folds: **{headline['n_folds_per_strategy']}**",
        f"- Top-k: **{headline['top_k']}**",
        f"- Strategies: {', '.join(headline['strategies'])}",
        "",
        "## Spearman ρ (15-D vs each 14-D drop)",
        "",
        "Two paired correlations are reported:",
        "",
        "- **Full universe** — all molecules in the strategy ranking pool.",
        "- **Top-100 baseline** — only molecules that were in the baseline",
        "  top-100 **rows** (their ranks under 15-D vs 14-D).",
        "",
        "Set overlap (retention) is separate from Spearman.",
        "",
        "| Strategy | Full mean ρ | Full min ρ | Top-100 mean ρ | Top-100 min ρ | Worst full drop | Worst top-100 drop | Top-100 retention | Filtered retention |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for _, r in strat_summary.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['spearman_rho_mean']:.4f} | "
            f"{r['spearman_rho_min']:.4f} | {r['spearman_top100_mean']:.4f} | "
            f"{r['spearman_top100_min']:.4f} | `{r['worst_drop_descriptor']}` "
            f"({r['worst_drop_spearman']:.4f}) | "
            f"`{r['worst_drop_descriptor_top100']}` "
            f"({r['worst_drop_spearman_top100']:.4f}) | "
            f"{r['top100_retention_mean']:.3f} | "
            f"{r['filtered_retention_mean']:.3f} |"
        )
    lines += [
        "",
        "## Descriptors whose removal hurts ranking most (mean across strategies)",
        "",
        "| Left-out descriptor | Full mean ρ | Full min ρ | Top-100 mean ρ | Top-100 min ρ | Mean top-100 retention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in desc_impact.iterrows():
        lines.append(
            f"| `{r['left_out_descriptor']}` | {r['spearman_rho_mean']:.4f} | "
            f"{r['spearman_rho_min']:.4f} | {r['spearman_top100_mean']:.4f} | "
            f"{r['spearman_top100_min']:.4f} | {r['top100_retention_mean']:.3f} |"
        )
    lines += [
        "",
        "## Compounds A–E",
        "",
        "Same style as fluid LOO (`loo_mean_profile` ranks): universe rank /",
        "distance under 15-D baseline, then stability across the 15 descriptor drops.",
        "",
        "| Code | Strat | Base rank | Base dist | Base rank in filtered | % LODO top-100 | Mean LODO rank | Mean LODO dist | % LODO filter | Mean filt-rank when passed | Worst drop (max abs Δrank) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in ae_summary.iterrows():
        brf = r["baseline_rank_in_filtered"]
        brf_s = f"{int(brf)}" if pd.notna(brf) else "—"
        mrf = r["mean_rank_in_filtered_when_passed"]
        mrf_s = f"{mrf:.1f}" if pd.notna(mrf) else "—"
        lines.append(
            f"| {r['code']} | {r['strategy']} | {int(r['baseline_rank'])} | "
            f"{r['baseline_distance']:.4f} | {brf_s} | "
            f"{r['pct_folds_in_top100']:.0f}% | {r['mean_rank_lodo']:.1f} | "
            f"{r['mean_distance_lodo']:.4f} | {r['pct_folds_in_filtered']:.0f}% | "
            f"{mrf_s} | `{r['worst_drop_descriptor']}` "
            f"({r['max_abs_delta_rank']:.0f}) |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `ranks/strategy_*/baseline_top100.csv`, `baseline_filtered.csv`, "
        "`fold_spearman_retention.csv`",
        "- `ranks/aggregate/spearman_retention_by_fold.csv`",
        "- `ranks/aggregate/spearman_summary_by_strategy.csv`",
        "- `ranks/aggregate/descriptor_drop_impact.csv`",
        "- `ranks/aggregate/matrix_spearman_rho.csv`, `matrix_spearman_rho_top100.csv`",
        "- `ranks/aggregate/matrix_top100_retention.csv`, `matrix_filtered_retention.csv`",
        "- `ranks/aggregate/AE_summary.csv`, `AE_distance_rank_by_fold.csv`",
        "- `ranks/aggregate/headline_claims.json`",
        "- `ranks/aggregate/figures/fig_spearman_heatmap*.pdf`, "
        "`fig_top100_retention_heatmap.pdf`, `fig_filtered_retention_heatmap.pdf`",
        "",
        "## Note on baseline distances",
        "",
        "Baseline ranks use **recomputed** 15-D Euclidean distances (not the",
        "historical CSV `RefrigerantDistance`) so Spearman vs 14-D folds is",
        "apples-to-apples. Filtered set sizes may therefore differ slightly",
        "from the paper/Excel Thermo handoff counts.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _plot_heatmap(
    pivot: pd.DataFrame,
    fig_dir: Path,
    *,
    title: str,
    stem: str,
    vmin: float,
    cbar_label: str = "ρ",
) -> None:
    fig, ax = plt.subplots(
        figsize=(8.5, max(4.5, 0.35 * len(pivot) + 1.5)),
        constrained_layout=True,
    )
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index), fontsize=8)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Left-out descriptor")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)
    for path in (fig_dir / f"{stem}.pdf", fig_dir / f"{stem}.png"):
        fig.savefig(path, dpi=150)
    plt.close(fig)


def _maybe_plot_spearman(
    pivot: pd.DataFrame,
    pivot100: pd.DataFrame,
    pivot_ret100: pd.DataFrame,
    pivot_ret_filt: pd.DataFrame,
    fig_dir: Path,
) -> None:
    if plt is None or pivot.empty:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_heatmap(
        pivot,
        fig_dir,
        title="Spearman ρ vs full 15-D ranking (full universe)",
        stem="fig_spearman_heatmap",
        vmin=0.8,
        cbar_label="ρ",
    )
    if not pivot100.empty:
        vmin100 = float(np.nanmin(pivot100.to_numpy(dtype=float)))
        _plot_heatmap(
            pivot100,
            fig_dir,
            title="Spearman ρ among baseline top-100 molecules",
            stem="fig_spearman_heatmap_top100",
            vmin=max(0.0, min(0.5, vmin100 - 0.05)),
            cbar_label="ρ",
        )
    if not pivot_ret100.empty:
        vmin_r = float(np.nanmin(pivot_ret100.to_numpy(dtype=float)))
        _plot_heatmap(
            pivot_ret100,
            fig_dir,
            title="Top-100 retention (unique SMILES vs 15-D baseline)",
            stem="fig_top100_retention_heatmap",
            vmin=max(0.5, min(0.85, vmin_r - 0.05)),
            cbar_label="retention",
        )
    if not pivot_ret_filt.empty:
        vmin_f = float(np.nanmin(pivot_ret_filt.to_numpy(dtype=float)))
        _plot_heatmap(
            pivot_ret_filt,
            fig_dir,
            title="Filtered-set retention vs 15-D baseline",
            stem="fig_filtered_retention_heatmap",
            vmin=max(0.5, min(0.85, vmin_f - 0.05)),
            cbar_label="retention",
        )

    # bar of mean ρ across strategies (full + top100)
    means = pivot.mean(axis=1).sort_values()
    means100 = pivot100.mean(axis=1).reindex(means.index)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    y = np.arange(len(means))
    ax.barh(y - 0.2, means.values, height=0.4, color="#3d5a80", label="Full universe")
    ax.barh(
        y + 0.2,
        means100.values,
        height=0.4,
        color="#98c1d9",
        label="Baseline top-100",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(means.index.astype(str))
    ax.set_xlabel("Mean Spearman ρ across strategies")
    ax.set_title("Impact of dropping each descriptor")
    ax.axvline(1.0, color="0.5", lw=0.8, ls="--")
    ax.legend(loc="lower right", frameon=False)
    lo = float(np.nanmin([means.min(), means100.min()]))
    ax.set_xlim(min(0.5, lo - 0.05), 1.0)
    for path in (
        fig_dir / "fig_descriptor_drop_spearman.pdf",
        fig_dir / "fig_descriptor_drop_spearman.png",
    ):
        fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote Spearman / retention figures under %s", fig_dir)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Leave-one-descriptor-out ranking ablation (14 vs 15 properties)."
        )
    )
    p.add_argument(
        "--combined-descriptors",
        type=Path,
        default=None,
        help="Path to combined_descriptors.csv",
    )
    p.add_argument(
        "--refrigerants-csv",
        type=Path,
        default=None,
        help="Path to refrigerants.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: profile_robustness/loo_descriptor_ablation)",
    )
    p.add_argument(
        "--strategies",
        type=str,
        default="I,IIa,IIb,III",
        help="Comma-separated strategies (default: I,IIa,IIb,III)",
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

    if not Path(combined).is_file():
        logger.error("combined_descriptors not found: %s", combined)
        return 1

    logger.info("Loading descriptors from %s", combined)
    _, refrigerants, _ = load_combined_descriptors(
        Path(combined), Path(refs_csv) if refs_csv else None
    )
    logger.info(
        "Loaded %d profile fluids; running LODO over %d descriptors; strategies=%s",
        len(refrigerants),
        len(FIXED_15),
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
        top_k=int(args.top_k),
        baseline_distance_source="recomputed_euclidean_15",
        scale_mode="freeze_refrigerants_union_pool_active_descriptors",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output_dir / "run_config.json", asdict(cfg))

    readme = """# Leave-one-descriptor-out (LODO) ranking ablation

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
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    headline = run_ablation(
        refrigerants,
        list(FIXED_15),
        output_dir,
        refrigerants_csv=Path(refs_csv),
        strategies=strategies,
        top_k=args.top_k,
    )
    logger.info(
        "Done. Mean Spearman by strategy: %s. See %s",
        headline["spearman_rho_mean_by_strategy"],
        output_dir / "ranks" / "aggregate" / "SUMMARY.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
