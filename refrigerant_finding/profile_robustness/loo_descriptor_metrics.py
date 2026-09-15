#!/usr/bin/env python3
"""
Leave-one-out (LOO) robustness analysis for descriptor-selection metrics.

For each leave-one-out of the 21 size-matched working fluids, recomputes
Jensen–Shannon (JS) distances and Mann–Whitney U tests vs the fixed 177
size-matched reference molecules, using the same methodology as
``refrigerant_comparison.py`` / ``statistical_comparison/``.

This script does NOT re-rank candidates or recompute mean profiles — only
descriptor-level JS / MW metrics and near-substitute discovery.

Typical usage (from repo root or this file's directory)::

    python refrigerant_finding/profile_robustness/loo_descriptor_metrics.py

    python refrigerant_finding/profile_robustness/loo_descriptor_metrics.py \\
        --combined-descriptors refrigerant_finding/statistical_comparison/combined_descriptors.csv \\
        --output-dir refrigerant_finding/profile_robustness/loo_on_profile
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import entropy, mannwhitneyu

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:  # pragma: no cover - RDKit optional for name mapping only
    Chem = None
    Descriptors = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper-fixed 15 descriptors (curated; not auto top-15)
# ---------------------------------------------------------------------------
FIXED_15: list[str] = [
    "MaxPartialCharge",
    "SMR_VSA1",
    "MinEStateIndex",
    "MolMR",
    "FpDensityMorgan3",
    "MinPartialCharge",
    "fr_halogen",
    "fr_alkyl_halide",
    "TPSA",
    "NumHAcceptors",
    "VSA_EState1",
    "LabuteASA",
    "MolLogP",
    "NumHeteroatoms",
    "NumRotatableBonds",
]

REFRIGERANT_CATEGORY = "refrigerant"
REFERENCE_CATEGORY = "database_size_matched"
META_COLUMNS = {"smiles", "category", "compound", "refrigerant_id", "fold_id"}

# Match refrigerant_comparison.py continuous/discrete switch
CATEGORICAL_NUNIQUE_THRESHOLD = 20
HISTOGRAM_BINS = 30
DEFAULT_P_THRESHOLD = 0.05
DEFAULT_NEAR_SUB_TOP_N = 30
HEAVY_ATOM_MIN = 5
HEAVY_ATOM_MAX = 10


@dataclass(frozen=True)
class RunConfig:
    combined_descriptors: str
    refrigerants_csv: Optional[str]
    output_dir: str
    p_threshold: float
    near_substitute_top_n: int
    near_substitute_js_mode: str
    heavy_atom_min: int
    heavy_atom_max: int
    fixed_15: list[str]
    n_refrigerants: int
    n_references: int
    n_descriptors: int
    categorical_nunique_threshold: int
    histogram_bins: int
    seed: int
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _project_refrigerant_finding_dir() -> Path:
    return _package_dir().parent


def default_combined_descriptors_path() -> Path:
    return (
        _project_refrigerant_finding_dir()
        / "statistical_comparison"
        / "combined_descriptors.csv"
    )


def default_refrigerants_csv_path() -> Path:
    return _project_refrigerant_finding_dir() / "refrigerants.csv"


def default_output_dir() -> Path:
    return _package_dir() / "loo_on_profile"


# ---------------------------------------------------------------------------
# JS + Mann–Whitney (match refrigerant_comparison.py)
# ---------------------------------------------------------------------------
def jensen_shannon_distance(
    distribution1: Sequence[float], distribution2: Sequence[float]
) -> float:
    """Jensen–Shannon distance (sqrt of JS divergence), matching original code."""
    distribution1 = np.asarray(distribution1, dtype=float)
    distribution2 = np.asarray(distribution2, dtype=float)

    if np.sum(distribution1) == 0 or np.sum(distribution2) == 0:
        return 1.0

    distribution1 = distribution1 / np.sum(distribution1)
    distribution2 = distribution2 / np.sum(distribution2)
    avg = (distribution1 + distribution2) / 2.0
    js_divergence = (
        entropy(distribution1, avg) + entropy(distribution2, avg)
    ) / 2.0
    return float(np.sqrt(js_divergence))


def _pair_categories(cat_a: str, cat_b: str) -> tuple[str, str]:
    """
    Reproduce refrigerant_comparison.calculate_js_distances category pairing.

    The original loops ``for cat1 in categories: for cat2 in categories`` and
    skips when ``cat1 >= cat2`` (lexicographic). Histogram bin edges are taken
    from ``cat1``, so for refrigerant vs database_size_matched the edges come
    from the database set.
    """
    if cat_a < cat_b:
        return cat_a, cat_b
    if cat_b < cat_a:
        return cat_b, cat_a
    raise ValueError(f"Categories must differ: {cat_a!r}")


def js_distance_for_feature(
    df: pd.DataFrame,
    feature: str,
    category1: str,
    category2: str,
    *,
    categorical_threshold: int = CATEGORICAL_NUNIQUE_THRESHOLD,
    bins: int = HISTOGRAM_BINS,
) -> Optional[float]:
    """
    JS distance for one feature between two categories.

    NaN policy (original): ``dropna()`` per group; skip if either group has
    fewer than 2 finite values; skip if overall ``nunique() <= 1``.
    Discrete path when overall ``nunique() <= categorical_threshold``.
    """
    if feature not in df.columns:
        return None
    series = df[feature]
    if series.nunique(dropna=True) <= 1:
        return None

    cat1, cat2 = _pair_categories(category1, category2)
    data1 = df.loc[df["category"] == cat1, feature].dropna()
    data2 = df.loc[df["category"] == cat2, feature].dropna()
    if len(data1) < 2 or len(data2) < 2:
        return None

    # Replace ±inf with NaN then drop (defensive; original CSV has no infs)
    data1 = data1.replace([np.inf, -np.inf], np.nan).dropna()
    data2 = data2.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data1) < 2 or len(data2) < 2:
        return None

    if series.nunique(dropna=True) <= categorical_threshold:
        unique_values = series.dropna().unique()
        probs1 = [(data1 == val).sum() / len(data1) for val in unique_values]
        probs2 = [(data2 == val).sum() / len(data2) for val in unique_values]
        return jensen_shannon_distance(probs1, probs2)

    hist1, edges = np.histogram(data1, bins=bins, density=True)
    hist2, _ = np.histogram(data2, bins=edges, density=True)
    return jensen_shannon_distance(hist1, hist2)


def mannwhitney_for_feature(
    df: pd.DataFrame,
    feature: str,
    refrigerant_category: str = REFRIGERANT_CATEGORY,
    reference_category: str = REFERENCE_CATEGORY,
) -> tuple[Optional[float], Optional[float], int, int]:
    """
    Two-sided Mann–Whitney U (refrigerant vs reference), matching figure scripts.

    Returns (statistic, p_value, n_refrigerant, n_reference).
    """
    if feature not in df.columns:
        return None, None, 0, 0

    ref_data = (
        df.loc[df["category"] == refrigerant_category, feature]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    db_data = (
        df.loc[df["category"] == reference_category, feature]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    n_ref, n_db = int(len(ref_data)), int(len(db_data))
    if n_ref < 1 or n_db < 1:
        return None, None, n_ref, n_db

    # Identical constant samples → MW undefined / p=1 by convention
    if ref_data.nunique() == 1 and db_data.nunique() == 1 and ref_data.iloc[0] == db_data.iloc[0]:
        return float("nan"), 1.0, n_ref, n_db

    try:
        stat, p_value = mannwhitneyu(ref_data, db_data, alternative="two-sided")
        return float(stat), float(p_value), n_ref, n_db
    except ValueError:
        return None, None, n_ref, n_db


def descriptor_columns(df: pd.DataFrame) -> list[str]:
    """Numeric descriptor columns (excludes smiles/category/id annotations)."""
    skip = META_COLUMNS | {"refrigerant_index"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        cols.append(c)
    return cols


def compute_all_descriptor_metrics(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    refrigerant_category: str = REFRIGERANT_CATEGORY,
    reference_category: str = REFERENCE_CATEGORY,
) -> pd.DataFrame:
    """JS + Mann–Whitney for every descriptor; ranked by JS descending."""
    rows: list[dict[str, Any]] = []
    for feature in features:
        js = js_distance_for_feature(
            df, feature, refrigerant_category, reference_category
        )
        mw_stat, mw_p, n_ref, n_db = mannwhitney_for_feature(
            df, feature, refrigerant_category, reference_category
        )
        if js is None and mw_p is None:
            continue
        rows.append(
            {
                "Feature": feature,
                "Category1": reference_category
                if reference_category < refrigerant_category
                else refrigerant_category,
                "Category2": refrigerant_category
                if reference_category < refrigerant_category
                else reference_category,
                "JS_Distance": js if js is not None else np.nan,
                "MW_Statistic": mw_stat if mw_stat is not None else np.nan,
                "MW_pvalue": mw_p if mw_p is not None else np.nan,
                "n_refrigerant": n_ref,
                "n_reference": n_db,
                "significant_p005": bool(mw_p is not None and mw_p < DEFAULT_P_THRESHOLD),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        by=["JS_Distance", "Feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    out["JS_rank"] = np.arange(1, len(out) + 1)
    # Recompute significance with caller's threshold later if needed; default col uses 0.05
    return out


def apply_significance(
    metrics: pd.DataFrame, p_threshold: float
) -> pd.DataFrame:
    out = metrics.copy()
    out["significant"] = out["MW_pvalue"].lt(p_threshold).fillna(False)
    return out


# ---------------------------------------------------------------------------
# Data loading / identity
# ---------------------------------------------------------------------------
def _heavy_atom_count(smiles: str) -> Optional[int]:
    if Chem is None or Descriptors is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles).replace(" ", ""))
    if mol is None:
        return None
    return int(Descriptors.HeavyAtomCount(mol))


def load_compound_names(
    refrigerants_csv: Path,
    refrigerant_smiles: Sequence[str],
    *,
    heavy_atom_min: int = HEAVY_ATOM_MIN,
    heavy_atom_max: int = HEAVY_ATOM_MAX,
) -> list[str]:
    """
    Map the 21 profile SMILES to compound names via refrigerants.csv order.

    Profile fluids are the refrigerants.csv entries with heavy-atom count in
    [heavy_atom_min, heavy_atom_max], in file order — which matches
    ``combined_descriptors.csv`` refrigerant row order.
    """
    if not refrigerants_csv.is_file():
        logger.warning("Refrigerants CSV not found: %s", refrigerants_csv)
        return [f"unknown_{i:02d}" for i in range(len(refrigerant_smiles))]

    refs = pd.read_csv(refrigerants_csv)
    if "SMILES" not in refs.columns or "Compound" not in refs.columns:
        logger.warning("Unexpected refrigerants.csv columns: %s", list(refs.columns))
        return [f"unknown_{i:02d}" for i in range(len(refrigerant_smiles))]

    names: list[str] = []
    for _, row in refs.iterrows():
        hac = _heavy_atom_count(row["SMILES"])
        if hac is None:
            continue
        if heavy_atom_min <= hac <= heavy_atom_max:
            names.append(str(row["Compound"]))

    if len(names) != len(refrigerant_smiles):
        logger.warning(
            "Name list length %d != refrigerant count %d; falling back to ids",
            len(names),
            len(refrigerant_smiles),
        )
        return [f"refrigerant_{i:02d}" for i in range(len(refrigerant_smiles))]

    # Verify SMILES alignment (space-insensitive)
    refs_size = refs.copy()
    refs_size["_hac"] = refs_size["SMILES"].map(_heavy_atom_count)
    profile = refs_size[
        refs_size["_hac"].between(heavy_atom_min, heavy_atom_max)
    ].reset_index(drop=True)
    for i, (s_csv, s_comb) in enumerate(
        zip(profile["SMILES"], refrigerant_smiles)
    ):
        a = str(s_csv).replace(" ", "")
        b = str(s_comb).replace(" ", "")
        if a != b:
            logger.warning(
                "SMILES mismatch at profile index %d: csv=%r combined=%r",
                i,
                s_csv,
                s_comb,
            )
    return names


def load_combined_descriptors(
    path: Path,
    refrigerants_csv: Optional[Path] = None,
    *,
    heavy_atom_min: int = HEAVY_ATOM_MIN,
    heavy_atom_max: int = HEAVY_ATOM_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load combined descriptors; return (full_df, refrigerants_df, references_df).

    Refrigerant rows gain ``refrigerant_id``, ``compound``, and stable
    ``refrigerant_index`` (0..20 in file order).
    """
    df = pd.read_csv(path)
    if "category" not in df.columns or "smiles" not in df.columns:
        raise ValueError(
            f"{path} must contain 'smiles' and 'category' columns"
        )

    cats = set(df["category"].unique())
    if REFRIGERANT_CATEGORY not in cats or REFERENCE_CATEGORY not in cats:
        raise ValueError(
            f"Expected categories {REFRIGERANT_CATEGORY!r} and "
            f"{REFERENCE_CATEGORY!r}; found {sorted(cats)}"
        )

    refrigerants = (
        df[df["category"] == REFRIGERANT_CATEGORY]
        .copy()
        .reset_index(drop=True)
    )
    references = (
        df[df["category"] == REFERENCE_CATEGORY].copy().reset_index(drop=True)
    )

    if len(refrigerants) != 21:
        logger.warning(
            "Expected 21 refrigerants, found %d in %s",
            len(refrigerants),
            path,
        )
    if len(references) != 177:
        logger.warning(
            "Expected 177 size-matched references, found %d in %s",
            len(references),
            path,
        )

    names_path = refrigerants_csv or default_refrigerants_csv_path()
    names = load_compound_names(
        Path(names_path),
        refrigerants["smiles"].tolist(),
        heavy_atom_min=heavy_atom_min,
        heavy_atom_max=heavy_atom_max,
    )
    refrigerants["refrigerant_index"] = np.arange(len(refrigerants))
    refrigerants["compound"] = names[: len(refrigerants)]
    refrigerants["refrigerant_id"] = [
        f"{i:02d}_{name}" for i, name in enumerate(refrigerants["compound"])
    ]

    # Rebuild full frame with annotated refrigerants + unchanged references
    full = pd.concat([refrigerants, references], ignore_index=True)
    return full, refrigerants, references


# ---------------------------------------------------------------------------
# Summaries / near-substitutes
# ---------------------------------------------------------------------------
def fixed15_summary(
    metrics: pd.DataFrame,
    fixed_descriptors: Sequence[str],
    *,
    p_threshold: float,
    baseline_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    m = apply_significance(metrics, p_threshold)
    by_feat = m.set_index("Feature")

    rows = []
    for rank_in_list, feat in enumerate(fixed_descriptors, start=1):
        if feat not in by_feat.index:
            rows.append(
                {
                    "Feature": feat,
                    "fixed15_order": rank_in_list,
                    "JS_Distance": np.nan,
                    "JS_rank": np.nan,
                    "MW_Statistic": np.nan,
                    "MW_pvalue": np.nan,
                    "significant": False,
                    "passes_p": False,
                    "n_refrigerant": np.nan,
                    "n_reference": np.nan,
                }
            )
            continue
        r = by_feat.loc[feat]
        # If duplicate Feature rows (should not happen), take first
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append(
            {
                "Feature": feat,
                "fixed15_order": rank_in_list,
                "JS_Distance": float(r["JS_Distance"]),
                "JS_rank": int(r["JS_rank"]),
                "MW_Statistic": float(r["MW_Statistic"])
                if pd.notna(r["MW_Statistic"])
                else np.nan,
                "MW_pvalue": float(r["MW_pvalue"])
                if pd.notna(r["MW_pvalue"])
                else np.nan,
                "significant": bool(r["significant"]),
                "passes_p": bool(r["significant"]),
                "n_refrigerant": int(r["n_refrigerant"]),
                "n_reference": int(r["n_reference"]),
            }
        )

    summary = pd.DataFrame(rows)

    if baseline_metrics is not None and not baseline_metrics.empty:
        base = apply_significance(baseline_metrics, p_threshold).set_index(
            "Feature"
        )
        js_base, rank_base, p_base, pass_base = [], [], [], []
        for feat in summary["Feature"]:
            if feat in base.index:
                b = base.loc[feat]
                if isinstance(b, pd.DataFrame):
                    b = b.iloc[0]
                js_base.append(float(b["JS_Distance"]))
                rank_base.append(int(b["JS_rank"]))
                p_base.append(float(b["MW_pvalue"]) if pd.notna(b["MW_pvalue"]) else np.nan)
                pass_base.append(bool(b["significant"]))
            else:
                js_base.append(np.nan)
                rank_base.append(np.nan)
                p_base.append(np.nan)
                pass_base.append(False)
        summary["JS_Distance_baseline"] = js_base
        summary["JS_rank_baseline"] = rank_base
        summary["MW_pvalue_baseline"] = p_base
        summary["passes_p_baseline"] = pass_base
        summary["delta_JS"] = summary["JS_Distance"] - summary["JS_Distance_baseline"]
        summary["delta_JS_rank"] = summary["JS_rank"] - summary["JS_rank_baseline"]

    return summary


def near_substitutes(
    metrics: pd.DataFrame,
    fixed_descriptors: Sequence[str],
    *,
    p_threshold: float,
    top_n: int,
    js_mode: str = "min_fixed15",
) -> pd.DataFrame:
    """
    Descriptors that are significant and high-JS but not in the fixed 15.

    ``js_mode``:
      - ``min_fixed15``: JS >= minimum JS among fixed-15 that have finite JS
      - ``top_n``: among significant non-fixed, keep those with JS_rank <= top_n
      - ``both``: significant AND (JS >= min fixed15 OR JS_rank <= top_n)
    """
    m = apply_significance(metrics, p_threshold)
    fixed_set = set(fixed_descriptors)
    fixed_js = m.loc[m["Feature"].isin(fixed_set), "JS_Distance"].dropna()
    min_fixed_js = float(fixed_js.min()) if len(fixed_js) else np.nan

    cand = m[~m["Feature"].isin(fixed_set)].copy()
    cand = cand[cand["significant"]].copy()

    if js_mode == "min_fixed15":
        cand = cand[cand["JS_Distance"] >= min_fixed_js]
    elif js_mode == "top_n":
        cand = cand[cand["JS_rank"] <= top_n]
    elif js_mode == "both":
        mask = cand["JS_rank"] <= top_n
        if np.isfinite(min_fixed_js):
            mask = mask | (cand["JS_Distance"] >= min_fixed_js)
        cand = cand[mask]
    else:
        raise ValueError(f"Unknown js_mode: {js_mode}")

    cand = cand.sort_values(
        by=["JS_Distance", "Feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    cand["min_fixed15_JS"] = min_fixed_js
    cand["near_substitute_rule"] = js_mode
    return cand[
        [
            "Feature",
            "JS_Distance",
            "JS_rank",
            "MW_Statistic",
            "MW_pvalue",
            "significant",
            "n_refrigerant",
            "n_reference",
            "min_fixed15_JS",
            "near_substitute_rule",
        ]
    ]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "baseline": output_dir / "baseline",
        "folds": output_dir / "folds",
        "aggregate": output_dir / "aggregate",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_molecule_manifest(
    path: Path,
    refrigerants: pd.DataFrame,
    references: pd.DataFrame,
) -> None:
    payload = {
        "n_refrigerants": int(len(refrigerants)),
        "n_references": int(len(references)),
        "refrigerants": [
            {
                "refrigerant_index": int(r["refrigerant_index"]),
                "refrigerant_id": r["refrigerant_id"],
                "compound": r["compound"],
                "smiles": r["smiles"],
                "HeavyAtomCount": int(r["HeavyAtomCount"])
                if "HeavyAtomCount" in refrigerants.columns
                and pd.notna(r.get("HeavyAtomCount"))
                else None,
            }
            for _, r in refrigerants.iterrows()
        ],
        "references": [
            {
                "reference_index": int(i),
                "smiles": s,
                "HeavyAtomCount": int(hac)
                if "HeavyAtomCount" in references.columns and pd.notna(hac)
                else None,
            }
            for i, (s, hac) in enumerate(
                zip(
                    references["smiles"],
                    references["HeavyAtomCount"]
                    if "HeavyAtomCount" in references.columns
                    else [None] * len(references),
                )
            )
        ],
    }
    _write_json(path, payload)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def build_aggregates(
    fold_fixed15: list[pd.DataFrame],
    fold_meta: list[dict[str, Any]],
    near_sub_tables: list[pd.DataFrame],
    baseline_fixed15: pd.DataFrame,
    fixed_descriptors: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Build aggregate matrices and summaries across LOO folds."""
    n_folds = len(fold_fixed15)
    fold_ids = [m["fold_id"] for m in fold_meta]
    left_out = [m["left_out_compound"] for m in fold_meta]

    pass_mat = pd.DataFrame(index=fold_ids, columns=list(fixed_descriptors), dtype=object)
    js_mat = pd.DataFrame(index=fold_ids, columns=list(fixed_descriptors), dtype=float)
    rank_mat = pd.DataFrame(index=fold_ids, columns=list(fixed_descriptors), dtype=float)
    p_mat = pd.DataFrame(index=fold_ids, columns=list(fixed_descriptors), dtype=float)

    pass_mat.index.name = "fold_id"
    js_mat.index.name = "fold_id"
    rank_mat.index.name = "fold_id"
    p_mat.index.name = "fold_id"

    for fid, table in zip(fold_ids, fold_fixed15):
        t = table.set_index("Feature")
        for feat in fixed_descriptors:
            if feat not in t.index:
                pass_mat.loc[fid, feat] = False
                js_mat.loc[fid, feat] = np.nan
                rank_mat.loc[fid, feat] = np.nan
                p_mat.loc[fid, feat] = np.nan
                continue
            row = t.loc[feat]
            pass_mat.loc[fid, feat] = bool(row["passes_p"])
            js_mat.loc[fid, feat] = float(row["JS_Distance"])
            rank_mat.loc[fid, feat] = float(row["JS_rank"])
            p_mat.loc[fid, feat] = float(row["MW_pvalue"]) if pd.notna(row["MW_pvalue"]) else np.nan

    # Annotate left-out compound as a column
    for mat in (pass_mat, js_mat, rank_mat, p_mat):
        mat.insert(0, "left_out_compound", left_out)
        mat.insert(1, "left_out_smiles", [m["left_out_smiles"] for m in fold_meta])
        mat.insert(2, "left_out_refrigerant_id", [m["left_out_refrigerant_id"] for m in fold_meta])

    # Per-descriptor LOO summary
    base = baseline_fixed15.set_index("Feature")
    summary_rows = []
    for feat in fixed_descriptors:
        js_vals = js_mat[feat].astype(float)
        rank_vals = rank_mat[feat].astype(float)
        pass_vals = pass_mat[feat].astype(bool)
        p_vals = p_mat[feat].astype(float)
        b = base.loc[feat] if feat in base.index else None
        summary_rows.append(
            {
                "Feature": feat,
                "baseline_JS": float(b["JS_Distance"]) if b is not None else np.nan,
                "baseline_JS_rank": int(b["JS_rank"]) if b is not None else np.nan,
                "baseline_MW_pvalue": float(b["MW_pvalue"])
                if b is not None and pd.notna(b["MW_pvalue"])
                else np.nan,
                "baseline_passes_p": bool(b["passes_p"]) if b is not None else False,
                "loo_pass_count": int(pass_vals.sum()),
                "loo_pass_rate": float(pass_vals.mean()),
                "loo_JS_mean": float(js_vals.mean()),
                "loo_JS_std": float(js_vals.std(ddof=0)),
                "loo_JS_min": float(js_vals.min()),
                "loo_JS_max": float(js_vals.max()),
                "loo_JS_mean_delta_vs_baseline": float(
                    js_vals.mean() - float(b["JS_Distance"])
                )
                if b is not None
                else np.nan,
                "loo_rank_mean": float(rank_vals.mean()),
                "loo_rank_std": float(rank_vals.std(ddof=0)),
                "loo_rank_min": float(rank_vals.min()),
                "loo_rank_max": float(rank_vals.max()),
                "loo_rank_mean_delta_vs_baseline": float(
                    rank_vals.mean() - float(b["JS_rank"])
                )
                if b is not None
                else np.nan,
                "loo_pvalue_max": float(p_vals.max()),
                "loo_pvalue_min": float(p_vals.min()),
                "fold_min_JS": str(js_vals.idxmin()) if n_folds else "",
                "fold_max_JS": str(js_vals.idxmax()) if n_folds else "",
                "fold_max_pvalue": str(p_vals.idxmax()) if n_folds else "",
            }
        )
    fixed15_loo_summary = pd.DataFrame(summary_rows)

    # Near-substitute frequency
    freq: dict[str, dict[str, Any]] = {}
    for meta, ns in zip(fold_meta, near_sub_tables):
        for _, row in ns.iterrows():
            feat = row["Feature"]
            slot = freq.setdefault(
                feat,
                {
                    "Feature": feat,
                    "n_folds_as_near_substitute": 0,
                    "frequency": 0.0,
                    "JS_mean": [],
                    "JS_rank_mean": [],
                    "MW_pvalue_mean": [],
                    "example_folds": [],
                },
            )
            slot["n_folds_as_near_substitute"] += 1
            slot["JS_mean"].append(float(row["JS_Distance"]))
            slot["JS_rank_mean"].append(float(row["JS_rank"]))
            slot["MW_pvalue_mean"].append(float(row["MW_pvalue"]))
            if len(slot["example_folds"]) < 5:
                slot["example_folds"].append(meta["fold_id"])

    freq_rows = []
    for feat, slot in freq.items():
        freq_rows.append(
            {
                "Feature": feat,
                "n_folds_as_near_substitute": slot["n_folds_as_near_substitute"],
                "frequency": slot["n_folds_as_near_substitute"] / max(n_folds, 1),
                "JS_mean": float(np.mean(slot["JS_mean"])),
                "JS_rank_mean": float(np.mean(slot["JS_rank_mean"])),
                "MW_pvalue_mean": float(np.mean(slot["MW_pvalue_mean"])),
                "example_folds": ";".join(slot["example_folds"]),
            }
        )
    near_freq = pd.DataFrame(freq_rows)
    if not near_freq.empty:
        near_freq = near_freq.sort_values(
            by=["n_folds_as_near_substitute", "JS_mean", "Feature"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    return {
        "fixed15_loo_summary": fixed15_loo_summary,
        "fixed15_pass_matrix": pass_mat.reset_index(),
        "fixed15_js_matrix": js_mat.reset_index(),
        "fixed15_rank_matrix": rank_mat.reset_index(),
        "fixed15_pvalue_matrix": p_mat.reset_index(),
        "near_substitutes_frequency": near_freq,
    }


# ---------------------------------------------------------------------------
# Main LOO runner
# ---------------------------------------------------------------------------
def run_loo_analysis(
    combined_path: Path,
    output_dir: Path,
    *,
    refrigerants_csv: Optional[Path] = None,
    p_threshold: float = DEFAULT_P_THRESHOLD,
    near_substitute_top_n: int = DEFAULT_NEAR_SUB_TOP_N,
    near_substitute_js_mode: str = "both",
    heavy_atom_min: int = HEAVY_ATOM_MIN,
    heavy_atom_max: int = HEAVY_ATOM_MAX,
    fixed_descriptors: Sequence[str] = FIXED_15,
    seed: int = 0,
) -> dict[str, Any]:
    paths = _ensure_dirs(output_dir)
    full, refrigerants, references = load_combined_descriptors(
        combined_path,
        refrigerants_csv,
        heavy_atom_min=heavy_atom_min,
        heavy_atom_max=heavy_atom_max,
    )
    features = descriptor_columns(full)
    missing_fixed = [f for f in fixed_descriptors if f not in features]
    if missing_fixed:
        raise ValueError(f"Fixed descriptors missing from data: {missing_fixed}")

    save_molecule_manifest(
        paths["root"] / "molecule_manifest.json", refrigerants, references
    )

    # --- Baseline (all 21) ---
    logger.info(
        "Computing baseline metrics: %d refrigerants vs %d references, %d descriptors",
        len(refrigerants),
        len(references),
        len(features),
    )
    drop_ann = [
        c
        for c in ("refrigerant_index", "compound", "refrigerant_id")
        if c in refrigerants.columns
    ]
    baseline_df = pd.concat(
        [refrigerants.drop(columns=drop_ann, errors="ignore"), references],
        ignore_index=True,
    )
    baseline_metrics = compute_all_descriptor_metrics(baseline_df, features)
    baseline_metrics = apply_significance(baseline_metrics, p_threshold)
    baseline_metrics["significant_p005"] = baseline_metrics["MW_pvalue"].lt(
        DEFAULT_P_THRESHOLD
    ).fillna(False)
    baseline_fixed = fixed15_summary(
        baseline_metrics, fixed_descriptors, p_threshold=p_threshold
    )
    baseline_metrics.to_csv(
        paths["baseline"] / "js_mannwhitney_all_descriptors.csv", index=False
    )
    baseline_fixed.to_csv(paths["baseline"] / "fixed15_summary.csv", index=False)
    near_substitutes(
        baseline_metrics,
        fixed_descriptors,
        p_threshold=p_threshold,
        top_n=near_substitute_top_n,
        js_mode=near_substitute_js_mode,
    ).to_csv(paths["baseline"] / "near_substitutes.csv", index=False)

    fold_fixed15_tables: list[pd.DataFrame] = []
    fold_meta: list[dict[str, Any]] = []
    near_sub_tables: list[pd.DataFrame] = []

    # --- LOO folds ---
    for _, row in refrigerants.iterrows():
        left_id = str(row["refrigerant_id"])
        fold_name = f"fold_{int(row['refrigerant_index']):02d}_{row['compound']}"
        # Sanitize folder name
        fold_name = (
            fold_name.replace("/", "-")
            .replace("\\", "-")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        fold_dir = paths["folds"] / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        keep_mask = refrigerants["refrigerant_index"] != row["refrigerant_index"]
        train_refs = refrigerants.loc[keep_mask].copy()
        # Drop annotation columns that are not descriptors / category / smiles
        drop_cols = [
            c
            for c in ("refrigerant_index", "compound", "refrigerant_id")
            if c in train_refs.columns
        ]
        train_for_metrics = train_refs.drop(columns=drop_cols, errors="ignore")
        fold_df = pd.concat([train_for_metrics, references], ignore_index=True)

        logger.info(
            "Fold %s: leave out %s (%s); train n=%d",
            fold_name,
            left_id,
            row["smiles"],
            len(train_refs),
        )

        left_out = {
            "fold_id": fold_name,
            "fold_index": int(row["refrigerant_index"]),
            "left_out_refrigerant_id": left_id,
            "left_out_compound": row["compound"],
            "left_out_smiles": row["smiles"],
            "left_out_refrigerant_index": int(row["refrigerant_index"]),
            "n_train_refrigerants": int(len(train_refs)),
            "n_references": int(len(references)),
            "train_refrigerant_ids": train_refs["refrigerant_id"].tolist(),
            "train_compounds": train_refs["compound"].tolist(),
            "train_smiles": train_refs["smiles"].tolist(),
        }
        _write_json(fold_dir / "left_out.json", left_out)

        metrics = compute_all_descriptor_metrics(fold_df, features)
        metrics = apply_significance(metrics, p_threshold)
        metrics.to_csv(
            fold_dir / "js_mannwhitney_all_descriptors.csv", index=False
        )

        fixed_sum = fixed15_summary(
            metrics,
            fixed_descriptors,
            p_threshold=p_threshold,
            baseline_metrics=baseline_metrics,
        )
        fixed_sum.to_csv(fold_dir / "fixed15_summary.csv", index=False)

        ns = near_substitutes(
            metrics,
            fixed_descriptors,
            p_threshold=p_threshold,
            top_n=near_substitute_top_n,
            js_mode=near_substitute_js_mode,
        )
        ns.to_csv(fold_dir / "near_substitutes.csv", index=False)

        fold_fixed15_tables.append(fixed_sum)
        fold_meta.append(left_out)
        near_sub_tables.append(ns)

    aggregates = build_aggregates(
        fold_fixed15_tables,
        fold_meta,
        near_sub_tables,
        baseline_fixed,
        fixed_descriptors,
    )
    for name, table in aggregates.items():
        table.to_csv(paths["aggregate"] / f"{name}.csv", index=False)

    config = RunConfig(
        combined_descriptors=str(combined_path.resolve()),
        refrigerants_csv=str(Path(refrigerants_csv).resolve())
        if refrigerants_csv
        else str(default_refrigerants_csv_path()),
        output_dir=str(output_dir.resolve()),
        p_threshold=p_threshold,
        near_substitute_top_n=near_substitute_top_n,
        near_substitute_js_mode=near_substitute_js_mode,
        heavy_atom_min=heavy_atom_min,
        heavy_atom_max=heavy_atom_max,
        fixed_15=list(fixed_descriptors),
        n_refrigerants=int(len(refrigerants)),
        n_references=int(len(references)),
        n_descriptors=int(len(features)),
        categorical_nunique_threshold=CATEGORICAL_NUNIQUE_THRESHOLD,
        histogram_bins=HISTOGRAM_BINS,
        seed=seed,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(paths["root"] / "run_config.json", asdict(config))

    # Console highlight
    summary = aggregates["fixed15_loo_summary"]
    logger.info("=== Fixed-15 LOO pass rates ===")
    for _, r in summary.iterrows():
        logger.info(
            "  %s: pass_rate=%.2f  JS_mean=%.4f (Δ=%.4f)  rank_mean=%.1f (Δ=%+.1f)",
            r["Feature"],
            r["loo_pass_rate"],
            r["loo_JS_mean"],
            r["loo_JS_mean_delta_vs_baseline"],
            r["loo_rank_mean"],
            r["loo_rank_mean_delta_vs_baseline"],
        )

    return {
        "config": config,
        "baseline_fixed15": baseline_fixed,
        "aggregates": aggregates,
        "n_folds": len(fold_meta),
    }


def write_readme(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = """# LOO descriptor-metrics robustness (profile fluids)

Leave-one-out analysis of Jensen–Shannon distances and Mann–Whitney U tests for
the working-fluid profile descriptors.

## What this is

- **Profile set:** 21 size-matched working fluids (heavy atoms 5–10)
- **Reference set:** 177 size-matched database molecules (fixed across folds)
- **Per fold:** drop one fluid → recompute JS + MW for all ~170 descriptors vs the 177
- **Fixed 15:** paper-curated descriptors; report pass/fail at `p < 0.05`, JS, rank, deltas vs full-21 baseline
- **Near-substitutes:** other descriptors that are significant and high-JS (see `run_config.json`)

This experiment does **not** recompute mean profiles or candidate rankings.

## How to run

From the `RefrigerantFinding` repo root:

```bash
python refrigerant_finding/profile_robustness/loo_descriptor_metrics.py
```

Useful flags:

```bash
python refrigerant_finding/profile_robustness/loo_descriptor_metrics.py \\
  --combined-descriptors refrigerant_finding/statistical_comparison/combined_descriptors.csv \\
  --refrigerants-csv refrigerant_finding/refrigerants.csv \\
  --output-dir refrigerant_finding/profile_robustness/loo_on_profile \\
  --p-threshold 0.05 \\
  --near-substitute-top-n 30 \\
  --near-substitute-js-mode both
```

## Methodology (matched to original pipeline)

Implements the same JS procedure as `refrigerant_finding/refrigerant_comparison.py`:

1. Per descriptor, drop NaNs independently in each group; require ≥2 values/side
2. If overall `nunique() <= 20` → discrete probability mass over shared unique values
3. Else → 30-bin histograms with `density=True`; **bin edges from the lexicographically
   first category** (`database_size_matched` before `refrigerant`)
4. JS distance = `sqrt(JS divergence)` via `scipy.stats.entropy`
5. Mann–Whitney U: two-sided (`scipy.stats.mannwhitneyu`), refrigerant vs reference

## Layout

```
loo_on_profile/
  README.md
  run_config.json
  molecule_manifest.json          # the exact 21 + 177 identifiers
  baseline/
    js_mannwhitney_all_descriptors.csv
    fixed15_summary.csv
    near_substitutes.csv
  folds/
    fold_00_<compound>/
      left_out.json
      js_mannwhitney_all_descriptors.csv
      fixed15_summary.csv
      near_substitutes.csv
  aggregate/
    fixed15_loo_summary.csv
    fixed15_pass_matrix.csv
    fixed15_js_matrix.csv
    fixed15_rank_matrix.csv
    fixed15_pvalue_matrix.csv
    near_substitutes_frequency.csv
```

## Inputs

- `statistical_comparison/combined_descriptors.csv` — must contain categories
  `refrigerant` (21) and `database_size_matched` (177)
- `refrigerants.csv` — used only to attach compound names to the 21 profile fluids
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Leave-one-out robustness for descriptor JS distances and "
            "Mann–Whitney tests (21 working fluids vs 177 size-matched refs)."
        )
    )
    p.add_argument(
        "--combined-descriptors",
        type=Path,
        default=default_combined_descriptors_path(),
        help="Path to combined_descriptors.csv",
    )
    p.add_argument(
        "--refrigerants-csv",
        type=Path,
        default=default_refrigerants_csv_path(),
        help="Path to refrigerants.csv (compound names)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Output root for loo_on_profile results",
    )
    p.add_argument(
        "--p-threshold",
        type=float,
        default=DEFAULT_P_THRESHOLD,
        help="Mann–Whitney significance threshold (default: 0.05)",
    )
    p.add_argument(
        "--near-substitute-top-n",
        type=int,
        default=DEFAULT_NEAR_SUB_TOP_N,
        help="JS_rank cutoff for near-substitute 'top_n' / 'both' modes",
    )
    p.add_argument(
        "--near-substitute-js-mode",
        choices=["min_fixed15", "top_n", "both"],
        default="both",
        help=(
            "Near-substitute rule: JS >= min(fixed15 JS), JS_rank <= top_n, or either"
        ),
    )
    p.add_argument(
        "--heavy-atom-min",
        type=int,
        default=HEAVY_ATOM_MIN,
        help="Min heavy atoms for naming map from refrigerants.csv",
    )
    p.add_argument(
        "--heavy-atom-max",
        type=int,
        default=HEAVY_ATOM_MAX,
        help="Max heavy atoms for naming map from refrigerants.csv",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reserved for reproducibility metadata (analysis is deterministic)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.combined_descriptors.is_file():
        logger.error("Combined descriptors not found: %s", args.combined_descriptors)
        return 1

    write_readme(args.output_dir)
    result = run_loo_analysis(
        args.combined_descriptors,
        args.output_dir,
        refrigerants_csv=args.refrigerants_csv,
        p_threshold=args.p_threshold,
        near_substitute_top_n=args.near_substitute_top_n,
        near_substitute_js_mode=args.near_substitute_js_mode,
        heavy_atom_min=args.heavy_atom_min,
        heavy_atom_max=args.heavy_atom_max,
        fixed_descriptors=FIXED_15,
        seed=args.seed,
    )
    logger.info(
        "Done. %d folds written under %s",
        result["n_folds"],
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
