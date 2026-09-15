#!/usr/bin/env python3
"""
Publication-oriented aggregation of already-computed LOO descriptor metrics.

Reads ``loo_on_profile/`` (baseline + 21 folds + optional pre-built aggregate
CSVs) and writes refined tables, figures, and a short SUMMARY for SI / rebuttal.

Does **not** re-run LOO metrics or mean-profile / candidate-ranking LOO.

Typical usage (from repo root)::

    python3 refrigerant_finding/profile_robustness/aggregate_loo_descriptor_metrics.py

    python3 refrigerant_finding/profile_robustness/aggregate_loo_descriptor_metrics.py \\
        --loo-dir refrigerant_finding/profile_robustness/loo_on_profile \\
        --output-dir refrigerant_finding/profile_robustness/loo_on_profile/aggregation
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

try:
    from refrigerant_finding.profile_robustness.loo_descriptor_metrics import (
        FIXED_15,
        DEFAULT_P_THRESHOLD,
    )
    from refrigerant_finding.profile_robustness.descriptor_taxonomy import (
        CATEGORY_SHORT,
        category_block_boundaries,
        fixed15_and_near_category_table,
        sort_descriptors_by_category,
    )
except ImportError:  # pragma: no cover - script run as file
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from refrigerant_finding.profile_robustness.loo_descriptor_metrics import (
        FIXED_15,
        DEFAULT_P_THRESHOLD,
    )
    from refrigerant_finding.profile_robustness.descriptor_taxonomy import (
        CATEGORY_SHORT,
        category_block_boundaries,
        fixed15_and_near_category_table,
        sort_descriptors_by_category,
    )

logger = logging.getLogger(__name__)

META_COLS = {
    "fold_id",
    "left_out_compound",
    "left_out_smiles",
    "left_out_refrigerant_id",
}

DEFAULT_NEAR_SUB_MIN_FREQUENCY = 0.9
DEFAULT_HIGH_CORR_THRESHOLD = 0.7
DEFAULT_LOW_CORR_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Paths / styling
# ---------------------------------------------------------------------------
def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def default_loo_dir() -> Path:
    return _package_dir() / "loo_on_profile"


def default_output_dir(loo_dir: Optional[Path] = None) -> Path:
    return (loo_dir or default_loo_dir()) / "aggregation"


def default_combined_descriptors_path() -> Path:
    return (
        _package_dir().parent
        / "statistical_comparison"
        / "combined_descriptors.csv"
    )


def configure_matplotlib() -> None:
    """Publication-oriented typography (Nimbus Sans when available)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Nimbus Sans",
                "Nimbus Sans L",
                "DejaVu Sans",
                "Helvetica",
                "Arial",
            ],
            "mathtext.fontset": "dejavusans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


@dataclass
class AggregationMeta:
    loo_dir: str
    output_dir: str
    n_folds: int
    n_fixed15: int
    p_threshold: float
    timestamp_utc: str
    source_run_config: Optional[str]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_run_config(loo_dir: Path) -> dict[str, Any]:
    path = loo_dir / "run_config.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_fold_dirs(loo_dir: Path) -> list[Path]:
    folds_root = loo_dir / "folds"
    if not folds_root.is_dir():
        raise FileNotFoundError(f"Missing folds directory: {folds_root}")
    dirs = sorted(
        p for p in folds_root.iterdir() if p.is_dir() and p.name.startswith("fold_")
    )
    if not dirs:
        raise FileNotFoundError(f"No fold_* directories under {folds_root}")
    return dirs


def load_fold_tables(
    loo_dir: Path,
    fixed_descriptors: Sequence[str],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    """Load left_out.json, fixed15_summary, near_substitutes for each fold."""
    metas: list[dict[str, Any]] = []
    fixed_tables: list[pd.DataFrame] = []
    near_tables: list[pd.DataFrame] = []

    for fold_dir in discover_fold_dirs(loo_dir):
        left_path = fold_dir / "left_out.json"
        fixed_path = fold_dir / "fixed15_summary.csv"
        near_path = fold_dir / "near_substitutes.csv"
        for required in (left_path, fixed_path):
            if not required.is_file():
                raise FileNotFoundError(f"Missing required fold file: {required}")

        meta = json.loads(left_path.read_text(encoding="utf-8"))
        if "fold_id" not in meta:
            meta["fold_id"] = fold_dir.name

        fixed = pd.read_csv(fixed_path)
        missing = [f for f in fixed_descriptors if f not in set(fixed["Feature"])]
        if missing:
            raise ValueError(f"{fold_dir.name}: missing fixed-15 rows: {missing}")

        near = (
            pd.read_csv(near_path)
            if near_path.is_file()
            else pd.DataFrame(columns=["Feature"])
        )

        metas.append(meta)
        fixed_tables.append(fixed)
        near_tables.append(near)

    return metas, fixed_tables, near_tables


def load_baseline_fixed15(loo_dir: Path) -> pd.DataFrame:
    path = loo_dir / "baseline" / "fixed15_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing baseline fixed15 summary: {path}")
    return pd.read_csv(path)


def _descriptor_matrix_from_folds(
    metas: Sequence[dict[str, Any]],
    fixed_tables: Sequence[pd.DataFrame],
    fixed_descriptors: Sequence[str],
    value_col: str,
) -> pd.DataFrame:
    rows = []
    for meta, table in zip(metas, fixed_tables):
        by_feat = table.set_index("Feature")
        row: dict[str, Any] = {
            "fold_id": meta["fold_id"],
            "left_out_compound": meta.get("left_out_compound", ""),
            "left_out_smiles": meta.get("left_out_smiles", ""),
            "left_out_refrigerant_id": meta.get("left_out_refrigerant_id", ""),
        }
        for feat in fixed_descriptors:
            val = by_feat.loc[feat, value_col]
            row[feat] = val
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def build_stability_table(
    metas: Sequence[dict[str, Any]],
    fixed_tables: Sequence[pd.DataFrame],
    baseline_fixed: pd.DataFrame,
    fixed_descriptors: Sequence[str],
    p_threshold: float,
) -> pd.DataFrame:
    """Per-descriptor stability: pass rate, JS/rank mean±std / min–max, Δs."""
    n_folds = len(metas)
    base = baseline_fixed.set_index("Feature")
    rows = []

    for feat in fixed_descriptors:
        js_vals = []
        rank_vals = []
        p_vals = []
        pass_vals = []
        delta_js = []
        delta_rank = []
        for table in fixed_tables:
            r = table.loc[table["Feature"] == feat].iloc[0]
            js_vals.append(float(r["JS_Distance"]))
            rank_vals.append(float(r["JS_rank"]))
            p_vals.append(float(r["MW_pvalue"]))
            passes = bool(r["passes_p"]) if "passes_p" in r.index else (
                float(r["MW_pvalue"]) < p_threshold
            )
            pass_vals.append(passes)
            if "delta_JS" in r.index and pd.notna(r["delta_JS"]):
                delta_js.append(float(r["delta_JS"]))
            else:
                delta_js.append(float(r["JS_Distance"]) - float(base.loc[feat, "JS_Distance"]))
            if "delta_JS_rank" in r.index and pd.notna(r["delta_JS_rank"]):
                delta_rank.append(float(r["delta_JS_rank"]))
            else:
                delta_rank.append(float(r["JS_rank"]) - float(base.loc[feat, "JS_rank"]))

        js_arr = np.asarray(js_vals, dtype=float)
        rank_arr = np.asarray(rank_vals, dtype=float)
        p_arr = np.asarray(p_vals, dtype=float)
        djs_arr = np.asarray(delta_js, dtype=float)
        drank_arr = np.asarray(delta_rank, dtype=float)
        b = base.loc[feat]

        rows.append(
            {
                "Feature": feat,
                "baseline_JS": float(b["JS_Distance"]),
                "baseline_JS_rank": int(b["JS_rank"]),
                "baseline_MW_pvalue": float(b["MW_pvalue"]),
                "baseline_passes_p": bool(b["passes_p"])
                if "passes_p" in b.index
                else float(b["MW_pvalue"]) < p_threshold,
                "loo_pass_count": int(sum(pass_vals)),
                "loo_pass_rate": float(np.mean(pass_vals)),
                "loo_JS_mean": float(js_arr.mean()),
                "loo_JS_std": float(js_arr.std(ddof=0)),
                "loo_JS_min": float(js_arr.min()),
                "loo_JS_max": float(js_arr.max()),
                "loo_JS_mean_delta_vs_baseline": float(
                    js_arr.mean() - float(b["JS_Distance"])
                ),
                "mean_abs_delta_JS": float(np.mean(np.abs(djs_arr))),
                "max_abs_delta_JS": float(np.max(np.abs(djs_arr))),
                "loo_rank_mean": float(rank_arr.mean()),
                "loo_rank_std": float(rank_arr.std(ddof=0)),
                "loo_rank_min": float(rank_arr.min()),
                "loo_rank_max": float(rank_arr.max()),
                "loo_rank_mean_delta_vs_baseline": float(
                    rank_arr.mean() - float(b["JS_rank"])
                ),
                "mean_abs_delta_rank": float(np.mean(np.abs(drank_arr))),
                "max_abs_delta_rank": float(np.max(np.abs(drank_arr))),
                "loo_pvalue_min": float(p_arr.min()),
                "loo_pvalue_max": float(p_arr.max()),
                "n_folds": n_folds,
            }
        )

    out = pd.DataFrame(rows)
    out["fixed15_order"] = out["Feature"].map(
        {f: i + 1 for i, f in enumerate(fixed_descriptors)}
    )
    return out


def build_extremes_table(
    metas: Sequence[dict[str, Any]],
    fixed_tables: Sequence[pd.DataFrame],
    baseline_fixed: pd.DataFrame,
    fixed_descriptors: Sequence[str],
) -> pd.DataFrame:
    """Per descriptor: which left-out fluid caused max |ΔJS|, max p, max |Δrank|."""
    base = baseline_fixed.set_index("Feature")
    rows = []

    for feat in fixed_descriptors:
        records = []
        for meta, table in zip(metas, fixed_tables):
            r = table.loc[table["Feature"] == feat].iloc[0]
            js = float(r["JS_Distance"])
            rank = float(r["JS_rank"])
            pval = float(r["MW_pvalue"])
            b_js = float(base.loc[feat, "JS_Distance"])
            b_rank = float(base.loc[feat, "JS_rank"])
            d_js = (
                float(r["delta_JS"])
                if "delta_JS" in r.index and pd.notna(r["delta_JS"])
                else js - b_js
            )
            d_rank = (
                float(r["delta_JS_rank"])
                if "delta_JS_rank" in r.index and pd.notna(r["delta_JS_rank"])
                else rank - b_rank
            )
            records.append(
                {
                    "fold_id": meta["fold_id"],
                    "left_out": meta.get("left_out_compound", ""),
                    "js": js,
                    "rank": rank,
                    "pval": pval,
                    "delta_js": d_js,
                    "abs_delta_js": abs(d_js),
                    "delta_rank": d_rank,
                    "abs_delta_rank": abs(d_rank),
                }
            )

        df = pd.DataFrame(records)
        i_js = int(df["abs_delta_js"].idxmax())
        i_p = int(df["pval"].idxmax())
        i_rank = int(df["abs_delta_rank"].idxmax())
        r_js, r_p, r_rank = df.loc[i_js], df.loc[i_p], df.loc[i_rank]

        rows.append(
            {
                "Feature": feat,
                "max_abs_delta_JS": float(r_js["abs_delta_js"]),
                "fold_max_abs_delta_JS": r_js["fold_id"],
                "left_out_max_abs_delta_JS": r_js["left_out"],
                "delta_JS_at_max": float(r_js["delta_js"]),
                "JS_at_max_abs_delta": float(r_js["js"]),
                "max_pvalue": float(r_p["pval"]),
                "fold_max_pvalue": r_p["fold_id"],
                "left_out_max_pvalue": r_p["left_out"],
                "max_abs_delta_rank": float(r_rank["abs_delta_rank"]),
                "fold_max_abs_delta_rank": r_rank["fold_id"],
                "left_out_max_abs_delta_rank": r_rank["left_out"],
                "delta_rank_at_max": float(r_rank["delta_rank"]),
                "rank_at_max_abs_delta": float(r_rank["rank"]),
            }
        )

    return pd.DataFrame(rows)


def build_near_substitutes_table(
    metas: Sequence[dict[str, Any]],
    near_tables: Sequence[pd.DataFrame],
    fixed_descriptors: Sequence[str],
) -> pd.DataFrame:
    n_folds = len(metas)
    fixed_set = set(fixed_descriptors)
    freq: dict[str, dict[str, Any]] = {}

    for meta, ns in zip(metas, near_tables):
        if ns.empty or "Feature" not in ns.columns:
            continue
        for _, row in ns.iterrows():
            feat = str(row["Feature"])
            if feat in fixed_set:
                continue  # should not appear, but guard
            slot = freq.setdefault(
                feat,
                {
                    "Feature": feat,
                    "n_folds_as_near_substitute": 0,
                    "JS_vals": [],
                    "rank_vals": [],
                    "p_vals": [],
                    "example_folds": [],
                    "example_left_out": [],
                },
            )
            slot["n_folds_as_near_substitute"] += 1
            slot["JS_vals"].append(float(row["JS_Distance"]))
            slot["rank_vals"].append(float(row["JS_rank"]))
            slot["p_vals"].append(float(row["MW_pvalue"]))
            if len(slot["example_folds"]) < 5:
                slot["example_folds"].append(meta["fold_id"])
                slot["example_left_out"].append(meta.get("left_out_compound", ""))

    rows = []
    for feat, slot in freq.items():
        rows.append(
            {
                "Feature": feat,
                "in_curated_fixed15": False,
                "n_folds_as_near_substitute": slot["n_folds_as_near_substitute"],
                "frequency": slot["n_folds_as_near_substitute"] / max(n_folds, 1),
                "JS_mean": float(np.mean(slot["JS_vals"])),
                "JS_std": float(np.std(slot["JS_vals"], ddof=0)),
                "JS_rank_mean": float(np.mean(slot["rank_vals"])),
                "MW_pvalue_mean": float(np.mean(slot["p_vals"])),
                "MW_pvalue_max": float(np.max(slot["p_vals"])),
                "example_folds": ";".join(slot["example_folds"]),
                "example_left_out": ";".join(slot["example_left_out"]),
                "note": "NOT in curated fixed-15; persistent high-JS significant alternative",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        by=["n_folds_as_near_substitute", "JS_mean", "Feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_js_delta_long(
    metas: Sequence[dict[str, Any]],
    fixed_tables: Sequence[pd.DataFrame],
    baseline_fixed: pd.DataFrame,
    fixed_descriptors: Sequence[str],
) -> pd.DataFrame:
    """Long-form 21×15 ΔJS / Δrank / p for heatmaps and distributions."""
    base = baseline_fixed.set_index("Feature")
    rows = []
    for meta, table in zip(metas, fixed_tables):
        for feat in fixed_descriptors:
            r = table.loc[table["Feature"] == feat].iloc[0]
            b_js = float(base.loc[feat, "JS_Distance"])
            b_rank = float(base.loc[feat, "JS_rank"])
            js = float(r["JS_Distance"])
            rank = float(r["JS_rank"])
            d_js = (
                float(r["delta_JS"])
                if "delta_JS" in r.index and pd.notna(r["delta_JS"])
                else js - b_js
            )
            d_rank = (
                float(r["delta_JS_rank"])
                if "delta_JS_rank" in r.index and pd.notna(r["delta_JS_rank"])
                else rank - b_rank
            )
            rows.append(
                {
                    "fold_id": meta["fold_id"],
                    "left_out_compound": meta.get("left_out_compound", ""),
                    "Feature": feat,
                    "JS": js,
                    "JS_baseline": b_js,
                    "delta_JS": d_js,
                    "JS_rank": rank,
                    "JS_rank_baseline": b_rank,
                    "delta_rank": d_rank,
                    "MW_pvalue": float(r["MW_pvalue"]),
                    "passes_p": bool(r["passes_p"])
                    if "passes_p" in r.index
                    else True,
                }
            )
    return pd.DataFrame(rows)


def build_headline(
    stability: pd.DataFrame,
    extremes: pd.DataFrame,
    near_freq: pd.DataFrame,
    n_folds: int,
    p_threshold: float,
) -> dict[str, Any]:
    all_pass = bool((stability["loo_pass_rate"] == 1.0).all())
    max_p = float(stability["loo_pvalue_max"].max())
    mean_abs_djs = float(stability["mean_abs_delta_JS"].mean())
    max_abs_djs = float(stability["max_abs_delta_JS"].max())
    feat_max_djs = str(
        stability.loc[stability["max_abs_delta_JS"].idxmax(), "Feature"]
    )
    mean_abs_drank = float(stability["mean_abs_delta_rank"].mean())
    max_abs_drank = float(stability["max_abs_delta_rank"].max())
    feat_max_drank = str(
        stability.loc[stability["max_abs_delta_rank"].idxmax(), "Feature"]
    )

    # Overall fold that most often drives extremes
    left_out_counts: dict[str, int] = {}
    for col in (
        "left_out_max_abs_delta_JS",
        "left_out_max_pvalue",
        "left_out_max_abs_delta_rank",
    ):
        for v in extremes[col]:
            left_out_counts[str(v)] = left_out_counts.get(str(v), 0) + 1
    top_influencer = max(left_out_counts, key=left_out_counts.get) if left_out_counts else ""

    persistent = near_freq[near_freq["frequency"] == 1.0] if not near_freq.empty else near_freq
    top_near = (
        near_freq.head(10)[["Feature", "frequency", "JS_mean", "JS_rank_mean"]].to_dict(
            orient="records"
        )
        if not near_freq.empty
        else []
    )

    # Zero / near-zero JS variance descriptors
    flat_js = stability.loc[
        stability["loo_JS_std"] < 1e-12, "Feature"
    ].tolist()

    claims = {
        "n_folds": n_folds,
        "n_fixed15": int(len(stability)),
        "p_threshold": p_threshold,
        "all_fixed15_pass_all_folds": all_pass,
        "worst_loo_pvalue": max_p,
        "worst_loo_pvalue_still_below_threshold": max_p < p_threshold,
        "mean_of_per_descriptor_mean_abs_delta_JS": mean_abs_djs,
        "max_abs_delta_JS_across_all": max_abs_djs,
        "feature_with_largest_max_abs_delta_JS": feat_max_djs,
        "mean_of_per_descriptor_mean_abs_delta_rank": mean_abs_drank,
        "max_abs_delta_rank_across_all": max_abs_drank,
        "feature_with_largest_max_abs_delta_rank": feat_max_drank,
        "most_frequent_extreme_left_out_fluid": top_influencer,
        "n_near_substitutes_in_all_folds": int(len(persistent)),
        "top_near_substitutes": top_near,
        "descriptors_with_zero_JS_loo_variance": flat_js,
        "interpretation": (
            "Statistical support (Mann–Whitney p and JS distance) for the curated "
            "fixed-15 descriptors is stable under leave-one-out of the 21 profile "
            "fluids. This tests metric stability, not automatic re-selection of the 15."
        ),
        "caveats": [
            "Fixed-15 was curated multi-criteria (not auto top-15 by JS alone).",
            "LOO here does not recompute mean profiles or candidate rankings.",
            "Near-substitutes are significant high-JS alternatives, not replacements claimed in the paper.",
            "Some descriptors (e.g. TPSA, NumHAcceptors) can show near-zero JS change when values are discrete/shared.",
        ],
    }
    return claims


def headline_table(claims: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("n_folds", claims["n_folds"]),
        ("n_fixed15", claims["n_fixed15"]),
        ("p_threshold", claims["p_threshold"]),
        ("all_fixed15_pass_all_folds", claims["all_fixed15_pass_all_folds"]),
        ("worst_loo_pvalue", f"{claims['worst_loo_pvalue']:.3e}"),
        (
            "worst_loo_pvalue_still_below_threshold",
            claims["worst_loo_pvalue_still_below_threshold"],
        ),
        (
            "mean_abs_delta_JS (avg over descriptors)",
            f"{claims['mean_of_per_descriptor_mean_abs_delta_JS']:.4f}",
        ),
        (
            "max_abs_delta_JS (any fold×descriptor)",
            f"{claims['max_abs_delta_JS_across_all']:.4f}",
        ),
        (
            "feature_largest_max_abs_delta_JS",
            claims["feature_with_largest_max_abs_delta_JS"],
        ),
        (
            "mean_abs_delta_rank (avg over descriptors)",
            f"{claims['mean_of_per_descriptor_mean_abs_delta_rank']:.2f}",
        ),
        (
            "max_abs_delta_rank (any fold×descriptor)",
            f"{claims['max_abs_delta_rank_across_all']:.0f}",
        ),
        (
            "feature_largest_max_abs_delta_rank",
            claims["feature_with_largest_max_abs_delta_rank"],
        ),
        (
            "most_frequent_extreme_left_out_fluid",
            claims["most_frequent_extreme_left_out_fluid"],
        ),
        (
            "n_near_substitutes_present_in_all_folds",
            claims["n_near_substitutes_in_all_folds"],
        ),
        (
            "zero_JS_variance_descriptors",
            ", ".join(claims["descriptors_with_zero_JS_loo_variance"]) or "(none)",
        ),
    ]
    return pd.DataFrame(rows, columns=["claim", "value"])


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------
def df_to_markdown(df: pd.DataFrame, float_fmt: str = ".4g") -> str:
    """Minimal markdown table (no external dependency)."""
    if df.empty:
        return "_empty_\n"
    cols = list(df.columns)

    def fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return ""
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return format(float(v), float_fmt)
        return str(v).replace("|", "\\|")

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([header, sep, *body]) + "\n"


def write_table_pair(df: pd.DataFrame, path_stem: Path, float_fmt: str = ".6g") -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    md = df_to_markdown(df, float_fmt=float_fmt)
    path_stem.with_suffix(".md").write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _pivot(
    delta_long: pd.DataFrame,
    value: str,
    fixed_descriptors: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    folds = delta_long.drop_duplicates("fold_id")
    fold_ids = folds["fold_id"].tolist()
    ylabels = [
        str(c) for c in folds["left_out_compound"].tolist()
    ]
    mat = np.full((len(fold_ids), len(fixed_descriptors)), np.nan)
    lookup = {
        (r.fold_id, r.Feature): getattr(r, value)
        for r in delta_long.itertuples(index=False)
    }
    for i, fid in enumerate(fold_ids):
        for j, feat in enumerate(fixed_descriptors):
            mat[i, j] = lookup.get((fid, feat), np.nan)
    return mat, ylabels, list(fixed_descriptors)


def plot_js_delta_heatmap(
    delta_long: pd.DataFrame,
    fixed_descriptors: Sequence[str],
    out_dir: Path,
) -> Path:
    mat, ylabels, xlabels = _pivot(delta_long, "delta_JS", fixed_descriptors)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 0.05
    vmax = max(vmax, 1e-4)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=55, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Fixed-15 descriptor")
    ax.set_ylabel("Left-out fluid")
    ax.set_title(r"LOO $\Delta$JS vs full-21 baseline (folds $\times$ fixed-15)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\Delta$JS")
    fig.tight_layout()
    path = out_dir / "fig_js_delta_heatmap"
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


def plot_pvalue_heatmap(
    delta_long: pd.DataFrame,
    fixed_descriptors: Sequence[str],
    out_dir: Path,
    p_threshold: float,
) -> Path:
    mat, ylabels, xlabels = _pivot(delta_long, "MW_pvalue", fixed_descriptors)
    # Use -log10(p) for readability
    with np.errstate(divide="ignore"):
        logp = -np.log10(np.clip(mat, 1e-300, 1.0))
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    im = ax.imshow(logp, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=55, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Fixed-15 descriptor")
    ax.set_ylabel("Left-out fluid")
    thr = -math.log10(p_threshold)
    ax.set_title(rf"LOO Mann–Whitney $-\log_{{10}}(p)$ (threshold $p<{p_threshold}$: {thr:.2f})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"$-\log_{10}(p)$")
    cbar.ax.axhline(thr, color="white", lw=1.2, ls="--")
    fig.tight_layout()
    path = out_dir / "fig_pvalue_neglog10_heatmap"
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


def plot_rank_stability(
    stability: pd.DataFrame,
    fixed_descriptors: Sequence[str],
    out_dir: Path,
) -> Path:
    s = stability.set_index("Feature").loc[list(fixed_descriptors)].reset_index()
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    # LOO range as horizontal bar
    ax.hlines(
        y,
        s["loo_rank_min"],
        s["loo_rank_max"],
        color="#4C78A8",
        lw=3.0,
        label="LOO rank min–max",
        zorder=2,
    )
    ax.scatter(
        s["baseline_JS_rank"],
        y,
        s=55,
        color="#E45756",
        zorder=3,
        label="Baseline rank",
        marker="D",
    )
    ax.scatter(
        s["loo_rank_mean"],
        y,
        s=40,
        color="#4C78A8",
        zorder=3,
        label="LOO mean rank",
        marker="o",
        facecolors="white",
        edgecolors="#4C78A8",
        linewidths=1.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(s["Feature"])
    ax.invert_yaxis()
    ax.set_xlabel("JS rank among all descriptors (1 = highest JS)")
    ax.set_title("Rank stability of curated fixed-15 under LOO")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "fig_rank_stability"
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


def plot_near_substitutes_bar(
    near_freq: pd.DataFrame,
    out_dir: Path,
    top_n: int = 20,
) -> Optional[Path]:
    if near_freq.empty:
        return None
    top = near_freq.head(top_n).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    colors = ["#54A24B" if f == 1.0 else "#4C78A8" for f in top["frequency"]]
    ax.barh(top["Feature"], top["frequency"], color=colors, height=0.7)
    ax.set_xlabel("Fraction of LOO folds as near-substitute")
    ax.set_xlim(0, 1.05)
    ax.set_title(
        f"Top {min(top_n, len(near_freq))} near-substitutes "
        "(not in curated fixed-15)"
    )
    ax.axvline(1.0, color="#333333", ls=":", lw=0.8, alpha=0.6)
    fig.tight_layout()
    path = out_dir / "fig_near_substitutes_frequency"
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


def plot_js_delta_distribution(
    delta_long: pd.DataFrame,
    fixed_descriptors: Sequence[str],
    out_dir: Path,
) -> Path:
    """Three-panel LOO JS view: raw JS vs baseline (top), relative % below.

    A (top): LOO JS boxplots with baseline marked, features ordered by baseline JS.
    B (bottom-left): relative ΔJS % zoomed symmetrically (at least ±1%).
    C (bottom-right): relative ΔJS % full range.
    """
    dl = delta_long.copy()
    dl["delta_JS_pct"] = 100.0 * dl["delta_JS"] / dl["JS_baseline"]

    # Low → high baseline JS (bottom → top on horizontal boxplots).
    order = (
        dl.groupby("Feature")["JS_baseline"]
        .first()
        .reindex(fixed_descriptors)
        .sort_values(ascending=True)
        .index.tolist()
    )
    data_js = [dl.loc[dl["Feature"] == f, "JS"].values for f in order]
    data_pct = [dl.loc[dl["Feature"] == f, "delta_JS_pct"].values for f in order]
    baselines = [
        float(dl.loc[dl["Feature"] == f, "JS_baseline"].iloc[0]) for f in order
    ]

    # Symmetric zoom around 0 so -1% and +1% are equally visible.
    max_drop = abs(float(np.min([np.min(a) for a in data_pct])))
    zoom_half = max(1.0, max_drop) * 1.1
    zoom_lo, zoom_hi = -zoom_half, zoom_half
    full_hi = float(np.max([np.max(a) for a in data_pct]))
    full_hi = max(full_hi, zoom_half) * 1.08
    full_lo = min(zoom_lo, -0.5)

    fig = plt.figure(figsize=(12.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    ax_js = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_full = fig.add_subplot(gs[1, 1], sharey=ax_zoom)

    box_kw = dict(
        vert=False,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )

    # --- A: raw LOO JS with baseline marks ---
    bp_js = ax_js.boxplot(data_js, tick_labels=order, **box_kw)
    for patch in bp_js["boxes"]:
        patch.set_facecolor("#c6dbef")
        patch.set_edgecolor("#3182bd")
    y_pos = np.arange(1, len(order) + 1)
    ax_js.scatter(
        baselines,
        y_pos,
        marker="D",
        s=28,
        c="#e6550d",
        zorder=5,
        label="baseline (all 21 fluids)",
    )
    ax_js.set_xlabel("JS distance")
    ax_js.set_title("A. LOO JS vs baseline (features ordered by baseline JS)")
    ax_js.grid(axis="x", alpha=0.25)
    ax_js.legend(loc="lower right", fontsize=8, frameon=False)
    js_lo = min(float(np.min(np.concatenate(data_js))), min(baselines))
    js_hi = max(float(np.max(np.concatenate(data_js))), max(baselines))
    pad = 0.02 * (js_hi - js_lo + 1e-9)
    ax_js.set_xlim(js_lo - pad, js_hi + pad)

    # --- B / C: relative % (share feature order with A via matching tick labels) ---
    bp_zoom = ax_zoom.boxplot(data_pct, tick_labels=order, **box_kw)
    bp_full = ax_full.boxplot(data_pct, tick_labels=[""] * len(order), **box_kw)
    for bp in (bp_zoom, bp_full):
        for patch in bp["boxes"]:
            patch.set_facecolor("#9ecae1")
            patch.set_edgecolor("#3182bd")

    for ax in (ax_zoom, ax_full):
        ax.axvline(0.0, color="#333333", lw=0.9, ls="--")
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("Change vs baseline (%)")

    ax_zoom.set_xlim(zoom_lo, zoom_hi)
    ax_full.set_xlim(full_lo, full_hi)

    n_out = int(np.sum(np.abs(dl["delta_JS_pct"].to_numpy()) > zoom_half))
    ax_zoom.set_title(f"B. Typical change (±{zoom_half:.1f}%)")
    ax_full.set_title("C. Full % range")
    ax_zoom.text(
        0.98,
        0.02,
        f"{n_out} fold(s) beyond ±{zoom_half:.1f}%\nshown only in C",
        transform=ax_zoom.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    ax_full.axvspan(
        zoom_lo,
        zoom_hi,
        color="#9ecae1",
        alpha=0.15,
        zorder=0,
        label="panel B window",
    )
    ax_full.legend(loc="lower right", fontsize=8, frameon=False)

    zoom_ticks = sorted(
        {
            round(zoom_lo, 1),
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            round(zoom_hi, 1),
        }
    )
    ax_zoom.set_xticks([t for t in zoom_ticks if zoom_lo - 1e-9 <= t <= zoom_hi + 1e-9])
    ax_zoom.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _p: f"{x:g}")
    )
    ax_full.set_xticks([-1.0, 0.0, 1.0, 5.0, 10.0, 15.0])
    ax_full.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _p: f"{x:g}")
    )
    for ax in (ax_js, ax_zoom, ax_full):
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    fig.suptitle(
        "How JS changes when one profile fluid is left out (21 folds)\n"
        "A = absolute JS; B/C = % change (orange ◆ = full-21 baseline)",
        fontsize=11,
    )
    path = out_dir / "fig_js_delta_distribution"
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


def plot_js_delta_distribution_v2(
    delta_long: pd.DataFrame,
    fixed_descriptors: Sequence[str],
    out_dir: Path,
    typical_pct: float = 1.0,
) -> Path:
    """Two-panel LOO JS view (preferred main-text layout).

    A (top): LOO JS vs baseline ◆, features ordered by baseline JS.
    B (bottom): relative ΔJS % on a full linear scale, with a shaded
    ±typical_pct band marking where most folds live.
    """
    dl = delta_long.copy()
    dl["delta_JS_pct"] = 100.0 * dl["delta_JS"] / dl["JS_baseline"]

    order = (
        dl.groupby("Feature")["JS_baseline"]
        .first()
        .reindex(fixed_descriptors)
        .sort_values(ascending=True)
        .index.tolist()
    )
    data_js = [dl.loc[dl["Feature"] == f, "JS"].values for f in order]
    data_pct = [dl.loc[dl["Feature"] == f, "delta_JS_pct"].values for f in order]
    baselines = [
        float(dl.loc[dl["Feature"] == f, "JS_baseline"].iloc[0]) for f in order
    ]

    pct_all = dl["delta_JS_pct"].to_numpy()
    full_hi = float(np.max(pct_all)) * 1.08
    full_lo = min(float(np.min(pct_all)) * 1.15, -typical_pct)
    n_outside = int(np.sum(np.abs(pct_all) > typical_pct))

    fig = plt.figure(figsize=(10.5, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0])
    ax_js = fig.add_subplot(gs[0, 0])
    ax_pct = fig.add_subplot(gs[1, 0], sharey=ax_js)

    box_kw = dict(
        vert=False,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )

    bp_js = ax_js.boxplot(data_js, tick_labels=order, **box_kw)
    for patch in bp_js["boxes"]:
        patch.set_facecolor("#c6dbef")
        patch.set_edgecolor("#3182bd")
    y_pos = np.arange(1, len(order) + 1)
    ax_js.scatter(
        baselines,
        y_pos,
        marker="D",
        s=28,
        c="#e6550d",
        zorder=5,
        label="baseline (all 21 fluids)",
    )
    ax_js.set_xlabel("JS distance")
    ax_js.set_title("A. LOO JS vs baseline (ordered by baseline JS)")
    ax_js.grid(axis="x", alpha=0.25)
    ax_js.legend(loc="lower right", fontsize=8, frameon=False)
    js_lo = min(float(np.min(np.concatenate(data_js))), min(baselines))
    js_hi = max(float(np.max(np.concatenate(data_js))), max(baselines))
    pad = 0.02 * (js_hi - js_lo + 1e-9)
    ax_js.set_xlim(js_lo - pad, js_hi + pad)

    bp_pct = ax_pct.boxplot(data_pct, tick_labels=order, **box_kw)
    for patch in bp_pct["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_edgecolor("#3182bd")
    ax_pct.axvspan(
        -typical_pct,
        typical_pct,
        color="#9ecae1",
        alpha=0.22,
        zorder=0,
        label=f"typical band (±{typical_pct:g}%)",
    )
    ax_pct.axvline(0.0, color="#333333", lw=0.9, ls="--")
    ax_pct.set_xlim(full_lo, full_hi)
    ax_pct.set_xlabel("Change vs baseline JS (%)   (negative = weaker separation)")
    ax_pct.set_title(
        f"B. Relative change (full scale; {n_outside} folds outside ±{typical_pct:g}%)"
    )
    ax_pct.grid(axis="x", alpha=0.25)
    ax_pct.legend(loc="lower right", fontsize=8, frameon=False)
    ax_pct.set_xticks([-1.0, 0.0, 1.0, 5.0, 10.0, 15.0])
    ax_pct.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _p: f"{x:g}")
    )
    for ax in (ax_js, ax_pct):
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    fig.suptitle(
        "How JS changes when one profile fluid is left out (21 folds)\n"
        "Most folds stay near baseline; rare large increases are outlier-fluid removals",
        fontsize=11,
    )
    path = out_dir / "fig_js_delta_distribution_v2"
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


def plot_js_abs_delta_distribution(
    delta_long: pd.DataFrame,
    fixed_descriptors: Sequence[str],
    out_dir: Path,
) -> Path:
    """Absolute |ΔJS|; magnitude of change without direction."""
    order = (
        delta_long.groupby("Feature")["delta_JS"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .reindex(fixed_descriptors)
        .sort_values(ascending=True)
        .index.tolist()
    )
    data = [
        np.abs(delta_long.loc[delta_long["Feature"] == f, "delta_JS"].values)
        for f in order
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=order,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#c7e9c0")
        patch.set_edgecolor("#238b45")
    ax.set_xlabel(
        r"$|\Delta\mathrm{JS}|$ $=$ $|\mathrm{JS}_{\mathrm{LOO}} - "
        r"\mathrm{JS}_{\mathrm{baseline}}|$ (baseline $=$ all 21 fluids)"
    )
    ax.set_title(r"Magnitude of JS change across 21 LOO folds")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "fig_js_abs_delta_distribution"
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


# ---------------------------------------------------------------------------
# Near-substitute filtering + correlation vs fixed-15
# ---------------------------------------------------------------------------
def filter_near_substitutes_by_frequency(
    near_freq: pd.DataFrame,
    min_frequency: float = DEFAULT_NEAR_SUB_MIN_FREQUENCY,
) -> pd.DataFrame:
    """Keep near-substitutes present in >= min_frequency of LOO folds."""
    if near_freq.empty:
        return near_freq.copy()
    out = near_freq.loc[near_freq["frequency"] >= min_frequency].copy()
    out = out.sort_values(
        by=["frequency", "JS_mean", "Feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["min_frequency_threshold"] = min_frequency
    return out


def compute_near_substitute_correlations(
    combined_descriptors: pd.DataFrame,
    near_features: Sequence[str],
    fixed_descriptors: Sequence[str],
    *,
    high_corr_threshold: float = DEFAULT_HIGH_CORR_THRESHOLD,
    low_corr_threshold: float = DEFAULT_LOW_CORR_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pearson correlations between filtered near-substitutes and fixed-15.

    Returns
    -------
    corr_matrix
        Rows = near-subs, columns = fixed-15 (Pearson r on combined_descriptors rows).
    summary
        Per near-sub: closest fixed-15, max |r|, corr class (high/mid/low), swap hints.
    groups
        One row per (fixed-15, near-sub) high-corr pair for within-group swap studies.
    """
    fixed = list(fixed_descriptors)
    near = [f for f in near_features if f in combined_descriptors.columns]
    missing = [f for f in near_features if f not in combined_descriptors.columns]
    if missing:
        logger.warning("Near-subs missing from combined_descriptors: %s", missing)
    fixed_present = [f for f in fixed if f in combined_descriptors.columns]
    if not near or not fixed_present:
        empty = pd.DataFrame()
        return empty, empty, empty

    sub = combined_descriptors[near + fixed_present].apply(pd.to_numeric, errors="coerce")
    # Full pairwise, then extract near × fixed block
    full = sub.corr(method="pearson", min_periods=max(10, sub.shape[0] // 10))
    corr_matrix = full.loc[near, fixed_present].copy()
    corr_matrix.index.name = "near_substitute"
    corr_matrix = corr_matrix.reset_index()

    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for feat in near:
        series = full.loc[feat, fixed_present].astype(float)
        abs_s = series.abs()
        best = str(abs_s.idxmax())
        best_r = float(series.loc[best])
        max_abs = float(abs_s.max())
        if max_abs >= high_corr_threshold:
            corr_class = "high"
            swap_role = "within_group_swap_candidate"
        elif max_abs < low_corr_threshold:
            corr_class = "low"
            swap_role = "profile_complement_candidate"
        else:
            corr_class = "mid"
            swap_role = "optional_mid_corr"
        rows.append(
            {
                "near_substitute": feat,
                "closest_fixed15": best,
                "corr_with_closest": best_r,
                "max_abs_corr_vs_fixed15": max_abs,
                "corr_class": corr_class,
                "swap_role": swap_role,
                "high_corr_threshold": high_corr_threshold,
                "low_corr_threshold": low_corr_threshold,
                "n_molecules_used": int(sub[[feat] + fixed_present].dropna().shape[0]),
            }
        )
        # All high-corr partners (not only the closest)
        for fixed_feat, r in series.items():
            if abs(float(r)) >= high_corr_threshold:
                group_rows.append(
                    {
                        "fixed15_group": fixed_feat,
                        "near_substitute": feat,
                        "pearson_r": float(r),
                        "abs_r": abs(float(r)),
                        "is_closest_for_near_sub": fixed_feat == best,
                    }
                )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["_class_ord"] = summary["corr_class"].map(
            {"high": 0, "mid": 1, "low": 2}
        )
        summary = summary.sort_values(
            by=["_class_ord", "max_abs_corr_vs_fixed15", "near_substitute"],
            ascending=[True, False, True],
            kind="mergesort",
        ).drop(columns=["_class_ord"]).reset_index(drop=True)

    groups = pd.DataFrame(group_rows)
    if not groups.empty:
        groups = groups.sort_values(
            by=["fixed15_group", "abs_r", "near_substitute"],
            ascending=[True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return corr_matrix, summary, groups


def _category_display_label(cat: str, block_size: int) -> str:
    """Short axis label; further abbreviate narrow (≤2) blocks."""
    short = CATEGORY_SHORT.get(cat, cat)
    if block_size > 2:
        return short
    # Avoid neighbor collisions on 1-wide / 1-tall blocks
    ultra = {
        "Electronic": "Electr.",
        "Surface": "Surf.",
        "Polarity": "Polar.",
        "Lipophilicity": "LogP",
        "Halogen": "Hal.",
        "Size": "Size",
        "Topology": "Topo.",
        "Complexity": "Compl.",
        "Other": "Other",
    }
    return ultra.get(short, short)


def _annotate_category_blocks(
    ax: Any,
    fig: Any,
    row_blocks: Sequence[tuple[str, int, int]],
    col_blocks: Sequence[tuple[str, int, int]],
) -> None:
    """
    Place category titles outside the heatmap with bracket cues.

    Row labels sit left of the y-tick names (axes-x < 0); column labels sit
    just above the axes spine (axes-y > 1). Call after colorbar + margins.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    inv_ax = ax.transAxes.inverted()

    ytick_left_disp = ax_bbox.x0
    for tick in ax.get_yticklabels():
        if not tick.get_visible() or not tick.get_text():
            continue
        ytick_left_disp = min(ytick_left_disp, tick.get_window_extent(renderer).x0)

    x_ticks_left, _ = inv_ax.transform((ytick_left_disp, ax_bbox.y0))
    x_bracket = x_ticks_left - 0.08
    x_label = x_bracket - 0.11

    pos = ax.get_position()
    # Room for category labels + a fig-level y-axis title to their left.
    extra = abs(min(x_label, 0.0)) * pos.width + 0.10
    new_left = min(0.55, max(0.40, pos.x0 + extra))
    fig.subplots_adjust(left=new_left, top=0.86, bottom=0.22, right=0.96)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    inv_ax = ax.transAxes.inverted()
    ytick_left_disp = ax_bbox.x0
    for tick in ax.get_yticklabels():
        if not tick.get_visible() or not tick.get_text():
            continue
        ytick_left_disp = min(ytick_left_disp, tick.get_window_extent(renderer).x0)
    x_ticks_left, _ = inv_ax.transform((ytick_left_disp, ax_bbox.y0))
    x_bracket = x_ticks_left - 0.08
    x_label = x_bracket - 0.11

    trans_row = blended_transform_factory(ax.transAxes, ax.transData)
    trans_col = blended_transform_factory(ax.transData, ax.transAxes)

    for cat, start, end in row_blocks:
        mid = 0.5 * (start + end - 1)
        y0, y1 = start - 0.4, end - 0.6
        ax.plot(
            [x_bracket, x_bracket - 0.018, x_bracket - 0.018, x_bracket],
            [y0, y0, y1, y1],
            transform=trans_row,
            color="0.25",
            linewidth=0.9,
            clip_on=False,
            solid_capstyle="butt",
            zorder=5,
        )
        ax.text(
            x_label,
            mid,
            _category_display_label(cat, end - start),
            transform=trans_row,
            ha="center",
            va="center",
            fontsize=6.5 if (end - start) == 1 else 7.0,
            fontweight="bold",
            color="0.2",
            rotation=90,
            clip_on=False,
            zorder=5,
        )

    # Alternate height for narrow column blocks so adjacent short labels clear.
    stagger_hi = False
    for cat, start, end in col_blocks:
        mid = 0.5 * (start + end - 1)
        width = end - start
        x0, x1 = start - 0.4, end - 0.6
        if width <= 2 and stagger_hi:
            y_br_lo, y_br_hi, y_text = 1.055, 1.085, 1.100
        else:
            y_br_lo, y_br_hi, y_text = 1.015, 1.045, 1.060
        if width <= 2:
            stagger_hi = not stagger_hi
        else:
            stagger_hi = False
        ax.plot(
            [x0, x0, x1, x1],
            [y_br_lo, y_br_hi, y_br_hi, y_br_lo],
            transform=trans_col,
            color="0.25",
            linewidth=0.9,
            clip_on=False,
            solid_capstyle="butt",
            zorder=5,
        )
        ax.text(
            mid,
            y_text,
            _category_display_label(cat, width),
            transform=trans_col,
            ha="center",
            va="bottom",
            fontsize=6.0 if width <= 2 else 7.0,
            fontweight="bold",
            color="0.2",
            clip_on=False,
            zorder=5,
        )

    # Y-axis title left of category titles (figure coords from measured axes-x).
    label_disp = ax.transAxes.transform((x_label, 0.5))
    fig_xy = fig.transFigure.inverted().transform(label_disp)
    fig_x = fig_xy[0] - 0.035
    # Expand left margin if the ylabel would be clipped.
    if fig_x < 0.04:
        shift = 0.04 - fig_x
        pos = ax.get_position()
        fig.subplots_adjust(left=min(0.58, pos.x0 + shift))
        label_disp = ax.transAxes.transform((x_label, 0.5))
        fig_xy = fig.transFigure.inverted().transform(label_disp)
        fig_x = fig_xy[0] - 0.035
    pos = ax.get_position()
    fig.text(
        max(0.02, fig_x),
        pos.y0 + 0.5 * pos.height,
        "Near-substitute (by chemical category)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
    )


def plot_near_vs_fixed_corr_heatmap(
    corr_matrix: pd.DataFrame,
    out_dir: Path,
    *,
    high_corr_threshold: float,
    fixed_order: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """
    Near-substitute × fixed-15 Pearson-r heatmap.

    Axes are ordered by the paper Methods descriptor taxonomy
    (``descriptor_taxonomy``): electronic → surface → polarity →
    lipophilicity → halogen → size → topology → complexity, with block
    separators between categories.
    """
    if corr_matrix.empty:
        return None
    mat = corr_matrix.set_index("near_substitute")

    col_pref = list(fixed_order) if fixed_order is not None else list(mat.columns)
    cols = sort_descriptors_by_category(list(mat.columns), within_category=col_pref)
    rows = sort_descriptors_by_category(list(mat.index), within_category=col_pref)
    mat = mat.loc[rows, cols]

    n_rows, n_cols = mat.shape
    fig_h = max(7.0, 0.28 * n_rows + 2.8)
    fig_w = max(11.0, 0.48 * n_cols + 5.5)
    fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 24.0)))

    row_blocks = category_block_boundaries(rows)
    col_blocks = category_block_boundaries(cols)

    # Light alternating column bands behind the heatmap (category cues)
    for i, (_cat, start, end) in enumerate(col_blocks):
        if i % 2 == 1:
            ax.axvspan(start - 0.5, end - 0.5, color="0.92", zorder=0)

    im = ax.imshow(
        mat.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        zorder=1,
    )
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cols, rotation=55, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Curated fixed-15 descriptor (by chemical category)")
    # Y-axis title drawn in _annotate_category_blocks (fig.text) to clear brackets.
    ax.set_ylabel("")

    for _, _start, end in row_blocks[:-1]:
        ax.axhline(end - 0.5, color="0.2", linewidth=1.15, zorder=3)
    for _, _start, end in col_blocks[:-1]:
        ax.axvline(end - 0.5, color="0.2", linewidth=1.15, zorder=3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pearson r")

    # Leave headroom for fig.suptitle above the column category strip.
    fig.subplots_adjust(left=0.40, top=0.86, bottom=0.22, right=0.96)
    _annotate_category_blocks(ax, fig, row_blocks, col_blocks)
    # Center title on the heatmap axes (not the whole figure incl. left margin).
    pos = ax.get_position()
    fig.suptitle(
        f"Pearson r: near-substitutes vs fixed-15 "
        f"(category-ordered; |r|≥{high_corr_threshold:g} = high-corr group)",
        x=pos.x0 + 0.5 * pos.width,
        y=0.98,
        fontsize=11,
    )

    path = out_dir / "fig_near_vs_fixed15_correlation"
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path.with_suffix(".png")



# ---------------------------------------------------------------------------
# SUMMARY.md + README
# ---------------------------------------------------------------------------
def write_summary_md(
    path: Path,
    claims: dict[str, Any],
    stability: pd.DataFrame,
    extremes: pd.DataFrame,
    near_freq: pd.DataFrame,
    *,
    near_freq_ge: Optional[pd.DataFrame] = None,
    corr_summary: Optional[pd.DataFrame] = None,
) -> None:
    # Biggest movers by max abs ΔJS and max abs Δrank
    top_js = stability.nlargest(5, "max_abs_delta_JS")[
        ["Feature", "max_abs_delta_JS", "mean_abs_delta_JS", "loo_JS_std"]
    ]
    top_rank = stability.nlargest(5, "max_abs_delta_rank")[
        ["Feature", "max_abs_delta_rank", "mean_abs_delta_rank", "loo_rank_min", "loo_rank_max"]
    ]

    influencers = (
        extremes.melt(
            id_vars=["Feature"],
            value_vars=[
                "left_out_max_abs_delta_JS",
                "left_out_max_pvalue",
                "left_out_max_abs_delta_rank",
            ],
            value_name="left_out",
        )["left_out"]
        .value_counts()
        .head(8)
    )

    if near_freq_ge is None:
        near_freq_ge = pd.DataFrame()
    if corr_summary is None:
        corr_summary = pd.DataFrame()

    freq_thr = float(claims.get("near_sub_min_frequency", DEFAULT_NEAR_SUB_MIN_FREQUENCY))
    persistent = (
        near_freq_ge.head(12)
        if not near_freq_ge.empty
        else (
            near_freq[near_freq["frequency"] == 1.0].head(12)
            if not near_freq.empty
            else near_freq
        )
    )

    lines = [
        "# LOO descriptor-metrics aggregation — summary",
        "",
        "Leave-one-out (LOO) of the **21** size-matched profile fluids, recomputing",
        "Jensen–Shannon distances and Mann–Whitney tests vs the fixed **177**",
        "size-matched references. Aggregation consumes existing fold outputs;",
        "it does **not** re-run LOO or mean-profile / candidate-ranking analyses.",
        "",
        "## Headline claims",
        "",
    ]

    if claims["all_fixed15_pass_all_folds"]:
        lines.append(
            f"- **All 15 curated descriptors pass** Mann–Whitney "
            f"`p < {claims['p_threshold']}` in **all {claims['n_folds']} folds** "
            f"(worst LOO *p* = `{claims['worst_loo_pvalue']:.3e}`)."
        )
    else:
        lines.append(
            f"- Not all fixed-15 descriptors pass in all folds "
            f"(worst LOO *p* = `{claims['worst_loo_pvalue']:.3e}`)."
        )

    lines.extend(
        [
            f"- Mean absolute ΔJS across descriptors ≈ "
            f"**{claims['mean_of_per_descriptor_mean_abs_delta_JS']:.4f}**; "
            f"largest single |ΔJS| = "
            f"**{claims['max_abs_delta_JS_across_all']:.4f}** "
            f"(`{claims['feature_with_largest_max_abs_delta_JS']}`).",
            f"- Rank movement is modest on average "
            f"(mean |Δrank| ≈ **{claims['mean_of_per_descriptor_mean_abs_delta_rank']:.2f}**); "
            f"largest |Δrank| = "
            f"**{claims['max_abs_delta_rank_across_all']:.0f}** "
            f"(`{claims['feature_with_largest_max_abs_delta_rank']}`).",
            f"- Left-out fluid most often associated with per-descriptor extremes: "
            f"**{claims['most_frequent_extreme_left_out_fluid']}**.",
            f"- **{claims.get('n_near_substitutes_freq_ge_threshold', claims['n_near_substitutes_in_all_folds'])}** "
            f"near-substitutes with frequency ≥ **{freq_thr:.0%}** "
            f"({claims['n_near_substitutes_in_all_folds']} appear in all folds).",
            "",
            "## Magnitude of JS / rank movement",
            "",
            "### What ΔJS means (not missing absolute value)",
            "",
            "- **Baseline** = metrics with **all 21** profile fluids.",
            "- **Signed** ΔJS = JS_LOO − JS_baseline.",
            "- Values **below 0** mean that leaving one fluid out **weakened** separation "
            "(JS decreased). Values above 0 mean separation strengthened.",
            "- `fig_js_delta_distribution`: **A** LOO JS with baseline mark; "
            "**B** relative % zoomed (± at least 1%); **C** full % range. "
            "`fig_js_abs_delta_distribution` shows **|ΔJS|** (magnitude only).",
            "",
            "### Largest |ΔJS| (top 5 descriptors)",
            "",
            df_to_markdown(top_js),
            "",
            "### Largest |Δrank| span (top 5 descriptors)",
            "",
            df_to_markdown(top_rank, float_fmt=".4g"),
            "",
            "### Fluids that most often drive extremes",
            "",
            "| left_out | n_extreme_slots |",
            "| --- | --- |",
        ]
    )
    for name, count in influencers.items():
        lines.append(f"| {name} | {int(count)} |")

    flat = claims["descriptors_with_zero_JS_loo_variance"]
    if flat:
        lines.extend(
            [
                "",
                "### Near-invariant JS under LOO",
                "",
                "These descriptors show essentially **zero JS variance** across folds "
                f"(discrete / shared values): `{', '.join(flat)}`.",
            ]
        )

    lines.extend(
        [
            "",
            f"## Near-substitutes (frequency ≥ {freq_thr:.0%})",
            "",
            "Full list: `aggregate/near_substitutes_frequency.csv`. "
            f"Filtered (≥{freq_thr:.0%}): `aggregate/near_substitutes_frequency_ge90.csv`.",
            "",
            "These are **not** automatic paper replacements; they are candidates for "
            "later swap / complement ablation studies after correlation filtering.",
            "",
        ]
    )
    if not persistent.empty:
        cols = [
            c
            for c in [
                "Feature",
                "n_folds_as_near_substitute",
                "frequency",
                "JS_mean",
                "JS_rank_mean",
            ]
            if c in persistent.columns
        ]
        lines.append(df_to_markdown(persistent[cols]))
    else:
        lines.append(f"_No near-substitutes with frequency ≥ {freq_thr:.0%}._\n")

    if not corr_summary.empty:
        high = corr_summary.loc[corr_summary["corr_class"] == "high"]
        low = corr_summary.loc[corr_summary["corr_class"] == "low"]
        lines.extend(
            [
                "",
                "## Correlation vs curated fixed-15",
                "",
                f"Pearson *r* on `combined_descriptors.csv` (21+177 molecules). "
                f"High: |r| ≥ {claims.get('high_corr_threshold', DEFAULT_HIGH_CORR_THRESHOLD)}; "
                f"low: |r| < {claims.get('low_corr_threshold', DEFAULT_LOW_CORR_THRESHOLD)} "
                "(max abs corr vs any fixed-15).",
                "",
                f"- High-corr (within-group swap pool): **{len(high)}**",
                f"- Low-corr (profile-complement pool): **{len(low)}**",
                f"- Mid: **{int((corr_summary['corr_class'] == 'mid').sum())}**",
                "",
                "### High-corr near-subs (closest fixed-15)",
                "",
            ]
        )
        if not high.empty:
            lines.append(
                df_to_markdown(
                    high[
                        [
                            "near_substitute",
                            "closest_fixed15",
                            "corr_with_closest",
                            "max_abs_corr_vs_fixed15",
                        ]
                    ].head(20)
                )
            )
        else:
            lines.append("_None._")
        lines.extend(["", "### Low-corr complements (max |r| vs fixed-15)", ""])
        if not low.empty:
            lines.append(
                df_to_markdown(
                    low[
                        [
                            "near_substitute",
                            "closest_fixed15",
                            "corr_with_closest",
                            "max_abs_corr_vs_fixed15",
                        ]
                    ].head(20)
                )
            )
        else:
            lines.append("_None._")

    lines.extend(
        [
            "",
            "## Interpretation for reviewers",
            "",
            claims["interpretation"],
            "",
            "In short: the **statistical support** for the curated descriptor set "
            "(significant separation + high JS) is **stable** when any one of the "
            "21 profile fluids is removed. Hydrocarbon outliers (e.g. R601 / R601a) "
            "can shift JS more than halogenated fluids for some descriptors, but "
            "none of the fixed-15 lose significance.",
            "",
            "## Caveats",
            "",
        ]
    )
    for c in claims["caveats"]:
        lines.append(f"- {c}")
    lines.append(
        "- Signed ΔJS can be negative; use |ΔJS| figure when only magnitude matters."
    )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Tables: `tables/` (filters/correlations also under `../aggregate/`)",
            "- Figures: `figures/` (PNG + PDF)",
            "- Machine-readable claims: `headline_claims.json`",
            "",
            f"_Generated {claims.get('_generated_utc', datetime.now(timezone.utc).isoformat())}_",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")



def update_loo_readme(loo_dir: Path) -> None:
    readme = loo_dir / "README.md"
    if not readme.is_file():
        return
    text = readme.read_text(encoding="utf-8")
    marker = "## Publication aggregation"
    block = """## Publication aggregation

Refined SI-oriented tables/figures (consumes existing LOO outputs; does not re-run LOO):

```bash
python3 refrigerant_finding/profile_robustness/aggregate_loo_descriptor_metrics.py
```

Writes `aggregation/` (`tables/`, `figures/`, `SUMMARY.md`, `headline_claims.json`).

"""
    if marker in text:
        # Replace from marker to end-of-section or append cleanly
        before = text.split(marker)[0].rstrip() + "\n\n"
        # Keep anything after a following ## if present in the remainder
        rest = text.split(marker, 1)[1]
        # Drop old aggregation section body until next ## or EOF
        parts = rest.split("\n## ")
        # parts[0] is old aggregation body; restore subsequent sections
        after = ""
        if len(parts) > 1:
            after = "\n## " + "\n## ".join(parts[1:])
        text = before + block + after.lstrip("\n")
    else:
        # Insert before ## Inputs if present
        if "\n## Inputs\n" in text:
            text = text.replace("\n## Inputs\n", "\n" + block + "## Inputs\n", 1)
        else:
            text = text.rstrip() + "\n\n" + block

    # Also extend layout tree if aggregation not listed
    if "aggregation/" not in text:
        text = text.replace(
            "    near_substitutes_frequency.csv\n```",
            "    near_substitutes_frequency.csv\n"
            "  aggregation/                     # SI-oriented re-aggregation\n"
            "    SUMMARY.md\n"
            "    headline_claims.json\n"
            "    tables/\n"
            "    figures/\n```",
        )
    readme.write_text(text, encoding="utf-8")


def write_aggregation_readme(output_dir: Path) -> None:
    text = """# LOO aggregation (descriptor metrics)

Publication-oriented products built from existing `loo_on_profile/` fold outputs.

## Re-run

From the `RefrigerantFinding` repo root:

```bash
python3 refrigerant_finding/profile_robustness/aggregate_loo_descriptor_metrics.py
```

Optional:

```bash
python3 refrigerant_finding/profile_robustness/aggregate_loo_descriptor_metrics.py \
  --loo-dir refrigerant_finding/profile_robustness/loo_on_profile \
  --output-dir refrigerant_finding/profile_robustness/loo_on_profile/aggregation \
  --near-sub-min-frequency 0.9 \
  --high-corr-threshold 0.7 \
  --low-corr-threshold 0.4 \
  --top-near-substitutes 20
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
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")



# ---------------------------------------------------------------------------
# Consistency checks vs pre-built aggregate/
# ---------------------------------------------------------------------------
def check_against_existing_aggregate(
    loo_dir: Path,
    stability: pd.DataFrame,
    near_freq: pd.DataFrame,
) -> list[str]:
    """Soft checks; return human-readable issue strings (empty = OK)."""
    issues: list[str] = []
    agg = loo_dir / "aggregate"
    summary_path = agg / "fixed15_loo_summary.csv"
    if summary_path.is_file():
        old = pd.read_csv(summary_path).set_index("Feature")
        new = stability.set_index("Feature")
        for feat in new.index:
            if feat not in old.index:
                issues.append(f"Feature {feat} missing from existing aggregate summary")
                continue
            if abs(float(old.loc[feat, "loo_pass_rate"]) - float(new.loc[feat, "loo_pass_rate"])) > 1e-9:
                issues.append(f"{feat}: pass_rate mismatch vs aggregate/fixed15_loo_summary.csv")
            if abs(float(old.loc[feat, "loo_JS_mean"]) - float(new.loc[feat, "loo_JS_mean"])) > 1e-9:
                issues.append(f"{feat}: loo_JS_mean mismatch vs existing aggregate")
    else:
        issues.append("No existing aggregate/fixed15_loo_summary.csv to cross-check")

    near_path = agg / "near_substitutes_frequency.csv"
    if near_path.is_file() and not near_freq.empty:
        old_n = pd.read_csv(near_path).set_index("Feature")
        for feat in near_freq["Feature"].head(20):
            if feat not in old_n.index:
                issues.append(f"Near-sub {feat} missing from existing frequency CSV")
                continue
            if int(old_n.loc[feat, "n_folds_as_near_substitute"]) != int(
                near_freq.set_index("Feature").loc[feat, "n_folds_as_near_substitute"]
            ):
                issues.append(f"Near-sub {feat}: fold count mismatch vs existing aggregate")
    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_aggregation(
    loo_dir: Path,
    output_dir: Path,
    *,
    fixed_descriptors: Sequence[str] = FIXED_15,
    p_threshold: float = DEFAULT_P_THRESHOLD,
    top_near_substitutes: int = 20,
    near_sub_min_frequency: float = DEFAULT_NEAR_SUB_MIN_FREQUENCY,
    high_corr_threshold: float = DEFAULT_HIGH_CORR_THRESHOLD,
    low_corr_threshold: float = DEFAULT_LOW_CORR_THRESHOLD,
    combined_descriptors_path: Optional[Path] = None,
) -> dict[str, Any]:
    configure_matplotlib()
    loo_dir = loo_dir.resolve()
    output_dir = output_dir.resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir = loo_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = load_run_config(loo_dir)
    if run_cfg.get("p_threshold") is not None:
        p_threshold = float(run_cfg["p_threshold"])
    if run_cfg.get("fixed_15"):
        # Prefer run_config order if it matches known set
        cfg15 = list(run_cfg["fixed_15"])
        if set(cfg15) == set(fixed_descriptors):
            fixed_descriptors = cfg15

    logger.info("Loading fold outputs from %s", loo_dir)
    metas, fixed_tables, near_tables = load_fold_tables(loo_dir, fixed_descriptors)
    baseline_fixed = load_baseline_fixed15(loo_dir)
    n_folds = len(metas)
    logger.info("Loaded %d folds, %d fixed descriptors", n_folds, len(fixed_descriptors))

    stability = build_stability_table(
        metas, fixed_tables, baseline_fixed, fixed_descriptors, p_threshold
    )
    extremes = build_extremes_table(
        metas, fixed_tables, baseline_fixed, fixed_descriptors
    )
    near_freq = build_near_substitutes_table(metas, near_tables, fixed_descriptors)
    near_freq_ge = filter_near_substitutes_by_frequency(
        near_freq, min_frequency=near_sub_min_frequency
    )
    delta_long = build_js_delta_long(
        metas, fixed_tables, baseline_fixed, fixed_descriptors
    )
    claims = build_headline(stability, extremes, near_freq, n_folds, p_threshold)
    claims["near_sub_min_frequency"] = near_sub_min_frequency
    claims["n_near_substitutes_freq_ge_threshold"] = int(len(near_freq_ge))
    claims["high_corr_threshold"] = high_corr_threshold
    claims["low_corr_threshold"] = low_corr_threshold
    generated = datetime.now(timezone.utc).isoformat()
    claims["_generated_utc"] = generated
    headline_df = headline_table(claims)

    # Correlations: filtered near-subs (≥ freq threshold) vs fixed-15
    desc_path = (
        Path(combined_descriptors_path)
        if combined_descriptors_path is not None
        else default_combined_descriptors_path()
    )
    if run_cfg.get("combined_descriptors"):
        cfg_path = Path(str(run_cfg["combined_descriptors"]))
        if cfg_path.is_file():
            desc_path = cfg_path
    corr_matrix = pd.DataFrame()
    corr_summary = pd.DataFrame()
    corr_groups = pd.DataFrame()
    if desc_path.is_file() and not near_freq_ge.empty:
        logger.info("Computing near-sub vs fixed-15 correlations from %s", desc_path)
        combined = pd.read_csv(desc_path)
        corr_matrix, corr_summary, corr_groups = compute_near_substitute_correlations(
            combined,
            near_freq_ge["Feature"].tolist(),
            fixed_descriptors,
            high_corr_threshold=high_corr_threshold,
            low_corr_threshold=low_corr_threshold,
        )
        claims["n_near_sub_high_corr"] = int(
            (corr_summary["corr_class"] == "high").sum()
        ) if not corr_summary.empty else 0
        claims["n_near_sub_low_corr"] = int(
            (corr_summary["corr_class"] == "low").sum()
        ) if not corr_summary.empty else 0
        claims["n_near_sub_mid_corr"] = int(
            (corr_summary["corr_class"] == "mid").sum()
        ) if not corr_summary.empty else 0
    else:
        logger.warning(
            "Skipping correlation analysis (missing descriptors file or empty filtered near-subs): %s",
            desc_path,
        )

    # Also write long matrix and rebuilt core matrices for convenience
    js_mat = _descriptor_matrix_from_folds(
        metas, fixed_tables, fixed_descriptors, "JS_Distance"
    )
    rank_mat = _descriptor_matrix_from_folds(
        metas, fixed_tables, fixed_descriptors, "JS_rank"
    )
    p_mat = _descriptor_matrix_from_folds(
        metas, fixed_tables, fixed_descriptors, "MW_pvalue"
    )
    pass_mat = _descriptor_matrix_from_folds(
        metas, fixed_tables, fixed_descriptors, "passes_p"
    )

    # Tables
    write_table_pair(stability, tables_dir / "fixed15_stability")
    write_table_pair(extremes, tables_dir / "per_fold_extremes")
    write_table_pair(near_freq, tables_dir / "near_substitutes_frequency")
    write_table_pair(
        near_freq_ge,
        tables_dir / "near_substitutes_frequency_ge90",
    )
    # Mirror full + filtered frequency into loo_on_profile/aggregate/
    near_freq.to_csv(aggregate_dir / "near_substitutes_frequency.csv", index=False)
    near_freq_ge.to_csv(
        aggregate_dir / "near_substitutes_frequency_ge90.csv", index=False
    )
    write_table_pair(headline_df, tables_dir / "headline_summary", float_fmt="")
    write_table_pair(
        near_freq_ge.head(top_near_substitutes)
        if not near_freq_ge.empty
        else near_freq.head(top_near_substitutes),
        tables_dir / "near_substitutes_top",
    )
    if not corr_matrix.empty:
        write_table_pair(corr_matrix, tables_dir / "near_vs_fixed15_correlation_matrix")
        corr_matrix.to_csv(
            aggregate_dir / "near_vs_fixed15_correlation_matrix.csv", index=False
        )
    if not corr_summary.empty:
        write_table_pair(corr_summary, tables_dir / "near_vs_fixed15_correlation_summary")
        corr_summary.to_csv(
            aggregate_dir / "near_vs_fixed15_correlation_summary.csv", index=False
        )
        high = corr_summary.loc[corr_summary["corr_class"] == "high"].copy()
        low = corr_summary.loc[corr_summary["corr_class"] == "low"].copy()
        write_table_pair(high, tables_dir / "near_substitutes_high_corr_groups")
        write_table_pair(low, tables_dir / "near_substitutes_low_corr_complements")
        high.to_csv(aggregate_dir / "near_substitutes_high_corr_groups.csv", index=False)
        low.to_csv(
            aggregate_dir / "near_substitutes_low_corr_complements.csv", index=False
        )
    if not corr_groups.empty:
        write_table_pair(corr_groups, tables_dir / "near_substitutes_high_corr_pairs")
        corr_groups.to_csv(
            aggregate_dir / "near_substitutes_high_corr_pairs.csv", index=False
        )
    cat_map = fixed15_and_near_category_table(
        fixed_descriptors,
        near_freq_ge["Feature"].tolist() if not near_freq_ge.empty else [],
    )
    if not cat_map.empty:
        write_table_pair(cat_map, tables_dir / "descriptor_category_map")
        cat_map.to_csv(aggregate_dir / "descriptor_category_map.csv", index=False)
    delta_long.to_csv(tables_dir / "fixed15_delta_long.csv", index=False)
    js_mat.to_csv(tables_dir / "fixed15_js_matrix.csv", index=False)
    rank_mat.to_csv(tables_dir / "fixed15_rank_matrix.csv", index=False)
    p_mat.to_csv(tables_dir / "fixed15_pvalue_matrix.csv", index=False)
    pass_mat.to_csv(tables_dir / "fixed15_pass_matrix.csv", index=False)

    (output_dir / "headline_claims.json").write_text(
        json.dumps(claims, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # Figures
    logger.info("Writing figures to %s", figures_dir)
    plot_js_delta_heatmap(delta_long, fixed_descriptors, figures_dir)
    plot_pvalue_heatmap(delta_long, fixed_descriptors, figures_dir, p_threshold)
    plot_rank_stability(stability, fixed_descriptors, figures_dir)
    plot_near_substitutes_bar(
        near_freq_ge if not near_freq_ge.empty else near_freq,
        figures_dir,
        top_n=top_near_substitutes,
    )
    plot_js_delta_distribution(delta_long, fixed_descriptors, figures_dir)
    plot_js_delta_distribution_v2(delta_long, fixed_descriptors, figures_dir)
    plot_js_abs_delta_distribution(delta_long, fixed_descriptors, figures_dir)
    plot_near_vs_fixed_corr_heatmap(
        corr_matrix,
        figures_dir,
        high_corr_threshold=high_corr_threshold,
        fixed_order=fixed_descriptors,
    )

    write_summary_md(
        output_dir / "SUMMARY.md",
        claims,
        stability,
        extremes,
        near_freq,
        near_freq_ge=near_freq_ge,
        corr_summary=corr_summary,
    )
    write_aggregation_readme(output_dir)
    update_loo_readme(loo_dir)

    meta = AggregationMeta(
        loo_dir=str(loo_dir),
        output_dir=str(output_dir),
        n_folds=n_folds,
        n_fixed15=len(fixed_descriptors),
        p_threshold=p_threshold,
        timestamp_utc=generated,
        source_run_config=str((loo_dir / "run_config.json").resolve())
        if (loo_dir / "run_config.json").is_file()
        else None,
    )
    (output_dir / "aggregation_meta.json").write_text(
        json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8"
    )

    issues = check_against_existing_aggregate(loo_dir, stability, near_freq)
    if issues:
        logger.warning("Cross-check notes (%d):", len(issues))
        for msg in issues:
            logger.warning("  - %s", msg)
    else:
        logger.info("Cross-check vs existing aggregate/: OK")

    # Console highlight
    logger.info("=== Headline ===")
    logger.info(
        "  all_pass_all_folds=%s  worst_p=%.3e  max_|ΔJS|=%.4f (%s)  max_|Δrank|=%.0f (%s)",
        claims["all_fixed15_pass_all_folds"],
        claims["worst_loo_pvalue"],
        claims["max_abs_delta_JS_across_all"],
        claims["feature_with_largest_max_abs_delta_JS"],
        claims["max_abs_delta_rank_across_all"],
        claims["feature_with_largest_max_abs_delta_rank"],
    )
    logger.info(
        "  near-subs freq>=%.2f: %d  (high-corr=%s  low-corr=%s)",
        near_sub_min_frequency,
        len(near_freq_ge),
        claims.get("n_near_sub_high_corr"),
        claims.get("n_near_sub_low_corr"),
    )

    return {
        "meta": meta,
        "claims": claims,
        "stability": stability,
        "extremes": extremes,
        "near_freq": near_freq,
        "near_freq_ge": near_freq_ge,
        "corr_summary": corr_summary,
        "issues": issues,
        "output_dir": output_dir,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate existing LOO descriptor-metric outputs into SI-oriented "
            "tables, figures, and SUMMARY.md (does not re-run LOO)."
        )
    )
    p.add_argument(
        "--loo-dir",
        type=Path,
        default=default_loo_dir(),
        help="Path to loo_on_profile/ with baseline + folds",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Aggregation output root (default: <loo-dir>/aggregation)",
    )
    p.add_argument(
        "--p-threshold",
        type=float,
        default=DEFAULT_P_THRESHOLD,
        help="Significance threshold (overridden by run_config.json if present)",
    )
    p.add_argument(
        "--top-near-substitutes",
        type=int,
        default=20,
        help="How many near-substitutes to show in top table / bar chart",
    )
    p.add_argument(
        "--near-sub-min-frequency",
        type=float,
        default=DEFAULT_NEAR_SUB_MIN_FREQUENCY,
        help="Keep near-substitutes with LOO frequency >= this (default 0.9)",
    )
    p.add_argument(
        "--high-corr-threshold",
        type=float,
        default=DEFAULT_HIGH_CORR_THRESHOLD,
        help="|r| >= this ⇒ high-corr within-group swap candidate (default 0.7)",
    )
    p.add_argument(
        "--low-corr-threshold",
        type=float,
        default=DEFAULT_LOW_CORR_THRESHOLD,
        help="max |r| < this ⇒ low-corr profile complement (default 0.4)",
    )
    p.add_argument(
        "--combined-descriptors",
        type=Path,
        default=None,
        help="combined_descriptors.csv for near-sub vs fixed-15 correlations",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    loo_dir = args.loo_dir
    if not loo_dir.is_dir():
        logger.error("LOO directory not found: %s", loo_dir)
        return 1
    output_dir = args.output_dir or default_output_dir(loo_dir)
    result = run_aggregation(
        loo_dir,
        output_dir,
        p_threshold=args.p_threshold,
        top_near_substitutes=args.top_near_substitutes,
        near_sub_min_frequency=args.near_sub_min_frequency,
        high_corr_threshold=args.high_corr_threshold,
        low_corr_threshold=args.low_corr_threshold,
        combined_descriptors_path=args.combined_descriptors,
    )
    logger.info("Done. Outputs under %s", result["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
