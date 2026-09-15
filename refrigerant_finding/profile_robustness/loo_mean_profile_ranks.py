#!/usr/bin/env python3
"""
Leave-one-out (LOO) sensitivity of the mean working-fluid profile (and, later,
candidate ranks).

Phase 1 (default, ``--stage profile``):
  Drop one of the 21 size-matched profile fluids, recompute the mean of the
  fixed 15 descriptors on the remaining 20, and measure how much the mean
  profile moves vs the full-21 baseline.

Phase 2 (``--stage ranks``):
  Re-rank full strategy candidate pools against each LOO mean and report
  top-100 retention, filtered Thermo-handoff retention (top-100 → drop
  known refrigerants + dups → ~65/56/64/23), and A–E distance/rank shifts.

Typical usage::

    python refrigerant_finding/profile_robustness/loo_mean_profile_ranks.py

    python refrigerant_finding/profile_robustness/loo_mean_profile_ranks.py \\
        --stage ranks \\
        --output-dir refrigerant_finding/profile_robustness/loo_mean_profile
"""

from __future__ import annotations

import argparse
import json
import logging
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

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None

logger = logging.getLogger(__name__)

# Ensure sibling imports work when launched as a script from any cwd.
_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

# Reuse descriptor list + refrigerant loader from the descriptor-metric LOO.
from loo_descriptor_metrics import (  # noqa: E402
    FIXED_15,
    default_combined_descriptors_path,
    default_refrigerants_csv_path,
    load_combined_descriptors,
)

# Strategy pools (relative to refrigerant_finding/)
STRATEGY_POOLS: dict[str, str] = {
    "I": "candidates",
    "IIa": "candidates_novel",
    "IIb": "candidates_novel_seguridad",
    "III": "candidates_refrigerantsmf",
}

# Hardcoded working-fluid mean profile actually used by the original
# screening/ranking pipeline (find_refrigerant_candidates.py, default
# behaviour when run without --use_calculated_means), sourced verbatim from
# REFRIGERANT_PROPERTIES.md. This is a *fixed* snapshot (rounded to 1
# significant decimal), NOT the mean dynamically recomputed from the 21
# size-matched profile fluids. It reproduces the historical CSV
# ``RefrigerantDistance`` column (and hence Tables S2-S5 / Table 1) exactly
# when combined with min-max scales taken from the full, unfiltered
# ``all_molecules_ranked.csv`` for the corresponding strategy pool (see
# ``paper_pool_minmax_scales`` below). Verified to reproduce the published
# distances for compounds A-E to >= 6 significant figures.
PAPER_HARDCODED_MEANS: dict[str, float] = {
    "MaxPartialCharge": 0.4,
    "SMR_VSA1": 16.5,
    "MinEStateIndex": -4.1,
    "MolMR": 17.9,
    "FpDensityMorgan3": 1.4,
    "MinPartialCharge": -0.2,
    "fr_halogen": 4.0,
    "fr_alkyl_halide": 3.3,
    "TPSA": 0.0,
    "NumHAcceptors": 0.0,
    "VSA_EState1": 40.2,
    "LabuteASA": 39.4,
    "MolLogP": 2.1,
    "NumHeteroatoms": 4.0,
    "NumRotatableBonds": 0.2,
}


def paper_hardcoded_mean(properties: list[str]) -> dict[str, float]:
    """Subset/order PAPER_HARDCODED_MEANS for a given descriptor list."""
    missing = [p for p in properties if p not in PAPER_HARDCODED_MEANS]
    if missing:
        raise ValueError(f"No hardcoded paper mean for descriptors: {missing}")
    return {p: PAPER_HARDCODED_MEANS[p] for p in properties}


def paper_pool_minmax_scales(
    pool_full: pd.DataFrame, properties: list[str]
) -> dict[str, dict[str, float]]:
    """
    Min-max scale reproducing the historical screening pipeline exactly.

    ``find_refrigerant_candidates.py`` computes min/max over
    ``concat(refrigerants_properties, database_properties)`` *before* any
    known-refrigerant/duplicate filtering, which for each strategy pool is
    equivalent to the full, unfiltered ``all_molecules_ranked.csv`` as saved
    (it already contains both the refrigerant-tagged and candidate-tagged
    rows). Do not pass the post-``prepare_strategy_pool`` pool here (e.g. for
    IIb, which is filtered down to the PubChem-absent subset) -- always pass
    the raw ``pool_full`` read straight from the CSV.
    """
    return compute_minmax_scales(pool_full, properties)
# Paper text: 65/56/64/23. Code/Excel reconstruction: 65/56/84/23 (IIb Excel).
PAPER_FILTERED_N = {"I": 65, "IIa": 56, "IIb": 64, "III": 23}
CODE_FILTERED_N = {"I": 65, "IIa": 56, "IIb": 84, "III": 23}
TOP_K = 100
# IIb (visualize_novel_molecules_grid.py): full novel ranked CSV → keep
# PubChem-absent in rank order → top-100 → dedup → Excel (84).
# NOT "IIa handoff (56) then PubChem" (that only yields ~17).
IIB_PUBCHEM_FILE = (
    "candidates_novel_seguridad/CID-SMILES-filtered-lt70_canonical.txt"
)
IIB_ABSENT_CACHE = "ranks/iib_pubchem_absent_candidate_smiles.txt"
# Shipped with this repo so type-IIb ranking does not require the ~4 GB PubChem dump.
SHIPPED_IIB_CACHE = (
    "candidates_novel_seguridad/iib_pubchem_absent_candidate_smiles.txt"
)


@dataclass(frozen=True)
class RunConfig:
    stage: str
    combined_descriptors: str
    refrigerants_csv: Optional[str]
    output_dir: str
    fixed_15: list[str]
    n_refrigerants: int
    scale_mode: str
    mean_mode: str
    timestamp_utc: str


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def default_output_dir() -> Path:
    return _package_dir() / "loo_mean_profile"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _safe_slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def compute_minmax_scales(
    values: pd.DataFrame, properties: list[str]
) -> dict[str, dict[str, float]]:
    """Per-descriptor min/max used to put the mean profile in [0, 1] space."""
    scales: dict[str, dict[str, float]] = {}
    for prop in properties:
        col = values[prop].astype(float)
        lo = float(np.nanmin(col.to_numpy()))
        hi = float(np.nanmax(col.to_numpy()))
        scales[prop] = {"min": lo, "max": hi}
    return scales


def mean_profile_raw(df: pd.DataFrame, properties: list[str]) -> dict[str, float]:
    return {p: float(df[p].astype(float).mean()) for p in properties}


def normalize_profile(
    profile: dict[str, float],
    scales: dict[str, dict[str, float]],
    properties: list[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in properties:
        lo = scales[p]["min"]
        hi = scales[p]["max"]
        if hi > lo:
            out[p] = (profile[p] - lo) / (hi - lo)
        else:
            out[p] = 0.0
    return out


def profile_to_series(profile: dict[str, float], properties: list[str]) -> pd.Series:
    return pd.Series({p: profile[p] for p in properties}, dtype=float)


def l2(a: dict[str, float], b: dict[str, float], properties: list[str]) -> float:
    va = np.array([a[p] for p in properties], dtype=float)
    vb = np.array([b[p] for p in properties], dtype=float)
    return float(np.linalg.norm(va - vb))


def max_abs(a: dict[str, float], b: dict[str, float], properties: list[str]) -> float:
    va = np.array([a[p] for p in properties], dtype=float)
    vb = np.array([b[p] for p in properties], dtype=float)
    return float(np.max(np.abs(va - vb)))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_profile_csv(
    path: Path,
    raw: dict[str, float],
    normalized: dict[str, float],
    properties: list[str],
) -> None:
    rows = [
        {
            "descriptor": p,
            "mean_raw": raw[p],
            "mean_normalized": normalized[p],
        }
        for p in properties
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def copy_or_write_manifest(
    output_dir: Path, refrigerants: pd.DataFrame
) -> None:
    """Write a compact refrigerant-only manifest for this experiment."""
    payload = {
        "n_refrigerants": int(len(refrigerants)),
        "refrigerants": [
            {
                "refrigerant_index": int(r["refrigerant_index"]),
                "refrigerant_id": r["refrigerant_id"],
                "compound": r["compound"],
                "smiles": r["smiles"],
                "HeavyAtomCount": (
                    int(r["HeavyAtomCount"])
                    if "HeavyAtomCount" in refrigerants.columns
                    and pd.notna(r["HeavyAtomCount"])
                    else None
                ),
            }
            for _, r in refrigerants.iterrows()
        ],
        "note": (
            "Same 21 size-matched profile fluids as loo_on_profile/"
            "molecule_manifest.json (refrigerant rows only)."
        ),
    }
    write_json(output_dir / "molecule_manifest.json", payload)


def run_profile_stage(
    refrigerants: pd.DataFrame,
    properties: list[str],
    output_dir: Path,
    *,
    scale_mode: str = "freeze_baseline_21",
) -> dict[str, Any]:
    """
    Baseline + 21 LOO folds for mean-profile variation only.

    scale_mode:
      freeze_baseline_21 — min/max from all 21 refrigerants (default; isolates
      mean shift in a fixed normalized space, good for profile-stability
      reporting before candidate re-ranking).
    """
    if scale_mode != "freeze_baseline_21":
        raise ValueError(f"Unsupported scale_mode for profile stage: {scale_mode}")

    baseline_dir = output_dir / "baseline"
    folds_dir = output_dir / "folds"
    agg_dir = output_dir / "aggregate"
    for d in (baseline_dir, folds_dir, agg_dir):
        d.mkdir(parents=True, exist_ok=True)

    missing = [p for p in properties if p not in refrigerants.columns]
    if missing:
        raise ValueError(f"Missing descriptors in refrigerant table: {missing}")

    scales = compute_minmax_scales(refrigerants, properties)
    write_json(baseline_dir / "minmax_scales.json", scales)

    baseline_raw = mean_profile_raw(refrigerants, properties)
    baseline_norm = normalize_profile(baseline_raw, scales, properties)
    write_profile_csv(
        baseline_dir / "mean_profile.csv", baseline_raw, baseline_norm, properties
    )
    write_json(
        baseline_dir / "mean_profile.json",
        {
            "n_fluids": int(len(refrigerants)),
            "raw": baseline_raw,
            "normalized": baseline_norm,
            "scale_mode": scale_mode,
        },
    )

    fold_rows: list[dict[str, Any]] = []
    delta_raw_rows: list[dict[str, Any]] = []
    delta_norm_rows: list[dict[str, Any]] = []
    per_desc_abs_norm: dict[str, list[float]] = {p: [] for p in properties}

    for _, row in refrigerants.iterrows():
        idx = int(row["refrigerant_index"])
        compound = str(row["compound"])
        rid = str(row["refrigerant_id"])
        fold_name = f"fold_{idx:02d}_{_safe_slug(compound)}"
        fold_dir = folds_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        train = refrigerants.loc[refrigerants["refrigerant_index"] != idx].copy()
        fold_raw = mean_profile_raw(train, properties)
        fold_norm = normalize_profile(fold_raw, scales, properties)

        write_json(
            fold_dir / "left_out.json",
            {
                "fold_name": fold_name,
                "left_out_refrigerant_index": idx,
                "left_out_refrigerant_id": rid,
                "left_out_compound": compound,
                "left_out_smiles": row["smiles"],
                "n_train_refrigerants": int(len(train)),
            },
        )
        write_profile_csv(
            fold_dir / "mean_profile.csv", fold_raw, fold_norm, properties
        )

        d_raw = {p: fold_raw[p] - baseline_raw[p] for p in properties}
        d_norm = {p: fold_norm[p] - baseline_norm[p] for p in properties}
        abs_norm = {p: abs(d_norm[p]) for p in properties}
        for p in properties:
            per_desc_abs_norm[p].append(abs_norm[p])

        delta_raw_rows.append(
            {
                "fold_name": fold_name,
                "left_out_compound": compound,
                "left_out_refrigerant_id": rid,
                **{f"delta_{p}": d_raw[p] for p in properties},
            }
        )
        delta_norm_rows.append(
            {
                "fold_name": fold_name,
                "left_out_compound": compound,
                "left_out_refrigerant_id": rid,
                **{f"delta_{p}": d_norm[p] for p in properties},
            }
        )

        summary = {
            "fold_name": fold_name,
            "left_out_compound": compound,
            "left_out_refrigerant_id": rid,
            "n_train": int(len(train)),
            "l2_delta_raw": l2(fold_raw, baseline_raw, properties),
            "l2_delta_normalized": l2(fold_norm, baseline_norm, properties),
            "max_abs_delta_raw": max_abs(fold_raw, baseline_raw, properties),
            "max_abs_delta_normalized": max_abs(fold_norm, baseline_norm, properties),
            "mean_abs_delta_normalized": float(np.mean(list(abs_norm.values()))),
            "argmax_abs_delta_normalized": max(abs_norm, key=abs_norm.get),
        }
        write_json(fold_dir / "profile_delta_summary.json", summary)
        fold_rows.append(summary)
        logger.info(
            "%s: L2_norm=%.4g  max|Δ|_norm=%.4g (%s)",
            fold_name,
            summary["l2_delta_normalized"],
            summary["max_abs_delta_normalized"],
            summary["argmax_abs_delta_normalized"],
        )

    fold_summary = pd.DataFrame(fold_rows).sort_values("fold_name")
    fold_summary.to_csv(agg_dir / "fold_profile_deltas.csv", index=False)
    pd.DataFrame(delta_raw_rows).to_csv(
        agg_dir / "delta_mean_raw_by_fold.csv", index=False
    )
    pd.DataFrame(delta_norm_rows).to_csv(
        agg_dir / "delta_mean_normalized_by_fold.csv", index=False
    )

    # Wide matrices: folds × descriptors
    mat_raw = pd.DataFrame(delta_raw_rows).set_index("left_out_compound")
    mat_norm = pd.DataFrame(delta_norm_rows).set_index("left_out_compound")
    raw_cols = [f"delta_{p}" for p in properties]
    mat_raw[raw_cols].to_csv(agg_dir / "matrix_delta_mean_raw.csv")
    mat_norm[raw_cols].to_csv(agg_dir / "matrix_delta_mean_normalized.csv")

    desc_stability = pd.DataFrame(
        [
            {
                "descriptor": p,
                "mean_abs_delta_normalized": float(np.mean(per_desc_abs_norm[p])),
                "max_abs_delta_normalized": float(np.max(per_desc_abs_norm[p])),
                "std_abs_delta_normalized": float(np.std(per_desc_abs_norm[p])),
            }
            for p in properties
        ]
    ).sort_values("mean_abs_delta_normalized", ascending=False)
    desc_stability.to_csv(agg_dir / "descriptor_mean_stability.csv", index=False)

    headline = {
        "n_folds": int(len(fold_rows)),
        "scale_mode": scale_mode,
        "l2_delta_normalized": {
            "mean": float(fold_summary["l2_delta_normalized"].mean()),
            "median": float(fold_summary["l2_delta_normalized"].median()),
            "max": float(fold_summary["l2_delta_normalized"].max()),
            "max_fold": str(
                fold_summary.loc[
                    fold_summary["l2_delta_normalized"].idxmax(), "left_out_compound"
                ]
            ),
        },
        "max_abs_delta_normalized": {
            "mean": float(fold_summary["max_abs_delta_normalized"].mean()),
            "median": float(fold_summary["max_abs_delta_normalized"].median()),
            "max": float(fold_summary["max_abs_delta_normalized"].max()),
        },
        "mean_abs_delta_normalized": {
            "mean": float(fold_summary["mean_abs_delta_normalized"].mean()),
            "median": float(fold_summary["mean_abs_delta_normalized"].median()),
            "max": float(fold_summary["mean_abs_delta_normalized"].max()),
        },
        "most_sensitive_descriptors": desc_stability.head(5)["descriptor"].tolist(),
        "least_sensitive_descriptors": desc_stability.tail(5)["descriptor"].tolist()[
            ::-1
        ],
    }
    write_json(agg_dir / "headline_claims.json", headline)

    summary_md = _format_profile_summary_md(headline, fold_summary, desc_stability)
    (agg_dir / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    return headline


def _format_profile_summary_md(
    headline: dict[str, Any],
    fold_summary: pd.DataFrame,
    desc_stability: pd.DataFrame,
) -> str:
    lines = [
        "# LOO mean-profile variation — summary",
        "",
        "Leave-one-out over the 21 size-matched working fluids; fixed curated 15",
        "descriptors. Each fold recomputes the **mean profile** on the remaining",
        "20 fluids. Candidate re-ranking is **not** included in this stage.",
        "",
        f"- Folds: **{headline['n_folds']}**",
        f"- Scale mode: `{headline['scale_mode']}` "
        "(min–max from all 21; applied to raw means for comparable L2)",
        "",
        "## Headline (normalized space)",
        "",
        f"- Mean L2(Δ mean): **{headline['l2_delta_normalized']['mean']:.4g}** "
        f"(median {headline['l2_delta_normalized']['median']:.4g}; "
        f"max {headline['l2_delta_normalized']['max']:.4g} "
        f"when leaving out **{headline['l2_delta_normalized']['max_fold']}**)",
        f"- Mean of per-fold max|Δ|: "
        f"**{headline['max_abs_delta_normalized']['mean']:.4g}** "
        f"(worst fold max|Δ| = {headline['max_abs_delta_normalized']['max']:.4g})",
        f"- Mean of per-fold mean|Δ|: "
        f"**{headline['mean_abs_delta_normalized']['mean']:.4g}**",
        "",
        "### Most sensitive descriptors (by mean |Δ| across folds)",
        "",
    ]
    for _, r in desc_stability.head(5).iterrows():
        lines.append(
            f"- `{r['descriptor']}`: mean|Δ|={r['mean_abs_delta_normalized']:.4g}, "
            f"max|Δ|={r['max_abs_delta_normalized']:.4g}"
        )
    lines += [
        "",
        "### Least sensitive descriptors",
        "",
    ]
    for _, r in desc_stability.iloc[::-1].head(5).iterrows():
        lines.append(
            f"- `{r['descriptor']}`: mean|Δ|={r['mean_abs_delta_normalized']:.4g}"
        )
    lines += [
        "",
        "## Per-fold L2 (normalized)",
        "",
        "| Left out | L2(Δ) | max\\|Δ\\| | argmax |",
        "|---|---:|---:|---|",
    ]
    for _, r in fold_summary.sort_values(
        "l2_delta_normalized", ascending=False
    ).iterrows():
        lines.append(
            f"| {r['left_out_compound']} | {r['l2_delta_normalized']:.4g} | "
            f"{r['max_abs_delta_normalized']:.4g} | "
            f"{r['argmax_abs_delta_normalized']} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `fold_profile_deltas.csv` — one row per fold",
        "- `matrix_delta_mean_normalized.csv` — folds × descriptors",
        "- `descriptor_mean_stability.csv` — per-descriptor sensitivity",
        "- `headline_claims.json`",
        "",
        "## Next stage",
        "",
        "Run `--stage ranks` for full candidate re-rank + retention / A–E metrics.",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stage 2 — full re-rank
# ---------------------------------------------------------------------------
def _rf_root() -> Path:
    return _package_dir().parent


def canonical_smiles(s: Any) -> Optional[str]:
    raw = str(s).replace(" ", "").strip()
    if not raw or raw.lower() == "nan":
        return None
    if Chem is None:
        return raw
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return raw
    return Chem.MolToSmiles(mol)


def load_known_refrigerant_smiles(refrigerants_csv: Path) -> set[str]:
    refs = pd.read_csv(refrigerants_csv)
    col = "SMILES" if "SMILES" in refs.columns else "smiles"
    out: set[str] = set()
    for s in refs[col]:
        c = canonical_smiles(s)
        if c:
            out.add(c)
    return out


def load_focal_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["compounds"])


def normalize_matrix(
    raw: np.ndarray, scales: dict[str, dict[str, float]], properties: list[str]
) -> np.ndarray:
    """raw: (n, p) → normalized with frozen min–max scales."""
    out = np.zeros_like(raw, dtype=float)
    for j, prop in enumerate(properties):
        lo = scales[prop]["min"]
        hi = scales[prop]["max"]
        col = raw[:, j]
        if hi > lo:
            out[:, j] = (col - lo) / (hi - lo)
        else:
            out[:, j] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_mean_vector(
    mean_raw: dict[str, float],
    scales: dict[str, dict[str, float]],
    properties: list[str],
) -> np.ndarray:
    return normalize_matrix(
        np.array([[mean_raw[p] for p in properties]], dtype=float),
        scales,
        properties,
    )[0]


def euclidean_distances(norm_X: np.ndarray, mean_norm: np.ndarray) -> np.ndarray:
    diff = norm_X - mean_norm.reshape(1, -1)
    return np.linalg.norm(diff, axis=1)


def apply_thermo_filter(
    ranked: pd.DataFrame,
    known: set[str],
    *,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Shared handoff path after a strategy-specific ranking universe is fixed:
    top-k by distance → drop known refrigerants → drop duplicate canonical
    SMILES (keep best / first after sort).

    For IIb the ranking universe must already be PubChem-absent candidates
    (see ``load_iib_pubchem_absent_smiles``).
    """
    top = ranked.nsmallest(top_k, "distance").copy()
    if "molecule_type" in top.columns:
        top = top[top["molecule_type"].astype(str) != "refrigerant"]
    top = top[~top["canon_smiles"].isin(known)].copy()
    top = top.drop_duplicates(subset=["canon_smiles"], keep="first")
    return top.reset_index(drop=True)


def load_iib_pubchem_absent_smiles(
    output_dir: Path,
    pool_csv: Path,
    *,
    pubchem_path: Optional[Path] = None,
    refresh: bool = False,
) -> set[str]:
    """
    SMILES (space-stripped) of type-II candidates absent from the local
    PubChem CID-SMILES dump.

    Caches the absent set under ``loo_mean_profile/ranks/`` because the
    PubChem file is ~96M lines.
    """
    cache = output_dir / IIB_ABSENT_CACHE
    cache_candidates = []
    if cache.is_file() and not refresh:
        cache_candidates.append(cache)
    shipped = _rf_root() / SHIPPED_IIB_CACHE
    if shipped.is_file() and not refresh:
        cache_candidates.append(shipped)
    if cache_candidates:
        smiles = {
            ln.strip()
            for ln in cache_candidates[0].read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        logger.info(
            "Loaded cached IIb PubChem-absent SMILES: %d (%s)",
            len(smiles),
            cache_candidates[0],
        )
        return smiles

    pub = pubchem_path or (_rf_root() / IIB_PUBCHEM_FILE)
    if not pub.is_file():
        raise FileNotFoundError(
            f"IIb PubChem dump not found: {pub}. Needed to define type-IIb "
            "PubChem-absent ranking universe."
        )

    pool = pd.read_csv(pool_csv, usecols=["smiles", "molecule_type"])
    cand_set = {
        str(s).replace(" ", "")
        for s, t in zip(pool["smiles"], pool["molecule_type"])
        if str(t) == "candidate"
    }
    logger.info(
        "Scanning PubChem dump for IIb novelty filter (%d candidate SMILES)…",
        len(cand_set),
    )
    found: set[str] = set()
    with open(pub, "r", buffering=1024 * 1024, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            s = line.strip()
            if s in cand_set:
                found.add(s)
            if i % 10_000_000 == 0:
                logger.info("  PubChem scan %d lines, hits=%d", i, len(found))
    absent = cand_set - found
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sorted(absent)) + "\n", encoding="utf-8")
    logger.info(
        "IIb PubChem-absent candidates: %d / %d (cached %s)",
        len(absent),
        len(cand_set),
        cache,
    )
    return absent


def prepare_strategy_pool(
    strategy: str,
    pool: pd.DataFrame,
    *,
    iib_absent: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Restrict ranking universe per strategy.

    - I / IIa / III: full ranked CSV (candidates + refrigerants in file)
    - IIb: PubChem-absent molecules kept in the **original distance order**
      of the full novel CSV (equivalent to walking the ranked list and
      dropping PubChem hits, as in ``visualize_novel_molecules_grid.py``).
      This is *not* "take the IIa Thermo handoff (56) then drop PubChem".
    """
    if strategy != "IIb":
        return pool
    if not iib_absent:
        raise ValueError("IIb requires PubChem-absent SMILES set")
    s = pool["smiles"].astype(str).str.replace(" ", "", regex=False)
    # Prefer candidates; PubChem-absent refrigerants are rare/irrelevant here.
    is_cand = pool["molecule_type"].astype(str) == "candidate"
    out = pool.loc[is_cand & s.isin(iib_absent)].copy()
    # Preserve original ranking order (CSV is pre-sorted by RefrigerantDistance)
    if "RefrigerantDistance" in out.columns:
        out = out.sort_values("RefrigerantDistance", kind="mergesort")
    logger.info(
        "IIb ranking universe: %d PubChem-absent candidates (from %d rows; "
        "same procedure as visualize_novel_molecules_grid.py)",
        len(out),
        len(pool),
    )
    return out.reset_index(drop=True)


def rank_from_precomputed_distance(pool_meta: pd.DataFrame) -> pd.DataFrame:
    """Baseline ranks from the CSV's original RefrigerantDistance column."""
    if "distance" not in pool_meta.columns:
        raise ValueError("pool_meta must include precomputed 'distance'")
    out = pool_meta.sort_values("distance", kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _rank_lookup(ranked: pd.DataFrame, canon: str) -> tuple[Optional[int], Optional[float]]:
    hits = ranked.index[ranked["canon_smiles"] == canon].tolist()
    if not hits:
        return None, None
    # ranked is sorted by distance ascending with a 1-based rank column
    row = ranked.loc[hits[0]]
    return int(row["rank"]), float(row["distance"])


def _filtered_rank_lookup(filtered: pd.DataFrame, canon: str) -> Optional[int]:
    """1-based position within the Thermo handoff list (sorted by distance)."""
    if filtered.empty or "canon_smiles" not in filtered.columns:
        return None
    hits = filtered.index[filtered["canon_smiles"] == canon].tolist()
    if not hits:
        return None
    # filtered is already distance-sorted; use positional rank in that list
    pos = list(filtered["canon_smiles"]).index(canon)
    return int(pos + 1)


def rank_pool_against_mean(
    pool_meta: pd.DataFrame,
    norm_X: np.ndarray,
    mean_norm: np.ndarray,
) -> pd.DataFrame:
    dist = euclidean_distances(norm_X, mean_norm)
    out = pool_meta.copy()
    out["distance"] = dist
    out = out.sort_values("distance", kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def spearman_aligned(
    base_ranked: pd.DataFrame, fold_ranked: pd.DataFrame
) -> dict[str, float]:
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


def _plot_rank_heatmap(
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
    fig, ax = plt.subplots(
        figsize=(8.5, max(5.0, 0.28 * len(pivot) + 1.8)),
        constrained_layout=True,
    )
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index), fontsize=7)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Left-out fluid")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)
    for path in (fig_dir / f"{stem}.pdf", fig_dir / f"{stem}.png"):
        fig.savefig(path, dpi=150)
    plt.close(fig)


def write_fluid_loo_rank_figures(retention_df: pd.DataFrame, agg_dir: Path) -> None:
    """Matrices + heatmaps for Spearman / top-100 / filtered retention."""
    fig_dir = agg_dir / "figures"
    # Stable fluid order from fold index embedded in fold_name
    order = (
        retention_df[["fold_name", "left_out_compound"]]
        .drop_duplicates()
        .sort_values("fold_name")
    )
    fluid_order = list(order["left_out_compound"])

    def _pivot(col: str) -> pd.DataFrame:
        p = retention_df.pivot(
            index="left_out_compound", columns="strategy", values=col
        )
        return p.reindex(fluid_order)

    pivot_sp = _pivot("spearman_rho")
    pivot_sp100 = _pivot("spearman_rho_top100_baseline")
    pivot_ret100 = _pivot("top100_retention")
    pivot_ret_filt = _pivot("filtered_retention")

    pivot_sp.to_csv(agg_dir / "matrix_spearman_rho.csv")
    pivot_sp100.to_csv(agg_dir / "matrix_spearman_rho_top100.csv")
    pivot_ret100.to_csv(agg_dir / "matrix_top100_retention.csv")
    pivot_ret_filt.to_csv(agg_dir / "matrix_filtered_retention.csv")

    def _vmin(p: pd.DataFrame, floor: float, default: float) -> float:
        m = float(np.nanmin(p.to_numpy(dtype=float)))
        return max(floor, min(default, m - 0.05))

    _plot_rank_heatmap(
        pivot_sp,
        fig_dir,
        title="Spearman ρ vs baseline ranking (full universe)",
        stem="fig_spearman_heatmap",
        vmin=_vmin(pivot_sp, 0.7, 0.85),
        cbar_label="ρ",
    )
    _plot_rank_heatmap(
        pivot_sp100,
        fig_dir,
        title="Spearman ρ among baseline top-100 molecules",
        stem="fig_spearman_heatmap_top100",
        vmin=_vmin(pivot_sp100, 0.0, 0.5),
        cbar_label="ρ",
    )
    _plot_rank_heatmap(
        pivot_ret100,
        fig_dir,
        title="Top-100 retention (unique SMILES vs baseline)",
        stem="fig_top100_retention_heatmap",
        vmin=_vmin(pivot_ret100, 0.5, 0.85),
        cbar_label="retention",
    )
    _plot_rank_heatmap(
        pivot_ret_filt,
        fig_dir,
        title="Filtered-set retention vs baseline",
        stem="fig_filtered_retention_heatmap",
        vmin=_vmin(pivot_ret_filt, 0.5, 0.85),
        cbar_label="retention",
    )
    logger.info("Wrote fluid-LOO rank heatmaps under %s", fig_dir)


def run_ranks_stage(
    refrigerants: pd.DataFrame,
    properties: list[str],
    output_dir: Path,
    *,
    refrigerants_csv: Path,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    Full re-rank of strategy pools under baseline + each LOO mean.

    Baseline uses each CSV's original ``RefrigerantDistance`` so the Thermo
    handoff sizes reproduce the historical artifacts (65 / 56 / 84 / 23;
    IIb Excel). LOO folds recompute Euclidean distances with a calculated
    mean and scales frozen on refrigerants ∪ ranking universe.
    """
    ranks_root = output_dir / "ranks"
    agg_dir = ranks_root / "aggregate"
    ranks_root.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    known = load_known_refrigerant_smiles(refrigerants_csv)
    focal_path = output_dir / "focal_candidates.json"
    if not focal_path.is_file():
        # fall back to package default next to this script's output template
        focal_path = _package_dir() / "loo_mean_profile" / "focal_candidates.json"
    focals = load_focal_candidates(focal_path)
    write_json(ranks_root / "known_refrigerants_canonical.json", sorted(known))
    write_json(
        ranks_root / "procedure_notes.json",
        {
            "I_IIa_III": (
                "full all_molecules_ranked.csv → top-100 by RefrigerantDistance "
                "→ drop molecule_type=refrigerant + known refrigerants.csv → "
                "drop duplicate canonical SMILES"
            ),
            "IIb": (
                "same novel ranked CSV as IIa → keep PubChem-absent SMILES in "
                "rank order (visualize_novel_molecules_grid.py) → top-100 → "
                "drop duplicate canonical SMILES → Excel_candidatos_novel_pubchem "
                "(84). NOT: IIa filtered-56 then PubChem filter (that yields ~17)."
            ),
            "baseline_distance_source": "CSV RefrigerantDistance (historical)",
            "loo_distance_source": "recomputed Euclidean vs LOO mean (frozen scales)",
        },
    )

    iib_pool_path = _rf_root() / STRATEGY_POOLS["IIb"] / "all_molecules_ranked.csv"
    iib_absent = load_iib_pubchem_absent_smiles(output_dir, iib_pool_path)

    # Precompute raw / LOO means (raw space) for fold re-ranks
    fold_means: list[dict[str, Any]] = []
    for _, row in refrigerants.iterrows():
        idx = int(row["refrigerant_index"])
        train = refrigerants.loc[refrigerants["refrigerant_index"] != idx]
        fold_means.append(
            {
                "fold_name": f"fold_{idx:02d}_{_safe_slug(str(row['compound']))}",
                "left_out_compound": str(row["compound"]),
                "left_out_refrigerant_id": str(row["refrigerant_id"]),
                "mean_raw": mean_profile_raw(train, properties),
            }
        )

    retention_rows: list[dict[str, Any]] = []
    ae_rows: list[dict[str, Any]] = []
    strategy_summaries: dict[str, Any] = {}

    for strategy, rel in STRATEGY_POOLS.items():
        pool_path = _rf_root() / rel / "all_molecules_ranked.csv"
        if not pool_path.is_file():
            logger.warning("Missing pool for %s: %s", strategy, pool_path)
            continue
        logger.info("Ranking strategy %s from %s", strategy, pool_path)
        pool_full = pd.read_csv(pool_path)
        missing = [p for p in properties if p not in pool_full.columns]
        if missing:
            raise ValueError(f"{pool_path}: missing raw descriptors {missing}")
        if "RefrigerantDistance" not in pool_full.columns:
            raise ValueError(f"{pool_path}: missing RefrigerantDistance")

        pool = prepare_strategy_pool(
            strategy, pool_full, iib_absent=iib_absent if strategy == "IIb" else None
        )
        pool = pool.copy()
        pool["canon_smiles"] = pool["smiles"].map(canonical_smiles)
        pool["mol_id"] = np.arange(len(pool), dtype=int)
        pool_meta = pool[
            ["mol_id", "smiles", "canon_smiles", "RefrigerantDistance"]
            + (["molecule_type"] if "molecule_type" in pool.columns else [])
            + (["MolName"] if "MolName" in pool.columns else [])
        ].copy()
        pool_meta = pool_meta.rename(columns={"RefrigerantDistance": "distance"})
        raw_X = pool[properties].astype(float).to_numpy()

        # Freeze scales: refrigerants ∪ this ranking universe (for LOO folds)
        scale_df = pd.concat(
            [refrigerants[properties], pool[properties]],
            ignore_index=True,
        )
        scales = compute_minmax_scales(scale_df, properties)
        norm_X = normalize_matrix(raw_X, scales, properties)

        strat_dir = ranks_root / f"strategy_{strategy}"
        strat_dir.mkdir(parents=True, exist_ok=True)
        write_json(strat_dir / "minmax_scales.json", scales)

        # ---- baseline from ORIGINAL CSV distances (reproduce historical handoff)
        base_ranked = rank_from_precomputed_distance(pool_meta)
        base_top = base_ranked.nsmallest(top_k, "distance")
        base_filtered = apply_thermo_filter(base_ranked, known, top_k=top_k)
        # Retention is over unique canonical SMILES (duplicate rows in top-100
        # collapse). Row count can be < top_k for small pools (e.g. III).
        base_top_set = set(base_top["canon_smiles"].dropna())
        base_filt_set = set(base_filtered["canon_smiles"].dropna())
        base_top_n_rows = int(len(base_top))
        base_top_n_unique = int(len(base_top_set))
        base_top_mol_ids = set(base_top["mol_id"].astype(int))

        base_top[["mol_id", "smiles", "canon_smiles", "distance", "rank"]].to_csv(
            strat_dir / "baseline_top100.csv", index=False
        )
        base_filt_out = base_filtered.copy()
        base_filt_out["rank_in_filtered"] = np.arange(1, len(base_filt_out) + 1)
        base_filt_out[
            ["mol_id", "smiles", "canon_smiles", "distance", "rank", "rank_in_filtered"]
        ].to_csv(strat_dir / "baseline_filtered.csv", index=False)

        paper_n = PAPER_FILTERED_N.get(strategy)
        code_n = CODE_FILTERED_N.get(strategy)
        strategy_summaries[strategy] = {
            "pool_path": str(pool_path),
            "n_pool_full_csv": int(len(pool_full)),
            "n_ranking_universe": int(len(pool)),
            "ranking_universe_note": (
                "PubChem-absent candidates in original novel-CSV rank order "
                "(visualize_novel_molecules_grid.py); NOT IIa-handoff→PubChem"
                if strategy == "IIb"
                else "full ranked CSV (candidates + refrigerants present in file)"
            ),
            "baseline_distance_source": "CSV RefrigerantDistance",
            "baseline_top_k_rows": int(len(base_top)),
            "baseline_top_k_unique": int(len(base_top_set)),
            "baseline_filtered_n": int(len(base_filtered)),
            "paper_filtered_n": paper_n,
            "code_excel_filtered_n": code_n,
            "filtered_n_matches_code_excel": (
                int(len(base_filtered)) == code_n if code_n is not None else None
            ),
            "filtered_n_matches_paper": (
                int(len(base_filtered)) == paper_n if paper_n is not None else None
            ),
        }
        logger.info(
            "  %s baseline filtered n=%d (code/Excel %s, paper %s); universe=%d",
            strategy,
            len(base_filtered),
            code_n,
            paper_n,
            len(pool),
        )

        # Focal compounds in this strategy
        focals_here = [f for f in focals if f["strategy"] == strategy]
        for f in focals_here:
            c = canonical_smiles(f["smiles"])
            rnk, dist = _rank_lookup(base_ranked, c)
            rnk_filt = _filtered_rank_lookup(base_filtered, c)
            ae_rows.append(
                {
                    "code": f["code"],
                    "strategy": strategy,
                    "fold": "baseline",
                    "left_out_compound": None,
                    "smiles": f["smiles"],
                    "canon_smiles": c,
                    "distance": dist,
                    "rank": rnk,
                    "rank_in_filtered": rnk_filt,
                    "in_top100": bool(rnk is not None and rnk <= top_k),
                    "in_filtered": bool(rnk_filt is not None),
                    "delta_distance": 0.0 if dist is not None else None,
                    "delta_rank": 0 if rnk is not None else None,
                    "entered_top100": False,
                    "left_top100": False,
                }
            )

        # ---- LOO folds ----
        for fm in fold_means:
            mean_n = normalize_mean_vector(fm["mean_raw"], scales, properties)
            ranked = rank_pool_against_mean(pool_meta, norm_X, mean_n)
            top = ranked.nsmallest(top_k, "distance")
            filtered = apply_thermo_filter(ranked, known, top_k=top_k)
            top_set = set(top["canon_smiles"].dropna())
            filt_set = set(filtered["canon_smiles"].dropna())

            n_top_kept = len(base_top_set & top_set)
            n_filt_kept = len(base_filt_set & filt_set)
            sp = spearman_aligned(base_ranked, ranked)
            sp100 = spearman_on_subset(base_ranked, ranked, base_top_mol_ids)
            retention_rows.append(
                {
                    "strategy": strategy,
                    "fold_name": fm["fold_name"],
                    "left_out_compound": fm["left_out_compound"],
                    "left_out_refrigerant_id": fm["left_out_refrigerant_id"],
                    **sp,
                    **sp100,
                    "top100_baseline_rows": base_top_n_rows,
                    "top100_baseline_unique": base_top_n_unique,
                    "top100_kept": n_top_kept,
                    "top100_baseline": base_top_n_unique,  # alias: unique SMILES
                    "top100_retention": n_top_kept / max(base_top_n_unique, 1),
                    "filtered_kept": n_filt_kept,
                    "filtered_baseline": len(base_filt_set),
                    "filtered_new_size": int(len(filtered)),
                    "filtered_retention": n_filt_kept / max(len(base_filt_set), 1),
                }
            )

            for f in focals_here:
                c = canonical_smiles(f["smiles"])
                base_ae = next(
                    r
                    for r in ae_rows
                    if r["code"] == f["code"] and r["fold"] == "baseline"
                )
                rnk, dist = _rank_lookup(ranked, c)
                rnk_filt = _filtered_rank_lookup(filtered, c)
                in_top = bool(rnk is not None and rnk <= top_k)
                was_top = bool(base_ae["in_top100"])
                ae_rows.append(
                    {
                        "code": f["code"],
                        "strategy": strategy,
                        "fold": fm["fold_name"],
                        "left_out_compound": fm["left_out_compound"],
                        "smiles": f["smiles"],
                        "canon_smiles": c,
                        "distance": dist,
                        "rank": rnk,
                        "rank_in_filtered": rnk_filt,
                        "in_top100": in_top,
                        "in_filtered": bool(rnk_filt is not None),
                        "delta_distance": (
                            None
                            if dist is None or base_ae["distance"] is None
                            else float(dist - base_ae["distance"])
                        ),
                        "delta_rank": (
                            None
                            if rnk is None or base_ae["rank"] is None
                            else int(rnk - base_ae["rank"])
                        ),
                        "entered_top100": bool(in_top and not was_top),
                        "left_top100": bool(was_top and not in_top),
                    }
                )

        # per-strategy fold table
        strat_ret = pd.DataFrame(
            [r for r in retention_rows if r["strategy"] == strategy]
        )
        strat_ret.to_csv(strat_dir / "fold_retention.csv", index=False)

    retention_df = pd.DataFrame(retention_rows)
    ae_df = pd.DataFrame(ae_rows)
    retention_df.to_csv(agg_dir / "fold_retention_all_strategies.csv", index=False)
    ae_df.to_csv(agg_dir / "AE_distance_rank_by_fold.csv", index=False)

    # Summary tables
    ret_summary = (
        retention_df.groupby("strategy")
        .agg(
            top100_baseline_rows=("top100_baseline_rows", "first"),
            top100_baseline_unique=("top100_baseline_unique", "first"),
            top100_baseline=("top100_baseline", "first"),
            filtered_baseline=("filtered_baseline", "first"),
            spearman_rho_mean=("spearman_rho", "mean"),
            spearman_rho_min=("spearman_rho", "min"),
            spearman_top100_mean=("spearman_rho_top100_baseline", "mean"),
            spearman_top100_min=("spearman_rho_top100_baseline", "min"),
            top100_kept_mean=("top100_kept", "mean"),
            top100_kept_min=("top100_kept", "min"),
            top100_retention_mean=("top100_retention", "mean"),
            top100_retention_min=("top100_retention", "min"),
            filtered_kept_mean=("filtered_kept", "mean"),
            filtered_kept_min=("filtered_kept", "min"),
            filtered_retention_mean=("filtered_retention", "mean"),
            filtered_retention_min=("filtered_retention", "min"),
            filtered_new_size_mean=("filtered_new_size", "mean"),
        )
        .reset_index()
    )
    ret_summary.to_csv(agg_dir / "retention_summary_by_strategy.csv", index=False)
    write_fluid_loo_rank_figures(retention_df, agg_dir)

    ae_base = ae_df[ae_df["fold"] == "baseline"].copy()
    ae_loo = ae_df[ae_df["fold"] != "baseline"].copy()
    ae_sum_rows = []
    for code, g in ae_loo.groupby("code"):
        base = ae_base[ae_base["code"] == code].iloc[0]
        n_folds = int(len(g))
        n_top = int(g["in_top100"].sum())
        n_filt = int(g["in_filtered"].sum())
        g_top = g.loc[g["in_top100"]]
        g_filt = g.loc[g["in_filtered"]]
        ae_sum_rows.append(
            {
                "code": code,
                "strategy": base["strategy"],
                # Baseline
                "baseline_rank": base["rank"],
                "baseline_rank_in_filtered": base["rank_in_filtered"],
                "baseline_distance": base["distance"],
                "baseline_in_top100": base["in_top100"],
                "baseline_in_filtered": base["in_filtered"],
                # LOO top-100
                "pct_folds_in_top100": 100.0 * n_top / max(n_folds, 1),
                "mean_rank_loo": g["rank"].mean(skipna=True),
                "mean_distance_loo": g["distance"].mean(skipna=True),
                "mean_rank_loo_when_in_top100": (
                    g_top["rank"].mean() if n_top else None
                ),
                "mean_distance_loo_when_in_top100": (
                    g_top["distance"].mean() if n_top else None
                ),
                # LOO filtered (Thermo handoff)
                "pct_folds_in_filtered": 100.0 * n_filt / max(n_folds, 1),
                "mean_rank_in_filtered_when_passed": (
                    g_filt["rank_in_filtered"].mean() if n_filt else None
                ),
                "mean_distance_when_filtered": (
                    g_filt["distance"].mean() if n_filt else None
                ),
                "mean_universe_rank_when_filtered": (
                    g_filt["rank"].mean() if n_filt else None
                ),
                # extras
                "n_folds_total": n_folds,
                "n_folds_in_top100": n_top,
                "n_folds_in_filtered": n_filt,
                "n_left_top100": int(g["left_top100"].sum()),
                "n_entered_top100": int(g["entered_top100"].sum()),
                "max_abs_delta_rank": g["delta_rank"].abs().max(skipna=True),
                "mean_delta_distance": g["delta_distance"].mean(skipna=True),
            }
        )
    ae_summary = pd.DataFrame(ae_sum_rows).sort_values("code")
    ae_summary.to_csv(agg_dir / "AE_summary.csv", index=False)

    headline = {
        "top_k": top_k,
        "n_folds": int(refrigerants["refrigerant_index"].nunique()),
        "strategies": strategy_summaries,
        "spearman_rho_mean_by_strategy": {
            r["strategy"]: float(r["spearman_rho_mean"])
            for _, r in ret_summary.iterrows()
        },
        "spearman_top100_mean_by_strategy": {
            r["strategy"]: float(r["spearman_top100_mean"])
            for _, r in ret_summary.iterrows()
        },
        "retention_mean_top100": {
            r["strategy"]: float(r["top100_retention_mean"])
            for _, r in ret_summary.iterrows()
        },
        "retention_mean_filtered": {
            r["strategy"]: float(r["filtered_retention_mean"])
            for _, r in ret_summary.iterrows()
        },
        "AE_baseline": ae_base[
            [
                "code",
                "strategy",
                "rank",
                "rank_in_filtered",
                "distance",
                "in_top100",
                "in_filtered",
            ]
        ].to_dict(orient="records"),
        "paper_filtered_note": (
            "Baseline uses CSV RefrigerantDistance (historical). "
            "I/IIa/III: top-100 → drop known refrigerants + dups → 65/56/23. "
            "IIb (visualize_novel_molecules_grid.py): same novel CSV as IIa → "
            "keep PubChem-absent in rank order → top-100 → dedup → Excel 84 "
            "(paper cites 64). NOT IIa-handoff-then-PubChem (~17). "
            "LOO folds recompute distances; retention vs this baseline set."
        ),
    }
    write_json(agg_dir / "headline_claims_ranks.json", headline)
    write_json(ranks_root / "strategy_summaries.json", strategy_summaries)

    summary_md = _format_ranks_summary_md(
        headline, ret_summary, ae_summary, strategy_summaries
    )
    (agg_dir / "SUMMARY_ranks.md").write_text(summary_md, encoding="utf-8")
    return headline


def _format_ranks_summary_md(
    headline: dict[str, Any],
    ret_summary: pd.DataFrame,
    ae_summary: pd.DataFrame,
    strategy_summaries: dict[str, Any],
) -> str:
    lines = [
        "# LOO mean-profile re-rank — summary",
        "",
        "Baseline = historical CSV distances; LOO folds = recomputed Euclidean "
        "vs leave-one-out mean (scales frozen on refrigerants ∪ universe).",
        "",
        f"- Folds: **{headline['n_folds']}**",
        f"- Top-k: **{headline['top_k']}**",
        "",
        "## Baseline filtered sizes (Thermo handoff path)",
        "",
        "| Strategy | Universe | Baseline filtered n | Code/Excel n | Paper n | Match code? |",
        "|---|---|---:|---:|---:|---|",
    ]
    for s, meta in strategy_summaries.items():
        lines.append(
            f"| {s} | {meta.get('n_ranking_universe', '?')} | "
            f"{meta['baseline_filtered_n']} | "
            f"{meta.get('code_excel_filtered_n', '?')} | "
            f"{meta['paper_filtered_n']} | "
            f"{meta.get('filtered_n_matches_code_excel')} |"
        )
    lines += [
        "",
        f"_{headline['paper_filtered_note']}_",
        "",
        "## Retention + Spearman under LOO (mean / min over 21 folds)",
        "",
        "| Strategy | Full mean ρ | Top-100 mean ρ | Top-100 rows (unique) | Top-100 retention | Filtered retention |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for _, r in ret_summary.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['spearman_rho_mean']:.4f} | "
            f"{r['spearman_top100_mean']:.4f} | "
            f"{int(r['top100_baseline_rows'])} "
            f"({int(r['top100_baseline_unique'])} unique) | "
            f"{r['top100_retention_mean']:.3f} "
            f"(min {r['top100_retention_min']:.3f}) | "
            f"{r['filtered_retention_mean']:.3f} "
            f"(min {r['filtered_retention_min']:.3f}) |"
        )
    lines += [
        "",
        "## Compounds A–E",
        "",
        "| Code | Strat | Base rank | Base dist | Base rank in filtered | % LOO top-100 | Mean LOO rank | Mean LOO dist | % LOO filter | Mean filt-rank when passed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in ae_summary.iterrows():
        brf = r["baseline_rank_in_filtered"]
        brf_s = f"{int(brf)}" if pd.notna(brf) else "—"
        mrf = r["mean_rank_in_filtered_when_passed"]
        mrf_s = f"{mrf:.1f}" if pd.notna(mrf) else "—"
        lines.append(
            f"| {r['code']} | {r['strategy']} | {int(r['baseline_rank'])} | "
            f"{r['baseline_distance']:.4f} | {brf_s} | "
            f"{r['pct_folds_in_top100']:.0f}% | {r['mean_rank_loo']:.1f} | "
            f"{r['mean_distance_loo']:.4f} | {r['pct_folds_in_filtered']:.0f}% | "
            f"{mrf_s} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `ranks/strategy_*/baseline_top100.csv`, `baseline_filtered.csv`, `fold_retention.csv`",
        "- `ranks/aggregate/fold_retention_all_strategies.csv`",
        "- `ranks/aggregate/retention_summary_by_strategy.csv`",
        "- `ranks/aggregate/matrix_spearman_rho.csv`, `matrix_spearman_rho_top100.csv`",
        "- `ranks/aggregate/matrix_top100_retention.csv`, `matrix_filtered_retention.csv`",
        "- `ranks/aggregate/figures/fig_spearman_heatmap*.pdf`, "
        "`fig_top100_retention_heatmap.pdf`, `fig_filtered_retention_heatmap.pdf`",
        "- `ranks/aggregate/AE_distance_rank_by_fold.csv`, `AE_summary.csv`",
        "- `ranks/aggregate/headline_claims_ranks.json`",
        "",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "LOO mean-profile variation (phase 1) and candidate re-rank (phase 2)."
        )
    )
    p.add_argument(
        "--stage",
        choices=("profile", "ranks", "both"),
        default="profile",
        help="profile | ranks | both",
    )
    p.add_argument(
        "--combined-descriptors",
        type=Path,
        default=None,
        help="Path to combined_descriptors.csv (default: statistical_comparison/)",
    )
    p.add_argument(
        "--refrigerants-csv",
        type=Path,
        default=None,
        help="Path to refrigerants.csv for compound names / known-fluid filter",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: profile_robustness/loo_mean_profile)",
    )
    p.add_argument(
        "--scale-mode",
        default="freeze_baseline_21",
        choices=("freeze_baseline_21",),
        help="Scale mode for profile stage (ranks uses freeze refrigerants∪pool)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Top-k for retention / Thermo filter path (default: 100)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    combined = args.combined_descriptors or default_combined_descriptors_path()
    refs_csv = args.refrigerants_csv or default_refrigerants_csv_path()
    output_dir = (args.output_dir or default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(combined).is_file():
        logger.error("combined_descriptors not found: %s", combined)
        return 1

    logger.info("Loading descriptors from %s", combined)
    _, refrigerants, _ = load_combined_descriptors(
        Path(combined), Path(refs_csv) if refs_csv else None
    )
    logger.info("Loaded %d profile fluids; fixed-15 descriptors", len(refrigerants))
    copy_or_write_manifest(output_dir, refrigerants)

    do_profile = args.stage in ("profile", "both")
    do_ranks = args.stage in ("ranks", "both")

    if do_profile:
        cfg = RunConfig(
            stage="profile",
            combined_descriptors=str(Path(combined).resolve()),
            refrigerants_csv=str(Path(refs_csv).resolve()) if refs_csv else None,
            output_dir=str(output_dir),
            fixed_15=list(FIXED_15),
            n_refrigerants=int(len(refrigerants)),
            scale_mode=args.scale_mode,
            mean_mode="calculated_from_loo_subset",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(output_dir / "run_config.json", asdict(cfg))
        headline = run_profile_stage(
            refrigerants,
            list(FIXED_15),
            output_dir,
            scale_mode=args.scale_mode,
        )
        logger.info(
            "Profile stage done. Mean L2(Δ normalized)=%.4g; max fold L2=%.4g (%s).",
            headline["l2_delta_normalized"]["mean"],
            headline["l2_delta_normalized"]["max"],
            headline["l2_delta_normalized"]["max_fold"],
        )

    if do_ranks:
        write_json(
            output_dir / "run_config_ranks.json",
            {
                "stage": "ranks",
                "combined_descriptors": str(Path(combined).resolve()),
                "refrigerants_csv": str(Path(refs_csv).resolve()),
                "output_dir": str(output_dir),
                "top_k": args.top_k,
                "strategies": STRATEGY_POOLS,
                "scale_mode_ranks": "freeze_refrigerants_union_pool",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        headline_r = run_ranks_stage(
            refrigerants,
            list(FIXED_15),
            output_dir,
            refrigerants_csv=Path(refs_csv),
            top_k=args.top_k,
        )
        logger.info(
            "Ranks stage done. Mean top-100 retention: %s. See %s",
            headline_r["retention_mean_top100"],
            output_dir / "ranks" / "aggregate" / "SUMMARY_ranks.md",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
