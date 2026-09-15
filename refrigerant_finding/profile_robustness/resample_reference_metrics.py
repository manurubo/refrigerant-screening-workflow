#!/usr/bin/env python3
"""
Reference-molecule-set robustness for descriptor-selection metrics.

Complements ``loo_descriptor_metrics.py`` (LOO on the 21 profile fluids).
Here the **21 refrigerants stay fixed** and we vary the **size-matched
comparison / reference pool** that JS + Mann–Whitney are computed against.

Modes
-----
- ``bootstrap``: resample the paper's 177 refs *with replacement* (cheap;
  uses only ``combined_descriptors.csv``).
- ``pool_redraw``: draw new size-matched sets of size ``n_references``
  *without replacement* from a larger precomputed descriptor pool
  (``--pool-descriptors``).
- ``candidate_redraw``: authentic re-draw — sample ``n_candidates`` SMILES
  from ``molecular_formulas.csv``, keep heavy atoms in [min, max], compute
  descriptors, then run metrics (expensive; needs RDKit + formulas CSV).

Typical usage (from repo root)::

    python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py

    python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py \\
        --mode bootstrap --n-replicates 50 \\
        --output-dir refrigerant_finding/profile_robustness/molecule_set_robustness
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

# Sibling LOO module (same directory); works when run as a script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import loo_descriptor_metrics as loo  # noqa: E402

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:  # pragma: no cover
    Chem = None
    Descriptors = None

logger = logging.getLogger(__name__)

DEFAULT_N_REPLICATES = 50
DEFAULT_N_CANDIDATES = 10_000
DEFAULT_N_REFERENCES = 177
HEAVY_ATOM_MIN = loo.HEAVY_ATOM_MIN
HEAVY_ATOM_MAX = loo.HEAVY_ATOM_MAX
MODES = ("bootstrap", "pool_redraw", "candidate_redraw")


@dataclass(frozen=True)
class RunConfig:
    mode: str
    combined_descriptors: str
    refrigerants_csv: Optional[str]
    pool_descriptors: Optional[str]
    molecular_formulas: Optional[str]
    output_dir: str
    n_replicates: int
    n_references: int
    n_candidates: int
    p_threshold: float
    near_substitute_top_n: int
    near_substitute_js_mode: str
    heavy_atom_min: int
    heavy_atom_max: int
    fixed_15: list[str]
    n_refrigerants: int
    n_baseline_references: int
    n_descriptors: int
    n_pool: Optional[int]
    categorical_nunique_threshold: int
    histogram_bins: int
    seed: int
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _project_refrigerant_finding_dir() -> Path:
    return _THIS_DIR.parent


def default_combined_descriptors_path() -> Path:
    return loo.default_combined_descriptors_path()


def default_refrigerants_csv_path() -> Path:
    return loo.default_refrigerants_csv_path()


def default_output_dir() -> Path:
    return _THIS_DIR / "molecule_set_robustness"


def default_molecular_formulas_candidates() -> list[Path]:
    """Best-effort locations for molecular_formulas.csv (not required for bootstrap)."""
    root = _project_refrigerant_finding_dir().parent
    sibling = root.parent  # MoleculeDifusion/
    return [
        root / "molecular_formulas.csv",
        sibling / "MoleculeDiffusion2" / "molecular_formulas.csv",
        sibling / "MoleculeDiffusion2" / "files_for_bsc" / "molecular_formulas.csv",
        sibling / "TuringTest" / "molecule_turing" / "molecular_formulas.csv",
    ]


def resolve_molecular_formulas(explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for cand in default_molecular_formulas_candidates():
        if cand.is_file():
            return cand
    return None


# ---------------------------------------------------------------------------
# I/O helpers
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
        "replicates": output_dir / "replicates",
        "aggregate": output_dir / "aggregate",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Descriptor computation (candidate_redraw / pool building)
# ---------------------------------------------------------------------------
def _require_rdkit() -> None:
    if Chem is None or Descriptors is None:
        raise RuntimeError(
            "RDKit is required for candidate_redraw / pool building. "
            "Use --mode bootstrap with combined_descriptors.csv instead."
        )


def heavy_atom_count(smiles: str) -> Optional[int]:
    _require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles).replace(" ", ""))
    if mol is None:
        return None
    return int(Descriptors.HeavyAtomCount(mol))


def compute_descriptor_row(smiles: str, category: str) -> Optional[dict[str, Any]]:
    """One molecule → RDKit descList + HeavyAtomCount, matching refrigerant_comparison."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles).replace(" ", ""))
    if mol is None:
        return None
    row: dict[str, Any] = {"smiles": smiles, "category": category}
    for desc_name, desc_fn in Descriptors.descList:
        try:
            row[desc_name] = desc_fn(mol)
        except Exception:
            row[desc_name] = np.nan
    row["HeavyAtomCount"] = int(Descriptors.HeavyAtomCount(mol))
    return row


def compute_descriptors_for_smiles(
    smiles_list: Sequence[str], category: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for smi in smiles_list:
        row = compute_descriptor_row(smi, category)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def filter_smiles_by_heavy_atoms(
    smiles_list: Sequence[str],
    *,
    heavy_atom_min: int,
    heavy_atom_max: int,
) -> list[str]:
    kept: list[str] = []
    for smi in smiles_list:
        hac = heavy_atom_count(smi)
        if hac is None:
            continue
        if heavy_atom_min <= hac <= heavy_atom_max:
            kept.append(smi)
    return kept


def draw_size_matched_from_formulas(
    formulas_path: Path,
    *,
    n_candidates: int,
    n_references: int,
    heavy_atom_min: int,
    heavy_atom_max: int,
    seed: int,
    force_exact_n: bool = False,
) -> pd.DataFrame:
    """
    Match ``refrigerant_comparison.py`` size-match path with ``--atom_range``:

    1. Sample ``n_candidates`` from molecular_formulas.csv
    2. Keep SMILES with HeavyAtomCount in [min, max]
    3. If more than ``n_references`` and ``force_exact_n``, subsample;
       otherwise use **all** that pass (original used all when < n_samples)
    """
    molecules = pd.read_csv(formulas_path)
    if "smiles" not in molecules.columns:
        raise ValueError(f"{formulas_path} must contain a 'smiles' column")
    n = min(int(n_candidates), len(molecules))
    sample = molecules.sample(n=n, random_state=seed)
    matched = filter_smiles_by_heavy_atoms(
        sample["smiles"].tolist(),
        heavy_atom_min=heavy_atom_min,
        heavy_atom_max=heavy_atom_max,
    )
    logger.info(
        "Seed %d: sampled %d → %d size-matched (heavy atoms %d–%d)",
        seed,
        n,
        len(matched),
        heavy_atom_min,
        heavy_atom_max,
    )
    if not matched:
        raise RuntimeError(
            f"No size-matched molecules for seed={seed}; "
            f"try larger --n-candidates (got {n})"
        )
    if force_exact_n and len(matched) > n_references:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(matched), size=n_references, replace=False)
        matched = [matched[i] for i in idx]
    elif len(matched) > n_references and not force_exact_n:
        # Original pipeline: get_size_filtered_molecules samples n_samples
        # when enough remain; with atom_range already applied and
        # size_buffer=1.0 the pool *is* the size-matched set, so if
        # len < n_samples they kept all (~177). If somehow more remain,
        # sample down to n_references to keep replicate sizes comparable.
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(matched), size=n_references, replace=False)
        matched = [matched[i] for i in idx]
        logger.info(
            "Seed %d: downsampled size-matched set to %d", seed, n_references
        )

    return compute_descriptors_for_smiles(matched, loo.REFERENCE_CATEGORY)


def build_reference_pool(
    formulas_path: Path,
    out_path: Path,
    *,
    n_candidates: int = 50_000,
    target_size: int = 2_000,
    heavy_atom_min: int = HEAVY_ATOM_MIN,
    heavy_atom_max: int = HEAVY_ATOM_MAX,
    seed: int = 0,
    max_rounds: int = 20,
) -> pd.DataFrame:
    """
    Accumulate size-matched molecules with full descriptors for later
    ``pool_redraw`` replicates (avoids re-parsing SMILES every replicate).
    """
    molecules = pd.read_csv(formulas_path)
    if "smiles" not in molecules.columns:
        raise ValueError(f"{formulas_path} must contain a 'smiles' column")

    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    round_i = 0
    while len(rows) < target_size and round_i < max_rounds:
        round_i += 1
        n = min(int(n_candidates), len(molecules))
        # sample indices without depending on pandas random_state alone
        idx = rng.choice(len(molecules), size=n, replace=False)
        batch = molecules.iloc[idx]["smiles"].tolist()
        for smi in batch:
            key = str(smi).replace(" ", "")
            if key in seen:
                continue
            hac = heavy_atom_count(smi)
            if hac is None or not (heavy_atom_min <= hac <= heavy_atom_max):
                continue
            seen.add(key)
            row = compute_descriptor_row(smi, loo.REFERENCE_CATEGORY)
            if row is not None:
                rows.append(row)
            if len(rows) >= target_size:
                break
        logger.info(
            "Pool build round %d: %d / %d size-matched descriptors",
            round_i,
            len(rows),
            target_size,
        )

    pool = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out_path, index=False)
    logger.info("Wrote reference pool (%d molecules) → %s", len(pool), out_path)
    meta = {
        "n_pool": int(len(pool)),
        "target_size": int(target_size),
        "n_candidates_per_round": int(n_candidates),
        "rounds": int(round_i),
        "heavy_atom_min": int(heavy_atom_min),
        "heavy_atom_max": int(heavy_atom_max),
        "seed": int(seed),
        "molecular_formulas": str(formulas_path.resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out_path.with_suffix(out_path.suffix + ".meta.json"), meta)
    return pool


# ---------------------------------------------------------------------------
# Reference draws
# ---------------------------------------------------------------------------
def draw_bootstrap_references(
    references: pd.DataFrame, *, n: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(references), size=n)
    out = references.iloc[idx].copy().reset_index(drop=True)
    out["category"] = loo.REFERENCE_CATEGORY
    return out


def draw_pool_references(
    pool: pd.DataFrame, *, n: int, seed: int
) -> pd.DataFrame:
    if len(pool) < n:
        raise ValueError(
            f"Pool has {len(pool)} molecules; need at least n_references={n}"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n, replace=False)
    out = pool.iloc[idx].copy().reset_index(drop=True)
    out["category"] = loo.REFERENCE_CATEGORY
    return out


def strip_refrigerant_annotations(refrigerants: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        c
        for c in ("refrigerant_index", "compound", "refrigerant_id")
        if c in refrigerants.columns
    ]
    return refrigerants.drop(columns=drop_cols, errors="ignore")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def build_aggregates(
    rep_fixed15: list[pd.DataFrame],
    rep_meta: list[dict[str, Any]],
    near_sub_tables: list[pd.DataFrame],
    baseline_fixed15: pd.DataFrame,
    fixed_descriptors: Sequence[str],
) -> dict[str, pd.DataFrame]:
    n_reps = len(rep_fixed15)
    rep_ids = [m["replicate_id"] for m in rep_meta]

    pass_mat = pd.DataFrame(
        index=rep_ids, columns=list(fixed_descriptors), dtype=object
    )
    js_mat = pd.DataFrame(
        index=rep_ids, columns=list(fixed_descriptors), dtype=float
    )
    rank_mat = pd.DataFrame(
        index=rep_ids, columns=list(fixed_descriptors), dtype=float
    )
    p_mat = pd.DataFrame(
        index=rep_ids, columns=list(fixed_descriptors), dtype=float
    )
    pass_mat.index.name = "replicate_id"
    js_mat.index.name = "replicate_id"
    rank_mat.index.name = "replicate_id"
    p_mat.index.name = "replicate_id"

    for rid, table in zip(rep_ids, rep_fixed15):
        t = table.set_index("Feature")
        for feat in fixed_descriptors:
            if feat not in t.index:
                pass_mat.loc[rid, feat] = False
                js_mat.loc[rid, feat] = np.nan
                rank_mat.loc[rid, feat] = np.nan
                p_mat.loc[rid, feat] = np.nan
                continue
            row = t.loc[feat]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            pass_mat.loc[rid, feat] = bool(row["passes_p"])
            js_mat.loc[rid, feat] = float(row["JS_Distance"])
            rank_mat.loc[rid, feat] = float(row["JS_rank"])
            p_mat.loc[rid, feat] = (
                float(row["MW_pvalue"]) if pd.notna(row["MW_pvalue"]) else np.nan
            )

    base = baseline_fixed15.set_index("Feature")
    summary_rows: list[dict[str, Any]] = []
    for feat in fixed_descriptors:
        js_vals = js_mat[feat].astype(float)
        rank_vals = rank_mat[feat].astype(float)
        p_vals = p_mat[feat].astype(float)
        passes = pass_mat[feat].astype(bool)
        js_base = (
            float(base.loc[feat, "JS_Distance"])
            if feat in base.index and pd.notna(base.loc[feat, "JS_Distance"])
            else np.nan
        )
        rank_base = (
            float(base.loc[feat, "JS_rank"])
            if feat in base.index and pd.notna(base.loc[feat, "JS_rank"])
            else np.nan
        )
        p_base = (
            float(base.loc[feat, "MW_pvalue"])
            if feat in base.index and pd.notna(base.loc[feat, "MW_pvalue"])
            else np.nan
        )
        pass_base = (
            bool(base.loc[feat, "passes_p"]) if feat in base.index else False
        )
        delta_js = js_vals - js_base
        delta_rank = rank_vals - rank_base
        summary_rows.append(
            {
                "Feature": feat,
                "baseline_JS": js_base,
                "baseline_JS_rank": rank_base,
                "baseline_MW_pvalue": p_base,
                "baseline_passes_p": pass_base,
                "rep_pass_rate": float(passes.mean()) if n_reps else np.nan,
                "rep_n_pass": int(passes.sum()),
                "rep_n_total": int(n_reps),
                "rep_JS_mean": float(js_vals.mean()),
                "rep_JS_std": float(js_vals.std(ddof=0)),
                "rep_JS_min": float(js_vals.min()),
                "rep_JS_max": float(js_vals.max()),
                "rep_JS_mean_delta_vs_baseline": float(delta_js.mean()),
                "rep_JS_max_abs_delta_vs_baseline": float(delta_js.abs().max()),
                "rep_rank_mean": float(rank_vals.mean()),
                "rep_rank_std": float(rank_vals.std(ddof=0)),
                "rep_rank_mean_delta_vs_baseline": float(delta_rank.mean()),
                "rep_rank_max_abs_delta_vs_baseline": float(
                    delta_rank.abs().max()
                ),
                "rep_pvalue_max": float(p_vals.max())
                if p_vals.notna().any()
                else np.nan,
                "rep_pvalue_min": float(p_vals.min())
                if p_vals.notna().any()
                else np.nan,
                "always_significant": bool(passes.all()) if n_reps else False,
                "never_significant": bool((~passes).all()) if n_reps else False,
            }
        )
    fixed15_summary = pd.DataFrame(summary_rows)

    # Near-substitute frequency across replicates
    freq: dict[str, dict[str, Any]] = {}
    for meta, ns in zip(rep_meta, near_sub_tables):
        for _, row in ns.iterrows():
            feat = row["Feature"]
            slot = freq.setdefault(
                feat,
                {
                    "Feature": feat,
                    "n_replicates_as_near_substitute": 0,
                    "frequency": 0.0,
                    "JS_mean": [],
                    "JS_rank_mean": [],
                    "MW_pvalue_mean": [],
                    "example_replicates": [],
                },
            )
            slot["n_replicates_as_near_substitute"] += 1
            slot["JS_mean"].append(float(row["JS_Distance"]))
            slot["JS_rank_mean"].append(float(row["JS_rank"]))
            slot["MW_pvalue_mean"].append(float(row["MW_pvalue"]))
            if len(slot["example_replicates"]) < 5:
                slot["example_replicates"].append(meta["replicate_id"])

    freq_rows = []
    for feat, slot in freq.items():
        freq_rows.append(
            {
                "Feature": feat,
                "n_replicates_as_near_substitute": slot[
                    "n_replicates_as_near_substitute"
                ],
                "frequency": slot["n_replicates_as_near_substitute"]
                / max(n_reps, 1),
                "JS_mean": float(np.mean(slot["JS_mean"])),
                "JS_rank_mean": float(np.mean(slot["JS_rank_mean"])),
                "MW_pvalue_mean": float(np.mean(slot["MW_pvalue_mean"])),
                "example_replicates": ";".join(slot["example_replicates"]),
            }
        )
    near_freq = pd.DataFrame(freq_rows)
    if not near_freq.empty:
        near_freq = near_freq.sort_values(
            by=["n_replicates_as_near_substitute", "JS_mean", "Feature"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    # Pairwise Jaccard of significant fixed-15 sets (stability of conclusions)
    sig_sets = []
    for rid in rep_ids:
        feats = [f for f in fixed_descriptors if bool(pass_mat.loc[rid, f])]
        sig_sets.append(set(feats))
    jaccard_rows = []
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            a, b = sig_sets[i], sig_sets[j]
            union = len(a | b)
            jaccard = (len(a & b) / union) if union else 1.0
            jaccard_rows.append(
                {
                    "replicate_a": rep_ids[i],
                    "replicate_b": rep_ids[j],
                    "jaccard_significant_fixed15": jaccard,
                    "n_sig_a": len(a),
                    "n_sig_b": len(b),
                    "n_shared": len(a & b),
                }
            )
    jaccard_df = pd.DataFrame(jaccard_rows)

    headline = {
        "n_replicates": int(n_reps),
        "n_fixed15": int(len(fixed_descriptors)),
        "all_fixed15_always_significant": bool(
            fixed15_summary["always_significant"].all()
        )
        if not fixed15_summary.empty
        else False,
        "mean_rep_pass_rate": float(fixed15_summary["rep_pass_rate"].mean())
        if not fixed15_summary.empty
        else np.nan,
        "min_rep_pass_rate": float(fixed15_summary["rep_pass_rate"].min())
        if not fixed15_summary.empty
        else np.nan,
        "mean_of_mean_abs_delta_JS": float(
            fixed15_summary["rep_JS_mean_delta_vs_baseline"].abs().mean()
        )
        if not fixed15_summary.empty
        else np.nan,
        "max_abs_delta_JS": float(
            fixed15_summary["rep_JS_max_abs_delta_vs_baseline"].max()
        )
        if not fixed15_summary.empty
        else np.nan,
        "feature_with_largest_max_abs_delta_JS": (
            fixed15_summary.loc[
                fixed15_summary["rep_JS_max_abs_delta_vs_baseline"].idxmax(),
                "Feature",
            ]
            if not fixed15_summary.empty
            else None
        ),
        "mean_pairwise_jaccard_significant_fixed15": float(
            jaccard_df["jaccard_significant_fixed15"].mean()
        )
        if not jaccard_df.empty
        else np.nan,
        "min_pairwise_jaccard_significant_fixed15": float(
            jaccard_df["jaccard_significant_fixed15"].min()
        )
        if not jaccard_df.empty
        else np.nan,
    }

    return {
        "fixed15_resample_summary": fixed15_summary,
        "fixed15_pass_matrix": pass_mat.reset_index(),
        "fixed15_js_matrix": js_mat.reset_index(),
        "fixed15_rank_matrix": rank_mat.reset_index(),
        "fixed15_pvalue_matrix": p_mat.reset_index(),
        "near_substitutes_frequency": near_freq,
        "pairwise_jaccard_significant_fixed15": jaccard_df,
        "headline_claims": headline,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_reference_set_analysis(
    combined_path: Path,
    output_dir: Path,
    *,
    mode: str = "bootstrap",
    refrigerants_csv: Optional[Path] = None,
    pool_descriptors: Optional[Path] = None,
    molecular_formulas: Optional[Path] = None,
    n_replicates: int = DEFAULT_N_REPLICATES,
    n_references: int = DEFAULT_N_REFERENCES,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    p_threshold: float = loo.DEFAULT_P_THRESHOLD,
    near_substitute_top_n: int = loo.DEFAULT_NEAR_SUB_TOP_N,
    near_substitute_js_mode: str = "both",
    heavy_atom_min: int = HEAVY_ATOM_MIN,
    heavy_atom_max: int = HEAVY_ATOM_MAX,
    fixed_descriptors: Sequence[str] = loo.FIXED_15,
    seed: int = 0,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")

    paths = _ensure_dirs(output_dir)
    full, refrigerants, references = loo.load_combined_descriptors(
        combined_path,
        refrigerants_csv,
        heavy_atom_min=heavy_atom_min,
        heavy_atom_max=heavy_atom_max,
    )
    features = loo.descriptor_columns(full)
    missing_fixed = [f for f in fixed_descriptors if f not in features]
    if missing_fixed:
        raise ValueError(f"Fixed descriptors missing from data: {missing_fixed}")

    pool: Optional[pd.DataFrame] = None
    formulas_path: Optional[Path] = None
    if mode == "pool_redraw":
        if pool_descriptors is None or not pool_descriptors.is_file():
            raise FileNotFoundError(
                "pool_redraw requires --pool-descriptors pointing to a CSV "
                "of size-matched reference descriptors (see --build-pool)."
            )
        pool = pd.read_csv(pool_descriptors)
        if "smiles" not in pool.columns:
            raise ValueError("Pool CSV must contain 'smiles'")
        if "category" not in pool.columns:
            pool = pool.copy()
            pool["category"] = loo.REFERENCE_CATEGORY
        # Prefer pool's numeric columns that overlap baseline features
        pool_feats = [c for c in features if c in pool.columns]
        if len(pool_feats) < len(fixed_descriptors):
            raise ValueError(
                "Pool is missing too many descriptor columns vs combined_descriptors"
            )
        features = [c for c in features if c in pool.columns]
        logger.info(
            "Loaded pool with %d molecules (%d overlapping descriptors)",
            len(pool),
            len(features),
        )
    elif mode == "candidate_redraw":
        formulas_path = resolve_molecular_formulas(molecular_formulas)
        if formulas_path is None:
            raise FileNotFoundError(
                "candidate_redraw requires molecular_formulas.csv "
                "(pass --molecular-formulas). Not found in default locations."
            )
        _require_rdkit()
        logger.info("Using molecular formulas: %s", formulas_path)

    loo.save_molecule_manifest(
        paths["root"] / "molecule_manifest.json", refrigerants, references
    )

    # --- Baseline (paper 21 vs paper 177) ---
    logger.info(
        "Baseline: %d refrigerants vs %d references, %d descriptors",
        len(refrigerants),
        len(references),
        len(features),
    )
    refs_for_metrics = strip_refrigerant_annotations(refrigerants)
    baseline_df = pd.concat([refs_for_metrics, references], ignore_index=True)
    # Restrict to shared feature columns (+ meta)
    meta_keep = [c for c in ("smiles", "category") if c in baseline_df.columns]
    baseline_df = baseline_df[meta_keep + features]
    baseline_metrics = loo.compute_all_descriptor_metrics(baseline_df, features)
    baseline_metrics = loo.apply_significance(baseline_metrics, p_threshold)
    baseline_fixed = loo.fixed15_summary(
        baseline_metrics, fixed_descriptors, p_threshold=p_threshold
    )
    baseline_metrics.to_csv(
        paths["baseline"] / "js_mannwhitney_all_descriptors.csv", index=False
    )
    baseline_fixed.to_csv(paths["baseline"] / "fixed15_summary.csv", index=False)
    loo.near_substitutes(
        baseline_metrics,
        fixed_descriptors,
        p_threshold=p_threshold,
        top_n=near_substitute_top_n,
        js_mode=near_substitute_js_mode,
    ).to_csv(paths["baseline"] / "near_substitutes.csv", index=False)

    rep_fixed15_tables: list[pd.DataFrame] = []
    rep_meta: list[dict[str, Any]] = []
    near_sub_tables: list[pd.DataFrame] = []

    for i in range(n_replicates):
        rep_seed = int(seed + i)
        rep_id = f"rep_{i:03d}"
        rep_dir = paths["replicates"] / rep_id
        rep_dir.mkdir(parents=True, exist_ok=True)

        if mode == "bootstrap":
            drawn = draw_bootstrap_references(
                references, n=n_references, seed=rep_seed
            )
            draw_info = {
                "draw_type": "bootstrap_with_replacement",
                "n_source": int(len(references)),
                "n_drawn": int(len(drawn)),
                "n_unique_smiles": int(drawn["smiles"].nunique()),
            }
        elif mode == "pool_redraw":
            assert pool is not None
            drawn = draw_pool_references(pool, n=n_references, seed=rep_seed)
            # Align columns to features available
            keep = [c for c in drawn.columns if c in meta_keep + features]
            drawn = drawn[keep]
            draw_info = {
                "draw_type": "pool_without_replacement",
                "n_source": int(len(pool)),
                "n_drawn": int(len(drawn)),
                "n_unique_smiles": int(drawn["smiles"].nunique()),
            }
        else:  # candidate_redraw
            assert formulas_path is not None
            drawn = draw_size_matched_from_formulas(
                formulas_path,
                n_candidates=n_candidates,
                n_references=n_references,
                heavy_atom_min=heavy_atom_min,
                heavy_atom_max=heavy_atom_max,
                seed=rep_seed,
            )
            # Intersect descriptors with baseline feature set
            keep = [c for c in drawn.columns if c in meta_keep + features]
            drawn = drawn[keep]
            draw_info = {
                "draw_type": "candidate_sample_then_filter",
                "n_candidates": int(n_candidates),
                "n_drawn": int(len(drawn)),
                "n_unique_smiles": int(drawn["smiles"].nunique()),
                "molecular_formulas": str(formulas_path.resolve()),
            }

        # drawn may only have a subset of columns; align before concat
        ref_block = refs_for_metrics[meta_keep + features].copy()
        drawn_aligned = drawn.reindex(columns=meta_keep + features)
        fold_df = pd.concat([ref_block, drawn_aligned], ignore_index=True)
        logger.info(
            "%s (%s): n_references=%d unique=%d",
            rep_id,
            mode,
            len(drawn),
            draw_info["n_unique_smiles"],
        )

        meta = {
            "replicate_id": rep_id,
            "replicate_index": int(i),
            "seed": rep_seed,
            "mode": mode,
            "n_refrigerants": int(len(refrigerants)),
            "n_references": int(len(drawn)),
            **draw_info,
            "reference_smiles": drawn["smiles"].tolist(),
        }
        _write_json(rep_dir / "draw_manifest.json", meta)

        metrics = loo.compute_all_descriptor_metrics(fold_df, features)
        metrics = loo.apply_significance(metrics, p_threshold)
        metrics.to_csv(
            rep_dir / "js_mannwhitney_all_descriptors.csv", index=False
        )

        fixed_sum = loo.fixed15_summary(
            metrics,
            fixed_descriptors,
            p_threshold=p_threshold,
            baseline_metrics=baseline_metrics,
        )
        fixed_sum.to_csv(rep_dir / "fixed15_summary.csv", index=False)

        ns = loo.near_substitutes(
            metrics,
            fixed_descriptors,
            p_threshold=p_threshold,
            top_n=near_substitute_top_n,
            js_mode=near_substitute_js_mode,
        )
        ns.to_csv(rep_dir / "near_substitutes.csv", index=False)

        rep_fixed15_tables.append(fixed_sum)
        rep_meta.append(meta)
        near_sub_tables.append(ns)

    aggregates = build_aggregates(
        rep_fixed15_tables,
        rep_meta,
        near_sub_tables,
        baseline_fixed,
        fixed_descriptors,
    )
    headline = aggregates.pop("headline_claims")
    for name, table in aggregates.items():
        table.to_csv(paths["aggregate"] / f"{name}.csv", index=False)
    _write_json(paths["aggregate"] / "headline_claims.json", headline)

    config = RunConfig(
        mode=mode,
        combined_descriptors=str(combined_path.resolve()),
        refrigerants_csv=str(Path(refrigerants_csv).resolve())
        if refrigerants_csv
        else str(default_refrigerants_csv_path()),
        pool_descriptors=str(pool_descriptors.resolve())
        if pool_descriptors
        else None,
        molecular_formulas=str(formulas_path.resolve())
        if formulas_path
        else None,
        output_dir=str(output_dir.resolve()),
        n_replicates=int(n_replicates),
        n_references=int(n_references),
        n_candidates=int(n_candidates),
        p_threshold=p_threshold,
        near_substitute_top_n=near_substitute_top_n,
        near_substitute_js_mode=near_substitute_js_mode,
        heavy_atom_min=heavy_atom_min,
        heavy_atom_max=heavy_atom_max,
        fixed_15=list(fixed_descriptors),
        n_refrigerants=int(len(refrigerants)),
        n_baseline_references=int(len(references)),
        n_descriptors=int(len(features)),
        n_pool=int(len(pool)) if pool is not None else None,
        categorical_nunique_threshold=loo.CATEGORICAL_NUNIQUE_THRESHOLD,
        histogram_bins=loo.HISTOGRAM_BINS,
        seed=seed,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(paths["root"] / "run_config.json", asdict(config))

    logger.info("=== Fixed-15 reference-set pass rates ===")
    summary = aggregates["fixed15_resample_summary"]
    for _, r in summary.iterrows():
        logger.info(
            "  %s: pass_rate=%.2f  JS_mean=%.4f (Δ=%.4f)  rank_mean=%.1f",
            r["Feature"],
            r["rep_pass_rate"],
            r["rep_JS_mean"],
            r["rep_JS_mean_delta_vs_baseline"],
            r["rep_rank_mean"],
        )
    logger.info(
        "Headline: always_sig=%s  mean_jaccard=%.3f  max|ΔJS|=%.4f (%s)",
        headline.get("all_fixed15_always_significant"),
        headline.get("mean_pairwise_jaccard_significant_fixed15"),
        headline.get("max_abs_delta_JS"),
        headline.get("feature_with_largest_max_abs_delta_JS"),
    )

    return {
        "config": config,
        "baseline_fixed15": baseline_fixed,
        "aggregates": aggregates,
        "headline": headline,
        "n_replicates": n_replicates,
    }


def write_readme(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = """# Reference molecule-set robustness

Varies the **size-matched comparison molecules** used for Jensen–Shannon
distances and Mann–Whitney tests, while keeping the **21 profile fluids fixed**.

This is the sibling of `loo_on_profile/` (which leaves out profile fluids and
holds the 177 references fixed).

## Question

If we had drawn a different size-matched reference set of similar size (~177)
from the same filtering pipeline, would fixed-15 significance / JS / ranks
change?

## How the paper's ~177 set is produced

From `refrigerant_finding/refrigerant_comparison.py` (typical flags:
`--size_match --atom_range 5 10 --n_samples 10000 --size_buffer 1.0`):

1. Sample ~10k SMILES from `molecular_formulas.csv`
2. Keep molecules with heavy-atom count in **[5, 10]**
3. That filtered set (~177 for the published run) becomes
   `category=database_size_matched` in
   `statistical_comparison/combined_descriptors.csv`

## Modes

| Mode | What varies | Needs |
|------|-------------|-------|
| `bootstrap` (default) | Resample the published 177 **with replacement** | `combined_descriptors.csv` only |
| `pool_redraw` | Draw 177 **without replacement** from a larger precomputed pool | `--pool-descriptors` |
| `candidate_redraw` | Re-run sample→filter→descriptors with new seeds | RDKit + `molecular_formulas.csv` |

Prefer `bootstrap` for a fast sensitivity check. Prefer `pool_redraw` /
`candidate_redraw` to mimic a different draw from the large candidate DB.

## How to run

From the `RefrigerantFinding` repo root:

```bash
# Fast default (bootstrap the published 177)
python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py \\
  --mode bootstrap --n-replicates 50

# Build a reusable size-matched pool, then redraw
python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py \\
  --build-pool refrigerant_finding/profile_robustness/molecule_set_robustness/reference_pool.csv \\
  --molecular-formulas /path/to/molecular_formulas.csv \\
  --pool-target-size 2000 --n-candidates 50000

python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py \\
  --mode pool_redraw --n-replicates 50 \\
  --pool-descriptors refrigerant_finding/profile_robustness/molecule_set_robustness/reference_pool.csv

# Authentic redraw from formulas (slower; RDKit per replicate; use n_candidates≈10000)
python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py \\
  --mode candidate_redraw --n-replicates 20 --n-candidates 10000 \\
  --molecular-formulas /path/to/molecular_formulas.csv
```

## Outputs

```
molecule_set_robustness/
  README.md
  run_config.json
  molecule_manifest.json
  baseline/                 # paper 21 vs paper 177
  replicates/rep_XXX/       # each reference draw
    draw_manifest.json
    js_mannwhitney_all_descriptors.csv
    fixed15_summary.csv
    near_substitutes.csv
  aggregate/
    fixed15_resample_summary.csv
    fixed15_pass_matrix.csv
    fixed15_js_matrix.csv
    fixed15_rank_matrix.csv
    fixed15_pvalue_matrix.csv
    near_substitutes_frequency.csv
    pairwise_jaccard_significant_fixed15.csv
    headline_claims.json
    figures/                # publication PDFs/PNGs (see below)
```

## Figures

Publication figures (LOO-style analogs + redraw-specific panels) are generated
from existing aggregate CSVs — they do **not** re-run metrics.

```bash
# Default: authentic candidate_redraw run
python3 refrigerant_finding/profile_robustness/plot_molecule_set_robustness.py

# Explicit paths (e.g. bootstrap root run)
python3 refrigerant_finding/profile_robustness/plot_molecule_set_robustness.py \\
  --run-dir refrigerant_finding/profile_robustness/molecule_set_robustness \\
  --output-dir refrigerant_finding/profile_robustness/molecule_set_robustness/aggregate/figures
```

Primary figures live under `candidate_redraw/aggregate/figures/`.

## What is measured

- Per fixed-15 descriptor: pass rate across replicates, JS/rank mean±spread,
  deltas vs the published baseline
- Pairwise Jaccard of which fixed-15 descriptors remain significant
- Near-substitute frequency under changing reference sets

Does **not** recompute mean profiles or candidate rankings (same scope as LOO).
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Robustness of descriptor JS / Mann–Whitney metrics to the "
            "size-matched reference molecule set (sibling of LOO-on-profile)."
        )
    )
    p.add_argument(
        "--mode",
        choices=list(MODES),
        default="bootstrap",
        help="How to vary the reference set (default: bootstrap)",
    )
    p.add_argument(
        "--combined-descriptors",
        type=Path,
        default=default_combined_descriptors_path(),
        help="Path to combined_descriptors.csv (21 + 177 baseline)",
    )
    p.add_argument(
        "--refrigerants-csv",
        type=Path,
        default=default_refrigerants_csv_path(),
        help="Path to refrigerants.csv (compound names)",
    )
    p.add_argument(
        "--pool-descriptors",
        type=Path,
        default=None,
        help="Precomputed size-matched pool CSV (required for pool_redraw)",
    )
    p.add_argument(
        "--molecular-formulas",
        type=Path,
        default=None,
        help="molecular_formulas.csv for candidate_redraw / --build-pool",
    )
    p.add_argument(
        "--build-pool",
        type=Path,
        default=None,
        help=(
            "If set, build a size-matched descriptor pool at this path and exit "
            "(unless --run-after-build is also set)"
        ),
    )
    p.add_argument(
        "--run-after-build",
        action="store_true",
        help="After --build-pool, continue with pool_redraw using that pool",
    )
    p.add_argument(
        "--pool-target-size",
        type=int,
        default=2000,
        help="Target number of size-matched molecules when building a pool",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Output root for molecule_set_robustness results",
    )
    p.add_argument(
        "--n-replicates",
        type=int,
        default=DEFAULT_N_REPLICATES,
        help="Number of reference-set draws (default: 50)",
    )
    p.add_argument(
        "--n-references",
        type=int,
        default=DEFAULT_N_REFERENCES,
        help="Reference set size per replicate (default: 177)",
    )
    p.add_argument(
        "--n-candidates",
        type=int,
        default=DEFAULT_N_CANDIDATES,
        help="SMILES sample size before heavy-atom filter (candidate_redraw / pool)",
    )
    p.add_argument(
        "--p-threshold",
        type=float,
        default=loo.DEFAULT_P_THRESHOLD,
        help="Mann–Whitney significance threshold (default: 0.05)",
    )
    p.add_argument(
        "--near-substitute-top-n",
        type=int,
        default=loo.DEFAULT_NEAR_SUB_TOP_N,
        help="JS_rank cutoff for near-substitute modes",
    )
    p.add_argument(
        "--near-substitute-js-mode",
        choices=["min_fixed15", "top_n", "both"],
        default="both",
        help="Near-substitute rule (same as LOO)",
    )
    p.add_argument("--heavy-atom-min", type=int, default=HEAVY_ATOM_MIN)
    p.add_argument("--heavy-atom-max", type=int, default=HEAVY_ATOM_MAX)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed; replicate i uses seed+i",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    write_readme(args.output_dir)

    if args.build_pool is not None:
        formulas = resolve_molecular_formulas(args.molecular_formulas)
        if formulas is None:
            logger.error(
                "Cannot build pool: molecular_formulas.csv not found. "
                "Pass --molecular-formulas."
            )
            return 1
        try:
            build_reference_pool(
                formulas,
                args.build_pool,
                n_candidates=args.n_candidates,
                target_size=args.pool_target_size,
                heavy_atom_min=args.heavy_atom_min,
                heavy_atom_max=args.heavy_atom_max,
                seed=args.seed,
            )
        except Exception as exc:
            logger.error("Pool build failed: %s", exc)
            return 1
        if not args.run_after_build:
            logger.info(
                "Pool ready. Re-run with --mode pool_redraw "
                "--pool-descriptors %s",
                args.build_pool,
            )
            return 0
        args.mode = "pool_redraw"
        args.pool_descriptors = args.build_pool

    if not args.combined_descriptors.is_file():
        logger.error("Combined descriptors not found: %s", args.combined_descriptors)
        return 1

    try:
        result = run_reference_set_analysis(
            args.combined_descriptors,
            args.output_dir,
            mode=args.mode,
            refrigerants_csv=args.refrigerants_csv,
            pool_descriptors=args.pool_descriptors,
            molecular_formulas=args.molecular_formulas,
            n_replicates=args.n_replicates,
            n_references=args.n_references,
            n_candidates=args.n_candidates,
            p_threshold=args.p_threshold,
            near_substitute_top_n=args.near_substitute_top_n,
            near_substitute_js_mode=args.near_substitute_js_mode,
            heavy_atom_min=args.heavy_atom_min,
            heavy_atom_max=args.heavy_atom_max,
            fixed_descriptors=loo.FIXED_15,
            seed=args.seed,
        )
    except Exception as exc:
        logger.error("Analysis failed: %s", exc)
        logger.debug("Traceback", exc_info=True)
        return 1

    logger.info(
        "Done. %d replicates written under %s",
        result["n_replicates"],
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
