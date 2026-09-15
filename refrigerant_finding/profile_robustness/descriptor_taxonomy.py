"""
Chemically meaningful RDKit descriptor categories for profile-robustness figures.

Taxonomy source
---------------
Paper Methods (``paper_final/submission/main.tex``, physicochemical profiling)
lists six families for the 170 RDKit descriptors:

  (i)   size and composition metrics
  (ii)  topological descriptors
  (iii) electronic properties
  (iv)  surface area descriptors
  (v)   pharmacophore features
  (vi)  structural complexity measures

The curated fixed-15 also includes halogen fragment counts and MolLogP, which
are not named in that six-way list. We add two thin extensions so every
fixed-15 / near-substitute name maps cleanly:

  (vii)  halogen-related (``fr_halogen``, ``fr_alkyl_halide``, …)
  (viii) lipophilicity (``MolLogP``, ``SlogP_VSA*``, ``BCUT2D_LOGP*``, …)

Selection criteria in the same Methods paragraph cite diversity across
"electronic, topological, surface area, structural" — this module is the
operational mapping used for near-vs-fixed15 heatmaps.

Ambiguous / hybrid descriptors
------------------------------
Several RDKit names blend two ideas (e.g. ``VSA_EState*`` = E-state on VdW
surface bins; ``PEOE_VSA*`` = charge-weighted surface; ``TPSA`` = polar
surface). Assignments below follow the paper's primary wording where
available, else the dominant RDKit family. See ``NOTES`` on each explicit
entry when relevant.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

import pandas as pd

# Display order for axes / tables (paper Methods order, then extensions).
CATEGORY_ORDER: list[str] = [
    "Electronic",
    "Surface area",
    "Pharmacophore / polarity",
    "Lipophilicity",
    "Halogen",
    "Size / composition",
    "Topology",
    "Structural complexity",
    "Other",
]

CATEGORY_SHORT: dict[str, str] = {
    "Electronic": "Electronic",
    "Surface area": "Surface",
    "Pharmacophore / polarity": "Polarity",
    "Lipophilicity": "Lipophilicity",
    "Halogen": "Halogen",
    "Size / composition": "Size",
    "Topology": "Topology",
    "Structural complexity": "Complexity",
    "Other": "Other",
}

# Explicit map for curated fixed-15 + common near-substitutes / overrides.
# Prefer this over pattern rules whenever a name is listed.
DESCRIPTOR_CATEGORY: dict[str, str] = {
    # --- fixed-15 ---
    "MaxPartialCharge": "Electronic",
    "MinPartialCharge": "Electronic",
    "MinEStateIndex": "Electronic",
    "SMR_VSA1": "Surface area",
    "LabuteASA": "Surface area",
    "TPSA": "Surface area",  # Methods: surface area (polar surface)
    "VSA_EState1": "Surface area",  # hybrid E-state×VSA; paper glosses as VdW SA E-state
    "NumHAcceptors": "Pharmacophore / polarity",
    "NumRotatableBonds": "Pharmacophore / polarity",  # Methods pharmacophore
    "MolLogP": "Lipophilicity",
    "fr_halogen": "Halogen",
    "fr_alkyl_halide": "Halogen",
    "NumHeteroatoms": "Size / composition",
    "MolMR": "Structural complexity",
    "FpDensityMorgan3": "Structural complexity",
    # --- frequent near-substitutes / siblings ---
    "MaxEStateIndex": "Electronic",
    "MaxAbsEStateIndex": "Electronic",
    "FpDensityMorgan1": "Structural complexity",
    "FpDensityMorgan2": "Structural complexity",
    "NumHDonors": "Pharmacophore / polarity",
    "NOCount": "Pharmacophore / polarity",
    "NHOHCount": "Pharmacophore / polarity",
    "HeavyAtomCount": "Size / composition",
    "NumValenceElectrons": "Size / composition",
    "RingCount": "Topology",
    "BalabanJ": "Topology",
    "BertzCT": "Topology",
    "Ipc": "Topology",
    "AvgIpc": "Topology",
    "Kappa2": "Topology",
    "qed": "Structural complexity",  # composite drug-likeness; nearest paper bucket
}


def categorize_descriptor(name: str) -> str:
    """Return the chemical category for an RDKit descriptor name."""
    if name in DESCRIPTOR_CATEGORY:
        return DESCRIPTOR_CATEGORY[name]

    # Halogen fragments
    if re.search(r"halogen|halide|alkyl_hal", name, flags=re.IGNORECASE):
        return "Halogen"

    # Lipophilicity
    if name.startswith("SlogP_VSA") or "LOGP" in name.upper():
        return "Lipophilicity"
    if name in {"MolLogP", "SlogP"}:
        return "Lipophilicity"

    # Charge-weighted / PEOE surface → electronic (Gasteiger charge bins)
    if name.startswith("PEOE_VSA"):
        return "Electronic"

    # E-state × surface hybrids → surface area (paper VSA_EState gloss)
    if name.startswith("VSA_EState") or name.startswith("EState_VSA"):
        return "Surface area"

    # Molar refractivity surface bins
    if name.startswith("SMR_VSA"):
        return "Surface area"

    # Pure E-state / partial charge / BCUT charge
    if (
        "PartialCharge" in name
        or "EState" in name
        or name.startswith("BCUT2D_CHG")
    ):
        return "Electronic"

    # BCUT lipophilicity / MR / MW
    if name.startswith("BCUT2D_LOGP"):
        return "Lipophilicity"
    if name.startswith("BCUT2D_MR"):
        return "Structural complexity"
    if name.startswith("BCUT2D_MW"):
        return "Size / composition"

    # Fingerprint density / MR global
    if name.startswith("FpDensity") or name in {"MolMR"}:
        return "Structural complexity"

    # Connectivity / shape topology
    if (
        name.startswith("Chi")
        or name.startswith("Kappa")
        or name.startswith("HallKier")
        or name in {"BalabanJ", "BertzCT", "Ipc", "AvgIpc"}
    ):
        return "Topology"

    # Size / composition counts
    if name in {
        "MolWt",
        "ExactMolWt",
        "HeavyAtomCount",
        "NumAtoms",
        "NumHeavyAtoms",
        "NumValenceElectrons",
        "NumHeteroatoms",
        "LabuteASA",
    } or name.endswith("AtomCount"):
        return "Size / composition"

    # Pharmacophore / polarity
    if name in {
        "TPSA",
        "NumHAcceptors",
        "NumHDonors",
        "NumRotatableBonds",
        "NOCount",
        "NHOHCount",
    }:
        return "Pharmacophore / polarity"

    if name.startswith("fr_"):
        # Remaining fragment counts: treat as pharmacophore-like structural flags
        return "Pharmacophore / polarity"

    return "Other"


def category_rank(category: str) -> int:
    try:
        return CATEGORY_ORDER.index(category)
    except ValueError:
        return len(CATEGORY_ORDER)


def sort_descriptors_by_category(
    names: Sequence[str],
    *,
    within_category: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Stable category-major sort.

    ``within_category``: preferred order inside a category (e.g. FIXED_15 order).
    Names not listed there sort alphabetically within the category.
    """
    pref = {n: i for i, n in enumerate(within_category or [])}

    def key(n: str) -> tuple[int, int, str]:
        cat = categorize_descriptor(n)
        return (category_rank(cat), pref.get(n, 10_000), n)

    return sorted(names, key=key)


def category_block_boundaries(ordered_names: Sequence[str]) -> list[tuple[str, int, int]]:
    """
    Return ``(category, start, end)`` half-open index spans for an ordered list.
    """
    if not ordered_names:
        return []
    blocks: list[tuple[str, int, int]] = []
    start = 0
    cur = categorize_descriptor(ordered_names[0])
    for i, name in enumerate(ordered_names[1:], start=1):
        cat = categorize_descriptor(name)
        if cat != cur:
            blocks.append((cur, start, i))
            start = i
            cur = cat
    blocks.append((cur, start, len(ordered_names)))
    return blocks


def descriptor_category_table(
    names: Iterable[str],
    *,
    role: Optional[str] = None,
) -> pd.DataFrame:
    """Build a companion Feature → category table."""
    rows = []
    for name in names:
        cat = categorize_descriptor(name)
        rows.append(
            {
                "Feature": name,
                "category": cat,
                "category_short": CATEGORY_SHORT.get(cat, cat),
                "role": role or "",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_ord"] = df["category"].map(category_rank)
    df = df.sort_values(
        by=["_ord", "Feature"], kind="mergesort"
    ).drop(columns=["_ord"]).reset_index(drop=True)
    return df


def fixed15_and_near_category_table(
    fixed15: Sequence[str],
    near_substitutes: Sequence[str],
) -> pd.DataFrame:
    """Combined map for fixed-15 + near-substitutes (deduped, roles tagged)."""
    fixed_set = set(fixed15)
    near_set = set(near_substitutes)
    all_names = sort_descriptors_by_category(
        list(fixed_set | near_set), within_category=fixed15
    )
    rows = []
    for name in all_names:
        cat = categorize_descriptor(name)
        if name in fixed_set and name in near_set:
            role = "fixed-15 & near-substitute"
        elif name in fixed_set:
            role = "fixed-15"
        else:
            role = "near-substitute"
        rows.append(
            {
                "Feature": name,
                "category": cat,
                "category_short": CATEGORY_SHORT.get(cat, cat),
                "role": role,
                "in_fixed15": name in fixed_set,
                "in_near_substitutes": name in near_set,
            }
        )
    return pd.DataFrame(rows)
