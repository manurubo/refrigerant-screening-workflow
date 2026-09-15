#!/usr/bin/env python3
"""Prepare seed SMILES for formula-matched (type III) CoCoGraph generation.

Type III candidates are generated with CoCoGraph by noising and denoising the
21 size-matched working fluids, which preserves each fluid's molecular formula.
This repository does not vendor CoCoGraph; clone it from
https://github.com/seeslab/cocograph and point --cocograph-root at that clone
(or set COCOGRAPH_ROOT).

This script only writes the seed CSV. Generation itself is run inside the
CoCoGraph tree with sample_scripts/sample_molecules_FPSmodel.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

HEAVY_ATOM_MIN = 5
HEAVY_ATOM_MAX = 10
DEFAULT_REFRIGERANTS = Path("refrigerant_finding/refrigerants.csv")
DEFAULT_OUTPUT = Path("refrigerant_finding/type_iii_seeds.csv")

FPS_DIFFUSION_CKPT = "models/FPS_diffusion/model_epoch_1_slice_22.pth"
FPS_TIME_CKPT = "models/FPS_time/model_epoch_2_slice_22.pth"


def size_matched_working_fluids(refrigerants_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(refrigerants_csv)
    if "SMILES" not in df.columns:
        raise ValueError(f"{refrigerants_csv} must contain a SMILES column")
    rows = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["SMILES"]))
        if mol is None:
            continue
        n_heavy = int(Descriptors.HeavyAtomCount(mol))
        if HEAVY_ATOM_MIN <= n_heavy <= HEAVY_ATOM_MAX:
            record = row.to_dict()
            record["heavy_atom_count"] = n_heavy
            record["smiles"] = Chem.MolToSmiles(mol, canonical=True)
            record["molecular_formula"] = Chem.rdMolDescriptors.CalcMolFormula(mol)
            rows.append(record)
    if len(rows) != 21:
        raise RuntimeError(
            f"Expected 21 size-matched working fluids "
            f"({HEAVY_ATOM_MIN}–{HEAVY_ATOM_MAX} heavy atoms); got {len(rows)}"
        )
    return pd.DataFrame(rows)


def cocograph_command(seed_csv: Path, cocograph_root: Path | None) -> str:
    seed_abs = seed_csv.resolve()
    ckpt = FPS_DIFFUSION_CKPT
    time_ckpt = FPS_TIME_CKPT
    prefix = ""
    if cocograph_root is not None:
        prefix = f"cd {cocograph_root.resolve()} && \\\n"
    return (
        f"{prefix}python sample_scripts/sample_molecules_FPSmodel.py \\\n"
        f"  --input_smiles_csv {seed_abs} \\\n"
        f"  --output_dir_suffix type_III_formula_matched \\\n"
        f"  --model_checkpoint_path {ckpt} \\\n"
        f"  --time_model_checkpoint_path {time_ckpt} \\\n"
        f"  --batch_size_sample 21 \\\n"
        f"  --batch_size_process 21"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the 21 type-III CoCoGraph seed SMILES and print the generation command."
    )
    parser.add_argument(
        "--refrigerants-file",
        type=Path,
        default=DEFAULT_REFRIGERANTS,
        help="Known working-fluids CSV (default: refrigerant_finding/refrigerants.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Seed CSV written for CoCoGraph (must contain a 'smiles' column)",
    )
    parser.add_argument(
        "--cocograph-root",
        type=Path,
        default=os.environ.get("COCOGRAPH_ROOT"),
        help="Path to a CoCoGraph clone (or set COCOGRAPH_ROOT). Used only to print the command.",
    )
    args = parser.parse_args()

    seeds = size_matched_working_fluids(args.refrigerants_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = seeds[["smiles", "molecular_formula", "Compound", "heavy_atom_count"]].copy()
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} type-III seed molecules to {args.output}")

    print("\nCoCoGraph is not included in this repository.")
    print("Clone it from https://github.com/seeslab/cocograph then run:\n")
    print(cocograph_command(args.output, args.cocograph_root))
    print(
        "\nThe generated SMILES can then be ranked with "
        "find_refrigerant_candidates.py --database_file <generated.csv> "
        "--output_dir refrigerant_finding/candidates_refrigerantsmf"
    )


if __name__ == "__main__":
    main()
