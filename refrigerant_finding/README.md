# Refrigerant Screening Code & Data Directory

Python scripts, candidate pools, and robustness analyses for the refrigerant screening pipeline. Run all commands from the repository root.

## Quick Summary of Directories

- `candidates_type_i_training_db/`: Strategy I ranking universe (28k molecules), top 100, and 65 filtered candidates.
- `candidates_type_iia_generated_pubchem/`: Strategy IIa ranking universe (24k molecules), top 100, and 56 filtered candidates.
- `candidates_type_iib_generated_novel/`: Strategy IIb ranking universe (10.6k PubChem-absent molecules) and 84 filtered candidates.
- `candidates_type_iii_formula_matched/`: Strategy III ranking universe (69 molecules) and 23 filtered candidates.
- `statistical_comparison/`: 170 RDKit descriptors for the 21 profile fluids and 177 size-matched reference database molecules.
- `profile_robustness/`: Supplementary robustness analyses (fluid LOO, descriptor ablation, metric sensitivity, bootstrap).

## Main Scripts

- `refrigerant_comparison.py`: Compute RDKit descriptors and Jensen–Shannon distance ranking.
- `find_refrigerant_candidates.py`: Rank database molecules against the 15-D refrigerant profile.
- `extract_properties.py`: Deduplicate canonical SMILES and filter out known refrigerants.
- `generate_type_iii_seeds.py`: Extract 21 formula-matched seeds for CoCoGraph generative sampling.
