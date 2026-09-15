# Working-fluid screening code

Python scripts and data for the screening workflow. Run them from the
repository root; see the top-level README for commands.

- `refrigerant_comparison.py` — RDKit descriptors, size matching, Jensen–Shannon ranking of descriptors
- `find_refrigerant_candidates.py` — Euclidean ranking against the 15-D working-fluid profile
- `extract_properties.py` — drop known working fluids and duplicate canonical SMILES
- `generate_type_iii_seeds.py` — write CoCoGraph seeds for formula-matched (type III) generation
- `profile_robustness/` — SI sensitivity analyses
