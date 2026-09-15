# Refrigerant screening workflow

Companion code and data for the downstream working-fluid screening pipeline
described in the manuscript. It reproduces candidate generation *queries*,
descriptor computation, ranking, filtering, and the published top-candidate
lists.

The generative model itself is **not** in this repository. CoCoGraph is
released separately at [https://github.com/seeslab/cocograph](https://github.com/seeslab/cocograph), including its
training/sampling code and the 8.2-million-molecule generated database used
for type II candidates.

## What is included

| Piece | Location |
| --- | --- |
| Known working fluids (38) | `refrigerant_finding/refrigerants.csv` |
| 15-descriptor profile (means used for ranking) | `refrigerant_finding/REFRIGERANT_PROPERTIES.md` |
| Descriptor computation + Jensen–Shannon selection | `refrigerant_finding/refrigerant_comparison.py` |
| Size-matched 21 + 177 descriptor table | `refrigerant_finding/statistical_comparison/` |
| Ranking / filtering against the mean profile | `refrigerant_finding/find_refrigerant_candidates.py`, `extract_properties.py` |
| Type III seed SMILES (formula-matched generation) | `refrigerant_finding/generate_type_iii_seeds.py` |
| Ranked pools and filtered top lists | `refrigerant_finding/candidates*/` (see below) |
| SI robustness analyses | `refrigerant_finding/profile_robustness/` |

## Candidate pools (types I–III)

| Type | Source | Directory | Filtered list |
| --- | --- | --- | --- |
| I | CoCoGraph training database (2.25 M) | `candidates/` | 65 molecules |
| IIa | Generated database, present in PubChem | `candidates_novel/` | 56 molecules |
| IIb | Generated database, absent from PubChem | `candidates_novel_seguridad/` | 84 molecules |
| III | Formula-matched CoCoGraph generation | `candidates_refrigerantsmf/` | 23 molecules |

Each pool ships `all_molecules_ranked.csv` (ranking universe) and
`top_*.csv`. The post-filter lists used in the SI (known working fluids and
duplicate canonical SMILES removed) are also in

`profile_robustness/loo_mean_profile/ranks/strategy_*/baseline_filtered.csv`.

Type IIb ranking needs the PubChem-absent subset of the type II universe.
That subset is shipped as
`candidates_novel_seguridad/iib_pubchem_absent_candidate_smiles.txt`
so the multi-gigabyte PubChem dump is not required.

## External data you still need from CoCoGraph

Place (or symlink) the CoCoGraph training SMILES table as
`molecular_formulas.csv` in the repository root, **or** pass
`--database_file` / `--molecular-formulas` explicitly.

Typical paper settings for the size-matched profile:

```bash
python refrigerant_finding/refrigerant_comparison.py \
  --size_match --atom_range 5 10 --n_samples 10000 --size_buffer 1.0 \
  --database_file /path/to/molecules_lt70atoms_annotated.csv
```

Type I / II ranking against the published 15-D profile:

```bash
python refrigerant_finding/find_refrigerant_candidates.py \
  --database_file /path/to/database.csv \
  --use_calculated_means \
  --min_atoms 5 --max_atoms 10 --n_candidates 100 \
  --output_dir refrigerant_finding/candidates
```

Omit `--use_calculated_means` to rank with the documented profile means in
`REFRIGERANT_PROPERTIES.md` (this is what produced the published distances).

Drop known working fluids and duplicate canonical SMILES from a top-100
table:

```bash
python refrigerant_finding/extract_properties.py \
  --csv_file refrigerant_finding/candidates/top_100_molecules.csv \
  --output_dir refrigerant_finding/candidates
```

## Type III generation

Write the 21 size-matched seed SMILES, then run CoCoGraph (not this repo):

```bash
python refrigerant_finding/generate_type_iii_seeds.py \
  --cocograph-root /path/to/cocograph
```

The printed command calls CoCoGraph's FPS sampler with those seeds, which
preserves each working-fluid molecular formula. Rank the generated SMILES
with `find_refrigerant_candidates.py` as above, using
`--output_dir refrigerant_finding/candidates_refrigerantsmf`.

## Robustness (SI §S1.3)

```bash
python refrigerant_finding/profile_robustness/loo_descriptor_metrics.py
python refrigerant_finding/profile_robustness/loo_mean_profile_ranks.py
python refrigerant_finding/profile_robustness/loo_descriptor_ablation_ranks.py
python refrigerant_finding/profile_robustness/distance_metric_sensitivity_ranks.py
python refrigerant_finding/profile_robustness/resample_reference_metrics.py --mode bootstrap --n-replicates 50
```

## Requirements

```bash
pip install -r requirements.txt
```

Run scripts from the repository root. RDKit is required for descriptor
computation and ranking; the robustness re-ranking steps can reuse the
shipped CSVs without regenerating the CoCoGraph databases.

## License

MIT. CoCoGraph remains under its own license in its own repository.
