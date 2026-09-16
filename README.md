# Working-Fluid Screening Workflow

Companion code and data for the downstream working-fluid screening pipeline described in the manuscript. It reproduces candidate generation *queries*, RDKit physicochemical descriptor computation, ranking against the 15-descriptor refrigerant profile, post-filtering, and the published candidate lists.

The generative model itself is released separately at [https://github.com/seeslab/cocograph](https://github.com/seeslab/cocograph), including its training/sampling code and the 8.2-million-molecule generated database used for type II candidates.

---

## 1. Directory & Data Structure

```
.
├── README.md                                     <- Top-level workflow guide (this file)
├── LICENSE                                       <- MIT License
├── requirements.txt                              <- Python dependencies
└── refrigerant_finding/
    ├── refrigerants.csv                          <- 38 known working fluids (ASHRAE reference)
    ├── refrigerants_extra.csv                    <- Additional working fluids
    ├── REFRIGERANT_PROPERTIES.md                 <- Curated 15-D profile target values
    │
    ├── statistical_comparison/                   <- Profiling & reference data
    │   ├── combined_descriptors.csv              <- 170 RDKit descriptors for 21 size-matched
    │   │                                            fluids + 177 size-matched reference molecules
    │   └── js_distances_ranked*.csv              <- Jensen–Shannon distances for descriptor selection
    │
    ├── candidates_type_i_training_db/            <- Strategy I candidate pool (training DB universe)
    │   ├── all_molecules_ranked.csv              <- Ranked universe (28,236 molecules)
    │   ├── top_100_molecules.csv                 <- Top 100 closest to refrigerant profile
    │   └── canonicalized_molecules_properties.csv<- 65 filtered candidates (passed to thermo analysis)
    │
    ├── candidates_type_iia_generated_pubchem/    <- Strategy IIa candidate pool (generated, in PubChem)
    │   ├── all_molecules_ranked.csv              <- Ranked universe (24,230 molecules)
    │   ├── top_100_molecules.csv                 <- Top 100 closest to refrigerant profile
    │   └── canonicalized_molecules_properties.csv<- 56 filtered candidates (passed to thermo analysis)
    │
    ├── candidates_type_iib_generated_novel/      <- Strategy IIb candidate pool (generated, novel / absent PubChem)
    │   ├── all_molecules_ranked.csv              <- Symlink to type II ranking universe
    │   └── iib_pubchem_absent_candidate_smiles.txt <- PubChem-absent subset cache (10,632 molecules; 84 filtered)
    │
    ├── candidates_type_iii_formula_matched/      <- Strategy III candidate pool (formula-matched CoCoGraph)
    │   ├── all_molecules_ranked.csv              <- Ranked universe (69 molecules)
    │   ├── top_100_molecules.csv                 <- Top closest molecules
    │   └── canonicalized_molecules_properties_69.csv <- 23 filtered candidates (passed to thermo analysis)
    │
    ├── refrigerant_comparison.py                 <- Descriptor computation & Jensen–Shannon ranking
    ├── find_refrigerant_candidates.py            <- Euclidean distance ranking against 15-D profile
    ├── extract_properties.py                     <- Deduplication & known-fluid filtering
    ├── generate_type_iii_seeds.py                <- 21 size-matched seeds for CoCoGraph type III sampling
    └── profile_robustness/                       <- Comprehensive SI sensitivity & robustness analyses
        ├── loo_descriptor_metrics.py             <- Leave-one-out on profile fluids (JS distance / MW test)
        ├── loo_mean_profile_ranks.py             <- LOO mean profile re-ranking across all strategies
        ├── loo_descriptor_ablation_ranks.py      <- 15-D descriptor ablation (leave-one-descriptor-out)
        ├── distance_metric_sensitivity_ranks.py  <- Metric sensitivity (Euclidean vs L1 vs Cosine vs Chebyshev)
        └── resample_reference_metrics.py         <- Reference set bootstrap & redraw robustness
```

---

## 2. Published Candidate Pools (Strategies I–III)

| Strategy | Candidate Source | Directory | Published Filtered List | Focal Molecule |
|---|---|---|---|---|
| **Type I** | CoCoGraph training database (2.25 M) | `refrigerant_finding/candidates_type_i_training_db/` | 65 molecules | Compound B (`R245cb`) |
| **Type IIa** | Generated database, present in PubChem | `refrigerant_finding/candidates_type_iia_generated_pubchem/` | 56 molecules | Compound D (`CF3CBr2H`) |
| **Type IIb** | Generated database, novel / absent from PubChem | `refrigerant_finding/candidates_type_iib_generated_novel/` | 84 (64 in paper text) | Compound E (`CH3NFCHFBr`) |
| **Type III** | Formula-matched CoCoGraph generation | `refrigerant_finding/candidates_type_iii_formula_matched/` | 23 molecules | Compounds A & C |

Each candidate directory ships the full ranking universe (`all_molecules_ranked.csv`), the `top_100_molecules.csv`, and the post-filter candidate properties passed to thermodynamic evaluation.

---

## 3. Installation & Quick Verification

### Requirements
```bash
pip install -r requirements.txt
```

### Quick Verification (10 seconds)
Verify that the screening and sensitivity pipelines run correctly and reproduce published baseline values:
```bash
python3 refrigerant_finding/profile_robustness/loo_mean_profile_ranks.py --stage both
```

---

## 4. Step-by-Step Screening Workflow

All commands are run from the repository root.

### Step 1: Compute Descriptors & Size-Matched Profiling
Compare known working fluids against database molecules to identify the most discriminative physicochemical descriptors via Jensen–Shannon divergence:

```bash
python3 refrigerant_finding/refrigerant_comparison.py \
  --size_match \
  --atom_range 5 10 \
  --n_samples 10000 \
  --size_buffer 1.0 \
  --database_file /path/to/molecules_lt70atoms_annotated.csv
```

### Step 2: Rank Candidates against the 15-Descriptor Refrigerant Profile
Rank any molecular database (Type I or Type II) against the curated working-fluid profile:

```bash
# Type I ranking
python3 refrigerant_finding/find_refrigerant_candidates.py \
  --database_file /path/to/training_database.csv \
  --min_atoms 5 --max_atoms 10 --n_candidates 100 \
  --output_dir refrigerant_finding/candidates_type_i_training_db

# Type II ranking
python3 refrigerant_finding/find_refrigerant_candidates.py \
  --database_file /path/to/generated_8M_database.csv \
  --min_atoms 5 --max_atoms 10 --n_candidates 100 \
  --output_dir refrigerant_finding/candidates_type_iia_generated_pubchem
```
*(Omit `--use_calculated_means` to rank using the documented target means in `REFRIGERANT_PROPERTIES.md`, which reproduced published distances).*

### Step 3: Filter Known Working Fluids & Duplicates
Extract canonical properties from the top-100 list, dropping known refrigerants and duplicate canonical SMILES:

```bash
python3 refrigerant_finding/extract_properties.py \
  --csv_file refrigerant_finding/candidates_type_i_training_db/top_100_molecules.csv \
  --output_dir refrigerant_finding/candidates_type_i_training_db \
  --output_file canonicalized_molecules_properties.csv
```

### Step 4: Formula-Matched (Type III) Generation
Write the 21 size-matched working-fluid seed SMILES for CoCoGraph:

```bash
python3 refrigerant_finding/generate_type_iii_seeds.py \
  --cocograph-root /path/to/cocograph
```
Then run CoCoGraph's FPS diffusion sampler using the command printed by the script, and rank the resulting molecules into `refrigerant_finding/candidates_type_iii_formula_matched/`.

---

## 5. Supplementary Information Robustness Analyses (SI §S1.3)

All robustness experiments can be reproduced immediately using the shipped data tables without needing to re-download the multi-gigabyte databases:

```bash
# 1. Leave-one-out descriptor selection stability (JS distances & Mann–Whitney tests)
python3 refrigerant_finding/profile_robustness/loo_descriptor_metrics.py

# 2. Leave-one-out profile fluid re-ranking & retention across Strategies I–III
python3 refrigerant_finding/profile_robustness/loo_mean_profile_ranks.py --stage both

# 3. Leave-one-descriptor-out ablation across Strategies I–III
python3 refrigerant_finding/profile_robustness/loo_descriptor_ablation_ranks.py

# 4. Sensitivity to distance metric (Euclidean vs Manhattan vs Cosine vs Chebyshev)
python3 refrigerant_finding/profile_robustness/distance_metric_sensitivity_ranks.py

# 5. Reference set bootstrapping (50 replicates)
python3 refrigerant_finding/profile_robustness/resample_reference_metrics.py --mode bootstrap --n-replicates 50
```

---

## License

MIT License. CoCoGraph remains under its own license in its repository ([https://github.com/seeslab/cocograph](https://github.com/seeslab/cocograph)).
