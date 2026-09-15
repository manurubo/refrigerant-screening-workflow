# Type IIb pool

Type IIb candidates are type II generated molecules that are **absent from PubChem**.

- `all_molecules_ranked.csv` is a symlink to the shared type II ranking universe in `../candidates_novel/`.
- `iib_pubchem_absent_candidate_smiles.txt` is the PubChem-absent subset used for ranking (so the multi-gigabyte CID–SMILES dump is not needed).
- The filtered list of 84 molecules is
  `../profile_robustness/loo_mean_profile/ranks/strategy_IIb/baseline_filtered.csv`.
