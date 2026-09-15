# Mean-profile LOO — SI tables (draft)

## Table Sx. Per-fold shift of the mean profile (normalized space)

| Left-out fluid | L2(d mean) | max abs d | mean abs d | Descriptor with max abs d |
| --- | --- | --- | --- | --- |
| R601 | 0.0996 | 0.0452 | 0.0213 | NumRotatableBonds |
| R601a | 0.0901 | 0.0344 | 0.0199 | MaxPartialCharge |
| R227ea | 0.0675 | 0.0272 | 0.0146 | FpDensityMorgan3 |
| R134a | 0.0622 | 0.0325 | 0.0110 | FpDensityMorgan3 |
| R1345fz | 0.0560 | 0.0238 | 0.0120 | LabuteASA |
| R1336yf | 0.0560 | 0.0238 | 0.0120 | LabuteASA |
| R1123 | 0.0484 | 0.0238 | 0.0101 | fr_alkyl_halide |
| R125 | 0.0357 | 0.0161 | 0.0081 | MolMR |
| R1224yd(E) | 0.0351 | 0.0198 | 0.0067 | MolLogP |
| R1224yd(Z) | 0.0351 | 0.0198 | 0.0067 | MolLogP |
| R245fa | 0.0322 | 0.0202 | 0.0066 | NumRotatableBonds |
| R1225ye(Z) | 0.0274 | 0.0114 | 0.0062 | MinPartialCharge |
| R1336mzz(Z) | 0.0252 | 0.0164 | 0.0050 | MolLogP |
| R1243zf | 0.0252 | 0.0164 | 0.0050 | MolLogP |
| R1233zd(E) | 0.0232 | 0.0156 | 0.0043 | FpDensityMorgan3 |
| R1336mzz(E) | 0.0224 | 0.0156 | 0.0041 | FpDensityMorgan3 |
| R13I1 | 0.0208 | 0.0114 | 0.0041 | FpDensityMorgan3 |
| R236fa | 0.0192 | 0.0102 | 0.0040 | LabuteASA |
| R1234ze(Z) | 0.0179 | 0.0124 | 0.0033 | MinPartialCharge |
| R1234ze(E) | 0.0179 | 0.0124 | 0.0033 | MinPartialCharge |
| R1234yf | 0.0146 | 0.0084 | 0.0029 | MinPartialCharge |

## Table Sy. Per-descriptor sensitivity of the mean under LOO

| Descriptor | Category | mean abs d | max abs d | std abs d |
| --- | --- | --- | --- | --- |
| FpDensityMorgan3 | Complexity | 0.0124 | 0.0325 | 0.0084 |
| MolLogP | Lipophilicity | 0.0115 | 0.0273 | 0.0079 |
| LabuteASA | Surface | 0.0114 | 0.0262 | 0.0079 |
| MinPartialCharge | Electronic | 0.0096 | 0.0340 | 0.0089 |
| MolMR | Complexity | 0.0095 | 0.0297 | 0.0070 |
| fr_alkyl_halide | Halogen | 0.0093 | 0.0262 | 0.0090 |
| VSA_EState1 | Surface | 0.0090 | 0.0264 | 0.0080 |
| SMR_VSA1 | Surface | 0.0087 | 0.0269 | 0.0080 |
| fr_halogen | Halogen | 0.0086 | 0.0282 | 0.0083 |
| NumHeteroatoms | Size | 0.0086 | 0.0282 | 0.0083 |
| NumRotatableBonds | Polarity | 0.0082 | 0.0452 | 0.0094 |
| MinEStateIndex | Electronic | 0.0081 | 0.0376 | 0.0098 |
| MaxPartialCharge | Electronic | 0.0072 | 0.0349 | 0.0093 |
| TPSA | Surface | 0.0000 | 0.0000 | 0.0000 |
| NumHAcceptors | Polarity | 0.0000 | 0.0000 | 0.0000 |

## Headline stats

- Mean L2(Δ) = 0.0396 (median 0.0322; max 0.0996 leaving out R601)
- Mean of fold-wise max|Δ| = 0.0199; mean of fold-wise mean|Δ| = 0.0081
- Descriptors with zero LOO shift: 2 (TPSA, NumHAcceptors)

