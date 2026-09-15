# Key Molecular Properties of Refrigerants

This document outlines the key molecular properties that differentiate refrigerants from non-refrigerant molecules, based on statistical analysis of molecular descriptors. These findings can be used to identify potential new refrigerant candidates or understand why certain molecules function well as refrigerants.

## Top Differentiating Properties

Below are the most significant properties that distinguish refrigerants from other molecules, ranked by Jensen-Shannon distance (a measure of distribution dissimilarity):

### 1. MaxPartialCharge (JS Distance: 0.76)
- **Description**: The maximum positive partial charge in the molecule
- **Refrigerants**: 0.4 (mean)
- **Non-refrigerants**: 0.2 (mean)
- **Significance**: Refrigerants have significantly higher maximum partial charges, indicating distinctive electronic charge distribution that affects their interaction with surfaces and other molecules.

### 2. SMR_VSA1 (JS Distance: 0.75)
- **Description**: Surface area contribution from atoms with specific molar refractivity range
- **Refrigerants**: 16.5 (mean)
- **Non-refrigerants**: 6.8 (mean)
- **Significance**: Indicates different molecular surface properties that may affect volatility and fluid behavior.

### 3. MinEStateIndex (JS Distance: 0.75)
- **Description**: The minimum electrotopological state index in the molecule
- **Refrigerants**: -4.1 (mean)
- **Non-refrigerants**: -0.2 (mean)
- **Significance**: Reflects electronic and topological characteristics of the most electronegative atoms, typically halogens in refrigerants.

### 4. MolMR (JS Distance: 0.75)
- **Description**: Molar refractivity, related to molecular volume and polarizability
- **Refrigerants**: 17.9 (mean)
- **Non-refrigerants**: 34.6 (mean)
- **Significance**: Lower values in refrigerants indicate specific optical and physical properties important for gas-liquid phase transitions.

### 5. FpDensityMorgan3 (JS Distance: 0.75)
- **Description**: Density of Morgan fingerprint features at radius 3
- **Refrigerants**: 1.4 (mean)
- **Non-refrigerants**: 2.4 (mean)
- **Significance**: Lower values indicate simpler molecular structures with less chemical feature diversity.

### 6. MinPartialCharge (JS Distance: 0.74)
- **Description**: The minimum (most negative) partial charge in the molecule
- **Refrigerants**: -0.2 (mean)
- **Non-refrigerants**: -0.4 (mean)
- **Significance**: Less negative values in refrigerants indicate different electron distribution patterns.

### 7. fr_halogen (JS Distance: 0.70)
- **Description**: Count of halogen atoms (F, Cl, Br, I)
- **Refrigerants**: 4.0 (mean)
- **Non-refrigerants**: 0.3 (mean)
- **Significance**: Refrigerants contain substantially more halogen atoms, critical for their thermodynamic properties.

### 8. fr_alkyl_halide (JS Distance: 0.69)
- **Description**: Count of alkyl halide groups
- **Refrigerants**: 3.3 (mean)
- **Non-refrigerants**: 0.1 (mean)
- **Significance**: High presence of alkyl halides is characteristic of many refrigerants (CFCs, HCFCs, HFCs).

### 9. TPSA (JS Distance: 0.68)
- **Description**: Topological Polar Surface Area, sum of surface contributions from polar atoms
- **Refrigerants**: 0.0 (mean)
- **Non-refrigerants**: 36.3 (mean)
- **Significance**: Refrigerants consistently have zero TPSA, indicating absence of polar groups like -OH, -NH, etc.

### 10. NumHAcceptors (JS Distance: 0.70)
- **Description**: Number of hydrogen bond acceptors
- **Refrigerants**: 0.0 (mean)
- **Non-refrigerants**: 2.0 (mean)
- **Significance**: Refrigerants lack hydrogen bond acceptors, contributing to their volatility and non-water-soluble nature.

## Additional Important Properties

### 11. VSA_EState1 (JS Distance: 0.70)
- **Description**: Surface area contribution from atoms with specific electrotopological state
- **Refrigerants**: 40.2 (mean)
- **Non-refrigerants**: 3.2 (mean)
- **Significance**: Much higher in refrigerants, indicating distinctive electronic distribution on molecular surface.

### 12. LabuteASA (JS Distance: 0.65)
- **Description**: Approximate surface area calculation
- **Refrigerants**: 39.4 (mean)
- **Non-refrigerants**: 54.3 (mean)
- **Significance**: Lower accessible surface area in refrigerants relates to their compact molecular structure.

### 13. MolLogP (JS Distance: 0.58)
- **Description**: Calculated octanol-water partition coefficient
- **Refrigerants**: 2.1 (mean)
- **Non-refrigerants**: 0.9 (mean)
- **Significance**: Higher values indicate greater lipophilicity/hydrophobicity, important for refrigerant behavior.

### 14. NumHeteroatoms (JS Distance: 0.44)
- **Description**: Count of atoms that are not C or H
- **Refrigerants**: 4.0 (mean)
- **Non-refrigerants**: 2.8 (mean)
- **Significance**: Refrigerants have more heteroatoms, primarily halogens, affecting their physical properties.

### 15. NumRotatableBonds (JS Distance: 0.41)
- **Description**: Count of rotatable bonds
- **Refrigerants**: 0.2 (mean)
- **Non-refrigerants**: 1.4 (mean)
- **Significance**: Fewer rotatable bonds indicate less molecular flexibility, contributing to refrigerants' physical properties.

## Chemical Profile of Refrigerants

Based on these properties, refrigerants can be characterized as:

- **Halogen-rich molecules**: High counts of halogen atoms and alkyl halide groups
- **Non-polar**: Zero topological polar surface area (TPSA)
- **No hydrogen bond acceptors**: Absence of groups that can accept hydrogen bonds
- **Higher lipophilicity**: Higher MolLogP values indicating hydrophobic nature
- **Limited flexibility**: Few rotatable bonds
- **Distinctive electronic properties**: Characteristic partial charge distribution
- **Simpler molecular structures**: Lower fingerprint density values
- **Compact structures**: Different surface area properties than non-refrigerants

These properties collectively contribute to the thermodynamic behavior that makes these molecules suitable as refrigerants, including appropriate boiling points, vapor pressures, and heat transfer characteristics.

## Data Sources

This analysis is based on statistical comparison between known refrigerants and a database of non-refrigerant molecules, using:

1. Jensen-Shannon distances to measure distribution differences
2. Statistical significance testing (p-values)
3. Calculated RDKit molecular descriptors

For the raw data, refer to:
- `refrigerant_finding/statistical_comparison/js_distances_ranked_database_size_matched.csv`
- `refrigerant_finding/statistical_comparison/tableone_batch_*.csv` files

## Usage for Screening Potential Refrigerants

When screening molecules as potential refrigerants, prioritize candidates that match these property profiles:
1. Contain multiple halogen atoms (especially F, Cl)
2. Lack polar functional groups
3. Have relatively simple, compact structures
4. Possess the specific electronic property distributions described above

These guidelines should help identify molecules that may possess the physical and thermodynamic properties desired in refrigerants. 