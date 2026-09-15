# Near-substitutes vs fixed-15: selection notes

Evidence base: LOO aggregation under `tables/` (freq ≥ 90%, Pearson *r* on 197 molecules), paper Methods selection criteria, `descriptor_taxonomy.py`. Near-substitutes are **not** claimed replacements.

## Operational definition

Per fold (`loo_descriptor_metrics.near_substitutes`, `run_config.json`):

1. Not in curated fixed-15
2. Mann–Whitney `p < 0.05`
3. High-JS gate (`js_mode=both`, `top_n=30`): `JS_rank ≤ 30` **or** `JS ≥ min(fixed-15 JS)`

Aggregation: frequency = fraction of 21 LOO folds; filtered table uses frequency ≥ 0.9 (52 descriptors; 50 in all folds). Correlation classes vs fixed-15: high `|r|≥0.7`, mid `0.4–0.7`, low `<0.4`.

## A. Should NOT treat as alternative features

| Near-sub | Why discard |
| --- | --- |
| **NOCount** | Near-duplicate of polarity already covered: *r*=0.95 vs TPSA, 0.86 vs NumHAcceptors. |
| **FpDensityMorgan2** | Same family as selected FpDensityMorgan3 (*r*=0.92); redundant radius twin, not extra information. |
| **Chi0v / Chi0n / Chi1 / Chi1v / Chi1n** (+ most Chi*) | Size/polarizability proxies: *r*≈0.78–0.92 vs MolMR or LabuteASA; topology Chi swarm excluded by Methods diversity + anti-correlation rules. |
| **HeavyAtomCount, NumValenceElectrons, Chi0** | Collinear with LabuteASA (*r*≈0.80–0.86); size already constrained (5–10 HAC). Chi0 also weaker MW (max *p*≈0.020). |
| **SlogP_VSA10** | Highest near-sub JS, but *r*=0.87 vs VSA_EState1 and *r*≈0.86 vs both halogen counts — opaque SlogP-VSA bin that largely restates already-selected surface + halogen signal. |
| **PEOE_VSA14, PEOE_VSA4** | Charge/surface bins highly aligned with MaxPartialCharge or VSA_EState1 (*r*≥0.71); less interpretable than selected electronic/surface features. |
| **EState_VSA10, VSA_EState7, EState_VSA1** | High-|r| E-state×VSA siblings of MinEStateIndex / SMR_VSA1; composite bins. |
| **NumHDonors, NHOHCount** | *r*≈0.74–0.75 vs TPSA; H-bond donors are chemically secondary for typical HFC/HFO working fluids vs acceptors (already selected). |
| **Kappa2** | *r*=0.83 vs NumRotatableBonds; shape/flexibility proxy, less direct than rotatable-bond count. |
| **MaxEStateIndex / MaxAbsEStateIndex** | Identical values here; paper already keeps **Min**EStateIndex (more refrigerant-relevant). Mid-corr only; no distinct gain. |
| **qed** | Composite drug-likeness; not motivated by Methods refrigerant criteria. |
| **RingCount** | Narrow/sparse for small acyclic refrigerants; mid corr only. |
| **BCUT2D_MRLOW, BCUT2D_MWHI, Chi0** | Borderline MW under LOO (max *p* up to ~0.03–0.046) and/or opaque BCUT bins. |
| **Low-corr VSA leftovers** (EState_VSA6/9, SMR_VSA6) | Low max-|r| ≠ good features: lower JS, opaque bins, weak chemical story vs fixed-15 counterparts. |

## B. Could have been selected (reasonable alternates)

Excluded mainly for within-family redundancy, interpretability, or diversity — not for failing JS/significance.

| Near-sub | Case for alternate |
| --- | --- |
| **FpDensityMorgan1** | Same complexity family as Morgan3; only mid corr (*r*=0.68); competitive persistent JS (~0.64). Cleaner within-family pick than Morgan2 (*r*=0.92). |
| **PEOE_VSA1** | Electronic alternate near MinPartialCharge (*r*=−0.69, just below high threshold); stable high-JS. Lost to clearer partial-charge interpretation. |
| **MaxEStateIndex** | Same electronic/E-state family as MinEStateIndex; significant and persistent. Reasonable sibling if one wanted “max” rather than “min”. |
| **BCUT2D_LOGPHI** | Lipophilicity family; only mid corr with MolLogP (*r*=0.50); adds somewhat distinct LogP-BCUT info. Lost to interpretable MolLogP. |
| **SlogP_VSA2** | Lipophilicity surface alternate (mid vs MolLogP); weaker JS than MolLogP’s role and more opaque. |
| **AvgIpc / BalabanJ / BertzCT** | Topology with mid max-|r|; competitive JS (esp. AvgIpc ~0.68). Plausible topology representatives; fixed-15 instead used NumHeteroatoms / NumRotatableBonds for clearer chemistry. |
| **BCUT2D_CHGHI** | Mid-corr electronic (*r*≈0.56); JS ~0.66. Viable electronic complement but opaque vs Max/MinPartialCharge. |
| **SMR_VSA10** | Surface/MR-VSA sibling of SMR_VSA1 with only mid corr to LabuteASA; interpretability favors the selected SMR_VSA1 + LabuteASA pair. |

**Not** strong “could select” picks despite low corr: BCUT2D_MRHI / BCUT2D_MWHI / EState_VSA* low-JS bins — complementary on *r* but weaker metrics or opaque.

## Fixed-15 gaps (no strong within-family near-subs)

| Fixed-15 | Gap |
| --- | --- |
| **fr_halogen, fr_alkyl_halide** | Never anyone’s closest fixed-15; **zero** halogen-named descriptors in the ≥90% near-sub pool. Halogenation is a unique, chemically central axis for refrigerants — LOO does not surface interchangeable fragment alternatives. (SlogP_VSA10 correlates with them but is not a halogen feature and is discarded above.) |
| **NumHeteroatoms** | No near-sub with `|r|≥0.7` (max ~0.64). Composition signal is relatively non-redundant among high-JS competitors. |
| **MinPartialCharge** | No high-corr near-sub (best PEOE_VSA1 at 0.69). Partial-charge pair in fixed-15 is not easily swapped. |

Other fixed-15 members (TPSA, MolMR, LabuteASA, FpDensityMorgan3, …) have clear high-corr swap pools — consistent with Methods criterion (iv).

## Paper narrative (one paragraph)

Leave-one-out near-substitutes show that many high-JS significant descriptors are **statistical peers** of the curated fifteen, not independent axes: connectivity indices, Morgan-radius twins, NOCount, and several PEOE/E-state VSA bins are highly collinear with MolMR, LabuteASA, TPSA, FpDensityMorgan3, or partial charges, while opaque BCUT/VSA bins and donor counts add little refrigerant-relevant interpretability. A smaller set (e.g. FpDensityMorgan1, PEOE_VSA1, BCUT2D_LOGPHI, AvgIpc/BalabanJ) are plausible within-family or mid-correlation alternates that were reasonably passed over for diversity and clarity. Critically, **halogen fragment counts have no near-substitute family in the LOO pool**, underscoring that the fixed-15 halogen axis is uniquely motivated rather than interchangeable — supporting the curated profile as a compact, non-redundant working-fluid signature rather than an arbitrary top-JS list.

## Related paths

- `tables/near_substitutes_frequency_ge90.*`
- `tables/near_vs_fixed15_correlation_summary.*`
- `tables/near_substitutes_high_corr_groups.*`
- `tables/near_substitutes_low_corr_complements.*`
- `figures/fig_near_vs_fixed15_correlation.*` (category-bracket layout present in aggregation code as of this note)
