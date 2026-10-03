# SiO2/Water Water-Only Structural Interpretation

## Paper-Derived Assignment Used

- The SiO2/water paper assigns the phase-resolved positive band near 3400 cm^-1 to H-bonded water whose hydrogens point toward the negatively charged silica surface.
- It assigns the 3660 cm^-1 feature to weakly H-bonded/quasi-free OH groups of water in the direct binding interfacial layer, with the H atom pointing toward siloxane bridge oxygen sites.
- DFT-MD separates two direct-interface water populations in the 3500-3800 cm^-1 region: OH toward siloxane bridges at 3660 cm^-1 and OH toward in-plane silanol sites as a broad band centered near 3470 cm^-1.
- The paper does not provide a water-structure assignment for the strong features below 3000 cm^-1 in this file; those are treated here as non-water/background/nuisance for the purpose of water-structure inference.

## Fit Summary

- Fit window: 3300-3800 cm^-1.
- Phase/sign relative to the Cyran convention: -1. A value of -1 means the supplied complex data are globally sign-flipped relative to the positive-Im convention in the paper.
- Complex R2 on smoothed data in the water window: 0.8600.
- Removing the quasi-free/siloxane-water components worsens SSE by 0.0080679 (22.49% of the no-free-OH SSE).

## Fitted Water Components

| component | center cm^-1 | group | weight fraction | expected sign in data |
| --- | ---: | --- | ---: | ---: |
| BIL water OH toward in-plane silanol | 3470 | silanol-site-water | 0.412 | -1 |
| H-bonded water with H toward silica | 3400 | surface-oriented-hbonded | 0.267 | -1 |
| diffuse-layer / strongly H-bonded water | 3200 | hbonded-water | 0.205 | -1 |
| heterogeneous weakly H-bonded OH | 3685 | siloxane-quasifree-water | 0.114 | -1 |
| quasi-free OH toward siloxane bridge | 3660 | siloxane-quasifree-water | 0.001 | -1 |

## Structural Model

The data are most consistently described as a weak water-OH response sitting on top of much larger non-water/background structure below 3000 cm^-1. Within the water window, the model points to interfacial water whose OH groups are biased toward the silica surface rather than toward a vapor phase.

Compared with water/air, the free-OH-like water is not a dangling OH pointing out into vacuum. It is quasi-free or very weakly H-bonded because it points toward hydrophobic siloxane bridge sites on SiO2. Dynamically and spectroscopically this resembles the free OH at water/air, but geometrically it is buried and surface-directed.

The broader 3400-3470 cm^-1 contribution corresponds to H-bonded binding-layer water, including water interacting with silanol/silanolate-like surface sites. This is the part that makes SiO2/water different from a pure hydrophobic interface: hydrophilic and hydrophobic patches coexist, so the spectrum contains both H-bonded surface-directed water and hydrophobic-patch quasi-free water.

## Caution

The component weights are spectroscopic amplitudes, not direct populations. A quantitative population model would need the original phase convention, calibration, and ideally SiO2/water motif-resolved simulation spectra over the same experimental window.