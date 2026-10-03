# Structure-Constrained SiO2/Water SFG Model

## Fit Summary

- Best affine mapping: observed x = 0.606667 * structural_frequency + 711.000
- Equivalent structural frequency: (observed x - 711.000) / 0.606667
- Gamma scale: 0.350
- Global sign convention relative to the water-air motif table: +1
- Complex R2 in fitted structured region: 0.0282
- Literal-as-written OH-window model R2, using x-axis directly from 3000-3800 cm^-1: 0.0240

The affine mapping is not claimed to be a physical vibrational scaling factor by itself. It is a diagnostic that the strongest SiO2/water features align with the motif frequencies from the paper only after the supplied `xaxis.txt` is treated as a compressed/shifted spectral coordinate.

## Dominant Spectroscopic Components

| component | group | observed center | structural center | weight fraction | sign |
| --- | --- | ---: | ---: | ---: | ---: |
| surface-oriented bonded OH | surface-oriented-bonded | 2791.9 | 3430 | 0.362 | +1 |
| I-III free OH | free-oh | 2931.4 | 3660 | 0.354 | +1 |
| heterogeneous free-OH shoulder | free-oh | 2961.7 | 3710 | 0.135 | +1 |
| VII second-layer bonded OH | subsurface-negative | 2767.6 | 3390 | 0.109 | -1 |
| IV/VIII high-frequency coupling | coupling | 2816.1 | 3470 | 0.032 | +1 |
| VIII / low-bonded positive | subsurface-positive | 2670.5 | 3230 | 0.008 | +1 |
| IV coupling low-frequency pair | coupling | 2688.7 | 3260 | 0.000 | -1 |

## Structural Interpretation

- The fitted spectrum requires a strong positive weakly bonded/free-OH-like component that maps to the 3660 cm^-1 motif. For SiO2/water this is naturally interpreted as hydrophobic-water character: OH groups pointing toward local siloxane/oxygen-bridge sites, analogous to the Cyran et al. assignment.
- A sizeable positive bonded-OH component near the 3400-3500 cm^-1 structural region is also required. This is consistent with water donating H bonds toward negatively polarized silica sites, i.e. hydrogens biased toward the surface.
- Negative bonded-OH and coupling components are retained only where needed by the complex line shape. They represent water-water bonded motifs whose OH vectors point back into the liquid network and second-layer/coupling cancellation.
- The inferred structure is therefore not a simple water-air-like top layer. It is a mixed buried-interface layer: surface-directed/free or weakly bonded OH groups coexist with a hydrogen-bonded water network whose opposing orientations partially cancel in Im chi.

## Caveats

- The inversion is not unique; the component weights are spectroscopic amplitudes, not direct populations.
- The supplied x-axis does not behave like the final physical OH wavenumber axis. Without the affine mapping, the main features sit below 3000 cm^-1 and cannot be assigned cleanly to H2O OH stretch motifs.
- A more definitive structural model would require the original phase convention, experimental calibration notes, and ideally independent MD-derived SiO2/water motif-resolved spectra rather than water-air motif priors.