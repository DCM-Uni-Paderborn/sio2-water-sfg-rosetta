# Source data and calibration

## Experimental input

- `newest_data/xaxis_fud.txt`: 1600 increasing frequencies, in cm⁻¹.
- `newest_data/heat2_fud.txt`: 1600 paired response values, in arbitrary units; 22 nonfinite values occur below 2900 cm⁻¹.
- Provider: J. D. Cyran, in connection with the silica/water measurements described by Cyran et al., PNAS 116, 1520–1525 (2019), DOI [10.1073/pnas.1819000116](https://doi.org/10.1073/pnas.1819000116).

These are numerical files supplied by the experimental author. The files themselves do not encode the response-channel designation, pH, or acquisition metadata. The review manuscript flags those details for confirmation; they should not be inferred solely from the filenames. The files do not contain a full real/imaginary pair or replicate uncertainty.

## Molecular dictionary

Kaliannan, N. K.; Elgabarty, H.; Henao Aristizabal, A.; Kühne, T. D. **Orientation and coupling in sum-frequency generation spectra of interfacial water.** ChemRxiv, May 25, 2026, [10.26434/chemrxiv.15003862/v1](https://doi.org/10.26434/chemrxiv.15003862/v1).

`SFG_A.eps`, `SFG_B.eps`, and `SFG_L2.eps` contain the Total curves for I, II–VI, and VII–VIII, respectively. The retained `A.pdf`, `B.pdf`, and `L2.pdf` supply the orientation maps. PNG references in `analysis/structure_figures/` provide the coordinate scale used for those crops. These are source materials; the published experimental paper is not redistributed.

In the Total column, X = 3057 corresponds to 3000 cm⁻¹ and X = 4110 corresponds to 3800 cm⁻¹. The actual path endpoints are near 2900 and 3900 cm⁻¹. The explicit horizontal zero line calibrates each vertical response. `digitize_fingerprints.py` records all coordinates in `analysis/fingerprint_calibration.csv`; figure-coordinate rounding limits the basis precision.

The dictionary uses a normal directed toward vapor; the manuscript maps this onto the water-to-silica direction. Its reference regions A and B belong to the same topmost instantaneous layer. The paper's L1–L3 schematic labels therefore do not establish metric depths. VII and VIII have different polar orientations; they are not azimuthal mirror variants.
