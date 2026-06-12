# Best Current Fits for the SiO2/Water SFG Dataset

## Best Spectral Reproduction

- Full nominal OH window, 3300-3800 cm^-1:
  high-quality complex resonance fit gives complex R2 = 0.9898 and Im R2 = 0.9830.
- Cleaner OH window, 3350-3800 cm^-1:
  high-quality complex resonance fit gives complex R2 = 0.9952 and Im R2 = 0.9933.
- These high-quality fits include additional line-shape/nuisance terms and should not be read directly as water motif populations.

## Best Paper-Fingerprint Structural Fit

- The EPS fingerprint parser was corrected so the paper fingerprints now cover the full 3000-3800 cm^-1 range.
- The full 3300-3800 cm^-1 paper-fingerprint fit is still limited: best layer-populated model I+VI+VII+VIII has R2 = 0.6442.
- The limitation is localized at the 3300-3350 cm^-1 lower edge.
- In the clean 3400-3800 cm^-1 window, the layer-populated paper-fingerprint model I+II+VIII reaches R2 = 0.9720 with p = 5.17e-155 against the linear-background model.

## Final Water-Only Layer Model

Depth order is taken from Fig. 1 of the paper, not from fixed slab thicknesses:

| depth order from SiO2 | paper region | selected species | spectral fraction in clean fit | interpretation |
| ---: | --- | --- | ---: | --- |
| 1 | L1 / Region A | I | 0.375 | SiO2-side weak/free-OH-like marker; paper-up direction maps to SiO2 |
| 2 | L1 / Region B | II | 0.395 | H-bonded upper bridge from the species-I side into the interfacial sheet |
| 3 | L2 | VIII | 0.230 | H-bonded second-layer motif |

Conservative note: species VI appears in the full-window constrained paper fit, so the robust structural statement is an L1B bridge family with II-like and possibly VI-like character. The clean water-fingerprint fit itself selects II.

## Files

- `high_quality_sfg_fit_report.md`: high-quality complex resonance fits.
- `paper_fingerprint_window_scan_report.md`: paper-fingerprint fit quality across windows.
- `sio2_layered_water_model.md`: final layer/depth model.
- `sio2_layered_water_model.png`: final figure using the paper's water structures.
