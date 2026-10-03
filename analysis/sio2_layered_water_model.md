# Final Layered Water Model for SiO2/Water

## Basis

- Structural motifs are taken from Fig. 1 and the species-resolved A/B/L2 figures of the paper.
- The fit uses only 3300-3800 cm^-1. Below 3000 cm^-1, the SiO2/water paper does not support a water-OH structural assignment.
- In this SiO2/window geometry, water motifs that point upward in the water/air paper are interpreted as pointing toward the SiO2 side.
- Positive Im(chi) is interpreted in the Cyran convention as net H/OH orientation toward SiO2. The fitted global sign is a file/normal convention correction.
- The model weights are SFG fingerprint weights in the selected fit window, not molecule counts.

## Spectral Result

- Best clean paper-fingerprint window: 3400-3800 cm^-1.
- Minimal clean scaffold I, VIII: R2 = 0.9703, shift = -28.0 cm^-1.
- Layer-populated clean model I, II, VIII: R2 = 0.9720, shift = -25.0 cm^-1.
- Full 3300-3800 cm^-1 paper-fingerprint model remains limited: I, VI, VII, VIII, R2 = 0.6442.
- Full 3300-3800 cm^-1 high-quality complex resonance fit: complex R2 = 0.9898, Im R2 = 0.9830.
- Clean 3350-3800 cm^-1 high-quality complex resonance fit: complex R2 = 0.9952, Im R2 = 0.9933.

## Layer Model

| depth order from SiO2 | paper region | selected species | spectral fraction | H-bond model |
| ---: | --- | --- | ---: | --- |
| 1 | L1 / Region A | I | 0.375 | one weak/free-OH-like arm toward SiO2; the second OH remains connected back to the water H-bond network |
| 2 | L1 / Region B | II | 0.395 | H-bonded Region-B bridge that connects the species-I side into the interfacial water sheet |
| 3 | L2 | VIII | 0.230 | H-bonded second-layer water; VII is allowed but not required in the best clean-window model |

## Interpretation

- The best clean-window water model is I+II+VIII. It populates L1A, L1B, and L2 while fitting the clean water-fingerprint window very well.
- The full 3300-3800 cm^-1 window requires additional low-edge nuisance terms for a really high total spectral fit, so it should not be forced entirely into water fingerprints.
- Species VI appears in the full-window constrained paper fit, while species II appears in the clean 3400-3800 cm^-1 fit. A conservative structural reading is therefore an L1B bridge family with an upper II-like part and a lower VI-like connector, but the clean water fingerprint itself selects II.
- Species IV is still structurally allowed as a parallel/coupling-active Region-B motif, but the present clean-window spectrum does not require a positive IV fingerprint once I, II, and VIII are present.
- This is a water-only model; silica determines the orientation/boundary condition but is not assigned as an OH water structure.