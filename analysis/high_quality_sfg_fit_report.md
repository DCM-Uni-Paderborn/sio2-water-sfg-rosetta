# High-Quality Phenomenological SFG Fits

## Purpose

These fits reproduce the measured complex SiO2/water spectrum with additional line-shape terms. They are used to separate fit quality from water-structure assignment. Components below about 3350 cm^-1 are treated as low-edge/background/nuisance unless independently assigned to water.

## Summary

| case | window cm^-1 | resonances | complex R2 | Im R2 | selected centers cm^-1 |
| --- | --- | ---: | ---: | ---: | --- |
| full_3300_3800 | 3300-3800 | 14 | 0.9898 | 0.9830 | 3320, 3595, 3350, 3275, 3265, 3675, 3440, 3150, 3255, 3245, 3235, 3380, 3290, 3770 |
| clean_3350_3800 | 3350-3800 | 10 | 0.9952 | 0.9933 | 3630, 3360, 3450, 3675, 3325, 3310, 3335, 3300, 3765, 3405 |

## Interpretation

- The full 3300-3800 cm^-1 window can be fit very well only when several low-edge nuisance resonances are allowed. This confirms that the 3300-3350 cm^-1 edge is the main obstacle for the pure paper-fingerprint model.
- The cleaner 3350-3800 cm^-1 window gives an even better fit and is the safer window for water-fingerprint comparison.
- These high-quality fits should not replace the species assignment. They show that the experimental spectrum is reproducible, while the species-resolved water model should still be read from the paper-fingerprint fits and the SiO2/water phase convention.