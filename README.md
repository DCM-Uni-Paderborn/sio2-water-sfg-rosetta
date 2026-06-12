# SiO2/water SFG Rosetta-stone analysis

This repository contains the numerical data, analysis scripts, processed fit outputs, and manuscript figure assets for the study:

**Elucidating the buried silica/water interface with a water-air SFG Rosetta stone**

The central analysis revisits the phase-resolved silica/water SFG spectrum of Cyran *et al.* using motif-resolved water-air SFG fingerprints as a structural dictionary. The preferred clean-window structural model uses motifs I+II+VIII as a three-motif layered hydrogen-bond topology rather than as three sharply defined geometric slabs.

## Repository layout

- `sio2_water_complex.csv` and `xaxis.txt`: digitized complex silica/water SFG spectrum and frequency axis.
- `analysis/`: processed spectra, fit curves, weights, summaries, window scans, and diagnostic plots.
- `SFG_Structure/`: source water-air fingerprint EPS files and structure panels used to digitize/re-render the Rosetta-stone figures.
- `figures/`: main-text figure outputs.
- `si_figures/`: supplementary figure outputs.
- `manuscript/`: LaTeX manuscript and supplementary material sources.
- `*.py`: analysis scripts used to process the spectrum, fit complex line shapes, fit transferred water-air fingerprints, and generate structural models.

## Key output files

- `analysis/paper_sfg_fingerprints_digitized.csv`: digitized motif-resolved water-air total-SFG fingerprints.
- `analysis/sfg_processed.csv`: processed silica/water spectrum with raw and smoothed columns.
- `analysis/high_quality_sfg_fit_summary.csv`: full complex-spectrum reproduction metrics.
- `analysis/paper_fingerprint_window_scan_summary.csv`: window scan for transferred fingerprint fits.
- `analysis/paper_window_3400_3800_bridge_pool_curve.csv`: preferred I+II+VIII clean-window fit curve.
- `analysis/paper_window_3400_3800_bridge_pool_weights.csv`: preferred clean-window nonnegative weights and RMS-normalized fractions.

## Reproducing the analysis

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Re-run the main analyses from the repository root:

```bash
python3 analyze_sfg.py
python3 fit_high_quality_sfg.py
python3 fit_paper_fingerprints_to_sio2.py
python3 scan_paper_fingerprint_windows.py
python3 build_layered_water_model.py
```

Regenerate manuscript figures:

```bash
cd manuscript
python3 make_manuscript_figures.py
```

The figure script uses `pdftocairo` from Poppler when available to render the source water-structure panels at high resolution. If it is not available, it falls back to the included rendered structure images in `analysis/structure_figures/`.

## Notes on the experimental reference

The silica/water spectrum was digitized from the published phase-resolved SFG data of Cyran *et al.*, *PNAS* **116**, 1520-1525 (2019), DOI: [10.1073/pnas.1819000116](https://doi.org/10.1073/pnas.1819000116). The published paper PDF is not redistributed here.

The structural fits use a 17-point moving-average-smoothed version of the digitized experimental imaginary spectrum. They do not fit to the phenomenological Lorentzian/resonance model.

## License

Code in this repository is distributed under the MIT License. Numerical data, processed outputs, and generated figures are made available under CC BY 4.0 unless a source file explicitly states otherwise. Please cite the associated manuscript and the original Cyran *et al.* data source when using the data.
