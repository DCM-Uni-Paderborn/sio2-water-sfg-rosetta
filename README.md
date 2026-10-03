# SiO2/water SFG Rosetta-stone analysis

This repository contains the numerical data, analysis scripts, processed fit outputs, and manuscript figure assets for the study:

**The Silica/Water Interface Revisited: Elucidating Interfacial Water Structure with an SFG Rosetta Stone**

The central analysis interprets the silica/water SFG response trace of Cyran *et al.* using motif-resolved water-air SFG fingerprints as a structural dictionary. The current primary model is a three-motif I+VI+VIII hydrogen-bonded water network in the 3300--3800 cm^-1 OH-transfer window. Motif I supplies a small silica-facing weak/free-OH marker, motif VI acts as a hydrogen-bond connector, and motif VIII dominates the deeper bonded-water response. Motif VII is retained as a near-degenerate L3 admixture that cannot be uniquely excluded from the one-dimensional SFG trace.

## Repository layout

- `newest_data/heat2_fud.txt` and `newest_data/xaxis_fud.txt`: current experimental silica/water response trace and frequency axis used by the manuscript.
- `sio2_water_complex.csv` and `xaxis.txt`: earlier digitized complex silica/water SFG spectrum and frequency axis retained for provenance.
- `analysis/`: processed spectra, fit curves, weights, summaries, window scans, and diagnostic plots.
- `SFG_Structure/`: source water-air fingerprint EPS files and structure panels used to digitize/re-render the Rosetta-stone figures.
- `figures/`: main-text figure outputs.
- `si_figures/`: supplementary figure outputs.
- `manuscript/`: LaTeX manuscript and supplementary material sources.
- `*.py`: analysis scripts used in the processing workflow. The final exported manuscript figures are included directly in `figures/` and are the authoritative figure assets for this release.

## Key output files

- `analysis/paper_sfg_fingerprints_digitized.csv`: digitized motif-resolved water-air total-SFG fingerprints.
- `analysis/newest_trace_processed.csv`: current raw, interpolated, and smoothed experimental trace.
- `analysis/newest_primary_3300_3800_I_VI_VIII_curve.csv`: primary I+VI+VIII fit curve, baseline, residual, and component curves.
- `analysis/newest_primary_3300_3800_I_VI_VIII_weights.csv`: primary nonnegative weights and RMS-normalized spectral fractions.
- `analysis/newest_primary_model_comparison.csv`: primary three-motif model and four-motif I+VI+VII+VIII control.
- `analysis/newest_exact_motif_candidate_comparison.csv`: candidate-model comparison across windows.
- `analysis/newest_smoothing_window_sensitivity.csv`: smoothing-window robustness diagnostics.
- `analysis/newest_window_choice_summary.csv`: trace-based window-quality summary.

## Reproducing the analysis

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The current manuscript-ready output tables and figures are included in the repository. Legacy scripts used during development can be re-run from the repository root:

```bash
python3 analyze_sfg.py
python3 fit_high_quality_sfg.py
python3 fit_paper_fingerprints_to_sio2.py
python3 scan_paper_fingerprint_windows.py
python3 build_layered_water_model.py
```

The final manuscript figure exports are stored in `figures/`. Some final layout adjustments were applied to the exported figure files during manuscript preparation, so the archived PNG/PDF files are the authoritative figure assets for this release. The manuscript layout helper in `manuscript/` is retained as development provenance and is not required to use the release.

## Notes on the experimental reference

The silica/water spectrum was digitized from the published phase-resolved SFG data of Cyran *et al.*, *PNAS* **116**, 1520-1525 (2019), DOI: [10.1073/pnas.1819000116](https://doi.org/10.1073/pnas.1819000116). The published paper PDF is not redistributed here.

The current structural fits use a 17-point moving-average-smoothed version of `newest_data/heat2_fud.txt` in the 3300--3800 cm^-1 OH-transfer window. They fit the experimental trace directly with transferred motif fingerprints, not a phenomenological Lorentzian/resonance model.

## License

Code in this repository is distributed under the MIT License. Numerical data, processed outputs, and generated figures are made available under CC BY 4.0 unless a source file explicitly states otherwise. Please cite the associated manuscript and the original Cyran *et al.* data source when using the data.
