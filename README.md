# Silica/water SFG motif analysis

Data and reproducible analysis for the Communication draft:

**The Silica/Water Interface Revisited: Elucidating Interfacial Water Structure with an SFG Rosetta Stone**

This review revision corrects the frequency calibration and nonnegative fitting of the earlier release. The source EPS curves extend approximately 2900–3900 cm⁻¹; the labeled ticks at 3000 and 3800 cm⁻¹ calibrate the axes. Calibrating the curve endpoints to those ticks incorrectly compressed the frequency scale.

With the calibrated dictionary, the I+VI+VIII candidate in 3300–3800 cm⁻¹ reduces to **I+VIII**: VI has zero coefficient and R² = 0.991811. Allowing VII gives a **VII+VIII** boundary solution with R² = 0.991971. In the 3400–3800 cm⁻¹ control, I+VIII gives R² = 0.997986. The weak-OH contribution is window dependent, and the primary fit does not establish an I–VI–VIII network. Main-text Fig. 4 illustrates a possible geometry.

## Files retained for the paper

| Location | Purpose |
|---|---|
| `newest_data/` | Original author-supplied response and frequency axis |
| `SFG_Structure/` | Three source EPS files and three orientation PDFs |
| `analysis/structure_figures/` | Three panel references needed for crop calibration |
| `analysis/*.csv` | Calibrated basis, fits, coefficients, scans, and diagnostics |
| `figures/` | Four current main-text figures, PDF and PNG |
| `si_figures/` | Four current supplementary figures, PDF and PNG |
| `manuscript/` | Main text, supplementary material, and cited bibliography |

Obsolete fits to an earlier digitized complex spectrum, exploratory resonance models, unused panels, and superseded figures were removed from the current tree. They remain recoverable from Git history, including the previous release commit `2b03344`.

## Reproduce the analysis and figures

Python 3.10 or newer is required. Install `requirements.txt`. Poppler's `pdfimages` command extracts native orientation images from the PDFs. It is included in Homebrew's `poppler` package and Debian/Ubuntu's `poppler-utils` package.

Run from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 digitize_fingerprints.py
python3 analyze_current_trace.py
python3 make_manuscript_figures.py
python3 make_supplementary_tables.py
```

The analysis examines every coefficient boundary, verifies NNLS optimality conditions, and checks that adding an allowed motif cannot worsen the residual. It also evaluates all 92 one-, two-, and three-motif dictionary subsets in each of two windows. The solver uses NumPy; SciPy is not required.

To compile the manuscripts, use a TeX installation containing REVTeX 4.2, `latexmk`, `mhchem`, and `siunitx`:

```bash
cd manuscript
latexmk -pdf main.tex
latexmk -pdf supporting_information.tex
```

Numerical SI tables are generated from the CSV records. Re-running the table generator preserves the surrounding prose. PNG files are previews; PDF figures are used by LaTeX. Temporary image caches and TeX build files are ignored.

## Interpretation and sources

R² measures descriptive agreement. Smoothed samples are correlated, shifts and motif sets are optimized, and coefficients may lie on boundaries; conventional F-test probabilities are not reported. RMS coefficient fractions are spectral weights, not molecular populations.

The experimental files were supplied by J. D. Cyran in connection with [Cyran et al., PNAS 116, 1520–1525 (2019)](https://doi.org/10.1073/pnas.1819000116). They are supplied numerical data, rather than a digitization of a plot. The dictionary comes from [Kaliannan et al., ChemRxiv (2026), version 1](https://doi.org/10.26434/chemrxiv.15003862/v1). See `SOURCE_DATA.md` for provenance and `DATA_LICENSE.md` for attribution and licensing.

## License and citation

Code: MIT. Author-produced numerical outputs and generated figures: CC BY 4.0 under the existing data license. Source materials retain their original rights and attribution. Cite the associated manuscript and both source studies.
