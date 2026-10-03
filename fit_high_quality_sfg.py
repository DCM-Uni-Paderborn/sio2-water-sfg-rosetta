#!/usr/bin/env python3
"""High-quality phenomenological fits for the SiO2/water SFG spectrum.

This script is deliberately more flexible than the paper-fingerprint model.
It answers a different question: how well can the measured complex spectrum be
reproduced if we allow additional line-shape terms for background/window/tail
structure.  The resulting nuisance resonances are not counted as water motifs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "analysis"
X_AXIS = ROOT / "xaxis.txt"
COMPLEX_CSV = ROOT / "sio2_water_complex.csv"


@dataclass(frozen=True)
class FitCase:
    name: str
    fit_start: float
    fit_end: float
    candidate_start: float
    candidate_end: float
    center_step: float
    gammas: tuple[float, ...]
    poly_order: int
    max_resonances: int
    min_center_separation: float
    description: str


FIT_CASES = [
    FitCase(
        name="full_3300_3800",
        fit_start=3300.0,
        fit_end=3800.0,
        candidate_start=3150.0,
        candidate_end=3820.0,
        center_step=5.0,
        gammas=(5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0, 75.0, 110.0, 160.0),
        poly_order=2,
        max_resonances=14,
        min_center_separation=8.0,
        description="Full nominal OH window. Low-edge features are allowed as nuisance tails.",
    ),
    FitCase(
        name="clean_3350_3800",
        fit_start=3350.0,
        fit_end=3800.0,
        candidate_start=3200.0,
        candidate_end=3820.0,
        center_step=5.0,
        gammas=(5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0, 75.0, 110.0, 160.0),
        poly_order=3,
        max_resonances=10,
        min_center_separation=8.0,
        description="Cleaner water-fingerprint window, avoiding the most problematic 3300 cm^-1 edge.",
    ),
]


def moving_average(values: np.ndarray, window: int = 17) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def load_data() -> pd.DataFrame:
    x = np.array([float(v) for v in X_AXIS.read_text(encoding="utf-8-sig").splitlines()[1:] if v.strip()])
    raw = pd.read_csv(COMPLEX_CSV)
    real = raw.iloc[:, 0].to_numpy(float)
    imag = raw.iloc[:, 1].to_numpy(float)
    return pd.DataFrame(
        {
            "wavenumber_cm-1": x,
            "real_raw": real,
            "imag_raw": imag,
            "real_smooth": moving_average(real),
            "imag_smooth": moving_average(imag),
        }
    )


def polynomial_columns(x: np.ndarray, order: int) -> list[np.ndarray]:
    z = (x - x.mean()) / max(np.ptp(x), 1.0)
    return [z**i for i in range(order + 1)]


def resonance_column(x: np.ndarray, center: float, gamma: float) -> np.ndarray:
    col = 1.0 / ((center - x) - 1j * gamma)
    rms = np.sqrt(np.mean(np.abs(col) ** 2))
    return col / rms if rms > 0 else col


def fit_complex_ls(columns: list[np.ndarray], y: np.ndarray) -> dict:
    design = np.column_stack(columns)
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_fit = design @ coefs
    residual = y - y_fit
    sse = float(np.sum(np.abs(residual) ** 2))
    tss = float(np.sum(np.abs(y - y.mean()) ** 2))
    im_tss = float(np.sum((y.imag - y.imag.mean()) ** 2))
    re_tss = float(np.sum((y.real - y.real.mean()) ** 2))
    return {
        "coefs": coefs,
        "fit": y_fit,
        "residual": residual,
        "sse": sse,
        "complex_r2": 1.0 - sse / tss,
        "imag_r2": 1.0 - float(np.sum(residual.imag**2)) / im_tss,
        "real_r2": 1.0 - float(np.sum(residual.real**2)) / re_tss,
    }


def bic(sse: float, n: int, n_complex_columns: int, n_resonances: int) -> float:
    # Complex coefficients contribute two real parameters each; each selected
    # resonance also consumes center/gamma choices.
    k = 2 * n_complex_columns + 2 * n_resonances
    return n * math.log(max(sse / n, 1e-300)) + math.log(n) * k


def classify_resonance(center: float) -> str:
    if center < 3350.0:
        return "low-edge / non-water-tail nuisance"
    if center < 3550.0:
        return "H-bonded water band / collective OH"
    if center < 3630.0:
        return "broad H-bonded-to-weakly-bonded OH"
    if center < 3725.0:
        return "weak/quasi-free OH region"
    return "high-edge nuisance or weak OH tail"


def greedy_fit_case(df: pd.DataFrame, case: FitCase) -> dict:
    x_all = df["wavenumber_cm-1"].to_numpy()
    y_all = df["real_smooth"].to_numpy() + 1j * df["imag_smooth"].to_numpy()
    mask = (x_all >= case.fit_start) & (x_all <= case.fit_end)
    x = x_all[mask]
    y = y_all[mask]

    base_cols = polynomial_columns(x, case.poly_order)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for center in np.arange(case.candidate_start, case.candidate_end + 0.5 * case.center_step, case.center_step):
        for gamma in case.gammas:
            candidates.append((float(center), float(gamma), resonance_column(x, float(center), float(gamma))))

    selected_cols: list[np.ndarray] = []
    selected: list[tuple[float, float]] = []
    fit = fit_complex_ls(base_cols, y)
    current_bic = bic(fit["sse"], len(x), len(base_cols), 0)
    history = [
        {
            "step": 0,
            "center_cm-1": np.nan,
            "gamma_cm-1": np.nan,
            "bic": current_bic,
            "sse": fit["sse"],
            "complex_r2": fit["complex_r2"],
            "imag_r2": fit["imag_r2"],
            "real_r2": fit["real_r2"],
            "classification": "polynomial baseline",
        }
    ]

    for step in range(1, case.max_resonances + 1):
        best_candidate = None
        for idx, (center, gamma, col) in enumerate(candidates):
            if any(abs(center - chosen_center) < case.min_center_separation for chosen_center, _ in selected):
                continue
            trial_fit = fit_complex_ls(base_cols + selected_cols + [col], y)
            trial_bic = bic(trial_fit["sse"], len(x), len(base_cols) + len(selected_cols) + 1, len(selected) + 1)
            if best_candidate is None or trial_bic < best_candidate["bic"]:
                best_candidate = {
                    "idx": idx,
                    "center": center,
                    "gamma": gamma,
                    "col": col,
                    "fit": trial_fit,
                    "bic": trial_bic,
                }
        if best_candidate is None:
            break

        # Require BIC improvement after a minimally flexible 3-resonance fit.
        if step > 3 and best_candidate["bic"] >= current_bic:
            break

        selected.append((best_candidate["center"], best_candidate["gamma"]))
        selected_cols.append(best_candidate["col"])
        fit = best_candidate["fit"]
        current_bic = best_candidate["bic"]
        history.append(
            {
                "step": step,
                "center_cm-1": best_candidate["center"],
                "gamma_cm-1": best_candidate["gamma"],
                "bic": current_bic,
                "sse": fit["sse"],
                "complex_r2": fit["complex_r2"],
                "imag_r2": fit["imag_r2"],
                "real_r2": fit["real_r2"],
                "classification": classify_resonance(best_candidate["center"]),
            }
        )

    final_columns = base_cols + selected_cols
    final_fit = fit_complex_ls(final_columns, y)
    return {
        "case": case,
        "x": x,
        "y": y,
        "mask": mask,
        "fit": final_fit,
        "base_column_count": len(base_cols),
        "selected": selected,
        "history": pd.DataFrame(history),
        "columns": final_columns,
    }


def component_table(result: dict) -> pd.DataFrame:
    case = result["case"]
    coefs = result["fit"]["coefs"]
    rows = []
    for idx, (center, gamma) in enumerate(result["selected"]):
        col_index = result["base_column_count"] + idx
        component = result["columns"][col_index] * coefs[col_index]
        rows.append(
            {
                "case": case.name,
                "component_index": idx + 1,
                "center_cm-1": center,
                "gamma_cm-1": gamma,
                "complex_amplitude_real": coefs[col_index].real,
                "complex_amplitude_imag": coefs[col_index].imag,
                "imag_peak_to_peak": float(component.imag.max() - component.imag.min()),
                "abs_rms": float(np.sqrt(np.mean(np.abs(component) ** 2))),
                "classification": classify_resonance(center),
            }
        )
    return pd.DataFrame(rows)


def save_case_outputs(df: pd.DataFrame, result: dict) -> None:
    case = result["case"]
    x = result["x"]
    y = result["y"]
    y_fit = result["fit"]["fit"]
    residual = result["fit"]["residual"]

    curve = pd.DataFrame(
        {
            "wavenumber_cm-1": x,
            "real_smooth": y.real,
            "imag_smooth": y.imag,
            "real_fit": y_fit.real,
            "imag_fit": y_fit.imag,
            "real_residual": residual.real,
            "imag_residual": residual.imag,
        }
    )
    curve.to_csv(OUT_DIR / f"high_quality_{case.name}_curve.csv", index=False)
    component_table(result).to_csv(OUT_DIR / f"high_quality_{case.name}_components.csv", index=False)
    result["history"].to_csv(OUT_DIR / f"high_quality_{case.name}_history.csv", index=False)

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 12), sharex=True, constrained_layout=True)
    axes[0].plot(x, y.imag, color="#b93d3d", lw=1.25, label="Im data, smoothed")
    axes[0].plot(x, y_fit.imag, color="#301414", lw=2.0, label="Im fit")
    axes[0].set_ylabel("Im(chi)")
    axes[0].legend(frameon=False, loc="best")
    axes[0].set_title(
        f"{case.description} complex R2={result['fit']['complex_r2']:.4f}, "
        f"Im R2={result['fit']['imag_r2']:.4f}"
    )

    axes[1].plot(x, y.real, color="#31717a", lw=1.25, label="Re data, smoothed")
    axes[1].plot(x, y_fit.real, color="#123d44", lw=2.0, label="Re fit")
    axes[1].set_ylabel("Re(chi)")
    axes[1].legend(frameon=False, loc="best")

    coefs = result["fit"]["coefs"]
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(result["selected"]), 1)))
    for idx, (center, gamma) in enumerate(result["selected"]):
        col = result["columns"][result["base_column_count"] + idx]
        comp = coefs[result["base_column_count"] + idx] * col
        alpha = 0.88 if center >= 3350.0 else 0.42
        axes[2].plot(x, comp.imag, lw=1.0, color=colors[idx], alpha=alpha, label=f"{center:.0f}/{gamma:.0f}")
    axes[2].axhline(0, color="#333333", lw=0.8, alpha=0.5)
    axes[2].set_ylabel("Im components")
    if result["selected"]:
        axes[2].legend(frameon=False, fontsize=7, ncol=4, loc="best")

    axes[3].plot(x, residual.imag, color="#b93d3d", lw=1.1, label="Im residual")
    axes[3].plot(x, residual.real, color="#31717a", lw=1.1, label="Re residual")
    axes[3].axhline(0, color="#333333", lw=0.8, alpha=0.5)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    axes[3].legend(frameon=False, loc="best")

    for ax in axes:
        for xpos in (3300, 3350, 3400, 3470, 3660, 3700):
            ax.axvline(xpos, color="#777777", lw=0.7, alpha=0.25)
        ax.grid(True, color="#d0d0d0", alpha=0.42, lw=0.7)
    fig.savefig(OUT_DIR / f"high_quality_{case.name}_fit.png", dpi=230)
    plt.close(fig)


def save_summary(results: list[dict]) -> None:
    rows = []
    for result in results:
        case = result["case"]
        rows.append(
            {
                "case": case.name,
                "fit_window": f"{case.fit_start:.0f}-{case.fit_end:.0f}",
                "description": case.description,
                "resonance_count": len(result["selected"]),
                "complex_r2": result["fit"]["complex_r2"],
                "imag_r2": result["fit"]["imag_r2"],
                "real_r2": result["fit"]["real_r2"],
                "sse": result["fit"]["sse"],
                "selected_centers": ", ".join(f"{c:.0f}" for c, _ in result["selected"]),
                "selected_gammas": ", ".join(f"{g:.0f}" for _, g in result["selected"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "high_quality_sfg_fit_summary.csv", index=False)

    report = [
        "# High-Quality Phenomenological SFG Fits",
        "",
        "## Purpose",
        "",
        "These fits reproduce the measured complex SiO2/water spectrum with additional line-shape terms. They are used to separate fit quality from water-structure assignment. Components below about 3350 cm^-1 are treated as low-edge/background/nuisance unless independently assigned to water.",
        "",
        "## Summary",
        "",
        "| case | window cm^-1 | resonances | complex R2 | Im R2 | selected centers cm^-1 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in summary.iterrows():
        report.append(
            f"| {row['case']} | {row['fit_window']} | {int(row['resonance_count'])} | "
            f"{row['complex_r2']:.4f} | {row['imag_r2']:.4f} | {row['selected_centers']} |"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The full 3300-3800 cm^-1 window can be fit very well only when several low-edge nuisance resonances are allowed. This confirms that the 3300-3350 cm^-1 edge is the main obstacle for the pure paper-fingerprint model.",
            "- The cleaner 3350-3800 cm^-1 window gives an even better fit and is the safer window for water-fingerprint comparison.",
            "- These high-quality fits should not replace the species assignment. They show that the experimental spectrum is reproducible, while the species-resolved water model should still be read from the paper-fingerprint fits and the SiO2/water phase convention.",
        ]
    )
    (OUT_DIR / "high_quality_sfg_fit_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    results = []
    for case in FIT_CASES:
        result = greedy_fit_case(df, case)
        save_case_outputs(df, result)
        results.append(result)
        print(
            f"{case.name}: complex R2={result['fit']['complex_r2']:.5f}, "
            f"Im R2={result['fit']['imag_r2']:.5f}, resonances={len(result['selected'])}"
        )
    save_summary(results)
    print(f"Wrote {OUT_DIR / 'high_quality_sfg_fit_report.md'}")


if __name__ == "__main__":
    main()
