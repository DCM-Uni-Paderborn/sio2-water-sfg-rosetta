#!/usr/bin/env python3
"""Structure-constrained SFG decomposition of the SiO2/water complex spectrum.

The model uses the orientational motif fingerprints described in
SFG_Structure/main_pccp.tex as priors.  It is deliberately conservative:
component signs and approximate structural frequencies are fixed, while an
affine mapping from structural OH frequency to the measured x-axis is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "analysis"
X_AXIS = ROOT / "xaxis.txt"
COMPLEX_CSV = ROOT / "sio2_water_complex.csv"


@dataclass(frozen=True)
class Motif:
    label: str
    center_cm: float
    gamma_cm: float
    sign: float
    group: str
    structural_assignment: str


MOTIFS = [
    Motif(
        "VIII / low-bonded positive",
        3230.0,
        34.0,
        +1.0,
        "subsurface-positive",
        "positive bonded-OH feature from upward-biased second-layer motifs",
    ),
    Motif(
        "IV coupling low-frequency pair",
        3260.0,
        58.0,
        -1.0,
        "coupling",
        "negative low-frequency part introduced by intermolecular coupling",
    ),
    Motif(
        "V-VI liquid-oriented bonded OH",
        3335.0,
        72.0,
        -1.0,
        "liquid-oriented-bonded",
        "bonded OH groups preferentially oriented toward the liquid network",
    ),
    Motif(
        "VII second-layer bonded OH",
        3390.0,
        55.0,
        -1.0,
        "subsurface-negative",
        "negative bonded-OH response of second-layer motifs",
    ),
    Motif(
        "surface-oriented bonded OH",
        3430.0,
        46.0,
        +1.0,
        "surface-oriented-bonded",
        "positive H-bonded OH response, analogous to H atoms oriented toward silica",
    ),
    Motif(
        "IV/VIII high-frequency coupling",
        3470.0,
        72.0,
        +1.0,
        "coupling",
        "positive high-frequency component of the collective bonded-OH response",
    ),
    Motif(
        "weakly bonded upward motif",
        3560.0,
        68.0,
        +1.0,
        "weakly-bonded",
        "weakly H-bonded top-layer water with upward/surface-directed OH character",
    ),
    Motif(
        "I-III free OH",
        3660.0,
        25.0,
        +1.0,
        "free-oh",
        "sharp free/quasi-free OH response from top-layer or siloxane-facing water",
    ),
    Motif(
        "heterogeneous free-OH shoulder",
        3710.0,
        45.0,
        +1.0,
        "free-oh",
        "broader distribution of weakly bonded/free OH orientations",
    ),
]


def moving_average(values: np.ndarray, window: int = 9) -> np.ndarray:
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def load_data() -> pd.DataFrame:
    x_lines = X_AXIS.read_text(encoding="utf-8-sig").splitlines()
    x = np.array([float(row.strip()) for row in x_lines[1:] if row.strip()])
    cdf = pd.read_csv(COMPLEX_CSV)
    if len(x) != len(cdf):
        raise ValueError(f"x-axis has {len(x)} points but CSV has {len(cdf)} rows")
    real = cdf.iloc[:, 0].to_numpy(float)
    imag = cdf.iloc[:, 1].to_numpy(float)
    y = real + 1j * imag
    y_smooth = moving_average(real, 9) + 1j * moving_average(imag, 9)
    return pd.DataFrame(
        {
            "x_observed_cm-1": x,
            "real": real,
            "imag": imag,
            "real_smooth": y_smooth.real,
            "imag_smooth": y_smooth.imag,
            "abs2": np.abs(y) ** 2,
        }
    )


def complex_lorentzian(w: np.ndarray, center: float, gamma: float) -> np.ndarray:
    # Normalized so that Im(L(center)) = +1 for a positive-amplitude component.
    return gamma / ((center - w) - 1j * gamma)


def design_matrix(
    x: np.ndarray,
    a: float,
    b: float,
    gamma_scale: float,
    global_sign: float,
    motifs: Iterable[Motif] = MOTIFS,
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    z = (x - x.mean()) / max(np.ptp(x), 1.0)
    structural_w = (x - b) / a
    complex_columns = [
        np.ones_like(x, dtype=complex),
        z.astype(complex),
    ]
    complex_labels = ["complex baseline", "complex slope"]

    motif_columns: list[np.ndarray] = []
    motif_labels: list[str] = []
    for motif in motifs:
        line = complex_lorentzian(
            structural_w,
            motif.center_cm,
            motif.gamma_cm * gamma_scale,
        )
        motif_columns.append(global_sign * motif.sign * line)
        motif_labels.append(motif.label)

    # Baseline columns are complex-valued with independent real and imaginary
    # coefficients. Motif amplitudes are real, sign-constrained coefficients.
    real_cols = []
    labels = []
    for col, label in zip(complex_columns, complex_labels):
        real_cols.append(np.r_[col.real, col.imag])
        labels.append(f"{label} real")
        real_cols.append(np.r_[-col.imag, col.real])
        labels.append(f"{label} imag")
    for col, label in zip(motif_columns, motif_labels):
        real_cols.append(np.r_[col.real, col.imag])
        labels.append(label)
    return np.column_stack(real_cols), labels, motif_columns


def solve_active_set(
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    b: float,
    gamma_scale: float,
    global_sign: float,
    ridge: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, float, list[str], list[np.ndarray]]:
    X, labels, motif_columns = design_matrix(x, a, b, gamma_scale, global_sign)
    target = np.r_[y.real, y.imag]
    baseline_cols = 4
    active = np.ones(len(MOTIFS), dtype=bool)

    coefs = np.zeros(X.shape[1])
    for _ in range(len(MOTIFS) + 1):
        keep = np.r_[np.ones(baseline_cols, dtype=bool), active]
        Xk = X[:, keep]
        lhs = Xk.T @ Xk + ridge * np.eye(Xk.shape[1])
        rhs = Xk.T @ target
        ck = np.linalg.solve(lhs, rhs)
        trial = np.zeros(X.shape[1])
        trial[keep] = ck
        motif_coefs = trial[baseline_cols:]
        if np.all(motif_coefs[active] >= -1e-10):
            coefs = trial
            break
        active_indices = np.where(active)[0]
        most_negative = active_indices[np.argmin(motif_coefs[active])]
        active[most_negative] = False
        coefs = trial

    fitted_vec = X @ coefs
    fitted = fitted_vec[: len(x)] + 1j * fitted_vec[len(x) :]
    sse = float(np.sum(np.abs(y - fitted) ** 2))
    return coefs, fitted, sse, labels, motif_columns


def fit_grid(df: pd.DataFrame) -> dict:
    x_all = df["x_observed_cm-1"].to_numpy()
    y_all = df["real"].to_numpy() + 1j * df["imag"].to_numpy()
    # The measured spectrum has almost all structured intensity below 3120 cm^-1.
    # The affine mapping tests whether this is a compressed structural OH axis.
    mask = (x_all >= 2645.0) & (x_all <= 3125.0)
    x = x_all[mask]
    y = y_all[mask]

    def scan(a_values: np.ndarray, b_values: np.ndarray, g_values: np.ndarray, best: dict | None = None) -> dict:
        best_local = best
        for a in a_values:
            for b in b_values:
                structural_min = (x.min() - b) / a
                structural_max = (x.max() - b) / a
                if structural_max < 3520 or structural_min > 3380:
                    continue
                for gamma_scale in g_values:
                    for global_sign in (-1.0, +1.0):
                        coefs, fitted, sse, labels, motif_columns = solve_active_set(
                            x, y, a, b, gamma_scale, global_sign
                        )
                        if best_local is None or sse < best_local["sse"]:
                            best_local = {
                                "a": float(a),
                                "b": float(b),
                                "gamma_scale": float(gamma_scale),
                                "global_sign": float(global_sign),
                                "coefs": coefs,
                                "fitted": fitted,
                                "sse": sse,
                                "labels": labels,
                                "motif_columns": motif_columns,
                                "mask": mask,
                                "x_fit": x,
                                "y_fit": y,
                            }
        if best_local is None:
            raise RuntimeError("No grid point produced a model")
        return best_local

    best = scan(
        np.linspace(0.46, 0.78, 49),
        np.linspace(480.0, 1080.0, 81),
        np.linspace(0.55, 1.65, 12),
    )
    best = scan(
        np.linspace(best["a"] - 0.035, best["a"] + 0.035, 57),
        np.linspace(best["b"] - 60.0, best["b"] + 60.0, 81),
        np.linspace(max(0.35, best["gamma_scale"] - 0.35), best["gamma_scale"] + 0.35, 29),
        best=None,
    )

    y = best["y_fit"]
    tss = float(np.sum(np.abs(y - y.mean()) ** 2))
    best["r2_complex"] = 1.0 - best["sse"] / tss
    return best


def fit_literal_oh_window(df: pd.DataFrame) -> dict:
    x_all = df["x_observed_cm-1"].to_numpy()
    y_all = df["real"].to_numpy() + 1j * df["imag"].to_numpy()
    mask = (x_all >= 3000.0) & (x_all <= 3800.0)
    x = x_all[mask]
    y = y_all[mask]
    coefs, fitted, sse, labels, motif_columns = solve_active_set(
        x, y, a=1.0, b=0.0, gamma_scale=1.0, global_sign=1.0
    )
    tss = float(np.sum(np.abs(y - y.mean()) ** 2))
    return {
        "a": 1.0,
        "b": 0.0,
        "gamma_scale": 1.0,
        "global_sign": 1.0,
        "coefs": coefs,
        "fitted": fitted,
        "sse": sse,
        "r2_complex": 1.0 - sse / tss,
        "labels": labels,
        "motif_columns": motif_columns,
        "mask": mask,
        "x_fit": x,
        "y_fit": y,
    }


def reconstruct_on_axis(df: pd.DataFrame, fit: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x = df["x_observed_cm-1"].to_numpy()
    coefs, fitted, _, labels, motif_columns = solve_active_set(
        x,
        df["real"].to_numpy() + 1j * df["imag"].to_numpy(),
        fit["a"],
        fit["b"],
        fit["gamma_scale"],
        fit["global_sign"],
    )
    # Keep the parameters from the fit window, but evaluate the corresponding
    # model on the full axis instead of refitting the whole mostly-flat region.
    X, labels_full, motif_columns_full = design_matrix(
        x, fit["a"], fit["b"], fit["gamma_scale"], fit["global_sign"]
    )
    coefs_full = np.zeros(X.shape[1])
    coefs_full[: len(fit["coefs"])] = fit["coefs"]
    fitted_vec = X @ coefs_full
    fitted_full = fitted_vec[: len(x)] + 1j * fitted_vec[len(x) :]
    components = {}
    baseline_vec = X[:, :4] @ coefs_full[:4]
    components["baseline"] = baseline_vec[: len(x)] + 1j * baseline_vec[len(x) :]
    for motif, coef, col in zip(MOTIFS, fit["coefs"][4:], motif_columns_full):
        components[motif.label] = coef * col
    return x, fitted_full, components


def make_outputs(df: pd.DataFrame, best: dict, literal: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    x, fitted, components = reconstruct_on_axis(df, best)
    y = df["real"].to_numpy() + 1j * df["imag"].to_numpy()
    structural_axis = (x - best["b"]) / best["a"]

    curve = pd.DataFrame(
        {
            "x_observed_cm-1": x,
            "structural_frequency_cm-1": structural_axis,
            "real_measured": y.real,
            "imag_measured": y.imag,
            "real_fit": fitted.real,
            "imag_fit": fitted.imag,
            "real_residual": (y - fitted).real,
            "imag_residual": (y - fitted).imag,
        }
    )
    curve.to_csv(OUT_DIR / "sfg_structural_fit_curve.csv", index=False)

    rows = []
    total_weight = 0.0
    for motif, coef in zip(MOTIFS, best["coefs"][4:]):
        spectral_weight = float(coef * motif.gamma_cm * best["gamma_scale"])
        total_weight += abs(spectral_weight)
        rows.append(
            {
                "component": motif.label,
                "group": motif.group,
                "structural_center_cm-1": motif.center_cm,
                "observed_center_cm-1": best["a"] * motif.center_cm + best["b"],
                "observed_gamma_cm-1": best["a"] * motif.gamma_cm * best["gamma_scale"],
                "expected_imag_sign_after_global_sign": best["global_sign"] * motif.sign,
                "nonnegative_amplitude": coef,
                "signed_spectral_weight": best["global_sign"] * motif.sign * spectral_weight,
                "assignment": motif.structural_assignment,
            }
        )
    comp_df = pd.DataFrame(rows)
    if total_weight > 0:
        comp_df["absolute_weight_fraction"] = (
            comp_df["signed_spectral_weight"].abs() / total_weight
        )
    else:
        comp_df["absolute_weight_fraction"] = 0.0
    comp_df.to_csv(OUT_DIR / "sfg_structural_components.csv", index=False)

    group_df = (
        comp_df.groupby("group", as_index=False)
        .agg(
            signed_spectral_weight=("signed_spectral_weight", "sum"),
            absolute_weight=("signed_spectral_weight", lambda s: float(np.sum(np.abs(s)))),
        )
        .sort_values("absolute_weight", ascending=False)
    )
    if group_df["absolute_weight"].sum() > 0:
        group_df["absolute_weight_fraction"] = group_df["absolute_weight"] / group_df["absolute_weight"].sum()
    group_df.to_csv(OUT_DIR / "sfg_structural_group_weights.csv", index=False)

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True, constrained_layout=True)
    fit_mask = best["mask"]
    axes[0].plot(x, y.imag, color="#d59c9c", lw=0.8, alpha=0.55, label="Im measured")
    axes[0].plot(x, fitted.imag, color="#b73434", lw=1.8, label="Im structural fit")
    axes[0].set_ylabel("Im(chi)")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(x, y.real, color="#8bb2b8", lw=0.8, alpha=0.65, label="Re measured")
    axes[1].plot(x, fitted.real, color="#24656d", lw=1.8, label="Re structural fit")
    axes[1].set_ylabel("Re(chi)")
    axes[1].legend(frameon=False, loc="upper right")

    colors = [
        "#a33f3f",
        "#d08b39",
        "#6b8e3d",
        "#3d7f72",
        "#3b6ba5",
        "#7155a3",
        "#9b4f86",
        "#7a7a7a",
        "#b45f45",
    ]
    for color, motif in zip(colors, MOTIFS):
        comp = components[motif.label]
        if np.max(np.abs(comp)) < 1e-7:
            continue
        axes[2].plot(x, comp.imag, lw=1.2, alpha=0.9, color=color, label=motif.label)
    axes[2].axhline(0, color="#333333", lw=0.8, alpha=0.5)
    axes[2].set_ylabel("Im components")
    axes[2].legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    residual = y - fitted
    axes[3].plot(x, residual.imag, color="#b73434", lw=1.0, alpha=0.8, label="Im residual")
    axes[3].plot(x, residual.real, color="#24656d", lw=1.0, alpha=0.8, label="Re residual")
    axes[3].axhline(0, color="#333333", lw=0.8, alpha=0.5)
    axes[3].set_ylabel("residual")
    axes[3].set_xlabel(r"Observed x-axis (cm$^{-1}$)")
    axes[3].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.grid(True, color="#d0d0d0", alpha=0.45, lw=0.7)
        ax.axvspan(x[fit_mask][0], x[fit_mask][-1], color="#d7dee8", alpha=0.16, lw=0)
    for motif in MOTIFS:
        obs = best["a"] * motif.center_cm + best["b"]
        if x.min() <= obs <= x.max():
            axes[0].axvline(obs, color="#666666", lw=0.7, alpha=0.28)
            axes[0].text(
                obs,
                0.98,
                f"{motif.center_cm:.0f}",
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="top",
                rotation=90,
                fontsize=7,
                color="#555555",
            )
    axes[0].set_title(
        "SiO2/water SFG decomposed with orientational motif fingerprints "
        f"(R2={best['r2_complex']:.3f})"
    )
    fig.savefig(OUT_DIR / "sfg_structural_fit.png", dpi=230)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    plot_mask = (structural_axis >= 3150) & (structural_axis <= 3820)
    ax.plot(structural_axis[plot_mask], y.imag[plot_mask], color="#d59c9c", lw=0.8, alpha=0.55, label="Im measured")
    ax.plot(structural_axis[plot_mask], fitted.imag[plot_mask], color="#b73434", lw=1.8, label="Im fit")
    ax.set_xlabel(r"Mapped structural OH frequency (cm$^{-1}$)")
    ax.set_ylabel("Im(chi)")
    ax.grid(True, color="#d0d0d0", alpha=0.45, lw=0.7)
    for xpos, label in [(3230, "3230"), (3430, "3430"), (3660, "3660 free OH")]:
        ax.axvline(xpos, color="#666666", lw=0.8, alpha=0.4)
        ax.text(xpos, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top", rotation=90, fontsize=8)
    ax.legend(frameon=False, loc="best")
    ax.set_title("Same fit on the inferred structural OH-frequency axis")
    fig.savefig(OUT_DIR / "sfg_structural_frequency_axis.png", dpi=230)
    plt.close(fig)

    top_components = comp_df.sort_values("absolute_weight_fraction", ascending=False).head(7)
    report = [
        "# Structure-Constrained SiO2/Water SFG Model",
        "",
        "## Fit Summary",
        "",
        f"- Best affine mapping: observed x = {best['a']:.6f} * structural_frequency + {best['b']:.3f}",
        f"- Equivalent structural frequency: (observed x - {best['b']:.3f}) / {best['a']:.6f}",
        f"- Gamma scale: {best['gamma_scale']:.3f}",
        f"- Global sign convention relative to the water-air motif table: {best['global_sign']:+.0f}",
        f"- Complex R2 in fitted structured region: {best['r2_complex']:.4f}",
        f"- Literal-as-written OH-window model R2, using x-axis directly from 3000-3800 cm^-1: {literal['r2_complex']:.4f}",
        "",
        "The affine mapping is not claimed to be a physical vibrational scaling factor by itself. It is a diagnostic that the strongest SiO2/water features align with the motif frequencies from the paper only after the supplied `xaxis.txt` is treated as a compressed/shifted spectral coordinate.",
        "",
        "## Dominant Spectroscopic Components",
        "",
        "| component | group | observed center | structural center | weight fraction | sign |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in top_components.iterrows():
        report.append(
            f"| {row['component']} | {row['group']} | "
            f"{row['observed_center_cm-1']:.1f} | {row['structural_center_cm-1']:.0f} | "
            f"{row['absolute_weight_fraction']:.3f} | {row['expected_imag_sign_after_global_sign']:+.0f} |"
        )
    report.extend(
        [
            "",
            "## Structural Interpretation",
            "",
            "- The fitted spectrum requires a strong positive weakly bonded/free-OH-like component that maps to the 3660 cm^-1 motif. For SiO2/water this is naturally interpreted as hydrophobic-water character: OH groups pointing toward local siloxane/oxygen-bridge sites, analogous to the Cyran et al. assignment.",
            "- A sizeable positive bonded-OH component near the 3400-3500 cm^-1 structural region is also required. This is consistent with water donating H bonds toward negatively polarized silica sites, i.e. hydrogens biased toward the surface.",
            "- Negative bonded-OH and coupling components are retained only where needed by the complex line shape. They represent water-water bonded motifs whose OH vectors point back into the liquid network and second-layer/coupling cancellation.",
            "- The inferred structure is therefore not a simple water-air-like top layer. It is a mixed buried-interface layer: surface-directed/free or weakly bonded OH groups coexist with a hydrogen-bonded water network whose opposing orientations partially cancel in Im chi.",
            "",
            "## Caveats",
            "",
            "- The inversion is not unique; the component weights are spectroscopic amplitudes, not direct populations.",
            "- The supplied x-axis does not behave like the final physical OH wavenumber axis. Without the affine mapping, the main features sit below 3000 cm^-1 and cannot be assigned cleanly to H2O OH stretch motifs.",
            "- A more definitive structural model would require the original phase convention, experimental calibration notes, and ideally independent MD-derived SiO2/water motif-resolved spectra rather than water-air motif priors.",
        ]
    )
    (OUT_DIR / "sfg_structural_fit_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    df = load_data()
    best = fit_grid(df)
    literal = fit_literal_oh_window(df)
    make_outputs(df, best, literal)
    print(f"R2 affine structural model: {best['r2_complex']:.5f}")
    print(f"Mapping observed = {best['a']:.6f} * structural + {best['b']:.3f}")
    print(f"Wrote {OUT_DIR / 'sfg_structural_fit_report.md'}")
    print(f"Wrote {OUT_DIR / 'sfg_structural_fit.png'}")


if __name__ == "__main__":
    main()
