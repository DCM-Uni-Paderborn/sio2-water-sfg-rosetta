#!/usr/bin/env python3
"""Water-only structural fit for the SiO2/water SFG data.

This model follows the SiO2/water paper in the folder:

  Cyran et al., "Molecular hydrophobicity at a macroscopically hydrophilic surface"

Only water OH contributions are structurally interpreted.  Strong features
below the OH-stretch window are treated as nuisance/background and are not
assigned to water motifs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "analysis"
X_AXIS = ROOT / "xaxis.txt"
COMPLEX_CSV = ROOT / "sio2_water_complex.csv"
FIT_START_CM = 3300.0
FIT_END_CM = 3800.0


@dataclass(frozen=True)
class WaterComponent:
    label: str
    center_cm: float
    gamma_cm: float
    sign_cyran: float
    group: str
    interpretation: str


WATER_COMPONENTS = [
    WaterComponent(
        "diffuse-layer / strongly H-bonded water",
        3200.0,
        135.0,
        +1.0,
        "hbonded-water",
        "hydrogen-bonded interfacial water; may include diffuse-layer contribution",
    ),
    WaterComponent(
        "H-bonded water with H toward silica",
        3400.0,
        95.0,
        +1.0,
        "surface-oriented-hbonded",
        "H-bonded water with hydrogens preferentially oriented toward the negatively charged silica surface",
    ),
    WaterComponent(
        "BIL water OH toward in-plane silanol",
        3470.0,
        125.0,
        +1.0,
        "silanol-site-water",
        "binding-interfacial-layer water with one OH oscillator pointing toward an in-plane silanol site; no dangling OH on average",
    ),
    WaterComponent(
        "quasi-free OH toward siloxane bridge",
        3660.0,
        32.0,
        +1.0,
        "siloxane-quasifree-water",
        "weakly H-bonded/quasi-free OH of water pointing its hydrogen toward a siloxane bridge",
    ),
    WaterComponent(
        "heterogeneous weakly H-bonded OH",
        3685.0,
        58.0,
        +1.0,
        "siloxane-quasifree-water",
        "broader distribution of weakly H-bonded water OH groups near hydrophobic silica patches",
    ),
]


def moving_average(values: np.ndarray, window: int = 17) -> np.ndarray:
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def load_data() -> pd.DataFrame:
    x = np.array([float(v) for v in X_AXIS.read_text(encoding="utf-8-sig").splitlines()[1:] if v.strip()])
    raw = pd.read_csv(COMPLEX_CSV)
    real = raw.iloc[:, 0].to_numpy(float)
    imag = raw.iloc[:, 1].to_numpy(float)
    if len(x) != len(real):
        raise ValueError("x-axis and complex spectrum have different lengths")
    return pd.DataFrame(
        {
            "wavenumber_cm-1": x,
            "real": real,
            "imag": imag,
            "real_smooth": moving_average(real),
            "imag_smooth": moving_average(imag),
        }
    )


def lorentzian_absorptive_imag_positive(x: np.ndarray, center: float, gamma: float) -> np.ndarray:
    # Im is +1 at resonance for a positive amplitude.  Re is the corresponding
    # dispersive part, so the complex phase is still constrained.
    return gamma / ((center - x) - 1j * gamma)


def design_matrix(
    x: np.ndarray,
    phase_sign: float,
    components: list[WaterComponent],
    include_slope: bool = True,
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    z = (x - x.mean()) / max(np.ptp(x), 1.0)
    baseline = [np.ones_like(x, dtype=complex)]
    baseline_labels = ["complex constant background"]
    if include_slope:
        baseline.append(z.astype(complex))
        baseline_labels.append("complex linear background")

    columns = []
    labels = []
    for col, label in zip(baseline, baseline_labels):
        columns.append(np.r_[col.real, col.imag])
        labels.append(f"{label} real")
        columns.append(np.r_[-col.imag, col.real])
        labels.append(f"{label} imag")

    component_curves: list[np.ndarray] = []
    for comp in components:
        curve = phase_sign * comp.sign_cyran * lorentzian_absorptive_imag_positive(
            x, comp.center_cm, comp.gamma_cm
        )
        component_curves.append(curve)
        columns.append(np.r_[curve.real, curve.imag])
        labels.append(comp.label)

    return np.column_stack(columns), labels, component_curves


def solve_nonnegative_components(
    x: np.ndarray,
    y: np.ndarray,
    phase_sign: float,
    components: list[WaterComponent],
    include_slope: bool = True,
) -> dict:
    X, labels, component_curves = design_matrix(x, phase_sign, components, include_slope)
    n_background = 4 if include_slope else 2
    target = np.r_[y.real, y.imag]
    active = np.ones(len(components), dtype=bool)
    coefs = np.zeros(X.shape[1])

    for _ in range(len(components) + 1):
        keep = np.r_[np.ones(n_background, dtype=bool), active]
        Xk = X[:, keep]
        lhs = Xk.T @ Xk + 1e-9 * np.eye(Xk.shape[1])
        rhs = Xk.T @ target
        ck = np.linalg.solve(lhs, rhs)
        trial = np.zeros(X.shape[1])
        trial[keep] = ck
        comp_coefs = trial[n_background:]
        if np.all(comp_coefs[active] >= -1e-12):
            coefs = trial
            break
        active_indices = np.where(active)[0]
        active[active_indices[np.argmin(comp_coefs[active])]] = False
        coefs = trial

    fit_vec = X @ coefs
    fit = fit_vec[: len(x)] + 1j * fit_vec[len(x) :]
    residual = y - fit
    sse = float(np.sum(np.abs(residual) ** 2))
    tss = float(np.sum(np.abs(y - y.mean()) ** 2))
    return {
        "coefs": coefs,
        "fit": fit,
        "residual": residual,
        "sse": sse,
        "r2": 1.0 - sse / tss if tss > 0 else np.nan,
        "labels": labels,
        "component_curves": component_curves,
        "n_background": n_background,
        "phase_sign": phase_sign,
        "components": components,
    }


def fit_water_window(df: pd.DataFrame) -> dict:
    x_all = df["wavenumber_cm-1"].to_numpy()
    y_all = df["real_smooth"].to_numpy() + 1j * df["imag_smooth"].to_numpy()
    # Below about 3000 cm^-1 the paper provides no water-OH assignment.  In
    # this file even the 3000-3300 cm^-1 region is strongly affected by the
    # large lower-frequency structure.  Cyran Fig. 1 and Fig. 3 provide the
    # most robust SiO2/water assignment from roughly 3300-3800 cm^-1, so that
    # is the conservative water-only window used for structural inference.
    mask = (x_all >= FIT_START_CM) & (x_all <= FIT_END_CM)
    x = x_all[mask]
    y = y_all[mask]
    candidates = [
        solve_nonnegative_components(x, y, phase_sign=+1.0, components=WATER_COMPONENTS),
        solve_nonnegative_components(x, y, phase_sign=-1.0, components=WATER_COMPONENTS),
    ]
    best = min(candidates, key=lambda item: item["sse"])

    no_free = [comp for comp in WATER_COMPONENTS if "siloxane-quasifree" not in comp.group]
    no_free_candidates = [
        solve_nonnegative_components(x, y, phase_sign=+1.0, components=no_free),
        solve_nonnegative_components(x, y, phase_sign=-1.0, components=no_free),
    ]
    best_no_free = min(no_free_candidates, key=lambda item: item["sse"])
    best["mask"] = mask
    best["x_fit"] = x
    best["y_fit"] = y
    best["no_free"] = best_no_free
    return best


def evaluate_on_full_axis(df: pd.DataFrame, fit: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    x = df["wavenumber_cm-1"].to_numpy()
    y = df["real_smooth"].to_numpy() + 1j * df["imag_smooth"].to_numpy()
    X, labels, curves = design_matrix(x, fit["phase_sign"], fit["components"], include_slope=True)
    coef = np.zeros(X.shape[1])
    coef[: len(fit["coefs"])] = fit["coefs"]
    fit_vec = X @ coef
    y_fit = fit_vec[: len(x)] + 1j * fit_vec[len(x) :]
    baseline_vec = X[:, : fit["n_background"]] @ coef[: fit["n_background"]]
    baseline = baseline_vec[: len(x)] + 1j * baseline_vec[len(x) :]
    components = {"background": baseline}
    for comp, amp, curve in zip(fit["components"], coef[fit["n_background"] :], curves):
        components[comp.label] = amp * curve
    return x, y_fit, components, y


def write_outputs(df: pd.DataFrame, fit: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    x, y_fit, components, y_smooth = evaluate_on_full_axis(df, fit)
    y_raw = df["real"].to_numpy() + 1j * df["imag"].to_numpy()
    residual = y_smooth - y_fit

    full = pd.DataFrame(
        {
            "wavenumber_cm-1": x,
            "real_raw": y_raw.real,
            "imag_raw": y_raw.imag,
            "real_smooth": y_smooth.real,
            "imag_smooth": y_smooth.imag,
            "real_water_model": y_fit.real,
            "imag_water_model": y_fit.imag,
            "real_residual": residual.real,
            "imag_residual": residual.imag,
        }
    )
    full.to_csv(OUT_DIR / "sio2_water_only_fit_curve.csv", index=False)

    rows = []
    total = 0.0
    amps = fit["coefs"][fit["n_background"] :]
    for comp, amp in zip(fit["components"], amps):
        signed_weight = amp * comp.gamma_cm * fit["phase_sign"] * comp.sign_cyran
        total += abs(signed_weight)
        rows.append(
            {
                "component": comp.label,
                "group": comp.group,
                "center_cm-1": comp.center_cm,
                "gamma_cm-1": comp.gamma_cm,
                "phase_sign_relative_to_cyran": fit["phase_sign"],
                "expected_imag_sign_in_data": fit["phase_sign"] * comp.sign_cyran,
                "amplitude_nonnegative": amp,
                "signed_spectral_weight": signed_weight,
                "interpretation": comp.interpretation,
            }
        )
    comp_df = pd.DataFrame(rows)
    if total > 0:
        comp_df["absolute_weight_fraction"] = comp_df["signed_spectral_weight"].abs() / total
    else:
        comp_df["absolute_weight_fraction"] = 0.0
    comp_df.to_csv(OUT_DIR / "sio2_water_only_components.csv", index=False)

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
    group_df.to_csv(OUT_DIR / "sio2_water_only_group_weights.csv", index=False)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12.5), sharex=True, constrained_layout=True)
    mask = fit["mask"]
    axes[0].plot(x, y_raw.imag, color="#ddb0b0", lw=0.7, alpha=0.45, label="Im raw")
    axes[0].plot(x, y_smooth.imag, color="#b93d3d", lw=1.1, alpha=0.75, label="Im smoothed")
    axes[0].plot(x, y_fit.imag, color="#681b1b", lw=1.9, label="Im water model")
    axes[0].set_ylabel("Im(chi)")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(x, y_raw.real, color="#9fc2c7", lw=0.7, alpha=0.45, label="Re raw")
    axes[1].plot(x, y_smooth.real, color="#31717a", lw=1.1, alpha=0.75, label="Re smoothed")
    axes[1].plot(x, y_fit.real, color="#123d44", lw=1.9, label="Re water model")
    axes[1].set_ylabel("Re(chi)")
    axes[1].legend(frameon=False, loc="upper right")

    colors = ["#6b8e3d", "#3b6ba5", "#7b5aa6", "#c07b32", "#8f4d68"]
    for color, comp in zip(colors, fit["components"]):
        curve = components[comp.label]
        if np.max(np.abs(curve)) > 1e-8:
            axes[2].plot(x, curve.imag, lw=1.4, color=color, label=comp.label)
    axes[2].axhline(0, color="#333333", lw=0.8, alpha=0.55)
    axes[2].set_ylabel("Im components")
    axes[2].legend(frameon=False, loc="upper right", fontsize=8)

    axes[3].plot(x, residual.imag, color="#b93d3d", lw=1.1, label="Im residual")
    axes[3].plot(x, residual.real, color="#31717a", lw=1.1, label="Re residual")
    axes[3].axhline(0, color="#333333", lw=0.8, alpha=0.55)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    axes[3].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.axvspan(2645, 3000, color="#ece2d0", alpha=0.35, lw=0)
        ax.axvspan(x[mask][0], x[mask][-1], color="#d7dee8", alpha=0.22, lw=0)
        for xpos, label in [(3200, "3200"), (3400, "3400"), (3470, "3470"), (3660, "3660")]:
            ax.axvline(xpos, color="#555555", lw=0.7, alpha=0.28)
        ax.grid(True, color="#d0d0d0", alpha=0.42, lw=0.7)
    axes[0].set_title(
        "SiO2/water: water-only OH structural fit "
        f"(fit window {FIT_START_CM:.0f}-{FIT_END_CM:.0f} cm^-1, R2={fit['r2']:.3f})"
    )
    fig.savefig(OUT_DIR / "sio2_water_only_fit.png", dpi=230)
    plt.close(fig)

    zoom = (x >= FIT_START_CM) & (x <= FIT_END_CM)
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    ax.plot(x[zoom], y_smooth.imag[zoom], color="#b93d3d", lw=1.3, label="Im data, smoothed")
    ax.plot(x[zoom], y_fit.imag[zoom], color="#681b1b", lw=2.0, label="Im water model")
    for color, comp in zip(colors, fit["components"]):
        curve = components[comp.label]
        if np.max(np.abs(curve[zoom])) > 1e-8:
            ax.plot(x[zoom], curve.imag[zoom], lw=1.1, color=color, alpha=0.9, label=comp.label)
    ax.axhline(0, color="#333333", lw=0.8, alpha=0.55)
    for xpos, label in [(3200, "3200 H-bonded"), (3400, "3400 H to silica"), (3470, "3470 silanol-site water"), (3660, "3660 siloxane/free OH")]:
        ax.axvline(xpos, color="#555555", lw=0.8, alpha=0.35)
        ax.text(xpos, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top", rotation=90, fontsize=8)
    ax.grid(True, color="#d0d0d0", alpha=0.42, lw=0.7)
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Im(chi)")
    ax.set_title("Water-only structural decomposition in the SiO2/water OH window")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(OUT_DIR / "sio2_water_only_fit_zoom.png", dpi=230)
    plt.close(fig)

    no_free = fit["no_free"]
    delta = no_free["sse"] - fit["sse"]
    pct = 100.0 * delta / no_free["sse"] if no_free["sse"] > 0 else 0.0
    top = comp_df.sort_values("absolute_weight_fraction", ascending=False)
    report = [
        "# SiO2/Water Water-Only Structural Interpretation",
        "",
        "## Paper-Derived Assignment Used",
        "",
        "- The SiO2/water paper assigns the phase-resolved positive band near 3400 cm^-1 to H-bonded water whose hydrogens point toward the negatively charged silica surface.",
        "- It assigns the 3660 cm^-1 feature to weakly H-bonded/quasi-free OH groups of water in the direct binding interfacial layer, with the H atom pointing toward siloxane bridge oxygen sites.",
        "- DFT-MD separates two direct-interface water populations in the 3500-3800 cm^-1 region: OH toward siloxane bridges at 3660 cm^-1 and OH toward in-plane silanol sites as a broad band centered near 3470 cm^-1.",
        "- The paper does not provide a water-structure assignment for the strong features below 3000 cm^-1 in this file; those are treated here as non-water/background/nuisance for the purpose of water-structure inference.",
        "",
        "## Fit Summary",
        "",
        f"- Fit window: {FIT_START_CM:.0f}-{FIT_END_CM:.0f} cm^-1.",
        f"- Phase/sign relative to the Cyran convention: {fit['phase_sign']:+.0f}. A value of -1 means the supplied complex data are globally sign-flipped relative to the positive-Im convention in the paper.",
        f"- Complex R2 on smoothed data in the water window: {fit['r2']:.4f}.",
        f"- Removing the quasi-free/siloxane-water components worsens SSE by {delta:.5g} ({pct:.2f}% of the no-free-OH SSE).",
        "",
        "## Fitted Water Components",
        "",
        "| component | center cm^-1 | group | weight fraction | expected sign in data |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for _, row in top.iterrows():
        report.append(
            f"| {row['component']} | {row['center_cm-1']:.0f} | {row['group']} | "
            f"{row['absolute_weight_fraction']:.3f} | {row['expected_imag_sign_in_data']:+.0f} |"
        )
    report.extend(
        [
            "",
            "## Structural Model",
            "",
            "The data are most consistently described as a weak water-OH response sitting on top of much larger non-water/background structure below 3000 cm^-1. Within the water window, the model points to interfacial water whose OH groups are biased toward the silica surface rather than toward a vapor phase.",
            "",
            "Compared with water/air, the free-OH-like water is not a dangling OH pointing out into vacuum. It is quasi-free or very weakly H-bonded because it points toward hydrophobic siloxane bridge sites on SiO2. Dynamically and spectroscopically this resembles the free OH at water/air, but geometrically it is buried and surface-directed.",
            "",
            "The broader 3400-3470 cm^-1 contribution corresponds to H-bonded binding-layer water, including water interacting with silanol/silanolate-like surface sites. This is the part that makes SiO2/water different from a pure hydrophobic interface: hydrophilic and hydrophobic patches coexist, so the spectrum contains both H-bonded surface-directed water and hydrophobic-patch quasi-free water.",
            "",
            "## Caution",
            "",
            "The component weights are spectroscopic amplitudes, not direct populations. A quantitative population model would need the original phase convention, calibration, and ideally SiO2/water motif-resolved simulation spectra over the same experimental window.",
        ]
    )
    (OUT_DIR / "sio2_water_only_fit_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    df = load_data()
    fit = fit_water_window(df)
    write_outputs(df, fit)
    print(f"Water-window R2: {fit['r2']:.5f}")
    print(f"Phase sign relative to Cyran: {fit['phase_sign']:+.0f}")
    print(f"Wrote {OUT_DIR / 'sio2_water_only_fit_report.md'}")
    print(f"Wrote {OUT_DIR / 'sio2_water_only_fit.png'}")


if __name__ == "__main__":
    main()
