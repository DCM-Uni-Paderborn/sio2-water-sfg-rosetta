#!/usr/bin/env python3
"""Quick-look analysis for the silica/water complex SFG data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
X_AXIS = ROOT / "xaxis.txt"
COMPLEX_CSV = ROOT / "sio2_water_complex.csv"
OUT_DIR = ROOT / "analysis"


@dataclass(frozen=True)
class Region:
    name: str
    low: float
    high: float


REGIONS = [
    Region("low-frequency / possible CH background", 2800.0, 3000.0),
    Region("H-bonded OH, lower band", 3000.0, 3300.0),
    Region("H-bonded OH, upper band", 3300.0, 3550.0),
    Region("weakly H-bonded / free OH", 3580.0, 3725.0),
    Region("high-frequency tail", 3725.0, 3900.0),
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


def read_xaxis(path: Path) -> pd.Series:
    rows = path.read_text(encoding="utf-8-sig").splitlines()
    if not rows:
        raise ValueError(f"{path} is empty")
    values = [float(row.strip()) for row in rows[1:] if row.strip()]
    return pd.Series(values, name="wavenumber_cm-1")


def load_data() -> pd.DataFrame:
    x = read_xaxis(X_AXIS)
    complex_df = pd.read_csv(COMPLEX_CSV)
    if len(x) != len(complex_df):
        raise ValueError(
            f"x-axis has {len(x)} points, but complex CSV has {len(complex_df)} rows"
        )
    df = pd.DataFrame(
        {
            "wavenumber_cm-1": x.to_numpy(float),
            "real": complex_df.iloc[:, 0].to_numpy(float),
            "imag": complex_df.iloc[:, 1].to_numpy(float),
        }
    )
    chi = df["real"].to_numpy() + 1j * df["imag"].to_numpy()
    df["abs_chi"] = np.abs(chi)
    df["intensity_abs2"] = np.abs(chi) ** 2
    df["phase_rad"] = np.unwrap(np.angle(chi))
    df["imag_smooth"] = moving_average(df["imag"].to_numpy(), 17)
    df["intensity_smooth"] = moving_average(df["intensity_abs2"].to_numpy(), 17)
    return df


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = df["wavenumber_cm-1"].to_numpy()
    for region in REGIONS:
        sub = df[(df["wavenumber_cm-1"] >= region.low) & (df["wavenumber_cm-1"] <= region.high)]
        if sub.empty:
            continue
        max_imag = sub.loc[sub["imag_smooth"].idxmax()]
        min_imag = sub.loc[sub["imag_smooth"].idxmin()]
        max_int = sub.loc[sub["intensity_smooth"].idxmax()]
        area_imag = np.trapezoid(sub["imag_smooth"].to_numpy(), sub["wavenumber_cm-1"].to_numpy())
        area_int = np.trapezoid(sub["intensity_smooth"].to_numpy(), sub["wavenumber_cm-1"].to_numpy())
        rows.append(
            {
                "region": region.name,
                "low_cm-1": region.low,
                "high_cm-1": region.high,
                "imag_max_cm-1": max_imag["wavenumber_cm-1"],
                "imag_max": max_imag["imag_smooth"],
                "imag_min_cm-1": min_imag["wavenumber_cm-1"],
                "imag_min": min_imag["imag_smooth"],
                "intensity_peak_cm-1": max_int["wavenumber_cm-1"],
                "intensity_peak_abs2": max_int["intensity_smooth"],
                "imag_area": area_imag,
                "intensity_area": area_int,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    formatted = []
    for _, row in df.iterrows():
        formatted_row = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted_row.append(f"{value:.5g}")
            else:
                formatted_row.append(str(value))
        formatted.append(formatted_row)
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in formatted)
    return "\n".join(lines)


def zero_crossings(x: np.ndarray, y: np.ndarray) -> list[float]:
    crossings: list[float] = []
    signs = np.sign(y)
    for i in range(len(y) - 1):
        if signs[i] == 0:
            crossings.append(float(x[i]))
        elif signs[i] * signs[i + 1] < 0:
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            crossings.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return crossings


def selected_peaks(df: pd.DataFrame, min_separation_cm: float = 35.0, limit: int = 12) -> pd.DataFrame:
    x = df["wavenumber_cm-1"].to_numpy()
    y = df["intensity_smooth"].to_numpy()
    candidates = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            candidates.append((y[i], x[i], i))
    chosen = []
    for _, peak_x, idx in sorted(candidates, reverse=True):
        if all(abs(peak_x - prev_x) >= min_separation_cm for prev_x, _ in chosen):
            chosen.append((peak_x, idx))
        if len(chosen) >= limit:
            break
    rows = []
    for peak_x, idx in sorted(chosen):
        rows.append(
            {
                "peak_cm-1": peak_x,
                "real": df.iloc[idx]["real"],
                "imag": df.iloc[idx]["imag"],
                "imag_smooth": df.iloc[idx]["imag_smooth"],
                "intensity_smooth": df.iloc[idx]["intensity_smooth"],
            }
        )
    return pd.DataFrame(rows)


def write_markdown_summary(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    peaks: pd.DataFrame,
    imag_zeros: list[float],
    real_zeros: list[float],
) -> None:
    lines = [
        "# SFG Quick-Look Summary",
        "",
        f"- Points: {len(df)}",
        f"- Wavenumber range: {df['wavenumber_cm-1'].min():.2f} to {df['wavenumber_cm-1'].max():.2f} cm^-1",
        f"- Real range: {df['real'].min():.4g} to {df['real'].max():.4g}",
        f"- Imag range: {df['imag'].min():.4g} to {df['imag'].max():.4g}",
        f"- |chi|^2 range: {df['intensity_abs2'].min():.4g} to {df['intensity_abs2'].max():.4g}",
        "- Diagnostic flag: the strongest separated |chi|^2 peaks occur below 3000 cm^-1, "
        "where neat H2O OH-stretch SFG is not normally expected. Check frequency calibration, "
        "model frequency scaling, or whether this file contains a different species/window.",
        "",
        "## Region Summary",
        "",
        markdown_table(summary),
        "",
        "## Strongest Separated |chi|^2 Peaks",
        "",
        markdown_table(peaks),
        "",
        "## Zero Crossings",
        "",
        "- Imaginary part, smoothed: "
        + ", ".join(f"{v:.1f}" for v in imag_zeros[:30])
        + (" ..." if len(imag_zeros) > 30 else ""),
        "- Real part: "
        + ", ".join(f"{v:.1f}" for v in real_zeros[:30])
        + (" ..." if len(real_zeros) > 30 else ""),
        "",
        "## Interpretation Notes",
        "",
        "- In phase-resolved SFG, the sign of Im(chi) carries orientational information.",
        "- The article context assigns broad hydrogen-bonded OH response near 3200/3400 cm^-1 and weakly hydrogen-bonded/free OH near 3660-3680 cm^-1.",
        "- The peak list is a numerical quick-look, not a Lorentzian fit.",
    ]
    (OUT_DIR / "sfg_quicklook_summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_plot(df: pd.DataFrame, summary: pd.DataFrame, peaks: pd.DataFrame) -> None:
    x = df["wavenumber_cm-1"].to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)

    colors = {
        "real": "#28666e",
        "imag": "#b63f3f",
        "abs": "#4f6f52",
        "phase": "#59546c",
        "region": "#d7dee8",
    }

    for ax in axes:
        for region in REGIONS:
            ax.axvspan(region.low, region.high, color=colors["region"], alpha=0.22, lw=0)
        ax.axhline(0, color="#333333", lw=0.8, alpha=0.55)
        ax.grid(True, color="#d0d0d0", alpha=0.45, lw=0.7)

    axes[0].plot(x, df["imag"], color="#d9a6a6", lw=0.8, alpha=0.55, label="Im raw")
    axes[0].plot(x, df["imag_smooth"], color=colors["imag"], lw=1.6, label="Im smoothed")
    axes[0].set_ylabel("Im(chi)")
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(x, df["real"], color=colors["real"], lw=1.0)
    axes[1].set_ylabel("Re(chi)")

    axes[2].plot(x, df["abs_chi"], color=colors["abs"], lw=1.0, label="|chi|")
    ax2 = axes[2].twinx()
    ax2.plot(x, df["intensity_smooth"], color="#8a6f2a", lw=1.2, alpha=0.9, label="|chi|^2 smoothed")
    axes[2].set_ylabel("|chi|")
    ax2.set_ylabel("|chi|^2")
    for _, row in peaks.iterrows():
        axes[2].axvline(row["peak_cm-1"], color="#8a6f2a", lw=0.7, alpha=0.35)

    axes[3].plot(x, df["phase_rad"], color=colors["phase"], lw=1.0)
    axes[3].set_ylabel("Unwrapped phase (rad)")
    axes[3].set_xlabel(r"Wavenumber (cm$^{-1}$)")

    for _, row in summary.iterrows():
        center = (row["low_cm-1"] + row["high_cm-1"]) / 2
        axes[0].text(
            center,
            0.97,
            row["region"].replace(" / ", "\n"),
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#222222",
        )

    fig.suptitle("Silica/water SFG complex response: quick-look", fontsize=14)
    fig.savefig(OUT_DIR / "sfg_quicklook.png", dpi=220)
    plt.close(fig)


def make_oh_zoom_plot(df: pd.DataFrame) -> None:
    zoom = df[(df["wavenumber_cm-1"] >= 3000) & (df["wavenumber_cm-1"] <= 3800)].copy()
    high = df[(df["wavenumber_cm-1"] >= 3300) & (df["wavenumber_cm-1"] <= 3800)].copy()
    if zoom.empty or high.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False, constrained_layout=True)
    references = [
        (3230, "3230"),
        (3270, "3270"),
        (3430, "3430"),
        (3660, "3660 free OH"),
    ]
    bands = [
        (3000, 3600, "bonded OH"),
        (3600, 3725, "free / weakly bonded OH"),
    ]

    for ax in axes:
        for low, high_edge, _ in bands:
            ax.axvspan(low, high_edge, color="#d7dee8", alpha=0.24, lw=0)
        for xpos, label in references:
            ax.axvline(xpos, color="#6b6b6b", lw=0.8, alpha=0.35)
            ax.text(
                xpos,
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                rotation=90,
                color="#555555",
            )
        ax.axhline(0, color="#333333", lw=0.8, alpha=0.55)
        ax.grid(True, color="#d0d0d0", alpha=0.45, lw=0.7)

    axes[0].plot(zoom["wavenumber_cm-1"], zoom["imag"], color="#d9a6a6", lw=0.8, alpha=0.5)
    axes[0].plot(zoom["wavenumber_cm-1"], zoom["imag_smooth"], color="#b63f3f", lw=1.6)
    axes[0].set_title("3000-3800 cm^-1: Im(chi) in the OH-stretch window")
    axes[0].set_ylabel("Im(chi)")

    axes[1].plot(high["wavenumber_cm-1"], high["imag_smooth"], color="#b63f3f", lw=1.6, label="Im")
    axes[1].plot(high["wavenumber_cm-1"], high["real"], color="#28666e", lw=1.0, alpha=0.85, label="Re")
    axes[1].set_title("3300-3800 cm^-1 magnified")
    axes[1].set_ylabel("Response")
    axes[1].legend(frameon=False, loc="best")

    axes[2].plot(
        zoom["wavenumber_cm-1"],
        np.maximum(zoom["intensity_smooth"], 1e-8),
        color="#8a6f2a",
        lw=1.4,
    )
    axes[2].set_yscale("log")
    axes[2].set_title("Smoothed |chi|^2, log scale")
    axes[2].set_ylabel("|chi|^2")
    axes[2].set_xlabel(r"Wavenumber (cm$^{-1}$)")

    for ax in axes:
        ax.set_xlim(3000, 3800)
    axes[1].set_xlim(3300, 3800)

    fig.suptitle("OH-region diagnostic zoom", fontsize=14)
    fig.savefig(OUT_DIR / "sfg_oh_zoom.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    summary = region_summary(df)
    peaks = selected_peaks(df)
    imag_zeros = zero_crossings(df["wavenumber_cm-1"].to_numpy(), df["imag_smooth"].to_numpy())
    real_zeros = zero_crossings(df["wavenumber_cm-1"].to_numpy(), df["real"].to_numpy())

    df.to_csv(OUT_DIR / "sfg_processed.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(OUT_DIR / "sfg_region_summary.csv", index=False)
    peaks.to_csv(OUT_DIR / "sfg_selected_peaks.csv", index=False)
    write_markdown_summary(df, summary, peaks, imag_zeros, real_zeros)
    make_plot(df, summary, peaks)
    make_oh_zoom_plot(df)

    print(f"Wrote {OUT_DIR / 'sfg_quicklook.png'}")
    print(f"Wrote {OUT_DIR / 'sfg_oh_zoom.png'}")
    print(f"Wrote {OUT_DIR / 'sfg_quicklook_summary.md'}")


if __name__ == "__main__":
    main()
