#!/usr/bin/env python3
"""Build the final SiO2/water layer model from the staged fingerprint fit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "analysis"
STRUCT_FIG_DIR = OUT_DIR / "structure_figures"

STAGE_SUMMARY = OUT_DIR / "paper_fingerprint_staged_fit_summary.csv"
WINDOW_SCAN = OUT_DIR / "paper_fingerprint_window_scan_summary.csv"
FINAL_WEIGHTS = OUT_DIR / "paper_window_3400_3800_bridge_pool_weights.csv"
MINIMAL_WEIGHTS = OUT_DIR / "paper_window_3400_3800_minimal_weights.csv"
FULL_WINDOW_WEIGHTS = OUT_DIR / "paper_fingerprint_bridge_ii_vi_pool_weights.csv"
HIGH_QUALITY_SUMMARY = OUT_DIR / "high_quality_sfg_fit_summary.csv"


def crop_light_border(img: np.ndarray) -> np.ndarray:
    rgb = img[..., :3]
    content = np.any(rgb < 0.985, axis=2)
    rows = np.where(np.any(content, axis=1))[0]
    cols = np.where(np.any(content, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    pad = 8
    r0 = max(int(rows[0]) - pad, 0)
    r1 = min(int(rows[-1]) + pad, img.shape[0] - 1)
    c0 = max(int(cols[0]) - pad, 0)
    c1 = min(int(cols[-1]) + pad, img.shape[1] - 1)
    return img[r0 : r1 + 1, c0 : c1 + 1]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in (WINDOW_SCAN, FINAL_WEIGHTS, MINIMAL_WEIGHTS, FULL_WINDOW_WEIGHTS, HIGH_QUALITY_SUMMARY) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing fit files: {names}. Run the paper scan and high-quality fit scripts first.")
    return (
        pd.read_csv(WINDOW_SCAN),
        pd.read_csv(FINAL_WEIGHTS),
        pd.read_csv(MINIMAL_WEIGHTS),
        pd.read_csv(FULL_WINDOW_WEIGHTS),
        pd.read_csv(HIGH_QUALITY_SUMMARY),
    )


def active_fraction(weights: pd.DataFrame, species: str) -> float:
    rows = weights[(weights["species"] == species) & (weights["active"])]
    if rows.empty:
        return 0.0
    return float(rows["rms_fraction"].iloc[0])


def build_layers(final_weights: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "depth_order_from_sio2": 1,
                "paper_region": "L1 / Region A",
                "selected_species": "I",
                "spectral_fraction_final": active_fraction(final_weights, "I"),
                "minimal_model_role": "part of the clean-window minimal I+VIII scaffold",
                "final_model_role": "SiO2-side species-I-like weak/free-OH marker",
                "orientation": "paper-up OH is read as pointing toward the SiO2/window side",
                "hbond_model": "one weak/free-OH-like arm toward SiO2; the second OH remains connected back to the water H-bond network",
            },
            {
                "depth_order_from_sio2": 2,
                "paper_region": "L1 / Region B",
                "selected_species": "II",
                "spectral_fraction_final": active_fraction(final_weights, "II"),
                "minimal_model_role": "absent from the strict clean-window minimal model",
                "final_model_role": "upper Region-B bridge selected in the 3400-3800 cm^-1 fingerprint fit",
                "orientation": "Region-B motif biased toward the SiO2-side/Region-A direction",
                "hbond_model": "H-bonded Region-B bridge that connects the species-I side into the interfacial water sheet",
            },
            {
                "depth_order_from_sio2": 3,
                "paper_region": "L2",
                "selected_species": "VIII",
                "spectral_fraction_final": active_fraction(final_weights, "VIII"),
                "minimal_model_role": "part of the clean-window minimal I+VIII scaffold",
                "final_model_role": "L2 motif retained in the best clean-window layer-populated model",
                "orientation": "second-layer motif with the SiO2-adapted orientation convention",
                "hbond_model": "H-bonded second-layer water; VII is allowed but not required in the best clean-window model",
            },
        ]
    )


def row_for(scan: pd.DataFrame, window_start: float, window_end: float, model: str) -> pd.Series:
    rows = scan[
        (scan["window_start_cm-1"] == window_start)
        & (scan["window_end_cm-1"] == window_end)
        & (scan["model"] == model)
    ]
    if rows.empty:
        raise ValueError(f"Missing scan row for {window_start}-{window_end} {model}")
    return rows.iloc[0]


def write_report(scan: pd.DataFrame, layers: pd.DataFrame, high_quality: pd.DataFrame) -> None:
    clean_minimal = row_for(scan, 3400.0, 3800.0, "minimal")
    clean_final = row_for(scan, 3400.0, 3800.0, "bridge_ii_vi_pool")
    full_struct = row_for(scan, 3300.0, 3800.0, "bridge_ii_vi_pool")
    full_hq = high_quality.loc[high_quality["case"] == "full_3300_3800"].iloc[0]
    clean_hq = high_quality.loc[high_quality["case"] == "clean_3350_3800"].iloc[0]

    lines = [
        "# Final Layered Water Model for SiO2/Water",
        "",
        "## Basis",
        "",
        "- Structural motifs are taken from Fig. 1 and the species-resolved A/B/L2 figures of the paper.",
        "- The fit uses only 3300-3800 cm^-1. Below 3000 cm^-1, the SiO2/water paper does not support a water-OH structural assignment.",
        "- In this SiO2/window geometry, water motifs that point upward in the water/air paper are interpreted as pointing toward the SiO2 side.",
        "- Positive Im(chi) is interpreted in the Cyran convention as net H/OH orientation toward SiO2. The fitted global sign is a file/normal convention correction.",
        "- The model weights are SFG fingerprint weights in the selected fit window, not molecule counts.",
        "",
        "## Spectral Result",
        "",
        f"- Best clean paper-fingerprint window: 3400-3800 cm^-1.",
        f"- Minimal clean scaffold {clean_minimal['active_species']}: R2 = {clean_minimal['r2']:.4f}, shift = {clean_minimal['shift_cm-1']:+.1f} cm^-1.",
        f"- Layer-populated clean model {clean_final['active_species']}: R2 = {clean_final['r2']:.4f}, shift = {clean_final['shift_cm-1']:+.1f} cm^-1.",
        f"- Full 3300-3800 cm^-1 paper-fingerprint model remains limited: {full_struct['active_species']}, R2 = {full_struct['r2']:.4f}.",
        f"- Full 3300-3800 cm^-1 high-quality complex resonance fit: complex R2 = {full_hq['complex_r2']:.4f}, Im R2 = {full_hq['imag_r2']:.4f}.",
        f"- Clean 3350-3800 cm^-1 high-quality complex resonance fit: complex R2 = {clean_hq['complex_r2']:.4f}, Im R2 = {clean_hq['imag_r2']:.4f}.",
        "",
        "## Layer Model",
        "",
        "| depth order from SiO2 | paper region | selected species | spectral fraction | H-bond model |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for _, row in layers.iterrows():
        lines.append(
            f"| {int(row['depth_order_from_sio2'])} | {row['paper_region']} | {row['selected_species']} | "
            f"{row['spectral_fraction_final']:.3f} | {row['hbond_model']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The best clean-window water model is I+II+VIII. It populates L1A, L1B, and L2 while fitting the clean water-fingerprint window very well.",
            "- The full 3300-3800 cm^-1 window requires additional low-edge nuisance terms for a really high total spectral fit, so it should not be forced entirely into water fingerprints.",
            "- Species VI appears in the full-window constrained paper fit, while species II appears in the clean 3400-3800 cm^-1 fit. A conservative structural reading is therefore an L1B bridge family with an upper II-like part and a lower VI-like connector, but the clean water fingerprint itself selects II.",
            "- Species IV is still structurally allowed as a parallel/coupling-active Region-B motif, but the present clean-window spectrum does not require a positive IV fingerprint once I, II, and VIII are present.",
            "- This is a water-only model; silica determines the orientation/boundary condition but is not assigned as an OH water structure.",
        ]
    )
    (OUT_DIR / "sio2_layered_water_model.md").write_text("\n".join(lines), encoding="utf-8")


def draw_model_figure(scan: pd.DataFrame, layers: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.25], height_ratios=[1, 1, 1])
    panels = [
        ("A.png", "L1A / species I", "SiO2-side weak/free-OH marker"),
        ("B.png", "L1B / species II", "upper H-bonded bridge selected in clean window"),
        ("L2.png", "L2 / species VIII", "deeper H-bonded motif retained"),
    ]
    for row, (fname, title, subtitle) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        img = crop_light_border(plt.imread(STRUCT_FIG_DIR / fname))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc="left")

    ax_dp = fig.add_subplot(gs[0, 1])
    dp = crop_light_border(plt.imread(STRUCT_FIG_DIR / "DP.png"))
    ax_dp.imshow(dp)
    ax_dp.set_axis_off()
    ax_dp.set_title("Depth ordering from Fig. 1: L1A -> L1B -> L2", fontsize=11, loc="left")

    ax_table = fig.add_subplot(gs[1:, 1])
    ax_table.axis("off")
    display_roles = {
        "I": "SiO2-side\nweak/free-OH-like",
        "II": "clean-window\nL1B bridge",
        "VIII": "retained L2\nH-bonded motif",
    }
    cell_text = [
        [
            int(row["depth_order_from_sio2"]),
            row["paper_region"],
            row["selected_species"],
            f"{row['spectral_fraction_final']:.3f}",
            display_roles[row["selected_species"]],
        ]
        for _, row in layers.iterrows()
    ]
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["depth", "paper region", "species", "fraction", "role"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.08, 0.19, 0.12, 0.12, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.85)

    minimal = row_for(scan, 3400.0, 3800.0, "minimal")
    final = row_for(scan, 3400.0, 3800.0, "bridge_ii_vi_pool")
    ax_table.set_title(
        f"Clean-window minimal {minimal['active_species']} R2={minimal['r2']:.3f}; final {final['active_species']} R2={final['r2']:.3f}",
        fontsize=11,
        loc="left",
    )
    fig.savefig(OUT_DIR / "sio2_layered_water_model.png", dpi=220)
    plt.close(fig)


def main() -> None:
    scan, final_weights, minimal_weights, full_window_weights, high_quality = load_inputs()
    layers = build_layers(final_weights)
    layers.to_csv(OUT_DIR / "sio2_layered_water_model.csv", index=False)
    write_report(scan, layers, high_quality)
    draw_model_figure(scan, layers)
    print(f"Wrote {OUT_DIR / 'sio2_layered_water_model.md'}")
    print(f"Wrote {OUT_DIR / 'sio2_layered_water_model.png'}")


if __name__ == "__main__":
    main()
