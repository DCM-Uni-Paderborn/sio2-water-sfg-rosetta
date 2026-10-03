from pathlib import Path
from collections import deque
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
STRUCTURES = ANALYSIS / "structure_figures"
PAPER_STRUCTURES = ROOT / "SFG_Structure"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
RENDERED_PAPER_STRUCTURES = OUT / "_paper_structure_panels"
RENDERED_PAPER_STRUCTURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 450,
})

COLORS = {
    "I": "#1f5aa6",
    "II": "#d66b00",
    "VI": "#c2410c",
    "VIII": "#2a9d55",
    "fit": "#b5121b",
    "data": "#111111",
    "baseline": "#6d6d6d",
    "edge": "#eeeeee",
    "window": "#e9f3ff",
}

PAPER_STRUCTURE_PDFS = {
    "A": PAPER_STRUCTURES / "A.pdf",
    "B": PAPER_STRUCTURES / "B.pdf",
    "L2": PAPER_STRUCTURES / "L2.pdf",
}

PAPER_NATIVE_IMAGES = {
    "A": ("A", 0),
    "B": ("B", 0),
    "L2": ("L2", 0),
}

REFERENCE_STRUCTURE_PNGS = {
    "A": STRUCTURES / "A.png",
    "B": STRUCTURES / "B.png",
    "L2": STRUCTURES / "L2.png",
}


WATER_MOTIFS = {
    "I": {
        "oxygen": (0.40, 0.37),
        "hydrogens": [(0.40, 0.74), (0.76, 0.23)],
        "r_o": 0.145,
        "r_h": 0.095,
    },
    "II": {
        "oxygen": (0.38, 0.36),
        "hydrogens": [(0.58, 0.61), (0.70, 0.53)],
        "r_o": 0.135,
        "r_h": 0.094,
    },
    "III": {
        "oxygen": (0.45, 0.36),
        "hydrogens": [(0.28, 0.70), (0.80, 0.35)],
        "r_o": 0.140,
        "r_h": 0.095,
    },
    "IV": {
        "oxygen": (0.33, 0.50),
        "hydrogens": [(0.70, 0.58), (0.72, 0.42)],
        "r_o": 0.140,
        "r_h": 0.095,
    },
    "V": {
        "oxygen": (0.56, 0.52),
        "hydrogens": [(0.18, 0.56), (0.72, 0.19)],
        "r_o": 0.145,
        "r_h": 0.096,
    },
    "VI": {
        "oxygen": (0.62, 0.60),
        "hydrogens": [(0.36, 0.37), (0.29, 0.27)],
        "r_o": 0.165,
        "r_h": 0.108,
    },
    "VII": {
        "oxygen": (0.38, 0.53),
        "hydrogens": [(0.71, 0.64), (0.39, 0.18)],
        "r_o": 0.135,
        "r_h": 0.094,
    },
    "VIII": {
        "oxygen": (0.38, 0.36),
        "hydrogens": [(0.58, 0.61), (0.70, 0.53)],
        "r_o": 0.135,
        "r_h": 0.094,
    },
}


def paper_structure_panel(key):
    """Return the native Fig.-2a image layer without PDF re-rendering."""
    source_key, image_index = PAPER_NATIVE_IMAGES[key]
    pdf = PAPER_STRUCTURE_PDFS[source_key]
    out = RENDERED_PAPER_STRUCTURES / f"{key}_native.jpg"
    if shutil.which("pdfimages") and (
        not out.exists() or out.stat().st_mtime < pdf.stat().st_mtime
    ):
        with tempfile.TemporaryDirectory(prefix="sfg_pdfimages_") as tmp:
            prefix = Path(tmp) / "img"
            subprocess.run(["pdfimages", "-j", str(pdf), str(prefix)], check=True)
            images = sorted(Path(tmp).glob("img-*.jpg"))
            if image_index >= len(images):
                raise RuntimeError(f"Could not find image {image_index} in {pdf}")
            shutil.copy2(images[image_index], out)
    if out.exists():
        return out
    return REFERENCE_STRUCTURE_PNGS[source_key]


def scale_box_to_paper_panel(key, box):
    ref = Image.open(REFERENCE_STRUCTURE_PNGS[key])
    panel = Image.open(paper_structure_panel(key))
    sx = panel.size[0] / ref.size[0]
    sy = panel.size[1] / ref.size[1]
    return (
        int(round(box[0] * sx)),
        int(round(box[1] * sy)),
        int(round(box[2] * sx)),
        int(round(box[3] * sy)),
    )


def crop_paper_image(key, box, whiteouts=None):
    image = crop_image(paper_structure_panel(key), scale_box_to_paper_panel(key, box), sharpen=False)
    if not whiteouts:
        return image

    image = image.copy()
    h, w = image.shape[:2]
    for x0, y0, x1, y1 in whiteouts:
        if max(x0, y0, x1, y1) <= 1.0:
            x0, x1 = int(round(x0 * w)), int(round(x1 * w))
            y0, y1 = int(round(y0 * h)), int(round(y1 * h))
        else:
            x0, x1 = int(round(x0)), int(round(x1))
            y0, y1 = int(round(y0)), int(round(y1))
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        image[y0:y1, x0:x1] = 255
    return image


def panel_label(ax, label):
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_sphere(ax, xy, radius, facecolor, edgecolor, zorder):
    ax.add_patch(
        Circle(
            xy,
            radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            lw=0.9,
            zorder=zorder,
        )
    )
    ax.add_patch(
        Circle(
            (xy[0] - radius * 0.35, xy[1] + radius * 0.38),
            radius * 0.32,
            facecolor="white",
            edgecolor="none",
            alpha=0.70,
            zorder=zorder + 0.2,
        )
    )


def draw_bond(ax, oxygen, hydrogen, zorder):
    ox, oy = oxygen
    hx, hy = hydrogen
    ax.plot(
        [ox, hx],
        [oy, hy],
        color="#c9ced1",
        lw=7.5,
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.plot(
        [ox, ox + 0.46 * (hx - ox)],
        [oy, oy + 0.46 * (hy - oy)],
        color="#d91f16",
        lw=5.0,
        solid_capstyle="round",
        zorder=zorder + 0.1,
    )


def draw_redrawn_water_motif(ax, species):
    motif = WATER_MOTIFS[species]
    oxygen = motif["oxygen"]
    hydrogens = motif["hydrogens"]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    for hydrogen in hydrogens:
        draw_bond(ax, oxygen, hydrogen, zorder=3)
    draw_sphere(ax, oxygen, motif["r_o"], "#e11d16", "#9f1110", zorder=5)
    for hydrogen in hydrogens:
        draw_sphere(ax, hydrogen, motif["r_h"], "#f7f7f4", "#c9c9c4", zorder=6)


def motif_to_extent(point, extent):
    xmin, xmax, ymin, ymax = extent
    x, y = point
    return (
        xmin + x * (xmax - xmin),
        ymin + y * (ymax - ymin),
    )


def water_motif_atoms_in_extent(species, extent, mirror_horizontal=False):
    motif = WATER_MOTIFS[species]
    scale = min(extent[1] - extent[0], extent[3] - extent[2])
    oxygen = motif_to_extent(motif["oxygen"], extent)
    hydrogens = [motif_to_extent(h, extent) for h in motif["hydrogens"]]
    if mirror_horizontal:
        # Reflect the covalent geometry about the vertical axis through oxygen.
        hydrogens = [(2 * oxygen[0] - hx, hy) for hx, hy in hydrogens]
    return {
        "oxygen": oxygen,
        "hydrogens": hydrogens,
        "r_o": motif["r_o"] * scale,
        "r_h": motif["r_h"] * scale,
    }


def draw_redrawn_water_motif_in_extent(
    ax, species, extent, zorder=10, mirror_horizontal=False
):
    atoms = water_motif_atoms_in_extent(species, extent, mirror_horizontal)
    oxygen = atoms["oxygen"]
    hydrogens = atoms["hydrogens"]
    for hydrogen in hydrogens:
        draw_bond(ax, oxygen, hydrogen, zorder=zorder)
    draw_sphere(ax, oxygen, atoms["r_o"], "#e11d16", "#9f1110", zorder=zorder + 2)
    for hydrogen in hydrogens:
        draw_sphere(ax, hydrogen, atoms["r_h"], "#f7f7f4", "#c9c9c4", zorder=zorder + 3)
    return atoms


def shortened_segment(start, end, start_radius, end_radius, gap=0.018):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    return (
        start + direction * (start_radius + gap),
        end - direction * (end_radius + gap),
    )


def draw_reference_structure_field(ax, species, show_layer_axis_labels=False):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    for y in (0.74, 0.46, 0.18):
        ax.plot([0.08, 0.94], [y, y], color="black", lw=1.05, ls=(0, (2.0, 2.6)), zorder=1)
    ax.annotate(
        "",
        xy=(0.08, 0.91),
        xytext=(0.08, 0.14),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.4, shrinkA=0, shrinkB=0),
        zorder=2,
    )
    if show_layer_axis_labels:
        for label, y in (("L1", 0.74), ("L2", 0.46), ("L3", 0.18)):
            ax.text(
                0.035,
                y,
                label,
                fontsize=8.2,
                fontweight="bold",
                ha="right",
                va="center",
                clip_on=False,
            )

    if species == "I":
        molecules = [("I", (0.42, 0.68, 0.64, 0.90), "")]
        labels = [
            (r"I$_{\mathrm{OH1}}$", 0.54, 0.87, "center", "bottom"),
            (r"I$_{\mathrm{OH2}}$", 0.66, 0.69, "left", "center"),
        ]
    elif species in ("II", "VI"):
        molecules = [
            ("II", (0.10, 0.28, 0.395, 0.575), "II"),
            ("III", (0.27, 0.45, 0.388, 0.588), "III"),
            ("IV", (0.44, 0.62, 0.360, 0.560), "IV"),
            ("V", (0.61, 0.79, 0.356, 0.556), "V"),
            ("VI", (0.76, 0.94, 0.340, 0.540), "VI"),
        ]
        labels = [
            ("II", 0.12, 0.59, "left", "bottom"),
            ("III", 0.28, 0.59, "left", "bottom"),
            ("IV", 0.51, 0.34, "center", "top"),
            ("V", 0.68, 0.34, "center", "top"),
            ("VI", 0.84, 0.34, "center", "top"),
        ]
    else:
        molecules = [
            ("VII", (0.15, 0.36, 0.07, 0.30), "VII"),
            ("VIII", (0.64, 0.85, 0.10, 0.32), "VIII"),
        ]
        labels = [
            (r"VII$_{\mathrm{OH1}}$", 0.31, 0.26, "left", "center"),
            (r"VII$_{\mathrm{OH2}}$", 0.15, 0.065, "left", "center"),
            ("VIII", 0.80, 0.13, "center", "top"),
        ]

    for motif, extent, _ in molecules:
        draw_redrawn_water_motif_in_extent(ax, motif, extent, zorder=3)
    for text, x, y, ha, va in labels:
        ax.text(x, y, text, fontsize=9.0, fontweight="bold", color="black", ha=ha, va=va, zorder=9)


def savefig(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def make_rosetta_pair_figure():
    fp = pd.read_csv(ANALYSIS / "paper_sfg_fingerprints_digitized.csv")
    pairs = [
        {
            "species": "I",
            "panel": "A",
            "box": (820, 25, 1065, 235),
            "cleanup": [(100, 146, 130, 171)],
            "color": COLORS["I"],
        },
        {
            "species": "II",
            "panel": "B",
            "box": (655, 225, 747, 350),
            "cleanup": [(48, 108, 73, 132), (82, 108, 103, 132)],
            "color": COLORS["II"],
        },
        {
            "species": "III",
            "panel": "B",
            "box": (725, 165, 880, 330),
            "cleanup": [(0, 118, 32, 143), (112, 124, 154, 146)],
            "color": "#b45309",
        },
        {
            "species": "IV",
            "panel": "B",
            "box": (875, 245, 1015, 350),
            "cleanup": [(85, 70, 138, 94)],
            "color": "#92400e",
        },
        {
            "species": "V",
            "panel": "B",
            "box": (1010, 245, 1135, 395),
            "cleanup": [(0, 90, 38, 118), (92, 92, 124, 116)],
            "color": "#a16207",
        },
        {
            "species": "VI",
            "panel": "B",
            "box": (1145, 220, 1236, 390),
            "cleanup": [(0, 118, 28, 150), (58, 118, 90, 150)],
            "color": "#c2410c",
        },
        {
            "species": "VII",
            "panel": "L2",
            "box": (685, 385, 900, 612),
            "cleanup": [(0, 120, 40, 148), (150, 118, 214, 148)],
            "color": "#15803d",
        },
        {
            "species": "VIII",
            "panel": "L2",
            "box": (1040, 350, 1225, 500),
            "cleanup": [(0, 158, 42, 176), (101, 158, 138, 176)],
            "color": COLORS["VIII"],
        },
    ]

    fig = plt.figure(figsize=(7.2, 6.7))
    gs = fig.add_gridspec(
        4,
        4,
        width_ratios=[0.74, 1.30, 0.74, 1.30],
        hspace=0.40,
        wspace=0.31,
    )

    for i, pair in enumerate(pairs):
        row = i // 2
        col = 2 * (i % 2)

        ax_struct = fig.add_subplot(gs[row, col])
        draw_redrawn_water_motif(ax_struct, pair["species"])
        ax_struct.set_axis_off()
        ax_struct.set_title(
            f"motif {pair['species']}",
            color=pair["color"],
            fontsize=9.0,
            fontweight="bold",
            pad=2,
        )
        ax_fp = fig.add_subplot(gs[row, col + 1])
        d = fp[fp["species"] == pair["species"]].sort_values("wavenumber_cm-1")
        x = d["wavenumber_cm-1"].to_numpy()
        y = d["fingerprint"].to_numpy()
        y = y / np.nanmax(np.abs(y))
        ax_fp.axvspan(3300, 3800, color=COLORS["window"], zorder=0)
        ax_fp.axhline(0, color="#c9c9c9", lw=0.75)
        ax_fp.plot(x, y, color=pair["color"], lw=1.85)
        ax_fp.set_xlim(3000, 3800)
        ax_fp.set_ylim(-1.08, 1.08)
        ax_fp.set_title("total SFG", fontsize=8.0, pad=3)
        ax_fp.set_xticks([3000, 3400, 3800])
        ax_fp.set_yticks([-1, 0, 1])
        ax_fp.tick_params(labelsize=6.7)
        if col == 0:
            ax_fp.set_ylabel("norm. Im", labelpad=1)
        if row == 3:
            ax_fp.set_xlabel(r"cm$^{-1}$", labelpad=1)
        else:
            ax_fp.set_xticklabels([])
        ax_fp.text(
            0.96,
            0.08,
            f"motif {pair['species']}",
            transform=ax_fp.transAxes,
            color=pair["color"],
            ha="right",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
        )

    savefig(fig, "fig1_motif_dictionary")


def sharpen_rgb(image):
    """Recover crispness after rendering Photoshop-PDF structure panels."""
    image = image.convert("RGB")
    image = ImageEnhance.Sharpness(image).enhance(1.65)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    return image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=95, threshold=3))


def crop_image(path, box, sharpen=False):
    image = Image.open(path).convert("RGB").crop(box)
    if sharpen:
        image = sharpen_rgb(image)
    return np.asarray(image)


def make_layer_rosetta_figure():
    fp = pd.read_csv(ANALYSIS / "paper_sfg_fingerprints_digitized.csv")

    columns = [
        {
            "species": "I",
            "title": "L1 / motif I",
            "subtitle": r"weak/free OH toward SiO$_2$",
            "panel": "A",
            "color": COLORS["I"],
            "ellipses": [(0.73, 0.55, 0.24, 0.31, -22), (0.32, 0.56, 0.25, 0.32, -28)],
            "orient_labels": [
                ("I", "OH1", 0.68, 0.58),
                ("I", "OH2", 0.27, 0.59),
            ],
        },
        {
            "species": "VI",
            "title": "L2 / motif VI",
            "subtitle": "H-bonded bridge",
            "panel": "B",
            "color": COLORS["VI"],
            "ellipses": [(0.34, 0.43, 0.27, 0.27, -12)],
        },
        {
            "species": "VIII",
            "title": "L3 / motif VIII",
            "subtitle": "second-layer H-bonded water",
            "panel": "L2",
            "color": COLORS["VIII"],
            "ellipses": [(0.55, 0.70, 0.32, 0.33, -28)],
        },
    ]

    # Crops are taken from the paper-derived panels:
    # left crop = cos(tau)-cos(xi) population map, right crop = representative structures.
    orient_boxes = {
        "I": (55, 112, 590, 550),
        "VI": (55, 112, 590, 575),
        "VIII": (55, 122, 590, 585),
    }
    fig = plt.figure(figsize=(7.2, 6.3))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.95, 1.05, 0.78],
        hspace=0.22,
        wspace=0.20,
    )

    for j, col in enumerate(columns):
        species = col["species"]
        color = col["color"]

        ax_struct = fig.add_subplot(gs[0, j])
        draw_reference_structure_field(ax_struct, species, show_layer_axis_labels=True)
        ax_struct.set_axis_off()
        ax_struct.set_title(f"{col['title']}\n{col['subtitle']}", fontsize=10.0, pad=4)
        if j == 0:
            panel_label(ax_struct, "a")

        ax_orient = fig.add_subplot(gs[1, j])
        ax_orient.imshow(crop_paper_image(col["panel"], orient_boxes[species]), interpolation="none")
        ax_orient.set_axis_off()
        if j == 0:
            ax_orient.text(
                -0.075,
                0.50,
                r"$\cos(\xi)$",
                transform=ax_orient.transAxes,
                rotation=90,
                fontsize=8.0,
                ha="center",
                va="center",
                clip_on=False,
            )
        for cx, cy, width, height, angle in col["ellipses"]:
            ax_orient.add_patch(
                Ellipse(
                    (cx, cy),
                    width,
                    height,
                    angle=angle,
                    transform=ax_orient.transAxes,
                    facecolor="none",
                    edgecolor="#c21f30",
                    lw=2.2,
                )
            )
        for main_text, sub_text, tx, ty in col.get("orient_labels", []):
            ax_orient.text(
                tx,
                ty,
                main_text,
                transform=ax_orient.transAxes,
                fontsize=13.4,
                fontweight="black",
                color="black",
                ha="center",
                va="center",
                zorder=8,
            )
            ax_orient.text(
                tx + 0.020,
                ty - 0.036,
                sub_text,
                transform=ax_orient.transAxes,
                fontsize=8.2,
                fontweight="black",
                color="black",
                ha="left",
                va="center",
                zorder=8,
            )
        if j == 0:
            panel_label(ax_orient, "b")

        ax_fp = fig.add_subplot(gs[2, j])
        d = fp[fp["species"] == species].sort_values("wavenumber_cm-1")
        x = d["wavenumber_cm-1"].to_numpy()
        y = d["fingerprint"].to_numpy()
        y = y / np.nanmax(np.abs(y))
        ax_fp.axvspan(3300, 3800, color=COLORS["window"], zorder=0)
        ax_fp.axhline(0, color="#c9c9c9", lw=0.75)
        ax_fp.plot(x, y, color=color, lw=2.0)
        ax_fp.set_xlim(3000, 3800)
        ax_fp.set_ylim(-1.08, 1.08)
        ax_fp.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_fp.set_yticklabels(["-1.0", "-0.5", "0", "0.5", "1.0"])
        ax_fp.set_title(f"total SFG fingerprint {species}", color=color, fontsize=9.0, pad=5)
        ax_fp.set_xlabel(r"wavenumber / cm$^{-1}$")
        if j == 0:
            ax_fp.set_ylabel("norm. Im")
            panel_label(ax_fp, "c")
        ax_fp.tick_params(labelsize=7.5)

    savefig(fig, "fig2_motif_context")


def draw_optical_wave_packets(ax):
    """Sketch the silica-side pulses with illustrative wavelengths and envelopes."""
    beam_focus = (4.10, 4.68)
    beam_specs = [
        ("IR", (0.82, 5.57), beam_focus, "#cf3b2e", 0.40, 0.42),
        ("vis", (2.20, 5.57), beam_focus, "#1f8f4e", 0.52, 0.13),
        ("SFG", beam_focus, (6.78, 5.54), "#7048a8", 0.54, 0.10),
    ]
    for label, start, end, color, packet_center, wavelength in beam_specs:
        start = np.asarray(start)
        end = np.asarray(end)
        axis = end - start
        length = np.linalg.norm(axis)
        direction = axis / length
        normal = np.array([-direction[1], direction[0]])
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="-|>", mutation_scale=10,
                            lw=0.85, color=color, alpha=0.75,
                            shrinkA=0, shrinkB=0),
            zorder=6.5,
        )
        distance = np.linspace(0, length, 1500)
        center = packet_center * length
        sigma = 0.28
        envelope = 0.14 * np.exp(-0.5 * ((distance - center) / sigma) ** 2)
        amplitude = envelope * np.sin(2 * np.pi * (distance - center) / wavelength)
        packet = start[:, None] + direction[:, None] * distance + normal[:, None] * amplitude
        visible = np.abs(distance - center) <= 3.1 * sigma
        ax.plot(packet[0, visible], packet[1, visible], color="white",
                lw=2.8, alpha=0.88, zorder=7.1)
        ax.plot(packet[0, visible], packet[1, visible], color=color,
                lw=1.15, zorder=7.2)
        label_position = start + direction * center + normal * 0.26
        ax.text(
            *label_position, label, color=color, fontsize=7.4,
            fontweight="bold", ha="center", va="center", zorder=8,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.82),
        )


def make_layer_model_figure():
    layers = [
        {
            "layer": "L1",
            "motif": "I",
            "role": "SiO2-facing weak/free OH",

            "panel": "A",
            "box": (820, 25, 1065, 235),
            "cleanup": [(100, 146, 130, 171)],
            "color": COLORS["I"],
            "extent": (4.10, 5.30, 3.58, 4.60),
        },
        {
            "layer": "L1_left",
            "motif": "I",
            "role": "H-bonded L1 motif-I water",

            "panel": "A",
            "box": (820, 25, 1065, 235),
            "cleanup": [(100, 146, 130, 171)],
            "color": COLORS["I"],
            "extent": (1.96, 3.16, 3.54, 4.56),
        },
        {
            "layer": "L2",
            "motif": "II",
            "role": "H-bonded bridge",

            "panel": "B",
            "box": (655, 225, 747, 350),
            "cleanup": [(48, 108, 73, 132), (82, 108, 103, 132)],
            "color": COLORS["II"],
            "extent": (2.88, 4.08, 2.34, 3.36),
        },
        {
            "layer": "L3",
            "motif": "VIII",
            "role": "second-layer H-bonded water",

            "panel": "L2",
            "box": (1040, 350, 1225, 500),
            "cleanup": [(0, 158, 42, 176), (101, 158, 138, 176)],
            "color": COLORS["VIII"],
            "extent": (2.05, 3.25, 1.14, 2.16),
        },
    ]

    layers[0]["extent"] = (4.83, 6.03, 3.58, 4.60)
    layers[0]["mirror_horizontal"] = True
    layers[2].update(
        motif="VI", role="H-bond connector",
        color=COLORS["VI"], extent=(3.356, 4.556, 2.213, 3.233),
    )

    layers[3]["extent"] = (2.02, 3.22, 1.12, 2.14)

    # Center L2 in its band and balance the outer oxygen positions about it.
    l2_oxygen = water_motif_atoms_in_extent("VI", layers[2]["extent"])["oxygen"]
    right_l1_oxygen = water_motif_atoms_in_extent("I", layers[0]["extent"])["oxygen"]
    left_oxygen_x = 2 * l2_oxygen[0] - right_l1_oxygen[0]
    for layer, oxygen_y in ((layers[1], right_l1_oxygen[1]), (layers[3], 1.60)):
        oxygen = water_motif_atoms_in_extent(layer["motif"], layer["extent"])["oxygen"]
        shift_x = left_oxygen_x - oxygen[0]
        shift_y = oxygen_y - oxygen[1]
        x0, x1, y0, y1 = layer["extent"]
        layer["extent"] = (x0 + shift_x, x1 + shift_x, y0 + shift_y, y1 + shift_y)

    # Keep the right L3 oxygen directly below the right L1 oxygen.
    x0, x1, y0, y1 = layers[3]["extent"]
    right_shift_x = right_l1_oxygen[0] - left_oxygen_x
    layers.append({
        **layers[3],
        "layer": "L3_right",
        "extent": (x0 + right_shift_x, x1 + right_shift_x, y0, y1),
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.85))
    ax.set_xlim(0, 8.2)
    ax.set_ylim(0.74, 6.08)
    ax.set_axis_off()

    band_specs = [
        (3.42, 4.62, COLORS["I"], "L1 / motif I", r"weak/free OH toward SiO$_2$", "3"),
        (2.23, 3.42, COLORS["II"], "L2 / motif II", "H-bond bridge", "6"),
        (0.98, 2.23, COLORS["VIII"], "L3 / motif VIII", "H-bonded water", "9"),
    ]
    band_specs[1] = (2.23, 3.42, COLORS["VI"], "L2 / motif VI", "H-bond connector", "6")
    for y0, y1, color, label, role, depth in band_specs:
        ax.add_patch(Rectangle((0, y0), 8.2, y1 - y0, facecolor=color, alpha=0.070, edgecolor="none", zorder=0))
        ax.plot([0.35, 7.95], [y0, y0], color=color, lw=0.8, alpha=0.45, ls=(0, (3, 3)), zorder=1)
        ax.text(
            0.28,
            y0,
            "",
            color="#505050",
            ha="right",
            va="center",
            fontsize=7.3,
            zorder=5,
        )
        ax.text(7.86, (y0 + y1) / 2, label, color=color, ha="right", va="center", fontsize=9.4, fontweight="bold")
        ax.text(7.86, (y0 + y1) / 2 - 0.22, role, color="#343434", ha="right", va="center", fontsize=7.3)

    silica_interface = [(0, 4.62), (0.8, 4.70), (1.65, 4.60), (2.55, 4.69), (3.35, 4.61),
                        (4.25, 4.71), (5.20, 4.62), (6.05, 4.70), (7.10, 4.61), (8.2, 4.68)]
    silica_poly = [(0, 5.75), (8.2, 5.75), *reversed(silica_interface)]
    ax.add_patch(Polygon(silica_poly, closed=True, facecolor="#dfe4e8", edgecolor="#8b949e", lw=1.0, zorder=2))
    ax.plot([p[0] for p in silica_interface], [p[1] for p in silica_interface], color="#6f7b86", lw=1.1, zorder=4)

    si_positions = [(0.55, 5.25), (1.45, 5.05), (2.42, 5.28), (3.35, 5.00),
                    (4.30, 5.23), (5.22, 5.03), (6.18, 5.26), (7.15, 5.05)]
    o_positions = [(0.25, 4.66), (0.95, 4.72), (1.88, 4.64), (2.80, 4.71),
                   (3.75, 4.65), (4.68, 4.72), (5.62, 4.64), (6.55, 4.71), (7.55, 4.66)]
    for sx, sy in si_positions:
        for ox, oy in o_positions:
            if (sx - ox) ** 2 + (sy - oy) ** 2 < 0.72:
                ax.plot([sx, ox], [sy, oy], color="#9ca5ad", lw=1.0, zorder=3)
    for ox, oy in o_positions:
        ax.add_patch(Circle((ox, oy), 0.115, facecolor="#d84b3a", edgecolor="#9d2e25", lw=0.45, zorder=5))
        ax.add_patch(Circle((ox - 0.035, oy + 0.035), 0.035, facecolor="white", edgecolor="none", alpha=0.60, zorder=6))
    for sx, sy in si_positions:
        ax.add_patch(Circle((sx, sy), 0.140, facecolor="#aeb7bf", edgecolor="#6e7781", lw=0.45, zorder=5))
        ax.add_patch(Circle((sx - 0.040, sy + 0.045), 0.040, facecolor="white", edgecolor="none", alpha=0.55, zorder=6))
    ax.text(
        7.72,
        5.35,
        r"SiO$_2$ surface",
        fontsize=7.2,
        color="#5d6872",
        ha="right",
        va="center",
        zorder=7,
    )

    draw_optical_wave_packets(ax)

    hbonds = [
        {
            "donor_layer": "L1_left",
            "donor_atom": "H2",
            "acceptor_layer": "L2",
            "acceptor_atom": "O",
            "label": "H(L1)...O(L2)",
            "text": (2.22, 3.24),
            "ha": "left",
            "show_label": False,
        },
        {
            "donor_layer": "L2",
            "donor_atom": "H1",
            "acceptor_layer": "L1",
            "acceptor_atom": "O",
            "label": "H(L2)...O(L1)",
            "text": (4.30, 3.58),
            "ha": "left",
        },
        {
            "donor_layer": "L3",
            "donor_atom": "H2",
            "acceptor_layer": "L2",
            "acceptor_atom": "O",
            "label": "H(L3)...O(L2)",
            "text": (3.62, 2.42),
            "ha": "left",
        },
    ]
    hbonds = [bond for bond in hbonds if bond["donor_layer"] != "L2"]
    for bond in hbonds:
        bond["show_label"] = False
        if bond["donor_layer"] == "L3":
            bond["donor_atom"] = "H1"
    hbonds.append({
        "donor_layer": "L1",
        "donor_atom": "H2",
        "acceptor_layer": "L2",
        "acceptor_atom": "O",
        "show_label": False,
    })
    hbonds.append({
        "donor_layer": "L2",
        "donor_atom": "H2",
        "acceptor_layer": "L3_right",
        "acceptor_atom": "O",
        "show_label": False,
    })

    motif_atoms = {}
    for layer in layers:
        atoms = water_motif_atoms_in_extent(
            layer["motif"], layer["extent"],
            layer.get("mirror_horizontal", False) or (layer["layer"] == "L2"),
        )
        motif_atoms[layer["layer"]] = {
            "O": atoms["oxygen"],
            "H1": atoms["hydrogens"][0],
            "H2": atoms["hydrogens"][1],
            "r_o": atoms["r_o"],
            "r_h": atoms["r_h"],
        }

    for hbond in hbonds:
        donor_atoms = motif_atoms[hbond["donor_layer"]]
        acceptor_atoms = motif_atoms[hbond["acceptor_layer"]]
        start, end = shortened_segment(
            donor_atoms[hbond["donor_atom"]],
            acceptor_atoms[hbond["acceptor_atom"]],
            donor_atoms["r_h"],
            acceptor_atoms["r_o"],
            gap=0.020,
        )
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        ax.plot(
            xs,
            ys,
            color="white",
            lw=3.6,
            alpha=0.82,
            solid_capstyle="butt",
            dash_capstyle="butt",
            zorder=9.0,
        )
        ax.plot(
            xs,
            ys,
            color="#3f82b4",
            lw=1.65,
            ls=(0, (2.1, 2.4)),
            solid_capstyle="butt",
            dash_capstyle="butt",
            zorder=9.2,
        )

    for layer in layers:
        draw_redrawn_water_motif_in_extent(
            ax, layer["motif"], layer["extent"], zorder=10,
            mirror_horizontal=layer.get("mirror_horizontal", False) or (layer["layer"] == "L2"),
        )

    for hbond in hbonds:
        if not hbond.get("show_label", True):
            continue
        ax.text(
            hbond["text"][0],
            hbond["text"][1],
            hbond["label"],
            color="#42759f",
            fontsize=6.5,
            ha=hbond["ha"],
            va="center",
            zorder=13,
            bbox=dict(boxstyle="round,pad=0.11", facecolor="white", edgecolor="none", alpha=0.82),
        )
    savefig(fig, "fig4_connectivity")


def current_curves(window, model):
    data = pd.read_csv(ANALYSIS / "newest_candidate_curves.csv")
    return data[(data.window == window) & (data.model == model)]


def make_current_fit():
    trace = pd.read_csv(ANALYSIS / "newest_trace_processed.csv")
    primary = current_curves("3300-3800", "I+VI+VIII")
    control = current_curves("3300-3800", "I+VI+VII+VIII")
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2, .85], "hspace": .17})
    for index, ax in enumerate(axes):
        panel_label(ax, "abc"[index])
        ax.axhline(0, color="#c7c7c7", lw=.6)
        ax.set_xlim(3300, 3800)
    x = primary["wavenumber_cm-1"]
    d = trace[trace["wavenumber_cm-1"].between(3300,3800)]
    axes[0].axvspan(3300,3400,color=COLORS["edge"])
    axes[0].plot(d["wavenumber_cm-1"],d.raw_trace,color="#aaaaaa",lw=.65,label="raw trace")
    axes[0].plot(x,primary.smooth_trace,color="black",lw=1.1,label="17-point average")
    axes[0].legend(fontsize=8,loc="upper right")
    axes[0].set_ylabel("response / a.u.")
    axes[1].plot(x,primary.smooth_trace,color="black",lw=1,label="smoothed trace")
    axes[1].plot(x,primary.fit,color=COLORS["fit"],lw=1.7,label=r"I+VIII: $R^2=0.9918$")
    axes[1].plot(x,control.fit,color="#ad8600",lw=1.4,ls="--",label=r"VII+VIII: $R^2=0.9920$")
    axes[1].plot(x,primary.baseline,color=COLORS["baseline"],lw=1,ls="--",label="I+VIII background")
    axes[1].legend(fontsize=7.8,loc="upper right")
    axes[1].set_ylabel("response / a.u.")
    for motif, label in (("I", "I: weak/free OH"),("VI", "VI: zero coefficient"),("VIII", "VIII: bonded water")):
        y=primary[f"component_{motif}"]
        axes[2].plot(x,y,color=COLORS[motif],lw=1.5,ls="--" if motif=="VI" else "-",label=label)
        axes[2].fill_between(x,0,y,color=COLORS[motif],alpha=.12)
    axes[2].set_ylabel("component / a.u.")
    axes[2].set_xlabel(r"wavenumber / cm$^{-1}$")
    axes[2].legend(fontsize=7.8,loc="upper right")
    savefig(fig,"fig3_spectral_fits")


def save_supplement(fig, stem):
    folder = ROOT / "si_figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{stem}.pdf")
    fig.savefig(folder / f"{stem}.png")
    plt.close(fig)


def make_supplementary_figures():
    trace=pd.read_csv(ANALYSIS / "newest_trace_processed.csv")
    d=trace[trace["wavenumber_cm-1"].between(3000,3900)]
    fig,axes=plt.subplots(2,1,figsize=(7.2,5.4),sharex=True)
    for i,ax in enumerate(axes):
        panel_label(ax,"ab"[i])
        ax.axvspan(3200,3300,color="#ffe7c5")
        ax.axvspan(3300,3800,color=COLORS["window"])
        ax.plot(d["wavenumber_cm-1"],d.raw_trace,color="#aaaaaa",lw=.55,label="raw trace")
        ax.plot(d["wavenumber_cm-1"],d.smooth_17,color="black",lw=1.05,label="17-point average")
        ax.set_ylabel("response / a.u.")
    axes[0].set_xlim(3000,3900)
    axes[1].set_ylim(-.05,.36)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel(r"wavenumber / cm$^{-1}$")
    fig.subplots_adjust(hspace=.18)
    save_supplement(fig,"figS1_trace")

    fig,axes=plt.subplots(2,1,figsize=(7.2,5.4),sharex=True,
                          gridspec_kw={"height_ratios":[1.5,1]})
    configs=(("I+VI+VII","#6c4d9b"),("I+VI+VIII",COLORS["fit"]),("I+VI+VII+VIII","#ad8600"))
    for i,ax in enumerate(axes):panel_label(ax,"ab"[i])
    for model,color in configs:
        c=current_curves("3300-3800",model)
        x=c["wavenumber_cm-1"]
        axes[0].plot(x,c.fit,lw=1.35,color=color,label=model)
        axes[1].plot(x,c.residual,lw=1,color=color,label=model)
    axes[0].plot(x,c.smooth_trace,color="black",lw=.9,label="smoothed trace")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("response / a.u.")
    axes[1].set_ylabel("residual / a.u.")
    axes[1].axhline(0,color="#aaaaaa",lw=.65)
    axes[1].set_xlabel(r"wavenumber / cm$^{-1}$")
    axes[1].set_xlim(3300,3800)
    fig.subplots_adjust(hspace=.17)
    save_supplement(fig,"figS2_l3_controls")

    fig,axes=plt.subplots(2,2,figsize=(9,6.1))
    for i,(window,ax) in enumerate(zip(("3200-3800","3300-3800","3400-3800","3350-3700"),axes.flat)):
        panel_label(ax,"abcd"[i])
        for model,color in (("I+VI+VIII",COLORS["fit"]),("I+II+VIII","#2379b5"),("I+VI+VII+VIII","#ad8600")):
            c=current_curves(window,model)
            ax.plot(c["wavenumber_cm-1"],c.fit,color=color,lw=1.3,label=model)
        ax.plot(c["wavenumber_cm-1"],c.smooth_trace,color="black",lw=.85,label="smoothed trace")
        ax.set_title(window+r" cm$^{-1}$",fontsize=9)
        ax.set_xlabel(r"wavenumber / cm$^{-1}$")
        ax.set_ylabel("response / a.u.")
    axes[0,0].legend(fontsize=7)
    fig.subplots_adjust(hspace=.35,wspace=.24)
    save_supplement(fig,"figS3_candidate_fits")

    stats=pd.read_csv(ANALYSIS / "newest_smoothing_window_sensitivity.csv")
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for model,color in (("I+VI+VIII",COLORS["fit"]),("I+II+VIII","#2379b5"),("I+VI+VII+VIII","#ad8600")):
        a=stats[(stats.model==model)&(stats.window_start_cm_1==3300)&(stats.window_end_cm_1==3800)]
        axes[0].plot(a.smoothing_points,a.r2,"o-",color=color,ms=3,label=model)
        b=stats[(stats.model==model)&(stats.smoothing_points==17)]
        windows=["3200-3800","3300-3800","3400-3800","3350-3700"]
        r=[b[(b.window_start_cm_1==int(w.split('-')[0]))&(b.window_end_cm_1==int(w.split('-')[1]))].r2.iloc[0] for w in windows]
        axes[1].plot(range(4),r,"o-",color=color,ms=3,label=model)
    axes[0].set_xlabel("moving-average width / points")
    axes[0].set_xticks([1,7,11,17,25,33])
    axes[0].legend(fontsize=7)
    axes[1].set_xticks(range(4),windows,rotation=20)
    axes[1].set_xlabel(r"fit window / cm$^{-1}$")
    for i,ax in enumerate(axes):
        panel_label(ax,"ab"[i]);ax.set_ylabel(r"$R^2$")
    fig.subplots_adjust(wspace=.27,bottom=.22)
    save_supplement(fig,"figS4_sensitivity")


if __name__ == "__main__":
    make_rosetta_pair_figure()
    make_layer_rosetta_figure()
    make_current_fit()
    make_layer_model_figure()
    make_supplementary_figures()
    print("Regenerated all four main and four supplementary figures.")
