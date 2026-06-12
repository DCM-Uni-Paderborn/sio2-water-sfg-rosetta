from pathlib import Path
from collections import deque
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
STRUCTURES = ANALYSIS / "structure_figures"
PAPER_STRUCTURES = ROOT / "SFG_Structure"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
RENDERED_PAPER_STRUCTURES = OUT / "_paper_structure_panels"
RENDERED_PAPER_STRUCTURES.mkdir(exist_ok=True)
PAPER_RENDER_DPI = 1440

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

REFERENCE_STRUCTURE_PNGS = {
    "A": STRUCTURES / "A.png",
    "B": STRUCTURES / "B.png",
    "L2": STRUCTURES / "L2.png",
}


def paper_structure_panel(key):
    """Return a high-resolution rendering of the original paper structure panel."""
    pdf = PAPER_STRUCTURE_PDFS[key]
    out = RENDERED_PAPER_STRUCTURES / f"{key}_{PAPER_RENDER_DPI}dpi.png"
    if shutil.which("pdftocairo") and (
        not out.exists() or out.stat().st_mtime < pdf.stat().st_mtime
    ):
        subprocess.run(
            [
                "pdftocairo",
                "-png",
                "-singlefile",
                "-r",
                str(PAPER_RENDER_DPI),
                str(pdf),
                str(out.with_suffix("")),
            ],
            check=True,
        )
    return out if out.exists() else REFERENCE_STRUCTURE_PNGS[key]


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


def crop_paper_image(key, box):
    return crop_image(paper_structure_panel(key), scale_box_to_paper_panel(key, box), sharpen=True)


def motif_cutout_from_paper(key, box, flip_vertical=True, flip_horizontal=False):
    return motif_cutout(
        paper_structure_panel(key),
        scale_box_to_paper_panel(key, box),
        flip_vertical=flip_vertical,
        flip_horizontal=flip_horizontal,
        sharpen=True,
    )


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


def savefig(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def make_rosetta_stone():
    fp = pd.read_csv(ANALYSIS / "paper_sfg_fingerprints_digitized.csv")
    species_info = [
        ("I", "L1 / motif I", r"weak/free OH toward SiO$_2$", "A"),
        ("II", "L2 / motif II", "H-bonded bridge", "B"),
        ("VIII", "L3 / motif VIII", "second-layer H-bonded water", "L2"),
    ]

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.35, 1.0],
        hspace=0.32,
        wspace=0.15,
    )

    for i, (species, title, role, panel_key) in enumerate(species_info):
        ax_img = fig.add_subplot(gs[i, 0])
        img = plt.imread(paper_structure_panel(panel_key))
        ax_img.imshow(img, interpolation="none")
        ax_img.set_axis_off()
        ax_img.set_title(f"{title}: {role}", loc="left", fontsize=8.6, pad=2)
        if i == 0:
            panel_label(ax_img, "a")

        ax = fig.add_subplot(gs[i, 1])
        d = fp[fp["species"] == species].sort_values("wavenumber_cm-1")
        x = d["wavenumber_cm-1"].to_numpy()
        y = d["fingerprint"].to_numpy()
        y = y / np.nanmax(np.abs(y))
        ax.axvspan(3400, 3800, color=COLORS["window"], zorder=0)
        ax.axhline(0, color="#bfbfbf", lw=0.7)
        ax.plot(x, y, color=COLORS[species], lw=1.8)
        ax.set_xlim(3000, 3800)
        ax.set_ylim(-1.08, 1.08)
        ax.set_ylabel("norm. Im")
        if i == 2:
            ax.set_xlabel(r"wavenumber / cm$^{-1}$")
        else:
            ax.set_xticklabels([])
        ax.text(
            0.02,
            0.91,
            f"fingerprint {species}",
            transform=ax.transAxes,
            color=COLORS[species],
            fontweight="bold",
        )
        ax.text(
            0.98,
            0.08,
            "clean transfer\nwindow",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="#305a7a",
        )
        if i == 0:
            panel_label(ax, "b")

    fig.suptitle(
        r"Rosetta-stone transfer: motif -> fingerprint -> buried SiO$_2$/water",
        fontsize=10.5,
        y=0.995,
    )
    savefig(fig, "fig1_rosetta_stone")


def make_rosetta_pair_figure():
    fp = pd.read_csv(ANALYSIS / "paper_sfg_fingerprints_digitized.csv")
    pairs = [
        {
            "species": "I",
            "panel": "A",
            "box": (820, 25, 1065, 235),
            "color": COLORS["I"],
        },
        {
            "species": "II",
            "panel": "B",
            "box": (655, 225, 747, 350),
            "color": COLORS["II"],
        },
        {
            "species": "III",
            "panel": "B",
            "box": (725, 165, 880, 330),
            "color": "#b45309",
        },
        {
            "species": "IV",
            "panel": "B",
            "box": (875, 245, 1015, 350),
            "color": "#92400e",
        },
        {
            "species": "V",
            "panel": "B",
            "box": (1010, 245, 1135, 395),
            "color": "#a16207",
        },
        {
            "species": "VI",
            "panel": "B",
            "box": (1145, 220, 1236, 390),
            "color": "#c2410c",
        },
        {
            "species": "VII",
            "panel": "L2",
            "box": (685, 385, 900, 612),
            "color": "#15803d",
        },
        {
            "species": "VIII",
            "panel": "L2",
            "box": (1040, 350, 1225, 500),
            "color": COLORS["VIII"],
        },
    ]

    fig = plt.figure(figsize=(12.6, 6.15))
    gs = fig.add_gridspec(
        2,
        8,
        width_ratios=[0.74, 1.30, 0.74, 1.30, 0.74, 1.30, 0.74, 1.30],
        hspace=0.34,
        wspace=0.18,
    )

    for i, pair in enumerate(pairs):
        row = i // 4
        col = 2 * (i % 4)

        ax_struct = fig.add_subplot(gs[row, col])
        cutout = motif_cutout_from_paper(pair["panel"], pair["box"], flip_vertical=False, flip_horizontal=False)
        visible = cutout[:, :, 3] > 0
        if np.any(visible):
            yy, xx = np.nonzero(visible)
            pad = 10
            y0, y1 = max(0, yy.min() - pad), min(cutout.shape[0], yy.max() + pad + 1)
            x0, x1 = max(0, xx.min() - pad), min(cutout.shape[1], xx.max() + pad + 1)
            cutout = cutout[y0:y1, x0:x1]
        ax_struct.imshow(cutout, interpolation="none")
        ax_struct.set_axis_off()
        ax_struct.set_title(
            f"motif {pair['species']}",
            color=pair["color"],
            fontsize=9.0,
            fontweight="bold",
            pad=2,
        )
        ax_struct.text(
            1.05,
            0.70,
            "->",
            transform=ax_struct.transAxes,
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#3c3c3c",
            clip_on=False,
        )

        ax_fp = fig.add_subplot(gs[row, col + 1])
        d = fp[fp["species"] == pair["species"]].sort_values("wavenumber_cm-1")
        x = d["wavenumber_cm-1"].to_numpy()
        y = d["fingerprint"].to_numpy()
        y = y / np.nanmax(np.abs(y))
        ax_fp.axvspan(3400, 3800, color=COLORS["window"], zorder=0)
        ax_fp.axhline(0, color="#c9c9c9", lw=0.75)
        ax_fp.plot(x, y, color=pair["color"], lw=1.85)
        ax_fp.set_xlim(3000, 3800)
        ax_fp.set_ylim(-1.08, 1.08)
        ax_fp.set_title("total SFG", fontsize=8.0, pad=2)
        ax_fp.set_xticks([3000, 3400, 3800])
        ax_fp.set_yticks([-1, 0, 1])
        ax_fp.tick_params(labelsize=6.7)
        if row == 1:
            ax_fp.set_xlabel(r"cm$^{-1}$", labelpad=1)
        else:
            ax_fp.set_xticklabels([])
        if col != 0:
            ax_fp.set_yticklabels([])
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

    fig.text(0.018, 0.50, "norm. Im total SFG", rotation=90, ha="center", va="center", fontsize=8.5)
    fig.suptitle(
        "Rosetta stone: every water motif maps to a total SFG fingerprint",
        fontsize=11.0,
        y=0.995,
    )
    savefig(fig, "fig1_rosetta_pairs")


def sharpen_rgb(image):
    """Recover crispness after rendering Photoshop-PDF structure panels."""
    image = image.convert("RGB")
    image = ImageEnhance.Sharpness(image).enhance(1.65)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    return image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=95, threshold=3))


def sharpen_rgba(image):
    rgb = sharpen_rgb(image.convert("RGB"))
    alpha = image.getchannel("A")
    return Image.merge("RGBA", (*rgb.split(), alpha))


def crop_image(path, box, sharpen=False):
    image = Image.open(path).convert("RGB").crop(box)
    if sharpen:
        image = sharpen_rgb(image)
    return np.asarray(image)


def motif_cutout(path, box, flip_vertical=True, flip_horizontal=False, sharpen=False):
    img = Image.open(path).convert("RGBA").crop(box)
    if flip_vertical:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if flip_horizontal:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    arr = np.asarray(img).copy()
    rgb = arr[:, :, :3]
    h, w = rgb.shape[:2]
    near_white = (rgb[:, :, 0] > 236) & (rgb[:, :, 1] > 236) & (rgb[:, :, 2] > 236)

    background = np.zeros((h, w), dtype=bool)
    queue = deque()
    for x in range(w):
        for y in (0, h - 1):
            if near_white[y, x] and not background[y, x]:
                background[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if near_white[y, x] and not background[y, x]:
                background[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for yn, xn in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yn < h and 0 <= xn < w and near_white[yn, xn] and not background[yn, xn]:
                background[yn, xn] = True
                queue.append((yn, xn))

    neutral_range = rgb.max(axis=2) - rgb.min(axis=2)
    dark_paper_marks = (rgb.max(axis=2) < 178) & (neutral_range < 62)
    transparent = background | dark_paper_marks
    arr[:, :, 3][transparent] = 0
    arr[:, :, :3][transparent] = 255

    visible = arr[:, :, 3] > 0
    visited = np.zeros((h, w), dtype=bool)
    keep = np.zeros((h, w), dtype=bool)
    red_like = (rgb[:, :, 0] > 130) & (rgb[:, :, 1] < 120) & (rgb[:, :, 2] < 120)

    for y0 in range(h):
        for x0 in range(w):
            if not visible[y0, x0] or visited[y0, x0]:
                continue
            coords = []
            queue = deque([(y0, x0)])
            visited[y0, x0] = True
            while queue:
                y, x = queue.popleft()
                coords.append((y, x))
                for yn, xn in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yn < h and 0 <= xn < w and visible[yn, xn] and not visited[yn, xn]:
                        visited[yn, xn] = True
                        queue.append((yn, xn))
            yy = np.fromiter((p[0] for p in coords), dtype=int)
            xx = np.fromiter((p[1] for p in coords), dtype=int)
            area = len(coords)
            has_water_oxygen = np.any(red_like[yy, xx])
            if has_water_oxygen:
                keep[yy, xx] = True

    remove = visible & ~keep
    arr[:, :, 3][remove] = 0
    arr[:, :, :3][remove] = 255
    if sharpen:
        arr = np.asarray(sharpen_rgba(Image.fromarray(arr)))
    return arr


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
        },
        {
            "species": "II",
            "title": "L2 / motif II",
            "subtitle": "H-bonded bridge",
            "panel": "B",
            "color": COLORS["II"],
            "ellipses": [(0.63, 0.62, 0.28, 0.27, -12)],
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
        "II": (55, 112, 590, 575),
        "VIII": (55, 122, 590, 585),
    }
    structure_boxes = {
        "I": (650, 0, 1230, 525),
        "II": (650, 0, 1230, 530),
        "VIII": (650, 0, 1230, 585),
    }

    fig = plt.figure(figsize=(10.6, 8.6))
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
        ax_struct.imshow(crop_paper_image(col["panel"], structure_boxes[species]), interpolation="none")
        ax_struct.set_axis_off()
        ax_struct.set_title(f"{col['title']}\n{col['subtitle']}", fontsize=10.0, pad=4)
        if j == 0:
            panel_label(ax_struct, "a")

        ax_orient = fig.add_subplot(gs[1, j])
        ax_orient.imshow(crop_paper_image(col["panel"], orient_boxes[species]), interpolation="none")
        ax_orient.set_axis_off()
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
        if j == 0:
            panel_label(ax_orient, "b")

        ax_fp = fig.add_subplot(gs[2, j])
        d = fp[fp["species"] == species].sort_values("wavenumber_cm-1")
        x = d["wavenumber_cm-1"].to_numpy()
        y = d["fingerprint"].to_numpy()
        y = y / np.nanmax(np.abs(y))
        ax_fp.axvspan(3400, 3800, color=COLORS["window"], zorder=0)
        ax_fp.axhline(0, color="#c9c9c9", lw=0.75)
        ax_fp.plot(x, y, color=color, lw=2.0)
        ax_fp.set_xlim(3000, 3800)
        ax_fp.set_ylim(-1.08, 1.08)
        ax_fp.set_title(f"total SFG fingerprint {species}", color=color, fontsize=9.0, pad=2)
        ax_fp.set_xlabel(r"wavenumber / cm$^{-1}$")
        if j == 0:
            ax_fp.set_ylabel("norm. Im")
            panel_label(ax_fp, "c")
        else:
            ax_fp.set_yticklabels([])
        ax_fp.tick_params(labelsize=7.5)

    fig.suptitle(
        "Selected water motifs, orientation maps, and total SFG fingerprints",
        fontsize=12.0,
        y=0.995,
    )
    savefig(fig, "fig1_layer_rosetta")


def make_sfg_fit_figure():
    hq = pd.read_csv(ANALYSIS / "high_quality_full_3300_3800_curve.csv")
    pf = pd.read_csv(ANALYSIS / "paper_window_3400_3800_bridge_pool_curve.csv")

    fig = plt.figure(figsize=(7.2, 6.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 0.82], hspace=0.16)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)

    xh = hq["wavenumber_cm-1"].to_numpy()
    ax0.axvspan(3300, 3400, color=COLORS["edge"], zorder=0)
    ax0.axvspan(3400, 3800, color=COLORS["window"], zorder=0)
    ax0.axhline(0, color="#c7c7c7", lw=0.7)
    ax0.plot(xh, hq["imag_smooth"], color=COLORS["data"], lw=1.15, label="experiment")
    ax0.plot(xh, hq["imag_fit"], color=COLORS["fit"], lw=1.55, label="full complex fit")
    ax0.set_xlim(3300, 3800)
    ax0.set_ylabel(r"Im $\chi^{(2)}$")
    ax0.legend(loc="lower right", ncol=2, fontsize=7.8)
    ax0.text(
        0.02,
        0.08,
        r"full 3300--3800 cm$^{-1}$: $R^2_\mathrm{complex}=0.990$, $R^2_\mathrm{Im}=0.983$",
        transform=ax0.transAxes,
        fontsize=7.6,
    )
    ax0.text(3331, ax0.get_ylim()[1] * 0.78, "edge /\nnuisance", ha="center", va="top", fontsize=7.1, color="#666666")
    panel_label(ax0, "a")

    xp = pf["wavenumber_cm-1"].to_numpy()
    ax1.axvspan(3400, 3800, color=COLORS["window"], zorder=0)
    ax1.plot(xp, pf["imag_exp_smooth"], color=COLORS["data"], lw=1.15, label="experiment")
    ax1.plot(xp, pf["imag_fit"], color=COLORS["fit"], lw=1.65, label="L1 + L2 + L3 fit")
    ax1.plot(xp, pf["imag_baseline"], color=COLORS["baseline"], lw=1.0, ls="--", label="floating baseline")
    ax1.set_ylabel(r"Im $\chi^{(2)}$")
    ax1.legend(loc="lower right", fontsize=7.8)
    ax1.text(
        0.02,
        0.08,
        r"paper fingerprints, 3400--3800 cm$^{-1}$: $R^2=0.972$, $p=5.2\times10^{-155}$",
        transform=ax1.transAxes,
        fontsize=7.6,
    )
    panel_label(ax1, "b")

    component_cols = {
        "L1 / I": ("I", "component_I / L1 Region A free-OH motif"),
        "L2 / II": ("II", "component_II / L1 Region B upward motif"),
        "L3 / VIII": ("VIII", "component_VIII / L2 positive bonded motif"),
    }
    ax2.axhline(0, color="#c7c7c7", lw=0.7)
    for label, (species, col) in component_cols.items():
        y = pf[col].to_numpy()
        ax2.plot(xp, y, lw=1.6, color=COLORS[species], label=label)
        ax2.fill_between(xp, 0, y, color=COLORS[species], alpha=0.12)
    ax2.set_ylabel("component")
    ax2.set_xlabel(r"wavenumber / cm$^{-1}$")
    ax2.legend(title="layer / motif", loc="upper left", ncol=3, fontsize=7.7, title_fontsize=7.7)
    ax2.text(
        0.98,
        0.80,
        r"L1/I: SiO$_2$-side weak/free OH" + "\nL2/II: H-bond bridge\nL3/VIII: H-bonded water",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    panel_label(ax2, "c")

    savefig(fig, "fig2_sfg_fit")


def make_layer_model_figure():
    layers = [
        {
            "layer": "L1",
            "motif": "I",
            "role": "SiO2-facing weak/free OH",
            "fraction": "0.375",
            "panel": "A",
            "box": (820, 25, 1065, 235),
            "color": COLORS["I"],
            "extent": (3.72, 5.82, 1.22, 3.02),
            "flip_horizontal": False,
        },
        {
            "layer": "L2",
            "motif": "II",
            "role": "H-bonded bridge",
            "fraction": "0.395",
            "panel": "B",
            "box": (655, 225, 747, 350),
            "color": COLORS["II"],
            "extent": (2.78, 4.03, 2.43, 3.95),
            "flip_horizontal": False,
        },
        {
            "layer": "L3",
            "motif": "VIII",
            "role": "second-layer H-bonded water",
            "fraction": "0.230",
            "panel": "L2",
            "box": (1040, 350, 1225, 500),
            "color": COLORS["VIII"],
            "extent": (1.84, 3.72, 3.38, 4.90),
            "flip_horizontal": False,
        },
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.85))
    ax.set_xlim(0, 8.2)
    ax.set_ylim(0, 5.75)
    ax.set_axis_off()

    band_specs = [
        (1.10, 2.48, COLORS["I"], "L1 / motif I", r"weak/free OH toward SiO$_2$"),
        (2.48, 3.72, COLORS["II"], "L2 / motif II", "H-bond bridge"),
        (3.72, 5.18, COLORS["VIII"], "L3 / motif VIII", "H-bonded water"),
    ]
    for y0, y1, color, label, role in band_specs:
        ax.add_patch(Rectangle((0, y0), 8.2, y1 - y0, facecolor=color, alpha=0.070, edgecolor="none", zorder=0))
        ax.plot([0.35, 7.95], [y1, y1], color=color, lw=0.8, alpha=0.45, ls=(0, (3, 3)), zorder=1)
        ax.text(7.86, (y0 + y1) / 2, label, color=color, ha="right", va="center", fontsize=9.4, fontweight="bold")
        ax.text(7.86, (y0 + y1) / 2 - 0.22, role, color="#343434", ha="right", va="center", fontsize=7.3)

    silica_top = [(0, 0.96), (0.8, 1.03), (1.65, 0.93), (2.55, 1.02), (3.35, 0.95),
                  (4.25, 1.05), (5.20, 0.96), (6.05, 1.03), (7.10, 0.95), (8.2, 1.02)]
    silica_poly = [(0, 0), (8.2, 0), *reversed(silica_top)]
    ax.add_patch(Polygon(silica_poly, closed=True, facecolor="#dfe4e8", edgecolor="#8b949e", lw=1.0, zorder=2))
    ax.plot([p[0] for p in silica_top], [p[1] for p in silica_top], color="#6f7b86", lw=1.1, zorder=4)

    si_positions = [(0.55, 0.35), (1.45, 0.55), (2.42, 0.33), (3.35, 0.62),
                    (4.30, 0.38), (5.22, 0.58), (6.18, 0.35), (7.15, 0.55)]
    o_positions = [(0.25, 0.92), (0.95, 1.03), (1.88, 0.95), (2.80, 1.03),
                   (3.75, 0.97), (4.68, 1.04), (5.62, 0.96), (6.55, 1.03), (7.55, 0.98)]
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
    ax.text(0.28, 0.24, r"SiO$_2$ surface", fontsize=9.5, color="#333333", ha="left", va="center", zorder=8)

    for layer in layers:
        cutout = motif_cutout_from_paper(
            layer["panel"],
            layer["box"],
            flip_vertical=True,
            flip_horizontal=layer["flip_horizontal"],
        )
        ax.imshow(cutout, extent=layer["extent"], interpolation="none", zorder=10)

    hbonds = [
        {
            "start": (3.55, 2.88),
            "end": (4.72, 2.05),
            "label": "H(L2)...O(L1)",
            "text": (4.12, 2.40),
        },
        {
            "start": (2.92, 4.06),
            "end": (3.18, 3.45),
            "label": "H(L3)...O(L2)",
            "text": (3.62, 4.03),
        },
    ]
    for hbond in hbonds:
        ax.annotate(
            "",
            xy=hbond["end"],
            xytext=hbond["start"],
            arrowprops=dict(arrowstyle="-", lw=1.3, color="#5a8fbf", ls=(0, (3, 3))),
            zorder=9,
        )
        ax.text(
            hbond["text"][0],
            hbond["text"][1],
            hbond["label"],
            color="#42759f",
            fontsize=6.8,
            ha="center",
            va="center",
            zorder=13,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.72),
        )
    label_specs = [
        ("L1", "I", 5.92, 2.76, COLORS["I"]),
        ("L2", "II", 2.34, 3.58, COLORS["II"]),
        ("L3", "VIII", 1.48, 4.86, COLORS["VIII"]),
    ]
    for layer, motif, x, y, color in label_specs:
        ax.text(
            x,
            y,
            f"{layer}/{motif}",
            color=color,
            fontsize=8.2,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=13,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, lw=0.8, alpha=0.88),
        )

    ax.annotate(
        r"away from SiO$_2$",
        xy=(0.45, 5.13),
        xytext=(0.45, 1.18),
        arrowprops=dict(arrowstyle="->", color="#404040", lw=1.1),
        color="#404040",
        fontsize=8.0,
        ha="left",
        va="bottom",
    )
    ax.text(
        4.05,
        5.50,
        r"Layered water model on SiO$_2$: L1/I $\rightarrow$ L2/II $\rightarrow$ L3/VIII",
        ha="center",
        va="center",
        fontsize=10.4,
        fontweight="bold",
    )
    savefig(fig, "fig3_layer_model")


if __name__ == "__main__":
    make_rosetta_pair_figure()
    make_layer_rosetta_figure()
    make_rosetta_stone()
    make_sfg_fit_figure()
    make_layer_model_figure()
    print(f"Wrote figures to {OUT}")
