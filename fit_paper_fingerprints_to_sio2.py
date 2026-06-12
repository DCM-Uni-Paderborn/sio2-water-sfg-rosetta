#!/usr/bin/env python3
"""Fit SFG fingerprints digitized from the author's paper to SiO2/water data.

The fingerprint curves are extracted from the gnuplot EPS source files
(`SFG_A.eps`, `SFG_B.eps`, `SFG_L2.eps`) rather than redrawn analytically.
Only the magenta "Total" curves are used as structural fingerprints.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STRUCT = ROOT / "SFG_Structure"
OUT_DIR = ROOT / "analysis"
X_AXIS = ROOT / "xaxis.txt"
COMPLEX_CSV = ROOT / "sio2_water_complex.csv"

FIT_START = 3300.0
FIT_END = 3800.0


@dataclass(frozen=True)
class FingerprintSpec:
    label: str
    eps_file: str
    total_index: int
    region: str
    species: str
    hbond_character: str


FINGERPRINTS = [
    FingerprintSpec(
        "I / L1 Region A free-OH motif",
        "SFG_A.eps",
        0,
        "L1 Region A",
        "I",
        "one nearly free OH plus one OH directed back to the H-bond network",
    ),
    FingerprintSpec(
        "II / L1 Region B upward motif",
        "SFG_B.eps",
        0,
        "L1 Region B",
        "II",
        "Region-B water connecting toward the dilute outer/weakly bonded side",
    ),
    FingerprintSpec(
        "III / L1 Region B upward H-bond motif",
        "SFG_B.eps",
        1,
        "L1 Region B",
        "III",
        "H-bonded Region-B motif with high-frequency/free-OH-like contribution",
    ),
    FingerprintSpec(
        "IV / L1 Region B parallel motif",
        "SFG_B.eps",
        2,
        "L1 Region B",
        "IV",
        "dipole and OH bonds nearly parallel to interface; weak locally, coupling-active",
    ),
    FingerprintSpec(
        "V / L1 Region B liquid-oriented motif",
        "SFG_B.eps",
        3,
        "L1 Region B",
        "V",
        "bonded OH groups oriented toward the liquid/H-bond network",
    ),
    FingerprintSpec(
        "VI / L1 Region B connector motif",
        "SFG_B.eps",
        4,
        "L1 Region B",
        "VI",
        "connector from Region B toward L2; mostly H-bonded with weakly bonded shoulder",
    ),
    FingerprintSpec(
        "VII / L2 negative bonded motif",
        "SFG_L2.eps",
        0,
        "L2",
        "VII",
        "second-layer H-bonded motif, opposite orientation to VIII",
    ),
    FingerprintSpec(
        "VIII / L2 positive bonded motif",
        "SFG_L2.eps",
        1,
        "L2",
        "VIII",
        "second-layer H-bonded motif, partially cancels VII",
    ),
]


TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?|[A-Za-z]+|%.*")


def moving_average(values: np.ndarray, window: int = 17) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def load_experiment() -> pd.DataFrame:
    x = np.array([float(v) for v in X_AXIS.read_text(encoding="utf-8-sig").splitlines()[1:] if v.strip()])
    raw = pd.read_csv(COMPLEX_CSV)
    return pd.DataFrame(
        {
            "wavenumber_cm-1": x,
            "real": raw.iloc[:, 0].to_numpy(float),
            "imag": raw.iloc[:, 1].to_numpy(float),
            "imag_smooth": moving_average(raw.iloc[:, 1].to_numpy(float), 17),
        }
    )


def parse_path_points(text: str) -> np.ndarray:
    tokens = TOKEN_RE.findall(text)
    points: list[tuple[float, float]] = []
    stack: list[float] = []
    current: tuple[float, float] | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("%"):
            i += 1
            continue
        try:
            stack.append(float(tok))
            i += 1
            continue
        except ValueError:
            pass

        if tok in {"M", "N"} and len(stack) >= 2:
            y = stack.pop()
            x = stack.pop()
            current = (x, y)
            points.append(current)
            stack.clear()
        elif tok == "L" and len(stack) >= 2:
            y = stack.pop()
            x = stack.pop()
            current = (x, y)
            points.append(current)
            stack.clear()
        elif tok == "V" and current is not None and len(stack) >= 2:
            dy = stack.pop()
            dx = stack.pop()
            current = (current[0] + dx, current[1] + dy)
            points.append(current)
            stack.clear()
        elif tok in {"stroke", "grestore", "PolyFill"}:
            stack.clear()
        else:
            stack.clear()
        i += 1
    if len(points) < 10:
        raise ValueError("Too few path points parsed")
    return np.array(points, dtype=float)


def total_plot_blocks(eps_path: Path) -> list[str]:
    lines = eps_path.read_text(errors="ignore").splitlines()
    blocks: list[str] = []
    capture = False
    buf: list[str] = []
    for i, line in enumerate(lines):
        if "% Begin plot #1" in line and "1.00 0.00 1.00 C" in "\n".join(lines[i : i + 6]):
            capture = True
            buf = []
            continue
        if capture and "% Begin plot #2" in line:
            blocks.append("\n".join(buf))
            capture = False
            buf = []
            continue
        if capture:
            buf.append(line)
    return blocks


def x_from_eps(px: np.ndarray) -> np.ndarray:
    # Each SFG panel is written at a different horizontal EPS offset.  The
    # total-curve path itself spans the data panel, so use that local extent
    # instead of assuming one hard-coded panel offset.  The EPS x-axis is
    # 3000-3800 cm^-1 in the author's figures.
    left = float(np.min(px))
    right = float(np.max(px))
    if right <= left:
        return np.full_like(px, 3000.0, dtype=float)
    return 3000.0 + (px - left) * (800.0 / (right - left))


def extract_fingerprint(spec: FingerprintSpec) -> pd.DataFrame:
    blocks = total_plot_blocks(STRUCT / spec.eps_file)
    if spec.total_index >= len(blocks):
        raise IndexError(f"{spec.eps_file} has only {len(blocks)} total curves")
    pts = parse_path_points(blocks[spec.total_index])
    # The filled gnuplot path is usually duplicated and contains a closing
    # vertical/baseline segment. Keep the first monotonically increasing trace.
    dx = np.diff(pts[:, 0])
    reset = np.where(dx < -100)[0]
    if len(reset):
        pts = pts[: reset[0] + 1]
    x = x_from_eps(pts[:, 0])
    y_px = pts[:, 1]
    order = np.argsort(x)
    x = x[order]
    y_px = y_px[order]
    keep = (x >= 2890.0) & (x <= 3910.0)
    x = x[keep]
    y_px = y_px[keep]

    # The zero line is estimated from the quiet ends of the curve. The EPS
    # y-axis is arbitrary and panel-specific, so only the relative shape is used.
    end_mask = (x < 3000.0) | (x > 3850.0)
    if np.sum(end_mask) >= 20:
        zero = float(np.median(y_px[end_mask]))
    else:
        zero = float(np.median(np.r_[y_px[:20], y_px[-20:]]))
    y = y_px - zero
    # Normalize on the water fit window. Do not flip sign here; the fit tests
    # both global signs relative to the paper convention.
    fit_mask = (x >= FIT_START) & (x <= FIT_END)
    rms = float(np.sqrt(np.mean(y[fit_mask] ** 2))) if np.any(fit_mask) else float(np.sqrt(np.mean(y**2)))
    if rms == 0:
        rms = 1.0
    y_norm = y / rms
    return pd.DataFrame(
        {
            "wavenumber_cm-1": x,
            "fingerprint": y_norm,
            "label": spec.label,
            "region": spec.region,
            "species": spec.species,
            "hbond_character": spec.hbond_character,
        }
    )


def interpolate_basis(
    exp_x: np.ndarray,
    fingerprints: list[pd.DataFrame],
    shift_cm: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    cols = []
    labels = []
    for fp in fingerprints:
        x = fp["wavenumber_cm-1"].to_numpy() + shift_cm
        y = fp["fingerprint"].to_numpy()
        # Average duplicate x points introduced by integer EPS line segments.
        grouped = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False).mean()
        yi = np.interp(exp_x, grouped["x"].to_numpy(), grouped["y"].to_numpy(), left=0.0, right=0.0)
        # Re-normalize after interpolation to prevent digitization density from
        # changing component weights.
        rms = np.sqrt(np.mean(yi**2))
        if rms > 0:
            yi = yi / rms
        cols.append(yi)
        labels.append(fp["label"].iloc[0])
    return np.column_stack(cols), labels


def solve_active_set(X: np.ndarray, y: np.ndarray, nonnegative_from: int = 2) -> np.ndarray:
    active = np.ones(X.shape[1] - nonnegative_from, dtype=bool)
    coefs = np.zeros(X.shape[1])
    for _ in range(len(active) + 1):
        keep = np.r_[np.ones(nonnegative_from, dtype=bool), active]
        Xk = X[:, keep]
        ck, *_ = np.linalg.lstsq(Xk, y, rcond=None)
        trial = np.zeros(X.shape[1])
        trial[keep] = ck
        constrained = trial[nonnegative_from:]
        if np.all(constrained[active] >= -1e-12):
            coefs = trial
            break
        idx = np.where(active)[0][np.argmin(constrained[active])]
        active[idx] = False
        coefs = trial
    return coefs


def f_survival(f_value: float, dfn: int, dfd: int) -> float:
    if not math.isfinite(f_value) or f_value <= 0:
        return 1.0
    try:
        import mpmath as mp

        mp.mp.dps = 120
        x = mp.mpf(dfd) / (mp.mpf(dfd) + mp.mpf(dfn) * mp.mpf(f_value))
        p = mp.betainc(mp.mpf(dfd) / 2, mp.mpf(dfn) / 2, 0, x, regularized=True)
        p_float = float(p)
        if math.isfinite(p_float):
            return p_float
    except Exception:
        pass
    if dfn == 1:
        # Large-df fallback: F(1,nu) is close to z^2 for small p.
        return math.erfc(math.sqrt(max(f_value, 0.0) / 2.0))
    if dfn == 2:
        # Exact survival for F(2,nu).
        return (dfd / (dfd + 2.0 * max(f_value, 0.0))) ** (dfd / 2.0)
    x = dfd / (dfd + dfn * f_value)
    return regularized_beta(x, dfd / 2.0, dfn / 2.0)


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iter = 300
    eps = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -((a + m) * (qab + m) * x) / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_bt) * beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(log_bt) * beta_continued_fraction(b, a, 1.0 - x) / b


def p_text(p_value: float) -> str:
    if not math.isfinite(p_value):
        return "not computed"
    if p_value == 0.0:
        return "<1e-300"
    return f"{p_value:.3e}"


def model_stats(y: np.ndarray, y_base: np.ndarray, sse_base: float, fit: np.ndarray, active_count: int) -> dict:
    n = len(y)
    p_base = 2
    p_full = p_base + active_count
    dfn = max(p_full - p_base, 1)
    dfd = max(n - p_full, 1)
    sse = float(np.sum((y - fit) ** 2))
    if sse_base > sse and active_count > 0:
        f_value = ((sse_base - sse) / dfn) / (sse / dfd)
        p_value = f_survival(f_value, dfn, dfd)
    else:
        f_value = 0.0
        p_value = 1.0
    tss = float(np.sum((y - y.mean()) ** 2))
    p_full = p_base + active_count
    return {
        "sse": sse,
        "r2": 1.0 - sse / tss,
        "r2_base": 1.0 - sse_base / tss,
        "f_value": f_value,
        "dfn": dfn,
        "dfd": dfd,
        "p_value": p_value,
        "aic": n * math.log(max(sse / n, 1e-300)) + 2 * p_full,
        "bic": n * math.log(max(sse / n, 1e-300)) + math.log(n) * p_full,
        "parameter_count": p_full,
    }


def incremental_stats(previous: dict, current: dict, n: int) -> dict:
    p_prev = int(previous["parameter_count"])
    p_current = int(current["parameter_count"])
    dfn = p_current - p_prev
    dfd = max(n - p_current, 1)
    if dfn <= 0 or previous["sse"] <= current["sse"]:
        return {"dfn": max(dfn, 0), "dfd": dfd, "f_value": float("nan"), "p_value": float("nan"), "delta_sse": previous["sse"] - current["sse"]}
    f_value = ((previous["sse"] - current["sse"]) / dfn) / (current["sse"] / dfd)
    return {
        "dfn": dfn,
        "dfd": dfd,
        "f_value": f_value,
        "p_value": f_survival(f_value, dfn, dfd),
        "delta_sse": previous["sse"] - current["sse"],
    }


def species_indices(species: list[str]) -> tuple[int, ...]:
    lookup = {spec.species: idx for idx, spec in enumerate(FINGERPRINTS)}
    return tuple(lookup[s] for s in species)


def nonempty_subsets(indices: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(1, len(indices) + 1)
        for subset in itertools.combinations(indices, size)
    ]


def fit_required_choice_model(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    fps: list[pd.DataFrame],
    X_base: np.ndarray,
    y_base: np.ndarray,
    sse_base: float,
    required_indices: tuple[int, ...],
    choice_groups: list[list[tuple[int, ...]]],
    description: str,
) -> dict:
    best = None
    for shift_cm in np.linspace(-80.0, 80.0, 161):
        basis, labels = interpolate_basis(x, fps, shift_cm=shift_cm)
        for global_sign in (-1.0, 1.0):
            basis_signed = global_sign * basis
            for choices in itertools.product(*choice_groups):
                selected = tuple(sorted(set(required_indices + tuple(itertools.chain.from_iterable(choices)))))
                solved = solve_selected_subset(X_base, basis_signed, y, selected)
                if solved is None:
                    continue
                coefs, y_fit = solved
                sse = float(np.sum((y - y_fit) ** 2))
                if best is None or sse < best["sse"]:
                    best = {
                        "kind": name,
                        "description": description,
                        "shift_cm": float(shift_cm),
                        "global_sign": global_sign,
                        "coefs": coefs,
                        "fit": y_fit,
                        "sse": sse,
                        "basis": basis,
                        "labels": labels,
                        "active_indices": selected,
                    }
    if best is None:
        raise RuntimeError(f"No feasible model found for {name}")
    best.update(model_stats(y, y_base, sse_base, best["fit"], len(best["active_indices"])))
    return best


def fit_required_optional_model(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    fps: list[pd.DataFrame],
    X_base: np.ndarray,
    y_base: np.ndarray,
    sse_base: float,
    required_indices: tuple[int, ...],
    optional_indices: tuple[int, ...],
    choice_groups: list[list[tuple[int, ...]]],
    description: str,
) -> dict:
    optional_subsets = [tuple()]
    optional_subsets.extend(nonempty_subsets(optional_indices))
    best = None
    for shift_cm in np.linspace(-80.0, 80.0, 161):
        basis, labels = interpolate_basis(x, fps, shift_cm=shift_cm)
        for global_sign in (-1.0, 1.0):
            basis_signed = global_sign * basis
            for opt_subset in optional_subsets:
                for choices in itertools.product(*choice_groups):
                    selected = tuple(sorted(set(required_indices + opt_subset + tuple(itertools.chain.from_iterable(choices)))))
                    solved = solve_selected_subset(X_base, basis_signed, y, selected)
                    if solved is None:
                        continue
                    coefs, y_fit = solved
                    sse = float(np.sum((y - y_fit) ** 2))
                    if best is None or sse < best["sse"]:
                        best = {
                            "kind": name,
                            "description": description,
                            "shift_cm": float(shift_cm),
                            "global_sign": global_sign,
                            "coefs": coefs,
                            "fit": y_fit,
                            "sse": sse,
                            "basis": basis,
                            "labels": labels,
                            "active_indices": selected,
                            "optional_allowed": optional_indices,
                        }
    if best is None:
        raise RuntimeError(f"No feasible model found for {name}")
    best.update(model_stats(y, y_base, sse_base, best["fit"], len(best["active_indices"])))
    return best


def fit_unconstrained(
    x: np.ndarray,
    y: np.ndarray,
    fps: list[pd.DataFrame],
    X_base: np.ndarray,
    y_base: np.ndarray,
    sse_base: float,
) -> dict:
    best = None
    for shift_cm in np.linspace(-80.0, 80.0, 161):
        basis, labels = interpolate_basis(x, fps, shift_cm=shift_cm)
        for global_sign in (-1.0, 1.0):
            X = np.column_stack([X_base, global_sign * basis])
            coefs = solve_active_set(X, y, nonnegative_from=2)
            y_fit = X @ coefs
            sse = float(np.sum((y - y_fit) ** 2))
            if best is None or sse < best["sse"]:
                active_indices = tuple(np.flatnonzero(coefs[2:] > 1e-12).tolist())
                best = {
                    "kind": "unconstrained",
                    "shift_cm": float(shift_cm),
                    "global_sign": global_sign,
                    "coefs": coefs,
                    "fit": y_fit,
                    "sse": sse,
                    "basis": basis,
                    "labels": labels,
                    "active_indices": active_indices,
                }
    assert best is not None
    best.update(model_stats(y, y_base, sse_base, best["fit"], len(best["active_indices"])))
    return best


def solve_selected_subset(
    X_base: np.ndarray,
    basis_signed: np.ndarray,
    y: np.ndarray,
    selected: tuple[int, ...],
    min_weight: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray] | None:
    X = np.column_stack([X_base, basis_signed[:, selected]])
    coefs_short, *_ = np.linalg.lstsq(X, y, rcond=None)
    weights = coefs_short[2:]
    if np.any(weights < min_weight):
        return None
    coefs_full = np.zeros(2 + basis_signed.shape[1])
    coefs_full[:2] = coefs_short[:2]
    coefs_full[2 + np.array(selected)] = weights
    return coefs_full, X @ coefs_short


def fit_constrained_layer_model(
    x: np.ndarray,
    y: np.ndarray,
    fps: list[pd.DataFrame],
    X_base: np.ndarray,
    y_base: np.ndarray,
    sse_base: float,
) -> dict:
    return fit_required_choice_model(
        "flexible_l1b_layer_populated",
        x,
        y,
        fps,
        X_base,
        y_base,
        sse_base,
        required_indices=species_indices(["I"]),
        choice_groups=[
            nonempty_subsets(species_indices(["II", "III", "IV", "V", "VI"])),
            nonempty_subsets(species_indices(["VII", "VIII"])),
        ],
        description="I fixed, any nonempty II-VI L1B mixture, and VII/VIII L2 present",
    )


def fit_minimal_l1a_l2_model(
    x: np.ndarray,
    y: np.ndarray,
    fps: list[pd.DataFrame],
    X_base: np.ndarray,
    y_base: np.ndarray,
    sse_base: float,
) -> dict:
    return fit_required_choice_model(
        "minimal_l1a_l2",
        x,
        y,
        fps,
        X_base,
        y_base,
        sse_base,
        required_indices=species_indices(["I"]),
        choice_groups=[nonempty_subsets(species_indices(["VII", "VIII"]))],
        description="Minimal H-bonded scaffold: L1A/I plus L2/VII and-or VIII",
    )


def fit_model() -> dict:
    exp = load_experiment()
    mask = (exp["wavenumber_cm-1"] >= FIT_START) & (exp["wavenumber_cm-1"] <= FIT_END)
    x = exp.loc[mask, "wavenumber_cm-1"].to_numpy()
    y = exp.loc[mask, "imag_smooth"].to_numpy()

    fps = [extract_fingerprint(spec) for spec in FINGERPRINTS]
    z = (x - x.mean()) / np.ptp(x)
    X_base = np.column_stack([np.ones_like(x), z])
    base_coefs, *_ = np.linalg.lstsq(X_base, y, rcond=None)
    y_base = X_base @ base_coefs
    sse_base = float(np.sum((y - y_base) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2_base = 1.0 - sse_base / tss
    unconstrained = fit_unconstrained(x, y, fps, X_base, y_base, sse_base)
    minimal = fit_minimal_l1a_l2_model(x, y, fps, X_base, y_base, sse_base)
    try:
        forced_bridge_ii_vi = fit_required_choice_model(
            "forced_bridge_ii_vi",
            x,
            y,
            fps,
            X_base,
            y_base,
            sse_base,
            required_indices=species_indices(["I", "II", "VI"]),
            choice_groups=[nonempty_subsets(species_indices(["VII", "VIII"]))],
            description="Diagnostic: force both L1B bridge species II and VI between L1A and L2",
        )
    except RuntimeError:
        forced_bridge_ii_vi = None
    bridge_ii_vi = fit_required_choice_model(
        "bridge_ii_vi_pool",
        x,
        y,
        fps,
        X_base,
        y_base,
        sse_base,
        required_indices=species_indices(["I"]),
        choice_groups=[
            nonempty_subsets(species_indices(["II", "VI"])),
            nonempty_subsets(species_indices(["VII", "VIII"])),
        ],
        description="Allow the first L1B bridge pool II/VI; at least one is populated",
    )
    bridge_with_iv = fit_required_optional_model(
        "bridge_ii_iv_vi",
        x,
        y,
        fps,
        X_base,
        y_base,
        sse_base,
        required_indices=species_indices(["I"]),
        optional_indices=species_indices(["IV"]),
        choice_groups=[
            nonempty_subsets(species_indices(["II", "VI"])),
            nonempty_subsets(species_indices(["VII", "VIII"])),
        ],
        description="Allow parallel/coupling-active species IV after the II/VI bridge",
    )
    bridge_with_iii_or_v = fit_required_optional_model(
        "bridge_ii_iv_vi_plus_iii_or_v",
        x,
        y,
        fps,
        X_base,
        y_base,
        sse_base,
        required_indices=species_indices(["I"]),
        optional_indices=species_indices(["IV", "III", "V"]),
        choice_groups=[
            nonempty_subsets(species_indices(["II", "VI"])),
            nonempty_subsets(species_indices(["VII", "VIII"])),
        ],
        description="Allow III and-or V after the II/VI bridge and the IV test",
    )
    constrained = fit_constrained_layer_model(x, y, fps, X_base, y_base, sse_base)
    staged_models = [
        minimal,
        bridge_ii_vi,
        bridge_with_iv,
        bridge_with_iii_or_v,
    ]
    n = len(y)
    for previous, current in zip(staged_models, staged_models[1:]):
        current["incremental_vs_previous"] = incremental_stats(previous, current, n)
    return {
        "exp": exp,
        "fit_mask": mask,
        "x": x,
        "y": y,
        "fingerprints": fps,
        "best": minimal,
        "minimal": minimal,
        "staged_models": staged_models,
        "forced_bridge_ii_vi": forced_bridge_ii_vi,
        "bridge_ii_vi": bridge_ii_vi,
        "bridge_with_iv": bridge_with_iv,
        "bridge_with_iii_or_v": bridge_with_iii_or_v,
        "constrained": constrained,
        "unconstrained": unconstrained,
        "base": y_base,
        "sse_base": sse_base,
        "r2": minimal["r2"],
        "r2_base": r2_base,
        "f_value": minimal["f_value"],
        "dfn": minimal["dfn"],
        "dfd": minimal["dfd"],
        "p_value": minimal["p_value"],
    }


def component_tables(result: dict, model: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    x = result["x"]
    y = result["y"]
    labels = model["labels"]
    coefs = model["coefs"]
    basis_signed = model["global_sign"] * model["basis"]
    contributions = basis_signed * coefs[2:]
    rows = []
    for spec, label, coef, contrib in zip(FINGERPRINTS, labels, coefs[2:], contributions.T):
        active = spec.species in [FINGERPRINTS[idx].species for idx in model["active_indices"]]
        rows.append(
            {
                "label": label,
                "region": spec.region,
                "species": spec.species,
                "hbond_character": spec.hbond_character,
                "nonnegative_weight": coef,
                "rms_contribution": float(np.sqrt(np.mean(contrib**2))),
                "active": bool(active),
            }
        )
    comp = pd.DataFrame(rows)
    total_rms = comp["rms_contribution"].sum()
    comp["rms_fraction"] = comp["rms_contribution"] / total_rms if total_rms > 0 else 0.0
    curve = pd.DataFrame(
        {
            "wavenumber_cm-1": x,
            "imag_exp_smooth": y,
            "imag_fit": model["fit"],
            "imag_baseline": result["base"],
            "residual": y - model["fit"],
        }
    )
    for label, contrib in zip(labels, contributions.T):
        curve[f"component_{label}"] = contrib
    return comp, curve, contributions


def save_overlay(
    result: dict,
    model: dict,
    comp: pd.DataFrame,
    contributions: np.ndarray,
    filename: str,
    title: str | None = None,
    fit_label: str | None = None,
) -> None:
    x = result["x"]
    y = result["y"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    axes[0].plot(x, y, color="#b93d3d", lw=1.4, label="SiO2/water Im data, smoothed")
    axes[0].plot(x, model["fit"], color="#301414", lw=2.2, label=fit_label or "paper-fingerprint fit")
    axes[0].plot(x, result["base"], color="#777777", lw=1.0, ls="--", label="linear background only")
    axes[0].set_ylabel("Im(chi), data units")
    axes[0].legend(frameon=False, loc="best")
    axes[0].set_title(
        (title or model.get("description", "SiO2/water paper-fingerprint fit"))
        + " "
        f"(R2={model['r2']:.3f}, p={p_text(model['p_value'])})"
    )

    colors = {
        "L1 Region A": "#8f4d68",
        "L1 Region B": "#356a8a",
        "L2": "#6b8e3d",
    }
    for spec, contrib, active in zip(FINGERPRINTS, contributions.T, comp["active"]):
        if not active:
            continue
        axes[1].plot(x, contrib, lw=1.45, color=colors.get(spec.region, "#555555"), alpha=0.9, label=f"{spec.species}: {spec.region}")
    axes[1].axhline(0, color="#333333", lw=0.8, alpha=0.5)
    axes[1].set_ylabel("Fingerprint contributions")
    axes[1].legend(frameon=False, loc="best", fontsize=8, ncol=2)

    axes[2].plot(x, y - model["fit"], color="#333333", lw=1.1)
    axes[2].axhline(0, color="#333333", lw=0.8, alpha=0.5)
    axes[2].set_ylabel("Residual")
    axes[2].set_xlabel(r"Wavenumber (cm$^{-1}$)")

    for ax in axes:
        for xpos in (3400, 3470, 3660):
            ax.axvline(xpos, color="#999999", lw=0.7, alpha=0.35)
        ax.grid(True, color="#d0d0d0", alpha=0.42, lw=0.7)
    fig.savefig(OUT_DIR / filename, dpi=230)
    plt.close(fig)


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


def save_structure_fit_figure(result: dict, model: dict, comp: pd.DataFrame, contributions: np.ndarray) -> None:
    struct_dir = OUT_DIR / "structure_figures"
    panels = [
        ("A.png", "Layer 1A: species I fixed", "paper up direction -> SiO2 side"),
        ("B.png", "Layer 1B: II-VI allowed", "best constrained motif shown in spectrum"),
        ("L2.png", "Layer 2: VII/VIII required", "second-layer H-bonded response"),
    ]
    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.35], height_ratios=[1, 1, 1])
    for row, (fname, title, subtitle) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        img = crop_light_border(plt.imread(struct_dir / fname))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc="left")

    x = result["x"]
    y = result["y"]
    ax_fit = fig.add_subplot(gs[:2, 1])
    ax_fit.plot(x, y, color="#b93d3d", lw=1.4, label="SiO2/water Im data, smoothed")
    ax_fit.plot(x, model["fit"], color="#301414", lw=2.2, label="constrained sum of paper fingerprints")
    ax_fit.plot(x, result["base"], color="#777777", lw=1.0, ls="--", label="linear background")
    ax_fit.set_title(
        f"Constrained fingerprint overlay: R2={model['r2']:.3f}, "
        f"shift={model['shift_cm']:+.0f} cm$^{{-1}}$, sign={model['global_sign']:+.0f}"
    )
    ax_fit.set_ylabel("Im(chi), data units")
    ax_fit.legend(frameon=False, loc="best")
    ax_fit.grid(True, color="#d0d0d0", alpha=0.42, lw=0.7)

    ax_comp = fig.add_subplot(gs[2, 1], sharex=ax_fit)
    colors = {
        "L1 Region A": "#8f4d68",
        "L1 Region B": "#356a8a",
        "L2": "#6b8e3d",
    }
    active_species = set(comp.loc[comp["active"], "species"])
    for spec, contrib in zip(FINGERPRINTS, contributions.T):
        if spec.species not in active_species:
            continue
        ax_comp.plot(x, contrib, lw=1.45, color=colors.get(spec.region, "#555555"), alpha=0.9, label=f"{spec.species}: {spec.region}")
    ax_comp.axhline(0, color="#333333", lw=0.8, alpha=0.5)
    ax_comp.set_ylabel("Contribution")
    ax_comp.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax_comp.legend(frameon=False, loc="best", fontsize=8, ncol=3)
    ax_comp.grid(True, color="#d0d0d0", alpha=0.42, lw=0.7)

    for ax in (ax_fit, ax_comp):
        for xpos in (3400, 3470, 3660):
            ax.axvline(xpos, color="#999999", lw=0.7, alpha=0.35)
    fig.savefig(OUT_DIR / "paper_structures_and_constrained_fit.png", dpi=220)
    plt.close(fig)


def model_species(model: dict) -> list[str]:
    return [FINGERPRINTS[idx].species for idx in model["active_indices"]]


def save_staged_overlay(result: dict, staged_models: list[dict]) -> None:
    x = result["x"]
    y = result["y"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    axes[0].plot(x, y, color="#111111", lw=1.4, label="SiO2/water Im data, smoothed")
    axes[0].plot(x, result["base"], color="#999999", lw=1.0, ls="--", label="linear background")
    colors = ["#8f4d68", "#356a8a", "#c17c2f", "#496f3a"]
    labels = [
        "minimal: I + L2",
        "+ II + VI bridge",
        "+ IV parallel",
        "+ III and/or V",
    ]
    for model, color, label in zip(staged_models, colors, labels):
        axes[0].plot(x, model["fit"], lw=1.7, color=color, label=f"{label} (R2={model['r2']:.3f})")
    axes[0].set_ylabel("Im(chi), data units")
    axes[0].set_title("Staged water-structure models from the paper fingerprints")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    for model, color, label in zip(staged_models, colors, labels):
        axes[1].plot(x, y - model["fit"], lw=1.1, color=color, label=label)
    axes[1].axhline(0, color="#333333", lw=0.8, alpha=0.5)
    axes[1].set_ylabel("Residual")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)

    rows = []
    for i, model in enumerate(staged_models):
        rows.append(
            {
                "stage": i,
                "label": labels[i],
                "active_species": ", ".join(model_species(model)),
                "r2": model["r2"],
                "sse": model["sse"],
                "delta_sse": model.get("incremental_vs_previous", {}).get("delta_sse", np.nan),
            }
        )
    table_df = pd.DataFrame(rows)
    axes[2].axis("off")
    table = axes[2].table(
        cellText=[
            [
                row["label"],
                row["active_species"],
                f"{row['r2']:.3f}",
                "" if pd.isna(row["delta_sse"]) else f"{row['delta_sse']:.3g}",
            ]
            for _, row in table_df.iterrows()
        ],
        colLabels=["model", "species", "R2", "Delta SSE"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.23, 0.42, 0.12, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)

    for ax in axes[:2]:
        for xpos in (3400, 3470, 3660):
            ax.axvline(xpos, color="#999999", lw=0.7, alpha=0.35)
        ax.grid(True, color="#d0d0d0", alpha=0.42, lw=0.7)
    fig.savefig(OUT_DIR / "paper_fingerprint_staged_fit_overlay.png", dpi=230)
    plt.close(fig)


def save_staged_structure_fit_figure(result: dict, staged_models: list[dict]) -> None:
    struct_dir = OUT_DIR / "structure_figures"
    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.35], height_ratios=[1, 1, 1])
    panels = [
        ("A.png", "Start: L1A/species I", "SiO2-side analogue of paper-up OH"),
        ("L2.png", "Start: L2/species VII/VIII", "H-bonded second-layer partner"),
        ("B.png", "Optional bridge: L1B/species II-VI", "II/VI first, IV, then III/V"),
    ]
    for row, (fname, title, subtitle) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        img = crop_light_border(plt.imread(struct_dir / fname))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc="left")

    x = result["x"]
    y = result["y"]
    ax_fit = fig.add_subplot(gs[:2, 1])
    ax_fit.plot(x, y, color="#111111", lw=1.4, label="SiO2/water Im data, smoothed")
    colors = ["#8f4d68", "#356a8a", "#c17c2f", "#496f3a"]
    labels = ["I + L2", "+ II/VI", "+ IV", "+ III/V"]
    for model, color, label in zip(staged_models, colors, labels):
        ax_fit.plot(x, model["fit"], lw=1.7, color=color, label=f"{label}: {', '.join(model_species(model))}")
    ax_fit.set_title("Layer-building sequence using the paper's water structures")
    ax_fit.set_ylabel("Im(chi), data units")
    ax_fit.legend(frameon=False, fontsize=8, ncol=2)
    ax_fit.grid(True, color="#d0d0d0", alpha=0.42, lw=0.7)

    ax_res = fig.add_subplot(gs[2, 1], sharex=ax_fit)
    for model, color, label in zip(staged_models, colors, labels):
        ax_res.plot(x, y - model["fit"], lw=1.1, color=color, label=label)
    ax_res.axhline(0, color="#333333", lw=0.8, alpha=0.5)
    ax_res.set_ylabel("Residual")
    ax_res.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax_res.grid(True, color="#d0d0d0", alpha=0.42, lw=0.7)
    for ax in (ax_fit, ax_res):
        for xpos in (3400, 3470, 3660):
            ax.axvline(xpos, color="#999999", lw=0.7, alpha=0.35)
    fig.savefig(OUT_DIR / "paper_structures_and_staged_fit.png", dpi=220)
    plt.close(fig)


def save_layer_model(comp: pd.DataFrame) -> None:
    active = comp[comp["active"]]
    rows = []
    for depth_order, region, allowed, note in [
        (
            1,
            "L1 Region A",
            "I",
            "closest SiO2-side analogue of the protruding/free-OH motif; upward in the paper is read as toward SiO2",
        ),
        (
            2,
            "L1 Region B",
            "II, III, IV, V, VI",
            "dense interfacial H-bond sheet; includes parallel/coupling-active IV and connector motifs",
        ),
        (
            3,
            "L2",
            "VII, VIII",
            "one layer deeper; oppositely oriented motifs can cancel in the net SFG signal",
        ),
    ]:
        sub = comp[comp["region"] == region]
        selected = active[active["region"] == region]
        rows.append(
            {
                "depth_order_from_sio2": depth_order,
                "paper_region": region,
                "allowed_species": allowed,
                "selected_species": ", ".join(selected["species"]) if len(selected) else "",
                "rms_fraction": float(sub["rms_fraction"].sum()),
                "structural_interpretation": note,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "paper_constrained_layer_model.csv", index=False)


def write_outputs(result: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    fps = result["fingerprints"]
    pd.concat(fps, ignore_index=True).to_csv(OUT_DIR / "paper_sfg_fingerprints_digitized.csv", index=False)

    staged_models = result["staged_models"]
    minimal = result["minimal"]
    forced_bridge = result["forced_bridge_ii_vi"]
    flexible = result["constrained"]
    unconstrained = result["unconstrained"]

    stage_labels = [
        "0 minimal I+L2",
        "1 + II/VI bridge",
        "2 + IV parallel",
        "3 + III/V",
    ]
    summary_rows = []
    for i, (label, model) in enumerate(zip(stage_labels, staged_models)):
        comp, curve, contributions = component_tables(result, model)
        safe = model["kind"]
        comp.to_csv(OUT_DIR / f"paper_fingerprint_{safe}_weights.csv", index=False)
        curve.to_csv(OUT_DIR / f"paper_fingerprint_{safe}_curve.csv", index=False)
        save_overlay(
            result,
            model,
            comp,
            contributions,
            f"paper_fingerprint_{safe}_overlay.png",
            title=model["description"],
            fit_label=label,
        )
        inc = model.get("incremental_vs_previous", {})
        summary_rows.append(
            {
                "stage": i,
                "model_label": label,
                "kind": model["kind"],
                "description": model["description"],
                "active_species": ", ".join(model_species(model)),
                "shift_cm-1": model["shift_cm"],
                "global_sign": model["global_sign"],
                "parameter_count": model["parameter_count"],
                "sse": model["sse"],
                "r2": model["r2"],
                "aic": model["aic"],
                "bic": model["bic"],
                "f_vs_background": model["f_value"],
                "p_vs_background": model["p_value"],
                "delta_sse_vs_previous": inc.get("delta_sse", np.nan),
                "f_vs_previous": inc.get("f_value", np.nan),
                "p_vs_previous": inc.get("p_value", np.nan),
            }
        )

    stage_summary = pd.DataFrame(summary_rows)
    stage_summary.to_csv(OUT_DIR / "paper_fingerprint_staged_fit_summary.csv", index=False)
    save_staged_overlay(result, staged_models)
    save_staged_structure_fit_figure(result, staged_models)

    minimal_comp, minimal_curve, minimal_contrib = component_tables(result, minimal)
    minimal_comp.to_csv(OUT_DIR / "paper_fingerprint_fit_weights.csv", index=False)
    minimal_curve.to_csv(OUT_DIR / "paper_fingerprint_fit_curve.csv", index=False)
    save_overlay(
        result,
        minimal,
        minimal_comp,
        minimal_contrib,
        "paper_fingerprint_fit_overlay.png",
        title=minimal["description"],
        fit_label="minimal model",
    )

    flexible_comp, flexible_curve, flexible_contrib = component_tables(result, flexible)
    flexible_comp.to_csv(OUT_DIR / "paper_fingerprint_flexible_l1b_weights.csv", index=False)
    flexible_comp.to_csv(OUT_DIR / "paper_fingerprint_constrained_fit_weights.csv", index=False)
    flexible_curve.to_csv(OUT_DIR / "paper_fingerprint_flexible_l1b_curve.csv", index=False)
    flexible_curve.to_csv(OUT_DIR / "paper_fingerprint_constrained_fit_curve.csv", index=False)
    save_overlay(
        result,
        flexible,
        flexible_comp,
        flexible_contrib,
        "paper_fingerprint_flexible_l1b_overlay.png",
        title=flexible["description"],
        fit_label="flexible L1B control",
    )
    save_overlay(
        result,
        flexible,
        flexible_comp,
        flexible_contrib,
        "paper_fingerprint_constrained_fit_overlay.png",
        title=flexible["description"],
        fit_label="flexible L1B control",
    )
    save_structure_fit_figure(result, flexible, flexible_comp, flexible_contrib)
    save_layer_model(flexible_comp)

    if forced_bridge is not None:
        forced_comp, forced_curve, forced_contrib = component_tables(result, forced_bridge)
        forced_comp.to_csv(OUT_DIR / "paper_fingerprint_forced_bridge_ii_vi_weights.csv", index=False)
        forced_curve.to_csv(OUT_DIR / "paper_fingerprint_forced_bridge_ii_vi_curve.csv", index=False)
        save_overlay(
            result,
            forced_bridge,
            forced_comp,
            forced_contrib,
            "paper_fingerprint_forced_bridge_ii_vi_overlay.png",
            title=forced_bridge["description"],
            fit_label="forced II+VI diagnostic",
        )

    layer = minimal_comp.groupby("region", as_index=False).agg(
        rms_contribution=("rms_contribution", "sum"),
        active_species=("active", "sum"),
    )
    if layer["rms_contribution"].sum() > 0:
        layer["rms_fraction"] = layer["rms_contribution"] / layer["rms_contribution"].sum()
    layer.to_csv(OUT_DIR / "paper_fingerprint_layer_weights.csv", index=False)

    flexible_layer = flexible_comp.groupby("region", as_index=False).agg(
        rms_contribution=("rms_contribution", "sum"),
        active_species=("active", "sum"),
    )
    if flexible_layer["rms_contribution"].sum() > 0:
        flexible_layer["rms_fraction"] = flexible_layer["rms_contribution"] / flexible_layer["rms_contribution"].sum()
    flexible_layer.to_csv(OUT_DIR / "paper_fingerprint_constrained_layer_weights.csv", index=False)

    minimal_active = minimal_comp[minimal_comp["active"]].sort_values("rms_fraction", ascending=False)
    flexible_active = flexible_comp[flexible_comp["active"]].sort_values("rms_fraction", ascending=False)
    unconstrained_active = [FINGERPRINTS[idx].species for idx in unconstrained["active_indices"]]
    best_staged = min(staged_models, key=lambda m: m["bic"])
    best_r2_staged = max(staged_models, key=lambda m: m["r2"])
    report = [
        "# Gestufter Fit der Paper-SFG-Fingerprints an SiO2/Wasser",
        "",
        "## Methode",
        "",
        "- Die magenta Total-Kurven wurden direkt aus den EPS-Dateien des eigenen Papers digitisiert; sie wurden nicht analytisch neu gezeichnet.",
        "- Gefittet wurde nur 3300-3800 cm^-1, also der OH-Streckbereich. Die starken Strukturen unter 3000 cm^-1 werden hier nicht als Wasserorientierung verwendet.",
        "- Jede aktive Struktur bekommt ein nichtnegatives Gewicht; wenn ein Motiv in einem Modell genannt ist, muss es im Fit positiv populiert sein.",
        "- Ein konstanter plus linearer Hintergrund ist immer enthalten.",
        "- Fuer jedes Modell wurden ein globaler Frequenzshift und ein globales Vorzeichen getestet. Das Vorzeichen ist eine Phasen-/Normalenkonvention der Dateien; physikalisch wird die Cyran-Konvention benutzt: positives Im(chi) bedeutet H/OH netto in Richtung SiO2.",
        "- Orientierung: Was im Paper nach oben zeigt, wird hier als Richtung SiO2/Fenster gelesen.",
        "- p-Werte sind F-Test-Diagnostik gegen den linearen Hintergrund bzw. in der Stufentabelle gegen die vorherige Stufe. Wegen Nichtnegativitaet, digitisierten Kurven und reoptimiertem Shift sind sie als Modellvergleich zu lesen, nicht als strenger experimenteller Signifikanztest.",
        "",
        "## Stufenmodell",
        "",
        "| Stufe | aktive Spezies | R2 | Shift / Vorzeichen | p gegen Hintergrund | Delta SSE zur Vorstufe | p zur Vorstufe |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in stage_summary.iterrows():
        delta = "" if pd.isna(row["delta_sse_vs_previous"]) else f"{row['delta_sse_vs_previous']:.3g}"
        p_prev = "" if pd.isna(row["p_vs_previous"]) else p_text(float(row["p_vs_previous"]))
        report.append(
            f"| {row['model_label']} | {row['active_species']} | {row['r2']:.4f} | "
            f"{row['shift_cm-1']:+.1f} / {row['global_sign']:+.0f} | "
            f"{p_text(float(row['p_vs_background']))} | {delta} | {p_prev} |"
        )
    report.extend(
        [
            "",
            "## Minimalmodell I + L2",
            "",
            f"- R2 Minimalmodell: {minimal['r2']:.4f}",
            f"- SSE Minimalmodell: {minimal['sse']:.6g}",
            f"- Aktive Spezies: {', '.join(model_species(minimal))}",
            f"- Bestes BIC innerhalb der gestuften Modelle: {best_staged['kind']} ({', '.join(model_species(best_staged))})",
            f"- Bestes R2 innerhalb der gestuften Modelle: {best_r2_staged['kind']} ({', '.join(model_species(best_r2_staged))})",
            "",
            "| species | region | rms fraction | structural note |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for _, row in minimal_active.iterrows():
        report.append(
            f"| {row['species']} | {row['region']} | {row['rms_fraction']:.3f} | {row['hbond_character']} |"
        )
    report.extend(
        [
            "",
            "## Flexible L1B-Kontrolle",
            "",
            "- Als Kontrollfit wurde auch eine freie, aber schichtbesetzte L1B-Mischung gerechnet: I ist gesetzt, mindestens eine Spezies aus II-VI ist vorhanden, und L2 enthaelt VII und/oder VIII.",
            f"- R2 flexible L1B-Kontrolle: {flexible['r2']:.4f}",
            f"- Aktive Spezies flexible L1B-Kontrolle: {', '.join(model_species(flexible))}",
            f"- R2 komplett unbeschraenkter Fingerprintfit: {unconstrained['r2']:.4f}; aktive Spezies: {', '.join(unconstrained_active)}",
            "",
            "| species | region | rms fraction | structural note |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for _, row in flexible_active.iterrows():
        report.append(
            f"| {row['species']} | {row['region']} | {row['rms_fraction']:.3f} | {row['hbond_character']} |"
        )
    report.extend(
        [
            "",
            "## Strukturinterpretation",
            "",
            "- Die Tiefeninformation kommt aus Fig. 1 des Papers: SiO2-Seite -> L1/Region A -> L1/Region B -> L2 -> tieferes Wasser. Es werden keine harten 3-Angstrom-Slabs angenommen.",
            "- Das Minimalmodell koppelt L1A/species I mit L2/species VII/VIII. Strukturell ist das der kleinste Wasser-only Ansatz, in dem die SiO2-seitige schwach/frei-OH-artige Orientierung und ein tieferes H-Brueckennetz gleichzeitig vorhanden sind.",
            "- II und VI sind der erste sinnvolle L1B-Zusatz, weil II an die L1A-Seite und VI an L2 koppeln kann. IV testet danach die parallel orientierte, lokal schwache aber kopplungsaktive Population. III/V testen zuletzt weitere Region-B-H-Brueckenvarianten.",
            (
                f"- Wenn II und VI beide zwingend erzwungen werden, faellt R2 auf {forced_bridge['r2']:.4f}. "
                "Der II/VI-Pool-Fit waehlt deshalb nur die spektral getragene Brueckenkomponente."
                if forced_bridge is not None
                else "- Wenn II und VI beide zwingend erzwungen werden, findet dieses Fenster keine zulaessige positive Loesung; der II/VI-Pool wird deshalb als Auswahlpool behandelt."
            ),
            "- Die Gewichte sind spektrale Fingerprint-Gewichte, keine direkten Molekuelzahlen.",
            "- Frequenzen unter 3000 cm^-1 bleiben ausserhalb dieses Wasserstrukturmodells; wenn sie im Rohspektrum dominieren, sind sie fuer diese Wasser-OH-Strukturzuordnung nicht belastbar.",
        ]
    )
    (OUT_DIR / "paper_fingerprint_fit_report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT_DIR / "paper_fingerprint_staged_fit_report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT_DIR / "paper_fingerprint_constrained_fit_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    result = fit_model()
    write_outputs(result)
    print(f"minimal I+L2 R2 {result['minimal']['r2']:.5f}")
    print(f"II/VI pool R2 {result['bridge_ii_vi']['r2']:.5f}")
    print(f"flexible L1B control R2 {result['constrained']['r2']:.5f}")
    print(f"flexible L1B control p {p_text(result['constrained']['p_value'])}")
    print(f"unconstrained R2 {result['unconstrained']['r2']:.5f}")
    print(f"Wrote {OUT_DIR / 'paper_fingerprint_staged_fit_report.md'}")
    print(f"Wrote {OUT_DIR / 'paper_structures_and_staged_fit.png'}")


if __name__ == "__main__":
    main()
