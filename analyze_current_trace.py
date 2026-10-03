"""Reproduce the Communication's fits to the author-supplied single-channel trace.

Only motif coefficients are nonnegative. The two baseline coefficients are free.
All faces of each small nonnegative cone, including zero-weight boundaries, are
examined; this avoids mistaking an infeasible interior solution for a bad model.
"""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"
WINDOWS = ((3200, 3800), (3300, 3800), (3400, 3800), (3350, 3700))
WIDTHS = (1, 7, 11, 17, 25, 33)
MODELS = ("I+VIII", "VI+VIII", "I+II+VIII", "I+VI+VIII",
          "I+II+VI+VIII", "I+IV+VI+VIII", "I+VI+VII", "I+VI+VII+VIII")


def moving_average(values, width):
    if width < 1 or width % 2 != 1:
        raise ValueError("The moving-average width must be a positive odd integer")
    return np.convolve(np.pad(values, width // 2, mode="edge"),
                       np.ones(width) / width, mode="valid")


def load_trace():
    x = np.loadtxt(ROOT / "newest_data/xaxis_fud.txt")
    y = np.loadtxt(ROOT / "newest_data/heat2_fud.txt")
    if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
        raise ValueError("The trace requires paired, finite, increasing frequencies")
    finite = np.isfinite(y)
    filled = np.interp(x, x[finite], y[finite])
    trace = pd.DataFrame({"wavenumber_cm-1": x, "raw_trace": y,
                          "finite": finite, "interp_trace": filled})
    for width in WIDTHS:
        trace[f"smooth_{width}"] = moving_average(filled, width)
    return trace


def load_fingerprints():
    fp = pd.read_csv(OUT / "paper_sfg_fingerprints_digitized.csv")
    return fp.groupby(["species", "wavenumber_cm-1"], as_index=False)["fingerprint"].mean()


def basis_at(fp, x, motifs, shift):
    cols = []
    for motif in motifs:
        d = fp[fp.species == motif].sort_values("wavenumber_cm-1")
        f = np.interp(x, d["wavenumber_cm-1"] + shift, d.fingerprint, left=0, right=0)
        rms = np.sqrt(np.mean(f * f))
        if rms == 0:
            raise ValueError(f"Zero fingerprint for {motif}")
        cols.append(f / rms)
    return np.column_stack(cols)


def solve_nonnegative_with_baseline(a, y, baseline):
    """Exact small NNLS by enumerating faces after projecting out the baseline."""
    q = np.linalg.qr(baseline, mode="reduced")[0]
    ar = a - q @ (q.T @ a)
    yr = y - q @ (q.T @ y)
    best = None
    for count in range(a.shape[1] + 1):
        for active in combinations(range(a.shape[1]), count):
            weights = np.zeros(a.shape[1])
            if active:
                w = np.linalg.lstsq(ar[:, active], yr, rcond=None)[0]
                if np.any(w < -1e-10):
                    continue
                weights[list(active)] = np.maximum(w, 0)
            residual = yr - ar @ weights
            sse = float(residual @ residual)
            if best is None or sse < best[0]:
                best = sse, weights
    sse, weights = best
    beta = np.linalg.lstsq(baseline, y - a @ weights, rcond=None)[0]
    fitted = a @ weights + baseline @ beta
    # Check optimality: gradient >= 0 at zero weights, gradient = 0 elsewhere.
    gradient = a.T @ (fitted - y)
    if (np.min(gradient) < -1e-7 or
            np.max(np.abs(gradient[weights > 1e-9]), initial=0) > 1e-7):
        raise AssertionError("NNLS optimality conditions failed")
    return sse, weights, beta, fitted


def fit_model(trace, fp, window, width, model, max_shift=20):
    d = trace[trace["wavenumber_cm-1"].between(*window)].copy()
    if not d.finite.all():
        raise ValueError("Fits must contain finite original measurements")
    x = d["wavenumber_cm-1"].to_numpy()
    y = d[f"smooth_{width}"].to_numpy()
    t = (x - x.mean()) / np.ptp(x)
    baseline = np.column_stack([np.ones(len(x)), t])
    motifs = model.split("+")
    best = None
    for shift in range(-max_shift, max_shift + 1):
        basis = basis_at(fp, x, motifs, shift)
        for sign in (1, -1):
            a = sign * basis
            sse, w, beta, fitted = solve_nonnegative_with_baseline(a, y, baseline)
            if best is None or sse < best["sse"] - 1e-14:
                best = dict(sse=sse, weights=w, beta=beta, fitted=fitted,
                            shift=shift, sign=sign, components=a*w)
    curve = pd.DataFrame({"wavenumber_cm-1": x, "raw_trace": d.raw_trace.to_numpy(),
                          "smooth_trace": y, "fit": best["fitted"],
                          "baseline": baseline @ best["beta"],
                          "residual": y - best["fitted"]})
    for i, motif in enumerate(motifs):
        curve[f"component_{motif}"] = best["components"][:, i]
    active = [m for m, w in zip(motifs, best["weights"]) if w > 1e-9]
    # R² describes line-shape agreement; smoothed samples are not independent.
    stats = dict(window_start_cm_1=window[0], window_end_cm_1=window[1],
                 smoothing_points=width, model=model, active_motifs="+".join(active),
                 r2=1-best["sse"] / np.sum((y-y.mean())**2), sse=best["sse"],
                 shift_cm_1=best["shift"], global_sign=best["sign"], points=len(x))
    weights = pd.DataFrame({"species": motifs, "weight": best["weights"],
                            "rms_fraction": best["weights"] / best["weights"].sum()})
    return stats, curve, weights


def run():
    OUT.mkdir(exist_ok=True)
    trace, fp = load_trace(), load_fingerprints()
    trace.to_csv(OUT / "newest_trace_processed.csv", index=False)
    results, cache = [], {}
    for window in WINDOWS:
        for width in WIDTHS:
            for model in MODELS:
                result = fit_model(trace, fp, window, width, model)
                results.append(result[0])
                cache[(window, width, model)] = result
        print(f"Completed {window[0]}–{window[1]} cm^-1", flush=True)
    summary = pd.DataFrame(results)
    summary.to_csv(OUT / "newest_smoothing_window_sensitivity.csv", index=False)
    candidates = summary[summary.smoothing_points == 17]
    candidates.to_csv(OUT / "newest_exact_motif_candidate_comparison.csv", index=False)
    primary_key = ((3300, 3800), 17, "I+VI+VIII")
    stats, curve, weights = cache[primary_key]
    curve.to_csv(OUT / "newest_primary_3300_3800_I_VI_VIII_curve.csv", index=False)
    weights.to_csv(OUT / "newest_primary_3300_3800_I_VI_VIII_weights.csv", index=False)
    pd.DataFrame([stats]).to_csv(OUT / "newest_primary_3300_3800_I_VI_VIII_stats.csv", index=False)
    primary_models = ("I+VI+VIII", "I+VI+VII+VIII")
    pd.DataFrame([cache[((3300, 3800), 17, m)][0] for m in primary_models]).to_csv(
        OUT / "newest_primary_model_comparison.csv", index=False)
    l3 = ("I+VI+VII",) + primary_models
    pd.DataFrame([cache[((3300, 3800), 17, m)][0] for m in l3]).to_csv(
        OUT / "newest_l3_vii_viii_control.csv", index=False)
    pd.concat([cache[((3300, 3800), 17, m)][1].assign(model=m) for m in l3]).to_csv(
        OUT / "newest_l3_vii_viii_3300_3800_curves.csv", index=False)
    # Publish curves for every window/model used in sensitivity figures.
    pd.concat([cache[(window, 17, m)][1].assign(model=m, window=f"{window[0]}-{window[1]}")
               for window in WINDOWS for m in MODELS]).to_csv(
        OUT / "newest_candidate_curves.csv", index=False)
    pd.concat([value[2].assign(model=key[2], smoothing_points=key[1],
                              window=f"{key[0][0]}-{key[0][1]}")
               for key, value in cache.items()]).to_csv(
        OUT / "newest_all_model_weights.csv", index=False)
    diagnostics = []
    for window in ((3000,3200),(3200,3300),(3300,3800),(3400,3800),(3350,3700)):
        d = trace[trace["wavenumber_cm-1"].between(*window) & trace.finite]
        rms = np.sqrt(np.mean((d.raw_trace-d.smooth_17)**2))
        diagnostics.append(dict(window_start_cm_1=window[0], window_end_cm_1=window[1],
                                points=len(d), raw_minus_smoothed_rms=rms,
                                relative_fluctuation=rms/np.ptp(d.smooth_17)))
    pd.DataFrame(diagnostics).to_csv(OUT / "newest_window_choice_summary.csv", index=False)
    # Scientific regression checks, including the boundary that previously failed.
    assert abs(stats["r2"] - 0.9918107440661879) < 1e-10
    for window in WINDOWS:
        for width in WIDTHS:
            for small, large in (("I+VI+VIII", "I+II+VI+VIII"),
                                 ("I+VI+VIII", "I+VI+VII+VIII"),
                                 ("I+VIII", "I+II+VIII")):
                assert cache[(window,width,large)][0]["sse"] <= cache[(window,width,small)][0]["sse"] + 1e-10
    print("Calibrated fit reproduced; all NNLS and nested-model checks passed.")


def run_dictionary_scan():
    trace, fp = load_trace(), load_fingerprints()
    motifs = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    rows = []
    for window in ((3300, 3800), (3400, 3800)):
        for count in (1, 2, 3):
            for subset in combinations(motifs, count):
                rows.append(fit_model(trace, fp, window, 17, "+".join(subset))[0])
    pd.DataFrame(rows).to_csv(OUT / "dictionary_small_model_scan.csv", index=False)
    print("Completed all 92 one-, two-, and three-motif subsets per window.")


if __name__ == "__main__":
    run()
    run_dictionary_scan()
