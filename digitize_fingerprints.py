"""Extract vector Total curves using the EPS frequency ticks and zero lines.

The curves extend approximately 2900–3900 cm^-1; their endpoints must not be
mistaken for the labeled 3000 and 3800 cm^-1 ticks. EPS vertical amplitudes are
arbitrary, since each fingerprint is subsequently RMS normalized for fitting.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCES = (("SFG_A.eps", ("I",)),
           ("SFG_B.eps", ("II", "III", "IV", "V", "VI")),
           ("SFG_L2.eps", ("VII", "VIII")))


def path_points(text):
    tokens = re.findall(r"-?\d+(?:\.\d+)?|[A-Za-z]+|%[^\n]*", text)
    points, stack, current = [], [], None
    for token in tokens:
        try:
            stack.append(float(token))
            continue
        except ValueError:
            pass
        if token in ("M", "N") and len(stack) >= 2:
            current = tuple(stack[-2:])
            points.append(current)
        elif token == "L" and len(stack) >= 2:
            current = tuple(stack[-2:])
            points.append(current)
        elif token == "V" and current is not None and len(stack) >= 2:
            current = tuple(np.asarray(current) + stack[-2:])
            points.append(current)
        stack = []
    return np.asarray(points)


def extract_file(path, motifs):
    text = path.read_text()
    markers = [m for m in re.finditer(r"% Begin plot #1\n(?:(?!% End plot #1).)*?1\.00 0\.00 1\.00 C", text, re.S)]
    if len(markers) != len(motifs):
        raise ValueError(f"Unexpected Total panel count in {path}")
    outputs, calibration = [], []
    for motif, marker in zip(motifs, markers):
        # Frequency labels occur only in the bottom row; Total is the rightmost column.
        ticks = {}
        for tick in (3000, 3800):
            matches = re.findall(r"(-?\d+)\s+(-?\d+)\s+M\s*\[\s*\[\s*\(Arial\)[^\n]*\(\s*" + str(tick) + r"\)\]", text)
            if not matches:
                raise ValueError(f"Missing frequency tick {tick}")
            ticks[tick] = float(matches[-1][0])
        end = text.index("% End plot #1", marker.end())
        curve_block = text[marker.end():end]
        # The filled polygon repeats the curve and adds artificial closing lines.
        if "PolyFill" in curve_block:
            curve_block = curve_block.split("PolyFill", 1)[1]
        pts = path_points(curve_block)
        zero_start = text.index("% Begin plot #2", end)
        zero_end = text.index("% End plot #2", zero_start)
        zero_points = path_points(text[zero_start:zero_end])
        if len(zero_points) != 2 or abs(zero_points[0,1]-zero_points[1,1]) > 1e-8:
            raise ValueError("Expected an explicit horizontal zero line")
        zero = float(zero_points[0,1])
        x = 3000 + (pts[:,0]-ticks[3000]) * 800 / (ticks[3800]-ticks[3000])
        y = pts[:,1] - zero
        data = pd.DataFrame({"species": motif, "wavenumber_cm-1": x, "fingerprint": y})
        data = data.groupby(["species", "wavenumber_cm-1"], as_index=False).mean()
        yscale = np.sqrt(np.mean(data.fingerprint**2))
        data.fingerprint /= yscale
        outputs.append(data)
        calibration.append(dict(species=motif, source=path.name, tick_3000_eps=ticks[3000],
                                tick_3800_eps=ticks[3800], zero_eps=zero,
                                eps_amplitude_rms=yscale, frequency_start=x.min(),
                                frequency_end=x.max()))
    return outputs, calibration


def run():
    data, calibration = [], []
    for source, motifs in SOURCES:
        curves, records = extract_file(ROOT / "SFG_Structure" / source, motifs)
        data.extend(curves)
        calibration.extend(records)
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    pd.concat(data).to_csv(out / "paper_sfg_fingerprints_digitized.csv", index=False)
    pd.DataFrame(calibration).to_csv(out / "fingerprint_calibration.csv", index=False)
    print("Extracted eight Total curves from explicit frequency ticks and zero lines.")


if __name__ == "__main__":
    run()
