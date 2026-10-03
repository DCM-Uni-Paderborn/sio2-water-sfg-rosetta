"""Refresh marked numerical table blocks in the supplementary LaTeX source."""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"


def tabular(columns, header, rows):
    return ("\\begin{tabular}{" + columns + "}\n\\toprule\n" +
            " & ".join(header) + " \\\\\n\\midrule\n" +
            "\n".join(" & ".join(map(str,row)) + " \\\\" for row in rows) +
            "\n\\bottomrule\n\\end{tabular}")


def run():
    tables = {}
    d = pd.read_csv(ANALYSIS / "newest_window_choice_summary.csv")
    rows = [[f"{int(r.window_start_cm_1)}--{int(r.window_end_cm_1)}",int(r.points),
             f"{r.raw_minus_smoothed_rms:.4f}",f"{r.relative_fluctuation:.3f}"] for r in d.itertuples()]
    tables["window_quality"] = tabular("lrrr",[r"Window / \si{cm^{-1}}","Points","Fluctuation RMS / a.u.","Relative fluctuation"],rows)
    d = pd.read_csv(ANALYSIS / "fingerprint_calibration.csv")
    tables["calibration"] = tabular("llr",["Motif","Source EPS","Zero coordinate"],
                                   [[r.species,r.source.replace("_",r"\_"),int(r.zero_eps)] for r in d.itertuples()])
    d = pd.read_csv(ANALYSIS / "newest_exact_motif_candidate_comparison.csv")
    selection = [(3300,"I+VIII"),(3300,"VI+VIII"),(3300,"I+II+VIII"),
                 (3300,"I+VI+VIII"),(3300,"I+II+VI+VIII"),(3300,"I+VI+VII"),
                 (3300,"I+VI+VII+VIII"),(3400,"I+VIII"),(3400,"I+VI+VIII"),
                 (3400,"I+IV+VI+VIII"),(3400,"I+VI+VII+VIII"),
                 (3200,"I+VI+VIII"),(3200,"I+VI+VII+VIII")]
    rows=[]
    for window,model in selection:
        r=d[(d.window_start_cm_1==window)&(d.window_end_cm_1==3800)&(d.model==model)].iloc[0]
        rows.append([f"{window}--3800",model,r.active_motifs,f"{r.r2:.6f}",f"{r.sse:.6f}",f"{r.shift_cm_1:+.0f}"])
    tables["candidates"] = tabular("lllrrr",[r"Window / \si{cm^{-1}}","Allowed","Active",r"$R^2$","SSE",r"$\Delta$ / \si{cm^{-1}}"],rows)
    d = pd.read_csv(ANALYSIS / "newest_all_model_weights.csv")
    d=d[(d.smoothing_points==17)&d.window.isin(["3300-3800","3400-3800"])&d.model.isin(["I+VI+VIII","I+VI+VII+VIII"])]
    rows=[[r.window.replace("-","--"),r.model,r.species,f"{r.weight:.6f}",f"{r.rms_fraction:.4f}"] for r in d.itertuples()]
    tables["weights"] = tabular("lllrr",[r"Window / \si{cm^{-1}}","Allowed","Motif","Weight","RMS fraction"],rows)
    d = pd.read_csv(ANALYSIS / "newest_smoothing_window_sensitivity.csv")
    rows=[]
    for window in (3300,3400):
        a=d[(d.model=="I+VI+VIII")&(d.window_start_cm_1==window)&(d.window_end_cm_1==3800)]
        rows.append([f"{window}--3800"]+[f"{a[a.smoothing_points==q].r2.iloc[0]:.6f}" for q in (1,7,11,17,25,33)])
    tables["smoothing"] = tabular("lrrrrrr",[r"Window / \si{cm^{-1}}","1 pt","7 pt","11 pt","17 pt","25 pt","33 pt"],rows)
    path=ROOT/"manuscript/supporting_information.tex"
    text=path.read_text()
    for name,value in tables.items():
        pattern=r"(% BEGIN GENERATED "+name+r"\n).*?(% END GENERATED "+name+r")"
        text,count=re.subn(pattern,lambda m:m.group(1)+value+"\n"+m.group(2),text,flags=re.S)
        if count!=1:raise ValueError(f"Expected one table block for {name}")
    path.write_text(text)
    print("Refreshed five SI tables from the calibrated analysis records.")


if __name__ == "__main__":
    run()
