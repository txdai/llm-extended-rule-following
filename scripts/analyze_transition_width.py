"""Width of the behavioral transition from the decoding replicates.

    python scripts/analyze_transition_width.py

Pools the five rescans of each replicate model (results/temperature-replicates/
rep-*/<slug>/trials/*.jsonl) into a per-length accuracy out of five, smooths it
with a centred window, and reports the span over which the smoothed accuracy
falls from 0.8 to 0.2, alongside the catalogue CC. Writes
paper/generated/transition_width_table.tex.
"""
from __future__ import annotations
import csv, json, glob, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "results/temperature-replicates"
SCC = ROOT / "results/counting-data/scc.csv"
OUT = ROOT / "paper/generated/transition_width_table.tex"
HALF = 5  # smoothing half-window in lengths

def catalogue_cc():
    """Catalogue CC keyed by model, using each replicate model's catalogue variant."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from counting.analysis.replicates import REPLICATE_MODELS
    variant = {m: v for m, v, _, _ in REPLICATE_MODELS}
    rows = {(r["model"], r["variant"]): r for r in csv.DictReader(open(SCC))}
    out = {}
    for m, v in variant.items():
        r = rows.get((m, v))
        if r and r.get("scc") not in (None, ""):
            out[m] = int(float(r["scc"]))
    return out

def load():
    per = defaultdict(lambda: defaultdict(list))   # model -> length -> [bool]
    for f in glob.glob(str(REP / "rep-*" / "*" / "trials" / "*.jsonl")):
        for line in open(f):
            try: d = json.loads(line)
            except json.JSONDecodeError: continue
            if d.get("phase") == "preflight": continue
            per[d["model"]][int(d["expected"])].append(bool(d["exact_match"]))
    return per

def smoothed(acc_by_len):
    Ls = sorted(acc_by_len); out = {}
    for L in Ls:
        win = [v for l in range(L - HALF, L + HALF + 1) for v in acc_by_len.get(l, [])]
        out[L] = sum(win) / len(win) if win else None
    return out

def width(sm):
    Ls = [L for L in sorted(sm) if sm[L] is not None]
    hi = max([L for L in Ls if sm[L] >= 0.8], default=None)
    if hi is None: return None, None, None
    lo = min([L for L in Ls if L > hi and sm[L] <= 0.2], default=None)
    return hi, lo, (lo - hi) if lo else None

if __name__ == "__main__":
    cc = catalogue_cc(); per = load(); rows = []
    for model, acc in sorted(per.items()):
        sm = smoothed(acc); hi, lo, w = width(sm)
        probes = sum(len(v) for v in acc.values())
        c = cc.get(model)
        rows.append((model, c, hi, lo, w, (w / c) if (w and c) else None, probes))
        print(f"{model:40s} CC={c}  last>=0.8 at {hi}  first<=0.2 at {lo}  width={w}  width/CC={(w/c) if (w and c) else float('nan'):.2f}  probes={probes}")
    ws = sorted(r[4] for r in rows if r[4]); fr = sorted(r[5] for r in rows if r[5])
    print(f"median width {ws[len(ws)//2]} items; median width/CC {fr[len(fr)//2]:.2f}; range {fr[0]:.2f}-{fr[-1]:.2f}")
    with open(OUT, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\caption{\\textbf{Width of the behavioral transition.} Per-length accuracy pooled over the five rescans of Appendix~\\ref{app:replicates} and smoothed over $\\pm 5$ lengths; the width is the span from the last length at which smoothed accuracy is at least 0.8 to the first at which it is at most 0.2.}\n\\label{tab:transition_width}\n\\begin{tabular}{lrrrrr}\n\\toprule\nModel & CC & last $\\ge 0.8$ & first $\\le 0.2$ & width & width / CC \\\\\n\\midrule\n")
        for m, c, hi, lo, w, fr_, _ in rows:
            name = m.split("/")[-1].replace("_", "\\_")
            f.write(f"\\texttt{{{name}}} & {c} & {hi if hi else '--'} & {lo if lo else '--'} & {w if w else '--'} & {fr_:.2f} \\\\\n" if fr_ else f"\\texttt{{{name}}} & {c} & {hi if hi else '--'} & {lo if lo else '--'} & -- & -- \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("wrote", OUT)
