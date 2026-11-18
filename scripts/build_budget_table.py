#!/usr/bin/env python3
"""Emit paper/generated/budget32k_table.tex: the sixteen budget-bounded reasoning
configurations rescanned at a 32,768-token cap (scripts/run_budget_rerun.py).

    python scripts/build_budget_table.py
"""
from __future__ import annotations

import csv
import glob
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from counting.analysis.capacity import fit_transition  # noqa: E402
from counting.analysis.reasoning_pairs import hybrid_pairs  # noqa: E402

OUT = REPO / "paper" / "generated" / "budget32k_table.tex"
LABEL = {"mistake": "resolved", "budget": "budget", "range_end": "scan end"}


def main() -> int:
    pairs = {p.model: p for p in hybrid_pairs() if p.classification == "verified"}
    # The released tables now carry the 32k measurement for these
    # configurations, so the 8k bound is refitted from the archived catalogue
    # scan (results/counting-data/models/<model>__think.json, untouched).
    def fit_8k(model: str) -> int:
        path = REPO / "results/counting-data/models" / (model.replace("/", "__") + "__think.json")
        payload = json.loads(path.read_text())
        trials = sorted([t for t in payload.get("trials", []) if t.get("phase") == "scan"],
                        key=lambda t: t["center_length"])
        t = fit_transition([x["center_length"] for x in trials], [bool(x.get("exact_match")) for x in trials],
                           [x.get("parsed") is None for x in trials])
        return int(t.scc or 0)
    rows = []
    for path in sorted(glob.glob(str(REPO / "results/budget-32k/*/trials/*.jsonl"))):
        recs = [json.loads(l) for l in open(path) if l.strip()]
        scan = sorted([r for r in recs if r.get("phase") == "scan" and r.get("response_status") != "error"],
                      key=lambda r: r["center_length"])
        L = [r["center_length"] for r in scan]
        ok = [bool(r.get("exact_match")) for r in scan]
        err = [r.get("parsed") is None for r in scan]
        hits = [r for r in scan if str(r.get("response_detail", "")).startswith("max_output")]
        t = fit_transition(L, ok, err)
        model = recs[0]["model"].split("#")[0]
        p = pairs[model]
        rows.append((model, p.cc_nothink, fit_8k(model), t.scc, t.ci_low, t.ci_high,
                     LABEL.get(t.bound_by, t.bound_by), len(hits), min((r["expected"] for r in hits), default=None)))
    lines = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{\textbf{Budget-bounded reasoning configurations rescanned at a 32,768-token cap.} The sixteen verified hybrid pairs whose reasoning member exhausted the catalogue's 8,192-token cap before failing on the count, rescanned with the cap raised fourfold and everything else unchanged. CC$_{\text{off}}$ is the no-reasoning capacity, CC$_{\text{8k}}$ the catalogue bound, CC$_{\text{32k}}$ the rescanned value. Bound: how the 32k scan ended. Hits: probes that exhausted 32,768 tokens, and the smallest count at which it first happened.}",
             r"\label{tab:budget32k}", r"\begin{tabular}{lrrlrr}", r"\toprule",
             r"Model & CC$_{\text{off}}$ & CC$_{\text{8k}}$ & CC$_{\text{32k}}$ [95\% CI] & Bound & Hits (first at) \\", r"\midrule"]
    for m, nt, c8, c32, lo, hi, bound, nh, first in rows:
        lines.append(f"\\texttt{{{m.split('/')[-1]}}} & {nt} & $\\geq${c8} & {'$\\geq$' if bound != 'resolved' else ''}{c32} [{lo}, {hi}] & {bound} & {nh} ({first}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n")
    resolved = sum(1 for r in rows if r[6] == "resolved")
    d = [(r[3] if r[3] else r[2]) - r[1] for r in rows]
    print(f"wrote {OUT}: {resolved}/{len(rows)} resolved at 32k; median dCC among the 16 = {statistics.median(d):+.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
