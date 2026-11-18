#!/usr/bin/env python3
"""Emit paper/generated/replicates_table.tex from the decoding-replicate scans.

    python scripts/build_replicates_table.py

One row per model: decoding control, catalogue CC with its profile interval,
the five replicate boundaries, their standard deviation in items and as a
fraction of CC, and the
share of commonly probed lengths on which all replicates agree.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from counting.analysis.replicates import load  # noqa: E402

OUT = REPO / "paper" / "generated" / "replicates_table.tex"
CONTROL = {"locked": "rejects", "ignored": "ignores", "pinned": "pinned at 0"}


def main() -> int:
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Decoding replicates.} Five independent rescans of the transition region per model under the catalogue protocol, with the same stimulus in every replicate. Temperature: whether the provider rejects the parameter, accepts and ignores it, or honours zero. Agreement is the share of commonly probed lengths on which all five replicates return the same outcome, against the share expected if each reply were an independent draw at that length's mean accuracy.}",
        r"\label{tab:replicates}",
        r"\begin{tabular}{llrlrrr}", r"\toprule",
        r"Model & Temperature & CC [95\% CI] & Replicate boundaries & SD & SD/CC & Agreement \\",
        r"\midrule",
    ]
    for m in load():
        if not m.replicates:
            continue
        taus = m.taus
        common = set.intersection(*[set(r.lengths) for r in m.replicates])
        outcome: dict[int, list[bool]] = {}
        for r in m.replicates:
            d = dict(zip(r.lengths, r.correct))
            for L in common:
                outcome.setdefault(L, []).append(d[L])
        unan = sum(1 for L in common if len(set(outcome[L])) == 1) / len(common)
        k = len(m.replicates)
        exp = statistics.mean(statistics.mean(v) ** k + (1 - statistics.mean(v)) ** k
                              for v in outcome.values())
        sd = statistics.pstdev(taus)
        lines.append(
            f"\\texttt{{{m.label}}} & {CONTROL[m.control]} & {m.cc} [{m.cc_low}, {m.cc_high}] & "
            f"{' '.join(str(t) for t in taus)} & {sd:.1f} & {sd / m.cc:.1%} & "
            f"{unan:.0%} ({exp:.0%}) \\\\".replace("%", "\\%")
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
