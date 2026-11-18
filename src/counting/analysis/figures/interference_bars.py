#!/usr/bin/env python3
"""Figure 3C: capacity displacement under calibrated concurrent loads.

    python -m counting.analysis.figures.interference_bars

For each interference condition, the counting boundary fitted under load is
compared with the model's solo CC, and the percent change is drawn as a bar.
Loads are the ladder's own tasks at fractions of each model's measured solo
capacity; inert (matched length, no work) and pointer (nameable state) are
the controls. gpt-5.6-terra is excluded: its calibration solos were
mis-bracketed, so its loads were degenerate (see results/excluded/).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from counting.analysis.capacity import fit_transition
from counting.analysis.figures.attractors import FOCUS_BLUE, FOCUS_RED, set_spine_width
from counting.paths import REPO_ROOT as ROOT

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

OUTPUT = ROOT / "paper" / "figures" / "interference-displacement.pdf"

# Solo CC of the exact configuration the interference scans ran (think variant
# for sonnet; flash has a single configuration).
MODELS = [("gemini-3p7-flash", "Gemini 3.7 Flash", 352, FOCUS_BLUE),
          ("claude-sonnet-5", "Claude Sonnet 5", 89, FOCUS_RED)]
CONDS = [("inert-90", "inert"), ("pointer-90", "pointer"),
         ("row2-60", "row 60%"), ("row3-60", "rule 60%"),
         ("grid-30", "grid 30%"), ("grid-60", "grid 60%"),
         ("grid-90", "grid 90%"), ("grid-120", "grid 120%")]


def fitted_boundary(run_dir: Path) -> int | None:
    rows = []
    for tf in glob.glob(str(run_dir / "trials" / "*.jsonl")):
        rows += [json.loads(line) for line in open(tf)]
    if not rows:
        return None
    rows.sort(key=lambda r: r["expected"])
    t = fit_transition([r["expected"] for r in rows],
                      [r.get("parsed") == r.get("expected") for r in rows])
    return None if (t.scc is None or t.censored) else int(t.scc)


def main() -> int:
    fig, ax = plt.subplots(figsize=(3.4, 3.1), constrained_layout=True)
    n_models = len(MODELS)
    width = 0.38
    for mi, (slug, label, solo, colour) in enumerate(MODELS):
        xs, ys = [], []
        for ci, (cond, _) in enumerate(CONDS):
            boundary = fitted_boundary(ROOT / "results" / "interference" / slug / cond)
            if boundary is None:
                continue
            xs.append(ci + (mi - (n_models - 1) / 2) * width)
            ys.append(100.0 * (boundary / solo - 1.0))
        ax.bar(xs, ys, width=width - 0.04, color=colour, label=label,
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(0, color="#666666", lw=1.0, zorder=2)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([short for _, short in CONDS], rotation=45,
                       ha="right", fontsize=7.2)
    ax.set_ylabel("change in CC under load (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(-42, 12)
    ax.grid(alpha=0.18, axis="y", zorder=0)
    ax.legend(fontsize=7.2, frameon=False, loc="lower left",
              handlelength=1.1, borderaxespad=0.2)
    set_spine_width(ax, 0.9)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, transparent=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=250, transparent=True)
    plt.close(fig)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
