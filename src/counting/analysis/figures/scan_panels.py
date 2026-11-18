#!/usr/bin/env python3
"""SI panels -- the SCC scan results, one panel per vendor.

For every model the scan records what was asked and what came back. Plotting
asked against answered puts exact counting on the diagonal, so each panel shows
a vendor's models tracking the diagonal to their counting capacity and then
leaving it. Models are separated by colour within the panel rather than given a
panel each: the comparison that matters is between a vendor's models, and one
panel per model buries it across pages.

Terminology follows the paper: SCC is the assay, CC the value it returns.

    python -m counting.analysis.figures.scan_panels
    python -m counting.analysis.figures.scan_panels --lab anthropic
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from counting.analysis.figures.search_panels import (
    PROVIDER_DISPLAY_NAMES,
    infer_provider_family,
)
from counting.paths import REPO_ROOT as ROOT

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

RESPONSES = ROOT / "results" / "counting-data" / "responses.csv"
SCC_CSV = ROOT / "results" / "counting-data" / "scc.csv"
OUT_DIR = ROOT / "paper" / "figures"

# Distinguishable at small size and colour-blind safe enough for a legend of
# this length; cycled when a vendor has more models than colours.
PALETTE = [
    "#1f78b4", "#e31a1c", "#33a02c", "#ff7f00", "#6a3d9a", "#b15928",
    "#a6cee3", "#fb9a99", "#b2df8a", "#fdbf6f", "#cab2d6", "#ffff99",
    "#4d4d4d", "#8dd3c7", "#bc80bd", "#d9d9d9",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]


def load():
    curves = defaultdict(list)
    with RESPONSES.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != "scan":
                continue
            answer = row["output_count"]
            curves[row["run_key"]].append(
                (int(row["input_count"]), int(answer) if answer else None)
            )
    cc = {}
    for row in csv.DictReader(SCC_CSV.open(encoding="utf-8")):
        cc[row["run_key"]] = (row["model"], int(row["scc"]))
    return curves, cc


def panel(ax, entries, curves):
    """One vendor: every measurement's asked-against-answered scan.

    The axes are scaled to the range of lengths *asked*, not of answers
    returned. After a model loses the count it can emit almost anything -- one
    Qwen measurement answers about 27,000 to a prompt of roughly 100 items --
    and letting that set the scale collapses every real point into the corner.
    Answers above the range are drawn on the top rail and counted in the corner
    rather than silently dropped.
    """
    top = 1
    for run_key, _, _ in entries:
        for length, _answer in curves[run_key]:
            top = max(top, length)
    lim = top * 1.12
    above = 0
    for index, (run_key, _label, _value) in enumerate(entries):
        colour = PALETTE[index % len(PALETTE)]
        marker = MARKERS[(index // len(PALETTE)) % len(MARKERS)]
        xs, ys = [], []
        for length, answer in curves[run_key]:
            if answer is None:
                continue          # no parseable integer; nothing to place on y
            if answer > lim:
                above += 1
                xs.append(length)
                ys.append(lim)
            else:
                xs.append(length)
                ys.append(answer)
        ax.scatter(xs, ys, s=2.4, marker=marker, color=colour, alpha=.85,
                   linewidths=0, zorder=2, rasterized=True)
    ax.plot([0, lim], [0, lim], color="#444444", lw=0.7, ls=(0, (4, 3)), zorder=3)
    if above:
        # Below the diagonal is empty by construction (undercounts are rare),
        # so the note sits there instead of over the tracking band.
        ax.annotate(f"{above} answers above range,\ndrawn on the top rail",
                    (0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=6.6, color="#6f6f6f", linespacing=1.35)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(labelsize=7.6, width=0.7, length=2.8)


def build(family, entries, curves, out_dir):
    n = len(entries)
    # Landscape: the panel is square, so the legend goes beside it rather than
    # underneath. Stacked vertically the figure was taller than wide and ate a
    # page each; this keeps eleven of them to a few pages.
    ncol = 1 if n <= 14 else (2 if n <= 40 else 3)
    rows = (n + ncol - 1) // ncol
    fig_h = max(3.5, rows * 0.128 + 0.85)
    plot_w = fig_h - 0.55
    legend_w = 1.85 * ncol
    fig_w = plot_w + legend_w + 0.30

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes((0.62 / fig_w, 0.60 / fig_h,
                       (plot_w - 0.75) / fig_w, (fig_h - 0.95) / fig_h))
    panel(ax, entries, curves)
    ax.set_xlabel("items asked", fontsize=9.0)
    ax.set_ylabel("count returned", fontsize=9.0)
    ax.set_title(f"{PROVIDER_DISPLAY_NAMES.get(family, family)} — {n} measurements",
                 fontsize=10.0, pad=5)

    handles = [
        Line2D([], [], marker=MARKERS[(i // len(PALETTE)) % len(MARKERS)], lw=0,
               markersize=3.6, color=PALETTE[i % len(PALETTE)],
               label=f"{label}  ({value})")
        for i, (_, label, value) in enumerate(entries)
    ]
    fig.legend(handles=handles, loc="center left",
               bbox_to_anchor=((plot_w + 0.02) / fig_w, 0.5), frameon=False,
               fontsize=6.4, ncol=ncol, handlelength=1.0, handletextpad=0.35,
               columnspacing=0.9, labelspacing=0.34)

    out = out_dir / f"scan-panels-{family}.pdf"
    fig.savefig(out, dpi=400)
    plt.close(fig)
    return out, n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lab", help="only this vendor")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    curves, cc = load()
    by_lab = defaultdict(list)
    for run_key, (model, value) in cc.items():
        label = run_key.split("/", 1)[-1]      # keeps the #think / #nothink suffix
        by_lab[infer_provider_family(model)].append((run_key, label, value))

    for family in sorted(by_lab):
        if args.lab and family != args.lab:
            continue
        # Strongest first, so the legend order matches how the panel reads.
        entries = sorted(by_lab[family], key=lambda e: (-e[2], e[1]))
        out, n = build(family, entries, curves, args.out_dir)
        print(f"  {family:<10} {n:>3} models -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
