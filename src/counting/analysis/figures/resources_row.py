#!/usr/bin/env python3
"""Figure 3 as one uniform 1x4 row: tokens, hybrid pairs, interference, replicates.

    python -m counting.analysis.figures.resources_row

Same geometry and type scale as the authoritative benchmark row, so the two
\\textwidth figures print with identical panel sizes. Panels carry their
letters, so the PDF drops straight into the manuscript.
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

from counting.analysis.figures.attractors import BAD, BASELINE_DARK, OK, set_spine_width
from counting.analysis.figures.benchmark_correlations import (
    ROW_FIGURE_WIDTH_IN,
    ROW_PLOT_LEFT_IN,
    ROW_PLOT_WIDTH_IN,
    add_manual_row_axes,
)
from counting.analysis.capacity import fit_transition
from counting.analysis.figures.interference_bars import MODELS
from counting.analysis.figures.search_panels import PROVIDER_COLORS, infer_provider_family
from counting.analysis.replicates import load as load_replicates
from counting.analysis.reasoning_pairs import (
    RUN,
    _probe_tokens,
    hybrid_pairs,
    multiplier_formatter,
)
from counting.paths import REPO_ROOT as ROOT

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

OUTPUT = ROOT / "paper" / "figures" / "resources-row.pdf"
# The manuscript's Figure 3: the same row with panel letters.
FIGURE3 = ROOT / "paper" / "figures" / "figure3.pdf"

TITLE_FS, LABEL_FS, TICK_FS, SMALL_FS = 16, 14, 12, 10


def draw_token_panel(ax) -> None:
    cc = {}
    with (RUN / "scc.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cc[row["run_key"]] = int(row["scc"])
    xs, ys, colours = [], [], []
    for key, probes in _probe_tokens().items():
        boundary = cc.get(key, 0)
        if boundary < 10:
            continue
        near = [p[1] for p in probes if 0.8 * boundary <= p[0] <= boundary and p[1] > 0]
        if near:
            xs.append(boundary)
            ys.append(float(np.mean(near)))
            colours.append(PROVIDER_COLORS.get(infer_provider_family(key.split("#")[0]), "#888888"))
    ax.scatter(xs, ys, s=26, c=colours, alpha=.8, edgecolors="white",
               linewidths=0.5, zorder=3)
    ax.axhline(2, color="#666666", lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate("2 tokens floor", xy=(0.97, 2.6), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=SMALL_FS, color="#444444")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Token Cost", fontsize=TITLE_FS)
    ax.set_xlabel("CC", fontsize=LABEL_FS)
    ax.set_ylabel("Completion tokens", fontsize=LABEL_FS, labelpad=2)


def draw_pair_panel(ax) -> None:
    shown = [p for p in hybrid_pairs()
             if p.classification == "verified" and np.isfinite(p.token_multiplier)]
    ax.axvline(0, color="#b8b8b8", lw=1.0, ls="--", zorder=1)
    ax.axhline(1.0, color="#b8b8b8", lw=1.0, ls="--", zorder=1)
    for p in shown:
        ax.scatter([p.delta_cc], [p.token_multiplier], s=30,
                   color=PROVIDER_COLORS.get(p.provider, "#888888"),
                   edgecolors="white", linewidths=0.5, alpha=.9, zorder=3)
    md = float(np.median([p.delta_cc for p in shown]))
    mm = float(np.median([p.token_multiplier for p in shown]))
    ax.scatter([md], [mm], s=90, marker="D", color=BASELINE_DARK,
               edgecolors="white", linewidths=1.0, zorder=5)
    ax.annotate(f"Median ({md:+.0f},\n{mm:.0f}x tokens)", xy=(md, mm), xytext=(-16, -14),
                textcoords="offset points", ha="right", va="top",
                fontsize=SMALL_FS, color="#222222")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_formatter(FuncFormatter(multiplier_formatter))
    ax.set_title("Reasoning On vs Off", fontsize=TITLE_FS)
    ax.set_xlabel(r"$\Delta$CC (reasoning on $-$ off)", fontsize=LABEL_FS)
    ax.set_ylabel("Token multiplier", fontsize=LABEL_FS, labelpad=2)


# Replicated concurrent-load scans: five scans per condition, each model
# against the mean of its own five solo scans. The single scans of the
# submitted version stay in results/interference and in the appendix table.
REPLICATE_ROOT = ROOT / "results" / "rebuttal-interference"
REPLICATE_CONDS = [("inert-90", "control"), ("semantic-90", "reading"),
                   ("pointer-90", "pointer"), ("row3-60", "rule 60%"),
                   ("grid-60", "grid 60%")]


def replicate_boundaries(slug: str, cond: str) -> list[int]:
    """Fitted boundary of each replicate scan, on scan-phase probes only."""
    out = []
    for rep_dir in sorted(REPLICATE_ROOT.glob("rep-*")):
        rows = []
        for path in (rep_dir / slug / cond / "trials").glob("*.jsonl"):
            rows += [json.loads(line) for line in path.open()]
        rows = sorted((r for r in rows if r.get("phase") == "scan"),
                      key=lambda r: r["center_length"])
        if not rows:
            continue
        fit = fit_transition([r["center_length"] for r in rows],
                             [bool(r["exact_match"]) for r in rows],
                             [r.get("parsed") is None for r in rows], calibrate=False)
        if not fit.censored:
            out.append(int(fit.scc))
    return out


def draw_interference_panel(ax) -> None:
    width = 0.38
    for mi, (slug, label, _, colour) in enumerate(MODELS):
        solo = float(np.mean(replicate_boundaries(slug, "solo")))
        for ci, (cond, _) in enumerate(REPLICATE_CONDS):
            scans = [100.0 * (b / solo - 1.0) for b in replicate_boundaries(slug, cond)]
            if not scans:
                continue
            x = ci + (mi - (len(MODELS) - 1) / 2) * width
            ax.bar([x], [float(np.mean(scans))], width=width - 0.04, color=colour,
                   label=label if ci == 0 else None, edgecolor="white",
                   linewidth=0.5, alpha=0.75, zorder=3)
            jitter = np.linspace(-0.11, 0.11, len(scans))
            ax.scatter(x + jitter, scans, s=9, color=BASELINE_DARK,
                       linewidths=0, alpha=0.85, zorder=4)
    ax.axhline(0, color="#666666", lw=1.0, zorder=2)
    ax.set_xticks(range(len(REPLICATE_CONDS)))
    ax.set_xticklabels([short for _, short in REPLICATE_CONDS], rotation=40,
                       ha="right", fontsize=SMALL_FS)
    ax.set_ylim(-45, 30)
    ax.legend(fontsize=SMALL_FS, frameon=False, loc="upper right",
              handlelength=1.1, borderaxespad=0.1, handletextpad=0.4)
    ax.set_title("Load Interference", fontsize=TITLE_FS)
    ax.set_ylabel(r"$\Delta$CC under load (%)", fontsize=LABEL_FS, labelpad=2)


def draw_replicate_panel(ax) -> None:
    """Five rescans of the transition region per model, one vertical strip each.

    Every probe is a short segment coloured by outcome (exact in blue, inexact
    in red, the language of Fig. 1b); the black tick is that replicate's fitted
    boundary; y is items asked relative to the catalogue CC, so every model's
    passage from mostly-right to mostly-wrong sits on one axis. Model names sit
    on the x axis, tilted like the load labels of the interference panel.
    """
    models = [m for m in load_replicates() if m.replicates]
    band, strip = 0.78, 0.12
    for mi, m in enumerate(models):
        x0 = mi - band / 2
        pitch = band / max(len(m.replicates), 1)
        for ri, rep in enumerate(m.replicates):
            x = x0 + ri * pitch
            runs, start, prev = [], None, None
            for length, ok in zip(rep.lengths, rep.correct):
                if prev is None or ok != prev:
                    if start is not None:
                        runs.append((start, length, prev))
                    start, prev = length, ok
            if start is not None:
                runs.append((start, rep.lengths[-1] + 1, prev))
            for lo, hi, ok in runs:
                ax.bar(x, (hi - lo) / m.cc, bottom=lo / m.cc, width=strip,
                       align="edge", color=OK if ok else BAD, linewidth=0,
                       alpha=0.9, zorder=2, rasterized=True)
            if rep.tau:
                ax.plot([x - 0.01, x + strip + 0.01], [rep.tau / m.cc] * 2,
                        color="black", lw=1.1, zorder=4, solid_capstyle="butt")
    ax.axhline(1.0, color="#666666", lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.label for m in models], rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=SMALL_FS - 1)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0.78, 1.56)
    ax.set_yticks([0.8, 1.0, 1.2, 1.4])
    ax.set_title("Decoding Replicates", fontsize=TITLE_FS)
    ax.set_ylabel("Items asked / catalogue CC", fontsize=LABEL_FS, labelpad=2)
    ax.grid(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.9)


def main() -> int:
    figure_height_in = 4.15
    fig = plt.figure(figsize=(ROW_FIGURE_WIDTH_IN, figure_height_in))
    axes = add_manual_row_axes(
        fig, panel_count=4,
        figure_width_in=ROW_FIGURE_WIDTH_IN, figure_height_in=figure_height_in,
        plot_left_in=ROW_PLOT_LEFT_IN, plot_bottom_in=1.34,
        plot_width_in=ROW_PLOT_WIDTH_IN, plot_height_in=2.24, panel_gap_in=0.82,
    )
    for ax, draw in zip(axes, (draw_token_panel, draw_pair_panel,
                               draw_interference_panel, draw_replicate_panel)):
        ax.grid(alpha=0.2, linewidth=0.9)
        draw(ax)
        ax.tick_params(axis="both", which="both", labelsize=TICK_FS)
        ax.set_box_aspect(1)
        set_spine_width(ax)
    fig.savefig(OUTPUT, dpi=200)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=200)
    for ax, letter in zip(axes, "ABCD"):
        box = ax.get_position()
        fig.text(box.x0 - 0.052, box.y1 + 0.035, letter, fontsize=TITLE_FS + 2,
                 fontweight="bold", ha="left", va="bottom")
    fig.savefig(FIGURE3, dpi=200)
    plt.close(fig)
    print(f"wrote {OUTPUT} and {FIGURE3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
