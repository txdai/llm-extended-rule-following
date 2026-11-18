#!/usr/bin/env python3
"""Fig. 1B -- how Stable Counting Capacity is estimated.

Three stacked rows over one length axis, drawn from a real measurement rather
than a schematic, so every feature shown is a feature of the data:

    record   one probe per length; exact answers in green, inexact in red
    fit      the two-regime model, accuracy p1 up to the transition and p2
             above it, with the transition and its 95% interval marked
    profile  the log-likelihood of every admissible cut, whose peak is the
             estimate and whose 1.92-unit contour is the interval

The panel exists to make one thing plain: the first inexact answer is not the
boundary. In the measurement shown it arrives at 65, the model keeps counting
correctly for most of the next 24 lengths, and exact counting stops being the
majority behaviour only at 89.

    python -m counting.analysis.figures.scc_schematic
    python -m counting.analysis.figures.scc_schematic --model x-ai/grok-4.3#think
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from counting.analysis.capacity import PROFILE_BAR, fit_transition, transition_profile
from counting.paths import REPO_ROOT as ROOT

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

RESPONSES = ROOT / "results" / "counting-data" / "responses.csv"
OUT_PDF = ROOT / "paper" / "figures" / "scc-estimation.pdf"

DEFAULT_MODEL = "anthropic/claude-sonnet-5#think"

# Sized for a slide panel: narrow, with type large relative to the frame.
FIG_W, FIG_H = 4.22, 2.55
FS = 9.3                        # body labels
FS_SMALL = 9.3                  # tick labels and in-plot values
FS_AXIS = 10.5                  # axis titles
# Window for the empirical rate drawn under the fit. Display only -- the
# estimator uses no window -- but the panel must show the measured rate as well
# as the fitted one, or the step reads as an instantaneous drop that the data
# does not show.
ROLL_W = 15

# Exact answers in figure-1 accent green (theme accent6 / box stroke); inexact
# in the figure-2 red. The fitted transition stays deep blue.
OK = "#70AD47"
BAD = "#c46a6a"
ACCENT = "#3d6a94"
INK = "#1a1a1a"
MUTED = "#6f6f6f"


def load(run_key: str):
    probes = []
    with RESPONSES.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["run_key"] != run_key or row["phase"] != "scan":
                continue
            probes.append((int(row["input_count"]), row["outcome"] == "correct"))
    if not probes:
        raise SystemExit(f"no scan probes for {run_key}")
    probes.sort()
    return [p[0] for p in probes], [p[1] for p in probes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    lengths, correct = load(args.model)
    fit = fit_transition(lengths, correct, [False] * len(lengths))
    cuts, ll_single, bar = transition_profile(lengths, correct)
    peak = max(ll for _, ll in cuts)
    first_bad = next(l for l, ok in zip(lengths, correct) if not ok)
    hi = lengths[-1]

    fig, axes = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H), sharex=True,
        gridspec_kw={"height_ratios": [0.5, 1.25, 1.25], "hspace": 0.42},
    )
    strip, regime, prof = axes
    for ax in axes:
        ax.set_xlim(0, hi + 1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(labelsize=FS_SMALL, width=0.7, length=2.5)

    # -- record ------------------------------------------------------------
    for length, ok in zip(lengths, correct):
        strip.plot([length, length], [0, 1], color=OK if ok else BAD, lw=0.85,
                   solid_capstyle="butt")
    strip.set_ylim(0, 1)
    strip.set_yticks([])
    strip.spines["left"].set_visible(False)
    strip.spines["bottom"].set_visible(False)
    strip.tick_params(bottom=False)


    # -- measured rate, then the fit over it -------------------------------
    roll_x, roll_y = [], []
    for i, length in enumerate(lengths):
        lo, hi_i = max(0, i - ROLL_W // 2), min(len(lengths), i + ROLL_W // 2 + 1)
        window = correct[lo:hi_i]
        roll_x.append(length)
        roll_y.append(sum(window) / len(window))
    regime.plot(roll_x, roll_y, color=MUTED, lw=0.9, alpha=.85, zorder=2)

    regime.axvspan(fit.ci_low, fit.ci_high, color=ACCENT, alpha=.16, lw=0)
    # regime.plot([0, fit.scc], [fit.p_stable] * 2, color=ACCENT, lw=1.5, zorder=3)
    # regime.plot([fit.scc, hi + 1], [fit.p_after] * 2, color=ACCENT, lw=1.5, zorder=3)
    # regime.plot([fit.scc, fit.scc], [fit.p_after, fit.p_stable], color=ACCENT, lw=1.5, zorder=3)
    regime.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    regime.annotate("½", (hi + 1, 0.5), xytext=(3, 0), textcoords="offset points",
                    fontsize=FS_SMALL, color=MUTED, va="center")
    for value, y, va in ((fit.p_stable, fit.p_stable, "bottom"), (fit.p_after, fit.p_after, "bottom")):
        regime.annotate(f"{value:.2f}", (1, y), xytext=(1, 2), textcoords="offset points",
                        fontsize=FS_SMALL, color=ACCENT, va=va)
    regime.set_ylim(-0.08, 1.12)
    regime.set_yticks([0, 1])
    regime.set_ylabel("exact rate", fontsize=FS_AXIS, color=MUTED)

    # -- likelihood profile ------------------------------------------------
    xs = [c[0] for c in cuts]
    ys = [c[1] - peak for c in cuts]
    prof.plot(xs, ys, color=INK, lw=1.0)
    prof.axvspan(fit.ci_low, fit.ci_high, color=ACCENT, alpha=.16, lw=0)
    prof.axhline(-PROFILE_BAR, color=ACCENT, lw=0.8, ls=(0, (3, 2)))
    # Inside the frame, left of where any admissible cut exists. Anchored to the
    # right edge it overhung the axes, and bbox_inches="tight" padded that
    # overhang into the figure width.
    prof.annotate("95% interval", (1, -PROFILE_BAR), xytext=(2, -2),
                  textcoords="offset points", fontsize=FS, color=ACCENT,
                  ha="left", va="top")
    prof.axhline(ll_single - peak, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    prof.set_ylim((ll_single - peak) * 1.15, 6)
    prof.set_ylabel("log-likelihood", fontsize=FS_AXIS, color=MUTED)
    prof.set_xlabel("sequence length (items)", fontsize=FS_AXIS)

    # transition marker, carried down the stack
    for ax in (regime, prof):
        ax.axvline(fit.scc, color=ACCENT, lw=0.9)
    regime.annotate(f"CC = {fit.scc}", (fit.scc, 1.1), xytext=(4, 0),
                    textcoords="offset points", fontsize=FS, color=ACCENT,
                    va="top", fontweight="bold")

    fig.align_ylabels(axes)
    out = args.out or OUT_PDF
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}")
    print(f"  {args.model}: first inexact {first_bad}, SCC {fit.scc} "
          f"[{fit.ci_low}-{fit.ci_high}], p1 {fit.p_stable:.3f}, p2 {fit.p_after:.3f}, "
          f"deltaLL {fit.delta_ll:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
