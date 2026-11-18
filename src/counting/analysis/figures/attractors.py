#!/usr/bin/env python3
"""Figure 2 -- what happens at and after the counting boundary.

Eight panels on a 2x4 grid, written as one figure:

    row 1   one model's whole scan; the cliff; the spread after it; off-by-k
    row 2   attractors; then the motif sweep, one panel per model

Rows 1's middle panels rescale length by each model's own CC. That is the only
way to overlay measurements whose capacities span 10 to 627, and it is what
shows the collapse happening at CC rather than at any absolute length.

Row 2's motif panels come from a separate sweep (scripts/run_motif_sweep.py):
the same three models counting eleven different item/delimiter combinations,
each scanned under the same protocol, with a rule at each variant's CC.

    python -m counting.analysis.figures.attractors
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from counting.analysis.capacity import fit_transition
from counting.paths import REPO_ROOT as ROOT

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# Kept for the modules that import them (landscape_triptych, scc_vs_params,
# benchmark_correlations, reasoning_pairs).
MUTED_BLUE = "#7f97b2"
MUTED_ORANGE = "#c6a78c"
MUTED_GRAY = "#aeb3b8"
MUTED_DARK = "#666666"
BASELINE_DARK = "#2f2f2f"
FOCUS_BLUE = "#5f84ad"
FOCUS_RED = "#c46a6a"

# The original figure's red/blue reading: exact answers and data in blue,
# failures and failure structure in red; deep blue for fitted rules and lines.
OK, BAD, INK, MUTED, ACCENT = FOCUS_BLUE, FOCUS_RED, "#1a1a1a", "#6f6f6f", "#3d6a94"

RUN = ROOT / "results" / "counting-data"
MOTIF_RUN = ROOT / "results" / "motif-sweep"
OUTPUT_DIR = ROOT / "paper" / "figures"
OUTPUT_BASENAME = "frontier-attractor-iteration"

FOCUS = "anthropic/claude-fable-5"
ROLL = 15
# Read off the wrong answers themselves rather than chosen: -1 alone is 19% of
# them, and 100 is the single most common answer at 13%.
OFFSETS = (-1, 1, -2, 2, -5, -10)
# Off-by-roughly-a-round-number: Anthropic models cluster at +98/+99/+198 and
# -100, Google at -100 and -50 (OpenAI and Qwen show no such structure). The
# window is +/-4 because the cluster sits near, not exactly on, the round
# offset -- +98 is the modal Anthropic error, an off-by-hundred with a slip.
OFFSETS_ROUND = (-100, -50, 100)
ROUND_WINDOW = 4


def offset_panel_takes(diff: int) -> bool:
    """Whether a non-attractor mistake belongs on the off-by-k panel."""
    if abs(diff) <= 12:
        return True
    return any(abs(diff - k) <= ROUND_WINDOW for k in OFFSETS_ROUND)
# Two tiers, from the measured counts among far-off wrong answers: the major
# values each occur 60+ times (100 alone 841 times, 500 107 times), the minor
# 40 to 60. The set also routes attractor-valued answers out of the off-by-k
# panel, where they painted horizontal runs through the offset structure.
ATTRACTORS_MAJOR = (30, 40, 50, 60, 80, 100, 200, 300, 500, 600, 1000)
ATTRACTORS_MINOR = (26, 49, 64, 70, 99, 150)
ATTRACTOR_VALUES = set(ATTRACTORS_MAJOR) | set(ATTRACTORS_MINOR) | {8}

MOTIF_ORDER = [
    ("baseline-a-comma", "a  ,"), ("letter-b-comma", "b  ,"),
    ("letter-x-comma", "x  ,"), ("double-aa-comma", "aa  ,"),
    ("greek-alpha-comma", "alpha  ,"), ("greek-beta-comma", "beta  ,"),
    ("chinese-han-comma", "han  ,"), ("chinese-zhong-comma", "zhong  ,"),
    ("letter-a-space", "a  space"), ("letter-a-pipe", "a  |"),
    ("letter-a-section", "a  section"),
]
MOTIF_COLOURS = [
    "#1f78b4", "#e31a1c", "#33a02c", "#ff7f00", "#6a3d9a", "#b15928",
    "#17becf", "#fb9a99", "#7f7f7f", "#bcbd22", "#4d4d4d",
]
MOTIF_MODELS = ["anthropic/claude-sonnet-5", "openai/gpt-5.6-terra",
                "google/gemini-3.7-flash"]


def set_spine_width(ax: plt.Axes, width: float = 1.5) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(width)


TITLE_FS = 15
LABEL_FS = 12.5
TICK_FS = 10.5
SOURCE_FS = 9


def style(ax: plt.Axes) -> None:
    set_spine_width(ax)
    ax.tick_params(labelsize=TICK_FS, width=1.1, length=4.0)
    ax.grid(alpha=0.18, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_box_aspect(1)


def add_panel_source_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.03, 0.965, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=SOURCE_FS, color="#555555",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.2))


# -- data -------------------------------------------------------------------

def load_scans():
    curves = defaultdict(list)
    with (RUN / "responses.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != "scan":
                continue
            answer = row["output_count"]
            curves[row["run_key"]].append(
                (int(row["input_count"]), int(answer) if answer else None, row["outcome"])
            )
    cc = {}
    with (RUN / "scc.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cc[row["run_key"]] = int(row["scc"])
    for key in curves:
        curves[key].sort()
    return curves, cc


def load_motifs() -> dict[tuple[str, str], tuple[list[int], list[int | None], int]]:
    data: dict[tuple[str, str], tuple[list[int], list[int | None], int]] = {}
    for motif_id, _label in MOTIF_ORDER:
        trials = MOTIF_RUN / motif_id / "trials"
        if not trials.is_dir():
            continue
        for path in sorted(trials.glob("*.jsonl")):
            lengths, answers, correct = [], [], []
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("phase") != "scan":
                    continue
                lengths.append(row["center_length"])
                answers.append(row.get("parsed"))
                correct.append(bool(row.get("exact_match")))
            if not lengths:
                continue
            order = sorted(range(len(lengths)), key=lambda i: lengths[i])
            lengths = [lengths[i] for i in order]
            answers = [answers[i] for i in order]
            correct = [correct[i] for i in order]
            fit = fit_transition(lengths, correct, [a is None for a in answers])
            model = path.stem.replace("__", "/", 1).split("__")[0]
            data[(motif_id, model)] = (lengths, answers, fit.scc)
    return data


# -- panels -----------------------------------------------------------------

def panel_single_scan(ax, curves, cc):
    key = next(k for k in curves if k.startswith(FOCUS))
    xs_ok, ys_ok, xs_bad, ys_bad = [], [], [], []
    for length, answer, outcome in curves[key]:
        if answer is None:
            continue
        (xs_ok if outcome == "correct" else xs_bad).append(length)
        (ys_ok if outcome == "correct" else ys_bad).append(answer)
    top = max(max(xs_ok + xs_bad), max(ys_ok + ys_bad)) * 1.04
    ax.plot([0, top], [0, top], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.scatter(xs_ok, ys_ok, s=4.5, color=OK, alpha=.85, linewidths=0, zorder=3)
    ax.scatter(xs_bad, ys_bad, s=4.5, color=BAD, alpha=.85, linewidths=0, zorder=3)
    boundary = cc[key]
    ax.axvline(boundary, color=ACCENT, lw=1.3, zorder=2)
    ax.annotate(f"CC = {boundary}", (boundary, 0.02 * top), xytext=(-5, 0),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=10.5, color=ACCENT, fontweight="bold")
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xlabel("Items asked", fontsize=LABEL_FS)
    ax.set_ylabel("Count returned", fontsize=LABEL_FS)
    ax.set_title("Representative Scan", fontsize=TITLE_FS)
    add_panel_source_label(ax, "claude-fable-5")
    style(ax)


def panel_cliff(ax, curves, cc):
    for key, probes in curves.items():
        boundary = cc.get(key, 0)
        if boundary < 10:
            continue
        xs, ys, window = [], [], []
        for length, _answer, outcome in probes:
            window.append(1 if outcome == "correct" else 0)
            if len(window) > ROLL:
                window.pop(0)
            if len(window) < ROLL:
                continue
            xs.append(length / boundary)
            ys.append(sum(window) / ROLL)
        ax.plot(xs, ys, color=ACCENT, lw=0.45, alpha=.055, zorder=2)
    ax.axvline(1.0, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=3)
    ax.set_xlim(0, 2.0); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Length / CC", fontsize=LABEL_FS)
    ax.set_ylabel("Exact-answer rate", fontsize=LABEL_FS)
    ax.set_title("Collapse Aligned at CC", fontsize=TITLE_FS)
    add_panel_source_label(ax, "All models")
    style(ax)


def panel_spread(ax, curves, cc):
    xs, ys = [], []
    for key, probes in curves.items():
        boundary = cc.get(key, 0)
        if boundary < 10:
            continue
        for length, answer, _outcome in probes:
            if answer is None:
                continue
            xs.append(length / boundary)
            ys.append(answer / boundary)
    ax.plot([0, 2], [0, 2], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.scatter(xs, ys, s=1.4, color=ACCENT, alpha=.16, linewidths=0, zorder=2,
               rasterized=True)
    ax.axvline(1.0, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=4)
    ax.set_xlim(0, 2.0); ax.set_ylim(0, 2.0)
    ax.set_xlabel("Length / CC", fontsize=LABEL_FS)
    ax.set_ylabel("Count returned / CC", fontsize=LABEL_FS)
    ax.set_title("Spread Past the Boundary", fontsize=TITLE_FS)
    add_panel_source_label(ax, "All models")
    style(ax)


def panel_offsets(ax, curves):
    from collections import Counter
    counts: Counter = Counter()
    for probes in curves.values():
        for length, answer, outcome in probes:
            if answer is None or outcome != "mistake":
                continue
            # Attractor-valued answers belong to the next panel whatever their
            # distance from the diagonal; a 50 answered to 43 is a jump to a
            # round number, not an off-by-seven.
            if answer in ATTRACTOR_VALUES:
                continue
            if answer != length and offset_panel_takes(answer - length):
                counts[(length, answer - length)] += 1
    # Plotting the offset itself keeps every off-by-k on one horizontal line at
    # any scale. On the asked-against-answered axes the lines converge into the
    # diagonal past a hundred items, exactly where the strongest models make
    # these errors, and the structure became unreadable.
    lo, hi = 1, 1300
    for k in OFFSETS:
        ax.axhline(k, color=BAD, lw=0.9, alpha=.75, zorder=3)
    for k in OFFSETS_ROUND:
        ax.axhline(k, color=ACCENT, lw=1.1, alpha=.85, zorder=3)
        ax.annotate(f"{k:+d}", (hi, k), xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color=ACCENT)
    for k in (-10, 10):
        ax.annotate(f"{k:+d}", (hi, k), xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=BAD)
    xs = [key[0] for key in counts]
    ys = [key[1] for key in counts]
    sizes = [min(3.5 + 1.6 * counts[key], 22.0) for key in counts]
    ax.scatter(xs, ys, s=sizes, color=INK, alpha=.5, linewidths=0, zorder=2,
               rasterized=True)
    ax.axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(-118, 118)
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300, 1000]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([-100, -50, -10, 0, 10, 50, 100])
    ax.set_xlabel("Items asked", fontsize=LABEL_FS)
    ax.set_ylabel("Count returned − asked", fontsize=LABEL_FS)
    ax.set_title("Off-by-k Errors", fontsize=TITLE_FS)
    add_panel_source_label(ax, "All models")
    style(ax)


def panel_attractors(ax, curves):
    xs, ys = [], []
    for probes in curves.values():
        for length, answer, outcome in probes:
            if answer is None or outcome != "mistake":
                continue
            if answer in ATTRACTOR_VALUES or not offset_panel_takes(answer - length):
                xs.append(length); ys.append(max(answer, 1))
    lo, hi = 8, 1300
    for value in ATTRACTORS_MAJOR:
        ax.axhline(value, color=BAD, lw=1.0, alpha=.8, zorder=3)
        ax.annotate(str(value), (hi, value), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, color=BAD)
    for value in ATTRACTORS_MINOR:
        ax.axhline(value, color=BAD, lw=0.55, alpha=.45, zorder=3)
    ax.scatter(xs, ys, s=2.2, color=INK, alpha=.28, linewidths=0, zorder=2,
               rasterized=True)
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator([10, 30, 100, 300, 1000]))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Items asked", fontsize=LABEL_FS)
    ax.set_ylabel("Count returned", fontsize=LABEL_FS)
    ax.set_title("Numerical Attractors", fontsize=TITLE_FS)
    add_panel_source_label(ax, "All models")
    style(ax)


def panel_motif(ax, data, model, show_ylabel: bool = False):
    top = 1
    for index, (motif_id, _label) in enumerate(MOTIF_ORDER):
        entry = data.get((motif_id, model))
        if entry is None:
            continue
        lengths, answers, boundary = entry
        colour = MOTIF_COLOURS[index % len(MOTIF_COLOURS)]
        xs = [l for l, a in zip(lengths, answers) if a is not None]
        ys = [a for a in answers if a is not None]
        if xs:
            top = max([top, *xs])
        ax.scatter(xs, ys, s=3.0, color=colour, alpha=.55, linewidths=0,
                   zorder=2, rasterized=True)
        ax.axvline(boundary, color=colour, lw=1.3, alpha=.9, zorder=3)
    lim = top * 1.06
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(model.split("/")[-1], fontsize=TITLE_FS)
    ax.set_xlabel("Items asked", fontsize=LABEL_FS)
    if show_ylabel:
        ax.set_ylabel("Count returned", fontsize=LABEL_FS)
    style(ax)


# -- figure -----------------------------------------------------------------

def build() -> tuple[Path, Path]:
    curves, cc = load_scans()
    motifs = load_motifs()

    fig, axes = plt.subplots(2, 4, figsize=(12.4, 7.0), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
    panel_single_scan(axes[0, 0], curves, cc)
    panel_cliff(axes[0, 1], curves, cc)
    panel_spread(axes[0, 2], curves, cc)
    panel_offsets(axes[0, 3], curves)
    panel_attractors(axes[1, 0], curves)
    for column, model in enumerate(MOTIF_MODELS, start=1):
        panel_motif(axes[1, column], motifs, model, show_ylabel=column == 1)

    if motifs:
        handles = [plt.Line2D([], [], marker="o", lw=0, markersize=4.6,
                              color=MOTIF_COLOURS[i % len(MOTIF_COLOURS)], label=label)
                   for i, (_mid, label) in enumerate(MOTIF_ORDER)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.638, 0.052),
                   frameon=False, fontsize=8.2, ncol=len(handles), handlelength=1.0,
                   handletextpad=0.3, columnspacing=0.9)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    fig.savefig(png_path, dpi=250, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    pdf_path, png_path = build()
    motifs = load_motifs()
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"  motif scans available: {len(motifs)} of {len(MOTIF_ORDER) * len(MOTIF_MODELS)}")


if __name__ == "__main__":
    main()
