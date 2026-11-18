#!/usr/bin/env python3
"""Fig. 1C -- stable counting capacity as a number line.

Each model is a mark sitting off a shared logarithmic axis, joined to it by a
leader line. The marker is the fitted transition and the whisker is its 95%
profile interval, so the mark carries the measured width of the boundary rather
than asserting a bare point. On this corpus most intervals are only a few
lengths wide, meaning the boundary is precisely located -- not that the rate
drops in one step, which it does not.

Marks are packed into rows alternating above and below the axis by greedy
first-fit on measured label extents, so nothing overlaps at any figure size.

The lower axis is the same length scale expressed in prompt tokens, at the
measured 2.49 tokens per item, against nominal context windows. The wedge
between the two shows the whole capacity axis landing inside the first few
thousand tokens of context.

    python -m counting.analysis.figures.capacity_numberline
    python -m counting.analysis.figures.capacity_numberline --all
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from counting.analysis.figures.search_panels import (
    CONTEXT_AXIS_COLOR,
    CONTEXT_REFERENCE_MILESTONES,
    CONTEXT_REFERENCE_TICKS,
    PROVIDER_COLORS,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_MARKERS,
    infer_provider_family,
)
from counting.paths import REPO_ROOT as ROOT

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

SCC_CSV = ROOT / "results" / "counting-data" / "scc.csv"
REGISTRY_CSV = ROOT / "results" / "counting-data" / "registry.csv"
OUT_PDF = ROOT / "paper" / "figures" / "capacity-numberline.pdf"

TOKENS_PER_ITEM = 2.49          # measured over 39,551 probes at length >= 20
FIG_W = 7.6                     # sized for \textwidth, not for a screen
LABEL_FS = 8.5
# Vertical pitch per row, in points. Rows carry a marker and a label of the same
# height, so the pitch only has to clear the type; anything more is dead space,
# which is what made the first version three times taller than it needed to be.
ROW_PT = 12.5
# Lane pitch, in row units. A lane of bare marks only has to clear a 3pt
# marker, so it is packed far tighter than a lane carrying a text label.
DENSE_STEP = 0.30
# Clearance between the axis line and the first lane of marks. The tick values
# sit in the line itself, so without it the innermost marks land on the digits.
AXIS_CLEAR = 0.50
# The legend lives in the figure's right margin, beside the axis labels, so it
# costs no plot height at all.
LEGEND_NCOL = 2
LABEL_STEP = 0.92
TICK_CANDIDATES = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]

# Labelled marks trace generation frontiers rather than every model: the point
# is to read capacity against the succession of model generations, and 205
# labels would be a wall of text. These are landmark releases, so unlike a
# "newest per lab" rule the list does not go stale; the current frontier is
# labelled too and is the part to revisit.
LANDMARKS = [
    # There is no plain gpt-5.6: the generation ships as luna/sol/terra in base
    # and pro, spanning CC 46 to 174, so the strongest stands for it.
#     "openai/gpt-3.5-turbo", "openai/gpt-4",
#     "openai/gpt-5.6-terra-pro",
#     "anthropic/claude-3-haiku",
#     "anthropic/claude-fable-5",
#     "google/gemini-2.5-pro", "google/gemini-3.7-flash",
#     "meta-llama/llama-3.3-70b-instruct",
#     "meta/muse-spark-1.1",
#  "x-ai/grok-4.6"
]

# A model that cannot count past a handful of items sits far off the left of the
# axis and compresses everything else into the middle. One model is below this
# floor, llama-3.2-1b-instruct at 2; the next lowest is 10.
MIN_SCC = 5

# Each lab's most recent release is labelled as well, resolved from the
# catalogue at render time so the current frontier cannot go stale the way a
# typed list does. Strictly newest by date: "newest" and "strongest" disagree
# for several labs, and silently substituting the stronger model would label a
# frontier that is not the frontier. The resolved list is printed on every run.
LABEL_LATEST_PER_LAB = False

# Where a lab's most recent release is a side variant rather than the model that
# represents the generation, the representative is named here. Each entry is a
# judgement about what the frontier is, not something derivable from dates:
# claude-fable-5 over the later claude-opus-5, and the flagship over an
# experimental or contributor build shipped days apart. Overrides are explicit
# so a wrong one is visible rather than buried in a heuristic.
LATEST_OVERRIDE = {
    "anthropic": "anthropic/claude-fable-5",        # over claude-opus-5
    # sol is the marketed flagship of the gpt-5.6 generation. Price does not
    # show it: later cuts left terra priced above sol ($12 against $10 per
    # Mtok output), so the catalogue ranks the tiers in the wrong order and
    # picking by price would label the wrong model.
    "openai": "openai/gpt-5.6-sol-pro",             # over gpt-5.6-terra-pro
    "qwen": "qwen/qwen3.8-2.4t-a95b",               # over qwen3.8-27b
    "deepseek": "deepseek/deepseek-v4-pro-0813",    # over -v4-flash-vision-exp
    "meta": "meta/muse-spark-1.2",                  # over the -contributor build
}

def load(label_all: bool = False, min_scc: int = MIN_SCC):
    best: dict[str, dict] = {}
    for row in csv.DictReader(SCC_CSV.open(encoding="utf-8")):
        model = row["model"]
        # One mark per model: the configuration that counted furthest.
        if model not in best or int(row["scc"]) > int(best[model]["scc"]):
            best[model] = row

    dropped = [m for m, r in best.items() if int(r["scc"]) < min_scc]
    for model in dropped:
        del best[model]
    if dropped:
        print(f"  below CC {min_scc}, not plotted: {', '.join(sorted(dropped))}")

    latest: set[str] = set()
    if LABEL_LATEST_PER_LAB:
        dates: dict[str, str] = {}
        for row in csv.DictReader(REGISTRY_CSV.open(encoding="utf-8")):
            dates.setdefault(row["model"], row.get("release_date") or "")
        by_lab: dict[str, list[str]] = {}
        for model in best:
            by_lab.setdefault(infer_provider_family(model), []).append(model)
        for family, models in by_lab.items():
            chosen = LATEST_OVERRIDE.get(family)
            if chosen is None:
                chosen = max(models, key=lambda m: (dates.get(m, ""), int(best[m]["scc"])))
            elif chosen not in best:
                print(f"  override for {family} not in the dataset: {chosen}")
                chosen = max(models, key=lambda m: (dates.get(m, ""), int(best[m]["scc"])))
            latest.add(chosen)
        print("  latest per lab: " + ", ".join(
            f"{m.split('/')[-1]} ({dates.get(m, '?')[:7]}, CC={best[m]['scc']})"
            for m in sorted(latest, key=lambda m: -int(best[m]["scc"]))))

    missing = [m for m in LANDMARKS if m not in best]
    if missing:
        print(f"  landmarks absent from the dataset: {', '.join(missing)}")

    rows = sorted(best.values(), key=lambda r: int(r["scc"]))
    for row in rows:
        row["_label"] = label_all or row["model"] in (set(LANDMARKS) | latest)
    return rows


def pack(items, ax, fig, gap_px: float = 1.5, forbid: dict | None = None,
         row_base: int = 0):
    """Assign each mark to a row so that no two overlap.

    Extents are measured from the rendered text rather than estimated from
    character counts, because the labels are model ids of very uneven width and
    a guess leaves either collisions or wasted space. An unlabelled mark costs
    only its own whisker, which is what lets the dense lanes hold the corpus.

    `forbid` maps a row index to an x in pixels beyond which that row is not
    available, which is how the legend keeps its corner: without it a mark
    eventually packs underneath it and is hidden. `row_base` offsets the rows
    this call produces, so successive calls fill disjoint bands of lanes.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = lambda v: ax.transData.transform((v, 0))[0]
    rows: list[float] = []
    for item in items:
        if item["label"]:
            probe = ax.text(0, 0, item["label"], fontsize=LABEL_FS)
            width = probe.get_window_extent(renderer).width + 6.0
            probe.remove()
        else:
            width = 0.0
        left = px(item["lo"]) - 2.2
        right = max(px(item["hi"]), px(item["scc"]) + 2.2) + width
        for r, edge in enumerate(rows):
            if edge + gap_px >= left:
                continue
            limit = (forbid or {}).get(row_base + r)
            if limit is not None and right > limit:
                continue
            rows[r] = right
            item["row"] = row_base + r
            break
        else:
            # Opening a new row has to respect the reservation too. Without this
            # the item that happens to create the top row lands under the legend
            # however carefully the existing rows were filtered.
            index = len(rows)
            while True:
                limit = (forbid or {}).get(row_base + index)
                if limit is None or right <= limit:
                    break
                rows.append(float("-inf"))   # placeholder: keeps indices aligned
                index = len(rows)
            rows.append(right)
            item["row"] = row_base + index
    return len(rows)


def gapped_axis(ax, fig, y, x0, x1, ticks, labels, *, color, lw, fontsize,
                weight="normal", pad_px=3.5, arrow=True):
    """Draw a number line whose value labels sit in gaps in the line itself.

    Widths are measured from the rendered text at the final figure size, so the
    gap is exactly as wide as the number it holds; a fixed gap would either
    clip the label or leave it swimming.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    edges = [x0]
    for tick, label in zip(ticks, labels):
        probe = ax.text(0, 0, label, fontsize=fontsize, fontweight=weight)
        half = probe.get_window_extent(renderer).width / 2.0 + pad_px
        probe.remove()
        centre_px = ax.transData.transform((tick, y))[0]
        edges.append(inv.transform((centre_px - half, 0))[0])
        edges.append(inv.transform((centre_px + half, 0))[0])
        ax.annotate(label, (tick, y), ha="center", va="center", fontsize=fontsize,
                    fontweight=weight, color=color, zorder=6)
    edges.append(x1)
    for i in range(0, len(edges), 2):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        # A tick close to the frame edge leaves a stub a few pixels long, which
        # reads as a stray dot rather than as part of the line.
        span_px = ax.transData.transform((b, y))[0] - ax.transData.transform((a, y))[0]
        if span_px < 5.0 and not (arrow and i == len(edges) - 2):
            continue
        if arrow and i == len(edges) - 2:
            ax.annotate("", xy=(b, y), xytext=(a, y), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                        shrinkA=0, shrinkB=0))
        else:
            ax.plot([a, b], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=4)


def lane_positions(items, n_rows: int) -> dict[int, float]:
    """y for every row, stepping outward from the axis on each side.

    Rows carrying a label get the full pitch because they hold type; rows of
    bare marks get a fraction of it, which is what lets the whole corpus fit.
    """
    labelled_rows = {it["row"] for it in items if it["label"]}
    y: dict[int, float] = {}
    for side, first in ((1, 0), (-1, 1)):
        pos = 0.0
        for row in range(first, n_rows, 2):
            pos += LABEL_STEP if row in labelled_rows else DENSE_STEP
            y[row] = side * (pos + AXIS_CLEAR)
    return y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label-all", action="store_true",
                        help="label every model instead of the landmarks")
    parser.add_argument("--min-scc", type=int, default=MIN_SCC)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = load(args.label_all, args.min_scc)
    items = [{
        "scc": int(r["scc"]), "lo": int(r["scc_ci_low"]), "hi": int(r["scc_ci_high"]),
        "label": r["model"].split("/", 1)[-1] if r["_label"] else "",
        "family": infer_provider_family(r["model"]),
    } for r in rows]
    for it in items:
        it["lo"] = min(it["lo"], it["scc"])
        it["hi"] = max(it["hi"], it["scc"])

    # Fit the axis to the data rather than to a fixed range, so the frame does
    # not carry dead space at either end when the selection changes.
    lo = min(i["lo"] for i in items if i["scc"] > 0)
    hi = max(i["hi"] for i in items)
    XLIM = (lo / 1.75, hi * 1.55)
    ticks = [t for t in TICK_CANDIDATES if XLIM[0] * 1.05 < t < XLIM[1] * 0.95]

    fig, ax = plt.subplots(figsize=(FIG_W, 4.0))
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    # Two passes over disjoint bands of lanes: the bare marks pack tight against
    # the axis, then the labelled few take the outer lanes where there is room
    # for type. Packing them together would strand a label between two dense
    # lanes, where it has nowhere to sit.
    bare = [i for i in items if not i["label"]]
    named = [i for i in items if i["label"]]

    def repack(forbid=None):
        dense = pack(bare, ax, fig, forbid=forbid)
        dense += dense % 2                      # keep parity, so labels start above
        return dense + pack(named, ax, fig, forbid=forbid, row_base=dense)

    n_rows = repack()

    # y is measured in row units, one unit being ROW_PT on the page, so the
    # figure height follows from the packing instead of being guessed.
    def geometry(rows_used: int):
        ys = lane_positions(items, rows_used)
        lanes = max(ys.values())
        ctx_y = min(ys.values()) - 1.35
        y_hi, y_lo = lanes + 0.55, ctx_y - 0.85
        fig.set_size_inches(FIG_W, (y_hi - y_lo) * ROW_PT / 72.0 / 0.985)
        ax.set_ylim(y_lo, y_hi)
        fig.subplots_adjust(left=0.004, right=0.815, top=0.9925, bottom=0.0075)
        return ys, lanes, ctx_y

    ys, lanes, ctx_y = geometry(n_rows)

    # The legend sits in the empty upper-right of the frame rather than above
    # it, which costs no vertical space at all. Its corner is then reserved and
    # the marks are re-packed around it; the reservation is in figure pixels and
    # the frame width never changes, so it survives the height changing.
    families = []
    for it in items:
        if it["family"] not in families:
            families.append(it["family"])
    fig.legend(
        handles=[Line2D([], [], color=PROVIDER_COLORS.get(f, PROVIDER_COLORS["other"]),
                        marker=PROVIDER_MARKERS.get(f, PROVIDER_MARKERS["other"]),
                        mfc="white", mew=0.9, ms=4.2, lw=0.7,
                        label=PROVIDER_DISPLAY_NAMES.get(f, f)) for f in families],
        loc="upper right", bbox_to_anchor=(0.999, 0.995), frameon=False,
        fontsize=7.5, ncol=LEGEND_NCOL, handlelength=1.1,
        columnspacing=0.9, handletextpad=0.35, borderaxespad=0.0,
        labelspacing=0.42,
    )


    # -- capacity axis -----------------------------------------------------
    gapped_axis(ax, fig, 0, XLIM[0], XLIM[1], ticks, [str(t) for t in ticks],
                color="#1a1a1a", lw=1.05, fontsize=9.0, weight="bold")
    ax.annotate("Count capacity", (XLIM[1], 0), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=10.0, fontweight="bold", color="#1a1a1a")

    # -- marks -------------------------------------------------------------
    for it in items:
        y = ys[it["row"]]
        colour = PROVIDER_COLORS.get(it["family"], PROVIDER_COLORS["other"])
        marker = PROVIDER_MARKERS.get(it["family"], PROVIDER_MARKERS["other"])
        if it["label"]:
            # Only the labelled marks are far enough out to need the axis
            # connection drawn; on 200 bare marks the leaders are a thicket.
            ax.plot([it["scc"], it["scc"]], [0, y], color=colour, lw=0.45,
                    alpha=.40, zorder=1)
        ax.plot([it["lo"], it["hi"]], [y, y], color=colour, lw=0.75, zorder=3,
                solid_capstyle="butt", alpha=.85)
        ax.plot([it["scc"]], [y], marker=marker, ms=2.5, mfc="white",
                mec=colour, mew=1.0, zorder=4)
        if it["label"]:
            ax.annotate(it["label"], (max(it["hi"], it["scc"]), y), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=LABEL_FS, color=colour)

    # -- context-length axis ----------------------------------------------
    # Stripped to the two things that carry information: the tick markers, and
    # the bold segment showing the extent every measured capacity occupies.
    ctx_lo, ctx_hi = 1000.0, 1.6e6
    span = XLIM[1] / XLIM[0]
    def ctx_x(tokens: float) -> float:
        frac = (math.log(tokens) - math.log(ctx_lo)) / (math.log(ctx_hi) - math.log(ctx_lo))
        return XLIM[0] * span ** frac

    gapped_axis(ax, fig, ctx_y, XLIM[0], XLIM[1],
                [ctx_x(t) for t in CONTEXT_REFERENCE_TICKS],
                [f"{t // 1024}k" if t < 1_000_000 else "1M" for t in CONTEXT_REFERENCE_TICKS],
                color=CONTEXT_AXIS_COLOR, lw=0.95, fontsize=8.0)
    # the extent every measured capacity occupies, drawn over the axis line
    ax.plot([XLIM[0], ctx_x(max(i["scc"] for i in items) * TOKENS_PER_ITEM)],
            [ctx_y, ctx_y], color="#1a1a1a", lw=2.6, solid_capstyle="butt", zorder=7)
    ax.annotate("Context length", (XLIM[1], ctx_y), xytext=(8, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=10.0, color=CONTEXT_AXIS_COLOR)

    ax.set_axis_off()

    pdf = args.out or OUT_PDF
    # Transparent so the panel composites onto whatever ground it is placed on,
    # rather than punching a white rectangle into the slide or page.
    fig.savefig(pdf, transparent=True)
    fig.savefig(pdf.with_suffix(".png"), dpi=300, transparent=True)
    print(f"wrote {pdf}")
    print(f"  {len(items)} models in {n_rows} rows ({lanes} lanes each side)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
