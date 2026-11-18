#!/usr/bin/env python3
"""Figures for the cross-model mechanistic suite (results/suite/).

    python -m counting.analysis.figures.mechanistic_suite

Outputs (paper/figures/):
    mechanistic-suite-panels.pdf   the eight main-text data panels, unlettered
                                   (the eight data panels; letters are composed in the
                                   figure source file)
    suite-readout-atlas.pdf        SI: readout vs true count, 6 models x 4 sites
    suite-clamp-detail.pdf         SI: decoded answer vs clamp target, all sites
    suite-chimera-matrix.pdf       SI: first-token transfer, 3 conditions
    suite-mixed-grids.pdf          SI: mixed-sequence correctness maps
    suite-margin-curves.pdf        SI: teacher-forced margin vs count
    suite-mode-scans.pdf           SI: immediate vs reasoning scan records
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from counting.analysis.figures.attractors import set_spine_width
from counting.paths import REPO_ROOT as ROOT

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

SUITE = ROOT / "results" / "suite"
OUT = ROOT / "paper" / "figures"

MODELS = ["gpt-oss-20b", "gpt-oss-120b", "gemma-4-26b-a4b-it",
          "muse-glimmer-30b", "qwen3.6-35b-a3b", "llama-3.3-70b-instruct"]
LABELS = {"gpt-oss-20b": "gpt-oss-20b", "gpt-oss-120b": "gpt-oss-120b",
          "gemma-4-26b-a4b-it": "gemma-4-26b", "muse-glimmer-30b": "muse-glimmer-30b",
          "qwen3.6-35b-a3b": "qwen3.6-35b", "llama-3.3-70b-instruct": "llama-3.3-70b"}
COLORS = {"gpt-oss-20b": "#7fa8d9", "gpt-oss-120b": "#2f6fb3",
          "gemma-4-26b-a4b-it": "#0f8f80", "muse-glimmer-30b": "#b07818",
          "qwen3.6-35b-a3b": "#7d5bb5", "llama-3.3-70b-instruct": "#c25555"}
N_LAYERS = {"gpt-oss-20b": 24, "gpt-oss-120b": 36, "gemma-4-26b-a4b-it": 30,
            "muse-glimmer-30b": 52, "qwen3.6-35b-a3b": 40,
            "llama-3.3-70b-instruct": 80}
IMMEDIATE = {"gpt-oss-20b": "final", "gpt-oss-120b": "final",
             "gemma-4-26b-a4b-it": "nothink", "muse-glimmer-30b": "final",
             "qwen3.6-35b-a3b": "nothink", "llama-3.3-70b-instruct": "final"}
REASONING = {"gpt-oss-20b": "default", "gpt-oss-120b": "default",
             "gemma-4-26b-a4b-it": "think", "muse-glimmer-30b": "default",
             "qwen3.6-35b-a3b": "think"}

TITLE_FS, LABEL_FS, TICK_FS, SMALL_FS = 15, 12.5, 10.5, 9


def load_tables() -> dict[str, dict]:
    rows = json.loads((SUITE / "tables.json").read_text())
    return {r["model"]: r for r in rows}


def sites_of(entry: dict) -> list[str]:
    keys = [k.split("|")[0] for k in entry["geometry"]]
    return sorted(keys, key=lambda s: int(s.rsplit("layer", 1)[1]))


def depth_of(model: str, site: str) -> float:
    return (int(site.rsplit("layer", 1)[1]) + 1) / N_LAYERS[model]


def load_projections(model: str) -> list[dict]:
    path = SUITE / model / "geometry.projections.csv"
    return list(csv.DictReader(path.open()))


def immediate_scc(entry: dict, model: str) -> float:
    return entry["scans"][IMMEDIATE[model]]["scc"]


def style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=TITLE_FS)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FS)
    ax.grid(alpha=0.2, linewidth=0.9)
    set_spine_width(ax)


# -- main-sheet panels -------------------------------------------------------

def panel_rmse_depth(ax, tables):
    for m in MODELS:
        e = tables[m]
        pts = sorted(((depth_of(m, k.split("|")[0]), v["cv_rmse_items"])
                      for k, v in e["geometry"].items()))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=4.5,
                lw=1.6, color=COLORS[m], label=LABELS[m])
    ax.set_ylim(0, 3.5)
    style(ax, "Readout Precision", "Layer depth (fraction)", "Held-out RMSE (items)")
    ax.legend(fontsize=7.2, frameon=False, handlelength=1.1, labelspacing=0.25,
              borderaxespad=0.15, loc="upper left")


def panel_exemplar(ax, tables, model="llama-3.3-70b-instruct"):
    rows = load_projections(model)
    sites = sites_of(tables[model])
    scc = immediate_scc(tables[model], model)
    for site, colour, lab in ((sites[0], "#2f6fb3", f"layer {sites[0].rsplit('layer',1)[1]}"),
                              (sites[-1], "#c25555", f"layer {sites[-1].rsplit('layer',1)[1]}")):
        pts = sorted((int(r["count"]), float(r["extrapolated_count"]))
                     for r in rows if r["site"] == site)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", lw=1.7,
                color=colour, label=lab)
    lim = max(p for r in rows for p in (int(r["count"]),)) * 1.02
    ax.plot([0, lim], [0, lim], ls="--", lw=1.1, color="#999999", zorder=1)
    ax.axvline(scc, color="#666666", lw=1.1, ls=":")
    ax.axvspan(0, scc, color="#2f6fb3", alpha=0.06, zorder=0)
    style(ax, "Boundary Extrapolation", "True count", "Linear readout")
    ax.text(0.97, 0.05, LABELS[model], transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.5, color="#555555")
    ax.legend(fontsize=8, frameon=False, loc="upper left", borderaxespad=0.15)


def panel_beyond_slopes(ax, tables):
    for i, m in enumerate(MODELS):
        e = tables[m]
        sites = sites_of(e)
        s_first = e["geometry"][f"{sites[0]}|decode_point"]["beyond_slopes"]["1.0-1.2x"]
        s_last = e["geometry"][f"{sites[-1]}|decode_point"]["beyond_slopes"]["1.0-1.2x"]
        ax.plot([i, i], [s_first, s_last], color="#aaaaaa", lw=1.2, zorder=2)
        ax.scatter([i], [s_first], s=42, color=COLORS[m], zorder=3)
        ax.scatter([i], [s_last], s=42, color=COLORS[m], marker="s",
                   facecolors="white", linewidths=1.6, edgecolors=COLORS[m], zorder=3)
    ax.axhline(0, color="#666666", lw=1.0)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=40, ha="right",
                       fontsize=7.6)
    style(ax, "Late Consolidation Failure", None, "Readout slope, 1.0-1.2x SCC")
    ax.scatter([], [], s=42, color="#555555", label="first layer")
    ax.scatter([], [], s=42, marker="s", facecolors="white", edgecolors="#555555",
               linewidths=1.6, label="last layer")
    ax.legend(fontsize=8, frameon=False, loc="lower left", borderaxespad=0.15)


def panel_margin(ax, tables):
    for m in MODELS:
        mc = tables[m]["margin_collapse"]
        ax.scatter([mc["first_negative_min_gap"]], [mc["first_greedy_divergence"]],
                   s=60, color=COLORS[m], edgecolors="white", linewidths=0.8,
                   zorder=3, label=LABELS[m])
    ax.plot([1, 80], [1, 80], ls="--", lw=1.1, color="#999999", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks([2, 5, 10, 20, 50]); ax.set_yticks([2, 5, 10, 20, 50])
    style(ax, "Margin-Behavior Coincidence", "First negative margin",
          "First behavioral error")


def panel_chimera(ax, tables):
    styles = {"chimera_full": ("-", "full state"),
              "chimera_residual_only": ("--", "nonlinear residual"),
              "chimera_linear_only": (":", "linear coordinate")}
    for cond, (ls, lab) in styles.items():
        xs_all, ys_all = [], []
        for m in MODELS:
            e = tables[m]
            sites = sites_of(e)
            xs = [depth_of(m, s) for s in sites]
            ys = [e["interventions"]["sites"][s][cond]["first_token_hit"]
                  / e["interventions"]["sites"][s][cond]["n"] for s in sites]
            ax.plot(xs, ys, ls, lw=0.9, color=COLORS[m], alpha=0.45)
            xs_all.append(xs); ys_all.append(ys)
        mean_x = np.mean(xs_all, axis=0)
        mean_y = np.mean(ys_all, axis=0)
        ax.plot(mean_x, mean_y, ls, lw=2.6, color="#1a1a1a", label=lab)
    ax.set_ylim(0, 1.05)
    style(ax, "Donor-State Transfer", "Layer depth (fraction)",
          "First-token transfer rate")
    ax.legend(fontsize=8, frameon=False, loc="upper left", borderaxespad=0.15)


def panel_inertness(ax, tables):
    fams = [("clamp", "clamp"), ("step_translate", "step"), ("random_control", "random")]
    width = 0.26
    for fi, (fam, lab) in enumerate(fams):
        xs, ys = [], []
        for i, m in enumerate(MODELS):
            moved = tot = 0
            for s in tables[m]["interventions"]["sites"].values():
                moved += s[fam]["moved"]
                tot += s[fam]["n"]
            xs.append(i + (fi - 1) * width)
            ys.append(moved)
        ax.bar(xs, ys, width=width - 0.03, zorder=3,
               color=["#2f6fb3", "#b07818", "#8d8d8d"][fi], label=lab,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=40, ha="right",
                       fontsize=7.6)
    ax.set_ylim(0, 33)
    style(ax, "Decode-Point Null", None, "Answers moved (count)")
    ax.legend(fontsize=8, frameon=False, loc="upper left", borderaxespad=0.15)


def panel_readable_usable(ax, tables):
    for i, m in enumerate(MODELS):
        e = tables[m]["mixed"]
        best_r2 = max(s["cv_r2_count"] for s in e["sites"].values())
        acc = e["behavioral_accuracy"]
        ax.plot([i, i], [100 * acc, 100 * best_r2], color="#aaaaaa", lw=1.2, zorder=2)
        ax.scatter([i], [100 * best_r2], s=46, color=COLORS[m], zorder=3)
        ax.scatter([i], [100 * acc], s=46, marker="s", facecolors="white",
                   edgecolors=COLORS[m], linewidths=1.6, zorder=3)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=40, ha="right",
                       fontsize=7.6)
    ax.set_ylim(-18, 106)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    style(ax, "Readability Gap", None, "Percent")
    ax.scatter([], [], s=46, color="#555555", label="count readout $R^2$")
    ax.scatter([], [], s=46, marker="s", facecolors="white", edgecolors="#555555",
               linewidths=1.6, label="mixed-task accuracy")
    ax.legend(fontsize=8, frameon=False, loc="lower left", borderaxespad=0.15)


def panel_mode_switch(ax, tables):
    for i, m in enumerate(MODELS):
        e = tables[m]
        imm = e["scans"][IMMEDIATE[m]]["scc"]
        ax.scatter([i], [imm], s=52, color=COLORS[m], zorder=4)
        if m in REASONING and REASONING[m] in e["scans"]:
            rea = e["scans"][REASONING[m]]["scc"]
            ax.annotate("", (i, rea), (i, imm),
                        arrowprops=dict(arrowstyle="-|>", lw=1.7,
                                        color=COLORS[m], shrinkA=4, shrinkB=1))
            ax.scatter([i], [rea], s=52, marker="s", facecolors="white",
                       edgecolors=COLORS[m], linewidths=1.7, zorder=4)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=40, ha="right",
                       fontsize=7.6)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(25, 124)
    style(ax, "Reasoning-Mode Capacity", None, "SCC")
    ax.scatter([], [], s=52, color="#555555", label="immediate answer")
    ax.scatter([], [], s=52, marker="s", facecolors="white", edgecolors="#555555",
               linewidths=1.7, label="reasoning mode")
    ax.legend(fontsize=8, frameon=False, loc="upper right", borderaxespad=0.15)


# -- round-2 panels (results/suite/round2_tables.json) ------------------------

def load_round2() -> dict[str, dict]:
    return json.loads((SUITE / "round2_tables.json").read_text())


def layer_num(site: str) -> int:
    return int(site.rsplit("layer", 1)[1])


def shift_sites(r2m: dict) -> list[str]:
    keys = {k.split("|")[0] for k in r2m["accumulate"]["shift"]}
    return sorted(keys, key=layer_num)


def best_shift_site(r2m: dict) -> tuple[str, float]:
    best = None
    for site in shift_sites(r2m):
        rec = r2m["accumulate"]["shift"].get(f"{site}|stable|S|local")
        if rec and (best is None or rec["slope"] > best[1]):
            best = (site, rec["slope"])
    return best


DELTAS = [-8, -4, -2, -1, 1, 2, 4, 8]
RESPONSIVE_SLOPE = 0.15


def panel_exact_depth(ax, r2):
    for m in MODELS:
        ex = r2[m]["exact"]
        pts = sorted((depth_of(m, site), v["in_regime"]["exact"])
                     for site, v in ex["sites"].items())
        ax.plot([q[0] for q in pts], [q[1] for q in pts], "-o", ms=4.5, lw=1.6,
                color=COLORS[m], label=LABELS[m])
    ax.set_ylim(0, 1.04)
    ax.set_xlim(0, 1.04)
    style(ax, "Readout Exactness", "Layer depth (fraction)",
          "Exact match of rounded readout")
    ax.legend(fontsize=7.2, frameon=False, handlelength=1.1, labelspacing=0.25,
              borderaxespad=0.15, loc="lower left")


def panel_shift(ax, r2):
    xs = DELTAS
    pooled = {"R": [], "S_decode_point": []}
    for m in MODELS:
        site, slope = best_shift_site(r2[m])
        sh = r2[m]["accumulate"]["shift"]
        ys = [sh[f"{site}|stable|S|local"]["mean_shift_by_delta"][str(d)] for d in xs]
        if slope >= RESPONSIVE_SLOPE:
            ax.plot(xs, ys, "-o", ms=4.2, lw=1.7, color=COLORS[m],
                    label=f"{LABELS[m]}, layer {layer_num(site)}")
            yb = [sh[f"{site}|beyond|S|local"]["mean_shift_by_delta"][str(d)] for d in xs]
            ax.plot(xs, yb, "--", lw=0.9, color=COLORS[m], alpha=0.45)
            for cond in pooled:
                pooled[cond].append([sh[f"{site}|stable|{cond}|local"]["mean_shift_by_delta"][str(d)] for d in xs])
        else:
            ax.plot(xs, ys, "-", lw=1.0, color="#bbbbbb", zorder=1)
    ax.plot(xs, np.mean(pooled["R"], axis=0), "x--", ms=5, lw=1.1, color="#555555",
            label="random direction")
    ax.plot(xs, np.mean(pooled["S_decode_point"], axis=0), "s:", ms=3.6, lw=1.1,
            color="#555555", label="decode point only")
    ax.plot([-8, 8], [-8, 8], ls="--", lw=1.0, color="#cccccc", zorder=0)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-7, 7)
    style(ax, "Shifting the Accumulation", "Shift of running count (items)",
          "Answer shift (items)")
    ax.plot([], [], "-", lw=1.0, color="#bbbbbb", label="non-responsive models")
    ax.plot([], [], "--", lw=0.9, color="#888888", alpha=0.7, label="same edit beyond CC")
    ax.legend(fontsize=6.8, frameon=False, loc="upper left", handlelength=1.6,
              labelspacing=0.2, borderaxespad=0.1)


def panel_depth_map(ax, r2):
    n = len(MODELS)
    for i, m in enumerate(MODELS):
        y = n - 1 - i
        e = r2[m]
        ax.plot([0, 1], [y, y], color="#dddddd", lw=1.0, zorder=1)
        for site in shift_sites(e):
            x = depth_of(m, site)
            rec = e["accumulate"]["shift"].get(f"{site}|stable|S|local")
            slope = rec["slope"] if rec else 0.0
            if slope >= RESPONSIVE_SLOPE:
                ax.scatter([x], [y], s=60 + 420 * slope, color=COLORS[m], zorder=4,
                           edgecolors="white", linewidths=0.8)
                ax.text(x, y + 0.34, f"{slope:.2f}", ha="center", va="bottom",
                        fontsize=7, color=COLORS[m])
            else:
                ax.scatter([x], [y], s=26, facecolors="white", edgecolors="#999999",
                           linewidths=1.0, zorder=3)
        nl = N_LAYERS[m]
        for task, marker, fc, dy in (("counting", "v", COLORS[m], 0.0), ("copy", "^", "white", 0.22),
                                     ("retrieval", "s", "white", -0.22)):
            layer = e[task]["localize"]["consolidation_layer"]
            ax.scatter([(layer + 1) / nl], [y + dy], s=36, marker=marker, facecolors=fc,
                       edgecolors=COLORS[m], linewidths=1.2, zorder=5)
        ax.text(0.01, y + 0.16, LABELS[m], fontsize=7.2, color=COLORS[m], va="bottom",
                ha="left", zorder=6)
    ax.set_yticks([])
    ax.set_xlim(0, 1.04)
    ax.set_ylim(-0.6, n - 0.2)
    style(ax, "Read Mid, Delivered Late", "Layer depth (fraction)", None)
    ax.scatter([], [], s=70, color="#555555", label="shift moves the answer")
    ax.scatter([], [], s=26, facecolors="white", edgecolors="#999999", linewidths=1.0,
               label="hooked site, no effect")
    ax.scatter([], [], s=36, marker="v", color="#555555", label="channel fills: counting")
    ax.scatter([], [], s=36, marker="^", facecolors="white", edgecolors="#555555",
               linewidths=1.2, label="channel fills: copy")
    ax.scatter([], [], s=36, marker="s", facecolors="white", edgecolors="#555555",
               linewidths=1.2, label="channel fills: retrieval")
    ax.legend(fontsize=6.4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              ncol=2, labelspacing=0.15, borderaxespad=0.0, handletextpad=0.4,
              columnspacing=0.7)


def panel_readable_usable_exact(ax, r2):
    for i, m in enumerate(MODELS):
        ex = r2[m]["exact"]
        best = max(ex["sites"].values(), key=lambda v: v["mixed_in_grid_cv"]["within1"])
        p_ex, p_w1 = best["mixed_in_grid_cv"]["exact"], best["mixed_in_grid_cv"]["within1"]
        m_ex, m_w1 = ex["model"]["mixed_exact"], ex["model"]["mixed_within1"]
        ax.plot([i - 0.16, i - 0.16], [m_ex, p_ex], color="#bbbbbb", lw=1.1, zorder=2)
        ax.plot([i + 0.16, i + 0.16], [m_w1, p_w1], color="#bbbbbb", lw=1.1, zorder=2)
        ax.scatter([i - 0.16], [p_ex], s=42, color=COLORS[m], zorder=3)
        ax.scatter([i - 0.16], [m_ex], s=42, facecolors="white", edgecolors=COLORS[m],
                   linewidths=1.5, zorder=3)
        ax.scatter([i + 0.16], [p_w1], s=48, marker="D", color=COLORS[m], zorder=3)
        ax.scatter([i + 0.16], [m_w1], s=48, marker="D", facecolors="white",
                   edgecolors=COLORS[m], linewidths=1.5, zorder=3)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=40, ha="right",
                       fontsize=7.6)
    ax.set_ylim(-0.34, 1.04)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    style(ax, "Readability Gap", None, "Rate on mixed grids")
    ax.scatter([], [], s=42, color="#555555", label="readout, exact")
    ax.scatter([], [], s=42, facecolors="white", edgecolors="#555555", linewidths=1.5,
               label="model, exact")
    ax.scatter([], [], s=48, marker="D", color="#555555", label="readout, within one")
    ax.scatter([], [], s=48, marker="D", facecolors="white", edgecolors="#555555",
               linewidths=1.5, label="model, within one")
    ax.legend(fontsize=6.8, frameon=False, loc="lower left", labelspacing=0.2,
              borderaxespad=0.1, ncol=2, columnspacing=0.8, handletextpad=0.4)


# -- boundary-window transplant: counting and an object history --------------

BOUNDARY_OPERATORS = {  # frozen operator per model (window, layers) and the sets it was run on
    "gpt-oss-20b": ("B1f:w8:L8-23", ("heldout_summary.json", "heldout_extra_summary.json")),
    "gemma-4-26b-a4b-it": ("B1f:w12:L8-29", ("heldout_extra_summary.json",)),
    "gpt-oss-120b-cuda": ("B1f:w8:L5-35", ("heldout_summary.json", "heldout_extra_summary.json")),
}
BOUNDARY_LABELS = {"gpt-oss-20b": "gpt-oss-20b", "gemma-4-26b-a4b-it": "gemma-4-26b",
                   "gpt-oss-120b-cuda": "gpt-oss-120b"}
BOUNDARY_COLORS = {"gpt-oss-20b": COLORS["gpt-oss-20b"], "gemma-4-26b-a4b-it": COLORS["gemma-4-26b-a4b-it"],
                   "gpt-oss-120b-cuda": COLORS["gpt-oss-120b"]}


def load_boundary() -> dict:
    """Pooled held-out rates of the frozen counting operator, and the object-task transplant."""
    out = {"counting": {}, "objects": {}}
    for m, (op, files) in BOUNDARY_OPERATORS.items():
        d = SUITE / m / "causal-crossing-v2" / "track_b"
        tot = {"edit": [0, 0], "sham": [0, 0], "b0": [0, 0]}
        for f in files:
            h = json.loads((d / f).read_text())
            for key, k in ((f"{op}|edit", "edit"), (f"{op}|sham", "sham"), ("B0|edit", "b0")):
                e = h[key]
                elig = e["n"] - e["classes"].get("unavailable", 0)
                tot[k][0] += e["unchanged"] if k == "sham" else e["predicted"]
                tot[k][1] += elig
        out["counting"][m] = {k: (v[0], v[1]) for k, v in tot.items()}
    g = SUITE / "gemma-4-26b-a4b-it" / "causal-crossing-v2" / "grid"
    early = json.loads((g / "early_summary.json").read_text())
    think = json.loads((g / "early_think_summary.json").read_text())
    for k in (3, 4, 5):
        c = early[f"k{k}|s2"]["replace"]
        row = {cond: (c[cond]["changed"]["follows_donor"], c[cond]["changed"]["n"])
               for cond in ("edit", "sham", "whole_cache")}
        row["factual"] = (c["edit"]["factual"]["unchanged"], c["edit"]["factual"]["n"])
        t = think.get(f"k{k}|s2|edit|changed")
        row["think"] = (t["donor"], t["complete"]) if t else None
        out["objects"][k] = row
    return out


def panel_boundary_transfer(ax, b=None):
    b = b or load_boundary()
    xs_c = [0.0, 1.0, 2.0]
    xs_o = [3.5, 4.5, 5.5]
    ticks, labels = [], []
    for x, m in zip(xs_c, BOUNDARY_OPERATORS):
        r = b["counting"][m]
        e, sh, b0 = r["edit"][0] / r["edit"][1], 1 - r["sham"][0] / r["sham"][1], r["b0"][0] / r["b0"][1]
        ax.plot([x, x], [sh, e], "-", lw=1.4, color=BOUNDARY_COLORS[m], zorder=2)
        ax.plot(x, b0, "D", ms=4.6, color="#555555", zorder=3)
        ax.plot(x, sh, "o", ms=6, mfc="white", mec=BOUNDARY_COLORS[m], mew=1.4, zorder=4)
        ax.plot(x, e, "o", ms=6.5, color=BOUNDARY_COLORS[m], zorder=5)
        ax.annotate(f"{r['edit'][0]}/{r['edit'][1]}", (x, e), xytext=(5, 2), textcoords="offset points",
                    fontsize=7, color=BOUNDARY_COLORS[m], va="bottom")
        ticks.append(x); labels.append(BOUNDARY_LABELS[m])
    col = COLORS["gemma-4-26b-a4b-it"]
    for x, k in zip(xs_o, (3, 4, 5)):
        r = b["objects"][k]
        e, sh, b0 = (r[c][0] / r[c][1] for c in ("edit", "sham", "whole_cache"))
        ax.plot([x, x], [sh, e], "-", lw=1.4, color=col, zorder=2)
        ax.plot(x, b0, "D", ms=4.6, color="#555555", zorder=3)
        ax.plot(x, sh, "o", ms=6, mfc="white", mec=col, mew=1.4, zorder=4)
        ax.plot(x, e, "o", ms=6.5, color=col, zorder=5)
        ax.plot(x, r["factual"][0] / r["factual"][1], "^", ms=6, color="#2a9d5c", zorder=5)
        if r["think"]:
            ax.plot(x, r["think"][0] / r["think"][1], "*", ms=8.5, color="#c25555", zorder=6)
        ticks.append(x); labels.append(f"{k} objects")
    ax.axvline(2.75, color="#cccccc", lw=0.9, ls="--", zorder=1)
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.0, -0.33, "counting", transform=tr, ha="center", va="top", fontsize=8, color="#444444")
    ax.text(4.5, -0.33, "object history (gemma-4-26b)", transform=tr, ha="center", va="top", fontsize=8,
            color="#444444")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlim(-0.6, 6.1)
    ax.set_ylim(-0.05, 1.42)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style(ax, "State Transplant", None, "Fraction following the donor")
    for h, lab in ((ax.plot([], [], "o", ms=6, color="#777777")[0], "window edit, layers \u2265 8"),
                   (ax.plot([], [], "o", ms=6, mfc="white", mec="#777777", mew=1.4)[0], "sham edit"),
                   (ax.plot([], [], "D", ms=4.6, color="#555555")[0], "whole donor cache"),
                   (ax.plot([], [], "^", ms=6, color="#2a9d5c")[0], "event reported (edit)"),
                   (ax.plot([], [], "*", ms=8.5, color="#c25555")[0], "edit, reasoning on")):
        h.set_label(lab)
    ax.legend(fontsize=6.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.2, labelspacing=0.25, columnspacing=1.0, borderaxespad=0.2)


def save_boundary_panel():
    b = load_boundary()
    for m, r in b["counting"].items():
        print(m, {k: f"{v[0]}/{v[1]}" for k, v in r.items()})
    for k, r in b["objects"].items():
        print(k, {c: f"{v[0]}/{v[1]}" if v else None for c, v in r.items()})
    fig, ax = plt.subplots(figsize=(3.4, 4.3))
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.28, top=0.9)
    panel_boundary_transfer(ax, b)
    ax.set_box_aspect(1)
    fig.savefig(OUT / "mechanistic-boundary-panel.pdf", transparent=True)
    fig.savefig(OUT / "mechanistic-boundary-panel.png", dpi=200)
    plt.close(fig)
    print("wrote mechanistic-boundary-panel.pdf")


def save_main_sheet(tables):
    from counting.analysis.figures.benchmark_correlations import add_manual_row_axes
    panel_h = 2.30
    bottom = 0.95      # rotated model names on the lower row
    row_gap = 1.30     # rotated names of the upper row plus lower-row titles
    top_pad = 0.45
    height = bottom + 2 * panel_h + row_gap + top_pad
    fig = plt.figure(figsize=(12.5, height))
    rows = [
        add_manual_row_axes(
            fig, panel_count=4, figure_width_in=12.5, figure_height_in=height,
            plot_left_in=0.72,
            plot_bottom_in=bottom + (panel_h + row_gap if row == 0 else 0),
            plot_width_in=11.43, plot_height_in=panel_h, panel_gap_in=0.72,
        )
        for row in (0, 1)
    ]
    axes = rows[0] + rows[1]
    r2 = load_round2()
    panel_exact_depth(axes[0], r2)
    panel_exemplar(axes[1], tables)
    panel_beyond_slopes(axes[2], tables)
    panel_shift(axes[3], r2)
    panel_depth_map(axes[4], r2)
    panel_boundary_transfer(axes[5], load_boundary())
    panel_readable_usable_exact(axes[6], r2)
    panel_mode_switch(axes[7], tables)
    for ax in axes:
        ax.set_box_aspect(1)
    fig.savefig(OUT / "mechanistic-suite-panels.pdf", transparent=True)
    fig.savefig(OUT / "mechanistic-suite-panels.png", dpi=200)
    plt.close(fig)
    print("wrote mechanistic-suite-panels.pdf")


# -- SI plates ---------------------------------------------------------------

def letter(ax, ch):
    ax.text(-0.22, 1.12, ch, transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="top")


def si_readout_atlas(tables):
    fig, axes = plt.subplots(6, 4, figsize=(12.4, 14.0), constrained_layout=True)
    for r, m in enumerate(MODELS):
        rows = load_projections(m)
        sites = sites_of(tables[m])
        scc = immediate_scc(tables[m], m)
        for c, site in enumerate(sites):
            ax = axes[r][c]
            pts = sorted((int(x["count"]), float(x["extrapolated_count"]))
                         for x in rows if x["site"] == site)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", lw=1.3,
                    color=COLORS[m])
            lim = max(p[0] for p in pts) * 1.02
            ax.plot([0, lim], [0, lim], ls="--", lw=0.9, color="#999999")
            ax.axvline(scc, color="#666666", lw=0.9, ls=":")
            ax.set_title(f"{LABELS[m]} · layer {site.rsplit('layer', 1)[1]}",
                         fontsize=10.5)
            ax.tick_params(labelsize=8.5)
            ax.grid(alpha=0.18)
            set_spine_width(ax, 1.0)
            if c == 0:
                ax.set_ylabel("Readout", fontsize=10)
                letter(ax, "ABCDEF"[r])
            if r == 5:
                ax.set_xlabel("True count", fontsize=10)
    fig.savefig(OUT / "suite-readout-atlas.pdf", transparent=True)
    plt.close(fig)
    print("wrote suite-readout-atlas.pdf")


def si_clamp_detail(tables):
    fig, axes = plt.subplots(6, 4, figsize=(12.4, 14.0), constrained_layout=True)
    for r, m in enumerate(MODELS):
        recs = [json.loads(l) for l in
                (SUITE / m / "interventions" / "interventions.jsonl").open()]
        sites = sites_of(tables[m])
        for c, site in enumerate(sites):
            ax = axes[r][c]
            pts = [(x["target"], x["parsed"]) for x in recs
                   if x["experiment"] == "clamp" and x["site"] == site
                   and x["target"] is not None and x["parsed"] is not None]
            base = next(x["base"] for x in recs if x["site"] == site)
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=13,
                       color=COLORS[m], alpha=0.85, zorder=3)
            lim = max((p[0] for p in pts), default=10) * 1.03
            ax.plot([0, lim], [0, lim], ls="--", lw=0.9, color="#999999")
            ax.axhline(base, color="#666666", lw=0.9, ls=":")
            ax.set_title(f"{LABELS[m]} · layer {site.rsplit('layer', 1)[1]}",
                         fontsize=10.5)
            ax.tick_params(labelsize=8.5)
            ax.grid(alpha=0.18)
            set_spine_width(ax, 1.0)
            if c == 0:
                ax.set_ylabel("Decoded answer", fontsize=10)
                letter(ax, "ABCDEF"[r])
            if r == 5:
                ax.set_xlabel("Clamp target", fontsize=10)
    fig.savefig(OUT / "suite-clamp-detail.pdf", transparent=True)
    plt.close(fig)
    print("wrote suite-clamp-detail.pdf")


def si_chimera_matrix(tables):
    conds = [("chimera_full", "Full state"),
             ("chimera_residual_only", "Nonlinear residual"),
             ("chimera_linear_only", "Linear coordinate")]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4), constrained_layout=True)
    for ci, (cond, title) in enumerate(conds):
        ax = axes[ci]
        grid = np.zeros((len(MODELS), 4))
        for r, m in enumerate(MODELS):
            for c, site in enumerate(sites_of(tables[m])):
                d = tables[m]["interventions"]["sites"][site][cond]
                grid[r][c] = d["first_token_hit"] / d["n"]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for r in range(len(MODELS)):
            for c in range(4):
                ax.text(c, r, f"{grid[r][c]:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if grid[r][c] > 0.6 else "#1a1a1a")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["site 1", "site 2", "site 3", "site 4"], fontsize=9)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([LABELS[m] for m in MODELS] if ci == 0 else [],
                           fontsize=9)
        ax.set_title(title, fontsize=12)
        letter(ax, "ABC"[ci])
    fig.colorbar(im, ax=axes, shrink=0.8, label="first-token transfer")
    fig.savefig(OUT / "suite-chimera-matrix.pdf", transparent=True)
    plt.close(fig)
    print("wrote suite-chimera-matrix.pdf")


def si_mixed_grids():
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.2), constrained_layout=True)
    for i, m in enumerate(MODELS):
        ax = axes[i // 3][i % 3]
        d = json.loads((SUITE / m / "mixed" / "mixed.json").read_text())
        for t in d["trials"]:
            ok = t["correct"]
            ax.scatter([t["length"]], [t["k"]], s=52,
                       marker="o" if ok else "x",
                       color="#2f6fb3" if ok else "#c25555",
                       linewidths=1.6, zorder=3)
        ax.set_title(f'{LABELS[m]} · accuracy {d["behavioral_accuracy"]:.0%}',
                     fontsize=12)
        ax.set_xlabel("Sequence length", fontsize=10.5)
        ax.set_ylabel("Target count k", fontsize=10.5)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.18)
        set_spine_width(ax, 1.1)
        letter(ax, "ABCDEF"[i])
    fig.savefig(OUT / "suite-mixed-grids.pdf", transparent=True)
    plt.close(fig)
    print("wrote suite-mixed-grids.pdf")


def si_margin_curves(tables):
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.2), constrained_layout=True)
    for i, m in enumerate(MODELS):
        ax = axes[i // 3][i % 3]
        rows = load_projections(m)
        site = sites_of(tables[m])[0]
        pts = sorted((int(r["count"]), float(r["min_gap"]))
                     for r in rows if r["site"] == site and r["min_gap"])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", lw=1.4,
                color=COLORS[m])
        ax.axhline(0, color="#666666", lw=1.0)
        mc = tables[m]["margin_collapse"]
        ax.axvline(mc["first_greedy_divergence"], color="#c25555", lw=1.1, ls=":")
        ax.axvline(immediate_scc(tables[m], m), color="#666666", lw=1.1, ls="--")
        ax.set_title(LABELS[m], fontsize=12)
        ax.set_xlabel("True count", fontsize=10.5)
        ax.set_ylabel("Min correct-answer margin", fontsize=10.5)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.18)
        set_spine_width(ax, 1.1)
        letter(ax, "ABCDEF"[i])
    fig.savefig(OUT / "suite-margin-curves.pdf", transparent=True)
    plt.close(fig)
    print("wrote suite-margin-curves.pdf")


def si_mode_scans(tables):
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.2), constrained_layout=True)
    for i, m in enumerate(MODELS):
        ax = axes[i // 3][i % 3]
        for mode, colour, lab in ((IMMEDIATE[m], "#c25555", "immediate"),
                                  (REASONING.get(m), "#2f6fb3", "reasoning")):
            if mode is None or mode not in tables[m]["scans"]:
                continue
            recs = [json.loads(l) for l in
                    (SUITE / m / f"scan-{mode}" / "records.jsonl").open()]
            xs = [r["length"] for r in recs]
            ys = [r["parsed"] if r["parsed"] is not None else 0 for r in recs]
            ax.plot(xs, ys, ".", ms=4.5, color=colour, alpha=0.8, label=lab)
        lim = max(r["length"] for mode in tables[m]["scans"]
                  for r in [json.loads(l) for l in
                            (SUITE / m / f"scan-{mode}" / "records.jsonl").open()])
        ax.plot([0, lim], [0, lim], ls="--", lw=0.9, color="#999999", zorder=1)
        ax.set_title(LABELS[m], fontsize=12)
        ax.set_xlabel("Items asked", fontsize=10.5)
        ax.set_ylabel("Answer returned", fontsize=10.5)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.18)
        set_spine_width(ax, 1.1)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        letter(ax, "ABCDEF"[i])
    fig.savefig(OUT / "suite-mode-scans.pdf", transparent=True)
    plt.close(fig)
    print("wrote suite-mode-scans.pdf")


def main() -> int:
    save_boundary_panel()
    tables = load_tables()
    save_main_sheet(tables)
    si_readout_atlas(tables)
    si_clamp_detail(tables)
    si_chimera_matrix(tables)
    si_mixed_grids()
    si_margin_curves(tables)
    si_mode_scans(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
