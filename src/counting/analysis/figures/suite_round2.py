#!/usr/bin/env python3
"""Plates for the round-2 mechanistic controls (results/suite/round2_tables.json).

    python -m counting.analysis.figures.suite_round2

Outputs (paper/figures/):
    round2-accumulation.pdf   E4: answer shift vs delta per model (rows) and site
                              (columns); running-count subspace, random subspace
                              and decode-point-only controls; in-regime and beyond-CC
    round2-output-channel.pdf E1: consolidation curves by task, U transfer and
                              complement by site and task, U ablation survival
    round2-logit-interp.pdf   E2: answer-token logits along the chord, two pairs per model
    round2-exact-match.pdf    E3: rounded-probe exact match against the model's own,
                              per band (the Fig. 4G replacement in exact-match units)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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
COLORS = {"S": "#2f6fb3", "R": "#9a9a9a", "S_decode_point": "#c25555"}
TASK_COLORS = {"counting": "#2f6fb3", "copy": "#0f8f80", "retrieval": "#b07818"}
TITLE_FS, LABEL_FS, TICK_FS, SMALL_FS = 12, 10.5, 9, 8.5


def letter(ax, ch):
    ax.text(-0.18, 1.06, ch, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left")


def style(ax, title=None, xlabel=None, ylabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=TICK_FS)
    if title:
        ax.set_title(title, fontsize=TITLE_FS)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)


def layer_num(site: str) -> int:
    return int(site.rsplit("layer", 1)[1])


def load():
    return json.loads((SUITE / "round2_tables.json").read_text())


# -- E4 ----------------------------------------------------------------------

def plate_accumulation(tables):
    models = [m for m in MODELS if tables.get(m, {}).get("accumulate")]
    if not models:
        return
    n_sites = max(len({k.split("|")[0] for k in tables[m]["accumulate"]["shift"]}) for m in models)
    fig, axes = plt.subplots(len(models), n_sites, figsize=(12.4, 2.4 * len(models)), squeeze=False)
    for row, m in enumerate(models):
        acc = tables[m]["accumulate"]
        sites = sorted({k.split("|")[0] for k in acc["shift"]}, key=layer_num)
        for col, site in enumerate(sites):
            ax = axes[row, col]
            for band, ls in (("stable", "-"), ("beyond", "--")):
                for fam in ("S", "R", "S_decode_point"):
                    st = acc["shift"].get(f"{site}|{band}|{fam}|local")
                    if not st or not st.get("mean_shift_by_delta"):
                        continue
                    ds = sorted(int(k) for k in st["mean_shift_by_delta"])
                    ys = [st["mean_shift_by_delta"][str(d)] for d in ds]
                    label = {"S": "running-count subspace", "R": "random subspace", "S_decode_point": "decode point only"}[fam]
                    ax.plot(ds, ys, ls, marker="o", ms=3, color=COLORS[fam], lw=1.4,
                            label=f"{label} ({band})" if row == 0 and col == 0 else None)
            lim = 9
            ax.plot([-lim, lim], [-lim, lim], ":", color="black", lw=0.8)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            s_st = acc["shift"].get(f"{site}|stable|S|local", {})
            slope = s_st.get("slope")
            style(ax, title=f"{LABELS[m]} {site.split('.')[-1]}" + (f", slope {slope:.2f}" if slope is not None else ""),
                  xlabel="delta (items) added from j onward", ylabel="answer shift (items)")
            ax.title.set_fontsize(9.5)
            if col == 0:
                letter(ax, chr(65 + row))
        for col in range(len(sites), n_sites):
            axes[row, col].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=SMALL_FS, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "round2-accumulation.pdf", transparent=True)
    fig.savefig(OUT / "round2-accumulation.png", dpi=130)
    plt.close(fig)


# -- E1 ----------------------------------------------------------------------

def plate_output_channel(tables):
    models = [m for m in MODELS if tables.get(m, {}).get("copy") or tables.get(m, {}).get("retrieval")]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 3, figsize=(12.4, 2.3 * len(models)), squeeze=False)
    for row, m in enumerate(models):
        entry = tables[m]
        ax = axes[row, 0]
        for task in ("counting", "copy", "retrieval"):
            t = entry.get(task)
            if t and t.get("localize"):
                curve = t["localize"]["curve"]
                layers = sorted(int(l) for l in curve)
                ax.plot(layers, [curve[str(l)] if str(l) in curve else curve[l] for l in layers], "-", color=TASK_COLORS[task], lw=1.6, label=task)
        style(ax, title=f"{LABELS[m]}: consolidation", xlabel="layer of full transplant", ylabel="donor answer transferred")
        ax.set_ylim(-0.03, 1.05)
        if row == 0:
            ax.legend(fontsize=SMALL_FS, frameon=False)
        letter(ax, chr(65 + row))
        # Sites are the union over tasks (the counting reference has the last two only).
        all_sites = sorted({s for task in ("counting", "copy", "retrieval")
                            for s in (entry.get(task) or {}).get("transplant", {})}, key=layer_num)
        pos_of = {s: i for i, s in enumerate(all_sites)}
        ax = axes[row, 1]
        width = 0.25
        for i, task in enumerate(("counting", "copy", "retrieval")):
            t = entry.get(task)
            if not t or not t.get("transplant"):
                continue
            sites = [s for s in all_sites if s in t["transplant"] and t["transplant"][s].get("group", "transplant") == "transplant"]
            xs = np.array([pos_of[s] for s in sites]) + (i - 1) * width
            ax.bar(xs, [t["transplant"][s]["U"] or 0 for s in sites], width, color=TASK_COLORS[task], label=f"{task}: U")
            ax.bar(xs, [t["transplant"][s]["U_comp"] or 0 for s in sites], width, bottom=0, fill=False,
                   edgecolor=TASK_COLORS[task], hatch="//", label=f"{task}: complement")
        ax.set_xticks(np.arange(len(all_sites)))
        ax.set_xticklabels([s.split(".")[-1] for s in all_sites])
        style(ax, title="U transplant (filled) and complement (hatched)", ylabel="donor answer transferred")
        ax.set_ylim(0, 1.05)
        if row == 0:
            ax.legend(fontsize=SMALL_FS, frameon=False, ncol=2)
        ax = axes[row, 2]
        for i, task in enumerate(("counting", "copy", "retrieval")):
            t = entry.get(task)
            if not t or not t.get("ablate"):
                continue
            sites = [s for s in all_sites if s in t["ablate"]]
            vals = []
            for s in sites:
                cell = t["ablate"][s].get("U")
                a, b = (int(v) for v in cell.split("/")) if cell else (0, 1)
                vals.append(a / b if b else 0)
            xs = np.array([pos_of[s] for s in sites]) + (i - 1) * width
            ax.bar(xs, vals, width, color=TASK_COLORS[task], label=task)
        ax.set_xticks(np.arange(len(all_sites)))
        ax.set_xticklabels([s.split(".")[-1] for s in all_sites])
        style(ax, title="own answer survives U ablation", ylabel="fraction")
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "round2-output-channel.pdf", transparent=True)
    fig.savefig(OUT / "round2-output-channel.png", dpi=130)
    plt.close(fig)


# -- E2 ----------------------------------------------------------------------

def plate_logits(tables):
    models = [m for m in MODELS if tables.get(m, {}).get("logits")]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 2, figsize=(12.4, 3.0 * len(models)), squeeze=False)
    for row, m in enumerate(models):
        rows = [json.loads(l) for l in (SUITE / m / "round2" / "logit_interp.jsonl").read_text().splitlines() if l]
        base = rows[0]["base"]
        donors = [d for d in sorted({r["donor"] for r in rows}) if str(d)[0] != str(base)[0] and
                  next(r for r in rows if r["donor"] == d)["n_intermediate_tokens"] > 0]
        picks = [donors[len(donors) // 4], donors[3 * len(donors) // 4]] if len(donors) >= 2 else donors
        for col, d in enumerate(picks):
            ax = axes[row, col]
            rs = sorted([r for r in rows if r["donor"] == d], key=lambda r: r["alpha"])
            al = [r["alpha"] for r in rs]
            ax.plot(al, [r["base_logit"] for r in rs], "-", color="#2f6fb3", lw=1.6, label=f"base {base}")
            ax.plot(al, [r["donor_logit"] for r in rs], "-", color="#c25555", lw=1.6, label=f"donor {d}")
            ax.plot(al, [r["max_intermediate_logit"] for r in rs], "--", color="#9a9a9a", lw=1.4, label="max intermediate")
            ax2 = ax.twinx()
            ax2.plot(al, [r["intermediate_mass"] for r in rs], ":", color="#b07818", lw=1.2)
            ax2.set_ylim(0, 1)
            ax2.tick_params(labelsize=TICK_FS)
            ax2.set_ylabel("intermediate mass", fontsize=SMALL_FS, color="#b07818")
            style(ax, title=f"{LABELS[m]}: {base} to {d}", xlabel="alpha", ylabel="answer-token logit")
            if row == 0 and col == 0:
                ax.legend(fontsize=SMALL_FS, frameon=False)
            if col == 0:
                letter(ax, chr(65 + row))
    fig.tight_layout()
    fig.savefig(OUT / "round2-logit-interp.pdf", transparent=True)
    fig.savefig(OUT / "round2-logit-interp.png", dpi=130)
    plt.close(fig)


# -- E3 ----------------------------------------------------------------------

def plate_exact(tables):
    models = [m for m in MODELS if tables.get(m, {}).get("exact")]
    if not models:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    bands = [("in_regime", "in_regime_exact", "in regime"), ("beyond", "beyond_exact", "1.0 to 1.25 x CC"),
             ("mixed_in_grid_cv", "mixed_exact", "mixed grids (in-grid CV)")]
    for ax, (probe_key, model_key, title) in zip(axes, bands):
        for i, m in enumerate(models):
            ex = tables[m]["exact"]
            sites = sorted(ex["sites"], key=layer_num)
            probe = [ex["sites"][s].get(probe_key, {}).get("exact") for s in sites]
            probe1 = [ex["sites"][s].get(probe_key, {}).get("within1") for s in sites]
            xs = np.arange(len(sites)) + i * 0.13 - 0.33
            ax.plot(xs, probe, "o-", color=plt.cm.tab10(i), ms=4, lw=1.2, label=LABELS[m])
            ax.plot(xs, probe1, "s--", color=plt.cm.tab10(i), ms=3, lw=0.8, alpha=0.6)
            mv = ex["model"].get(model_key)
            if mv is not None:
                ax.axhline(mv, color=plt.cm.tab10(i), lw=0.8, ls=":")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["site 1", "site 2", "site 3", "site 4"])
        ax.set_ylim(0, 1.02)
        style(ax, title=title, ylabel="exact match (circles), within 1 (squares)")
    axes[0].legend(fontsize=SMALL_FS, frameon=False)
    for ax, ch in zip(axes, "ABC"):
        letter(ax, ch)
    fig.suptitle("Rounded ridge readout vs the model's own exact match (dotted)", fontsize=TITLE_FS)
    fig.tight_layout()
    fig.savefig(OUT / "round2-exact-match.pdf", transparent=True)
    fig.savefig(OUT / "round2-exact-match.png", dpi=130)
    plt.close(fig)


def main() -> int:
    tables = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plate_accumulation(tables)
    plate_output_channel(tables)
    plate_logits(tables)
    plate_exact(tables)
    print("wrote round-2 plates to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
