#!/usr/bin/env python3
"""SI plate for the subspace and periodic-feature controls (results/suite/).

    python -m counting.analysis.figures.suite_subspace

Output (paper/figures/):
    suite-subspace.pdf   rows = models; columns = transfer vs subspace
                         dimension (trajectory / random-in-span / ambient /
                         decoder families, with the fixed hypothesis subspaces
                         as markers), ablation survival vs dimension, full
                         transplant vs layer (consolidation), and the
                         digit-shift matrix of the period-10 clamp at the
                         last site.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from counting.analysis.figures.attractors import set_spine_width
from counting.analysis.figures.mechanistic_suite import letter
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
FAMILY_STYLE = {
    "T": ("#2f6fb3", "trajectory T_k"),
    "R": ("#9a9a9a", "random in span(T)"),
    "A": ("#cfcfcf", "random ambient"),
    "U_k": ("#c25555", "decoder U_k"),
}
MARKER_STYLE = {"L": ("v", "#0f8f80", "linear L"), "F": ("s", "#b07818", "periodic F"),
                "G": ("D", "#7d5bb5", "digit G"), "U": ("*", "#c25555", "decoder U")}
# Same scale as the other six-row suite plates (readout atlas, margins).
TITLE_FS, LABEL_FS, TICK_FS, SMALL_FS = 12, 10.5, 9, 8.5


def load_tables() -> dict[str, dict]:
    rows = json.loads((SUITE / "tables.json").read_text())
    return {r["model"]: r for r in rows}


def style(ax, title=None, xlabel=None, ylabel=None):
    set_spine_width(ax, 1.0)
    ax.tick_params(labelsize=TICK_FS)
    if title:
        ax.set_title(title, fontsize=TITLE_FS)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)


def last_site(block: dict) -> str | None:
    if not block:
        return None
    return sorted(block, key=lambda s: int(s.rsplit("layer", 1)[1]))[-1]


def marker_rate(fam: dict, prefix: str) -> tuple[int | None, float | None]:
    for key, st in fam.items():
        parts = key.split("|")
        if len(parts) == 3 and parts[0] == prefix and parts[2] == "proj":
            return int(parts[1]), st["rate"]
    return None, None


def panel_transfer(ax, sub: dict, site: str) -> None:
    tr = sub["transplant"][site]
    full = tr["families"]["full"]["rate"]
    if full is not None:
        ax.axhline(full, color="black", lw=1, ls="--", label="full state")
    for family, (color, label) in FAMILY_STYLE.items():
        curve = tr["curves"].get(family)
        if not curve:
            continue
        ks = sorted(int(k) for k in curve["curve"])
        ys = [curve["curve"][str(k)] if str(k) in curve["curve"] else curve["curve"][k] for k in ks]
        ax.plot(ks, ys, "-o", color=color, ms=3.5, lw=1.4, label=label)
    for prefix, (marker, color, label) in MARKER_STYLE.items():
        k, rate = marker_rate(tr["families"], prefix)
        if k is not None and rate is not None:
            ax.plot([k], [rate], marker, color=color, ms=8, mec="black", mew=0.5, label=label, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.05)
    style(ax, xlabel="subspace dimension k", ylabel="donor answer transferred")


def panel_ablation(ax, sub: dict, site: str) -> None:
    ab = sub["ablate"][site]

    def survival(key: str) -> float | None:
        st = ab.get(key)
        return st["first_token_survive"] / st["n"] if st and st["n"] else None

    for family, (color, label) in FAMILY_STYLE.items():
        ks = sorted({int(k.split("|")[1]) for k in ab if k.split("|")[0] == family})
        if not ks:
            continue
        ys = [survival(f"{family}|{k}") for k in ks]
        ax.plot(ks, ys, "-o", color=color, ms=3.5, lw=1.4, label=label)
    for prefix, (marker, color, label) in MARKER_STYLE.items():
        keys = [k for k in ab if k.split("|")[0] == prefix]
        if keys:
            k = int(keys[0].split("|")[1])
            ax.plot([k], [survival(keys[0])], marker, color=color, ms=8, mec="black", mew=0.5, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.05)
    style(ax, xlabel="ablated dimension k", ylabel="own answer survives")


def panel_localize(ax, sub: dict, n_layers_hint: int | None = None) -> None:
    loc = sub["localize"]
    layers = sorted(int(l) for l in loc["curve"])
    ys = [loc["curve"][str(l)] for l in layers]
    ax.plot(layers, ys, "-", color="#2f6fb3", lw=1.6)
    ax.fill_between(layers, 0, ys, color="#2f6fb3", alpha=0.15)
    if loc["consolidation_layer"] is not None:
        ax.axvline(loc["consolidation_layer"], color="#c25555", lw=1, ls=":")
    ax.set_ylim(-0.03, 1.05)
    style(ax, xlabel="layer of full transplant", ylabel="donor answer transferred")


def panel_digit_shift(ax, sub: dict, site: str) -> None:
    pe = sub["periodic"][site].get("P10|F")
    if not pe:
        ax.axis("off")
        return
    mat = np.zeros((9, 10))
    for d, row in pe["matrix"].items():
        for shift, n in row.items():
            mat[int(d) - 1, int(shift)] = n
    mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.plot(range(1, 10), range(0, 9), "r.", ms=4)
    ax.set_yticks(range(9))
    ax.set_yticklabels(range(1, 10))
    ax.set_xticks(range(10))
    style(ax, xlabel="observed shift of last digit", ylabel="phase shift asked (d)")


def main() -> int:
    tables = load_tables()
    models = [m for m in MODELS if tables.get(m, {}).get("subspace", {}).get("transplant")]
    if not models:
        print("no subspace results yet")
        return 1
    fig, axes = plt.subplots(len(models), 4, figsize=(12.4, 14.0), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, model in enumerate(models):
        sub = tables[model]["subspace"]
        site = last_site(sub["transplant"])
        panel_transfer(axes[row, 0], sub, site)
        if sub.get("ablate", {}).get(site):
            panel_ablation(axes[row, 1], sub, site)
        else:
            axes[row, 1].axis("off")
        if sub.get("localize"):
            panel_localize(axes[row, 2], sub)
        else:
            axes[row, 2].axis("off")
        if sub.get("periodic", {}).get(site):
            panel_digit_shift(axes[row, 3], sub, site)
        else:
            axes[row, 3].axis("off")
        letter(axes[row, 0], chr(65 + row))
        for ax in axes[row]:
            if ax.axison:
                ax.set_title(f"{LABELS[model]} ({site.split('.')[-1]})", fontsize=TITLE_FS)
        if row == 0:
            # The family legend goes in the localization panel, whose upper
            # left is empty for every model (no transfer before consolidation).
            handles, labels = axes[0, 0].get_legend_handles_labels()
            axes[0, 2].legend(handles, labels, loc="upper left", fontsize=SMALL_FS,
                              frameon=False, handlelength=1.1, labelspacing=0.25,
                              borderaxespad=0.15)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "suite-subspace.pdf")
    fig.savefig(OUT / "suite-subspace.png", dpi=150)
    print(f"wrote {OUT / 'suite-subspace.pdf'} for {len(models)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
