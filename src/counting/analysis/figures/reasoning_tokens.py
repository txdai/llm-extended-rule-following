"""SI plate: how reasoning tokens grow with the count, at two output caps.

    python -m counting.analysis.figures.reasoning_tokens

Six representative reasoning configurations from the 32k-cap rescan
(results/budget-32k) overlaid on their catalogue scans at the 8,192-token cap
(results/counting-data). Each panel: completion tokens per probe (reasoning
included) against items asked, exact answers in blue and inexact in red,
catalogue-cap probes hollow and 32k-cap probes filled, the two caps as dashed
rules, and the 32k-cap fitted boundary as a vertical rule. The console prints
the stable-regime token slope (tokens per item) and the median tokens in the
last ten stable probes against the first ten probes past the boundary.
"""
from __future__ import annotations

import csv
import json
import statistics

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from counting.analysis.capacity import fit_transition
from counting.analysis.figures.attractors import BAD, OK, set_spine_width
from counting.analysis.figures.mechanistic_suite import letter
from counting.paths import REPO_ROOT as ROOT

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

OUTPUT = ROOT / "paper" / "figures" / "reasoning-tokens-panels.pdf"
CAPS = (8192, 32768)
MODELS = [
    ("deepseek/deepseek-v3.2", "deepseek-v3p2"),
    ("deepseek/deepseek-v4-pro-0813", "deepseek-v4-pro-0813"),
    ("google/gemma-4-26b-a4b-it", "gemma-4-26b-a4b-it"),
    ("anthropic/claude-sonnet-4.6", "claude-sonnet-4p6"),
    ("openai/gpt-5.6-luna", "gpt-5p6-luna"),
    ("moonshotai/kimi-k2.6", "kimi-k2p6"),
]


def catalogue_probes(model: str) -> list[tuple[int, float, bool]]:
    out = []
    with (ROOT / "results/counting-data/responses.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["run_key"] == f"{model}#think" and row["phase"] == "scan":
                tokens = float(row["output_tokens"] or 0)
                out.append((int(row["input_count"]), tokens, row["exact"] in ("True", "true", "1")))
    return sorted(out)


def rescan_probes(slug: str) -> tuple[list[tuple[int, float, bool]], int]:
    path = next((ROOT / "results/budget-32k" / slug / "trials").glob("*.jsonl"))
    recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    scan = sorted([r for r in recs if r.get("phase") == "scan" and r.get("response_status") != "error"],
                  key=lambda r: r["center_length"])
    fit = fit_transition([r["center_length"] for r in scan], [bool(r.get("exact_match")) for r in scan],
                         [r.get("parsed") is None for r in scan])
    probes = [(int(r["center_length"]), float(r.get("output_tokens") or 0), bool(r.get("exact_match"))) for r in scan]
    return probes, int(fit.scc or 0)


def main() -> int:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), constrained_layout=True)
    for ax, (model, slug), panel in zip(axes.flat, MODELS, "ABCDEF"):
        cat = catalogue_probes(model)
        res, tau = rescan_probes(slug)
        for probes, filled in ((cat, False), (res, True)):
            for length, tokens, exact in probes:
                colour = OK if exact else BAD
                ax.scatter([length], [max(tokens, 1)], s=14 if filled else 16,
                           facecolors=colour if filled else "none", edgecolors=colour,
                           linewidths=0.8, alpha=0.85 if filled else 0.6, zorder=3)
        for cap in CAPS:
            ax.axhline(cap, color="#666666", lw=1.0, ls=(0, (4, 3)), zorder=2)
        if tau:
            ax.axvline(tau, color="black", lw=1.1, zorder=2)
        stable = [t for l, t, e in res if l <= tau and e and t > 0]
        stable_l = [l for l, t, e in res if l <= tau and e and t > 0]
        slope = float(np.polyfit(stable_l, stable, 1)[0]) if len(stable) >= 5 else float("nan")
        def med(values: list[float]) -> float:
            return statistics.median(values) if values else float("nan")
        before = med([t for l, t, e in res if tau - 10 < l <= tau]) if tau else float("nan")
        after = med([t for l, t, e in res if tau < l <= tau + 10]) if tau else float("nan")
        hits = sum(1 for l, t, e in res if t >= CAPS[1] - 8)
        print(f"{panel} {model:32s} tau32k={tau:4d} slope={slope:6.1f} tok/item  median tokens last10 stable={before:7.0f} first10 past={after:7.0f}  32k hits={hits}")
        ax.set_yscale("log")
        ax.set_ylim(0.7, 6e4)
        letter(ax, panel)
        ax.set_title(model.split("/")[-1], fontsize=12)
        ax.set_xlabel("Items asked", fontsize=11)
        ax.set_ylabel("Completion tokens", fontsize=11)
        ax.tick_params(labelsize=9.5)
        ax.grid(alpha=0.2, linewidth=0.8)
        set_spine_width(ax)
    fig.savefig(OUTPUT, dpi=200)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=150)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
