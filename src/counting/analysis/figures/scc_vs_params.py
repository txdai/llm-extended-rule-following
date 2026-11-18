#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from counting.analysis.figures.attractors import set_spine_width
from counting.analysis.figures.search_panels import (
    PROVIDER_COLORS,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_MARKERS,
    infer_provider_family,
)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

from counting.datastore import load_registry
from counting.paths import REPO_ROOT as ROOT
RUN_DIRS = (
    ROOT / "results" / "counting-stable-search",
    ROOT / "results" / "counting-stable-search-open-source",
)
OUTPUT_PDF = ROOT / "paper" / "figures" / "scc-vs-active-params-si.pdf"
OUTPUT_PNG = ROOT / "paper" / "figures" / "scc-vs-active-params-si.png"


@dataclass(frozen=True)
class ParamPoint:
    model: str
    provider: str
    active_params_billion: float
    stable_count: float


def parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_points() -> list[ParamPoint]:
    """One point per model with both a capacity and a parameter count.

    Reads registry.csv, which already carries the parameter metadata inline, so
    there is no manifest lookup and no second source of truth.
    """
    points: list[ParamPoint] = []
    seen: set[str] = set()
    for run_dir in RUN_DIRS:
        for row in load_registry(run_dir):
            model = row.get("model") or ""
            if not model or model in seen:
                continue
            stable_count = parse_float(row.get("cc"))
            if stable_count is None or stable_count <= 0:
                continue
            total = parse_float(row.get("total_params_b"))
            active = parse_float(row.get("active_params_b")) or total
            if active is None or active <= 0:
                continue
            seen.add(model)
            points.append(
                ParamPoint(
                    model=model,
                    provider=infer_provider_family(model),
                    active_params_billion=active,
                    stable_count=stable_count,
                )
            )
    points.sort(key=lambda point: point.active_params_billion)
    return points


def make_figure(points: list[ParamPoint]) -> None:
    if not points:
        raise RuntimeError("No parameter-metadata points available for SCC vs active-parameter plot.")

    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)

    for point in points:
        ax.scatter(
            point.active_params_billion,
            point.stable_count,
            s=46,
            color=PROVIDER_COLORS[point.provider],
            marker=PROVIDER_MARKERS[point.provider],
            alpha=0.95,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    legend_handles = []
    legend_labels = []
    provider_order = (
        "openai",
        "anthropic",
        "google",
        "deepseek",
        "zai",
        "moonshot",
        "minimax",
        "meta",
        "mistral",
        "qwen",
        "microsoft",
        "other",
    )
    for provider in provider_order:
        if any(point.provider == provider for point in points):
            legend_handles.append(
                ax.scatter(
                    [],
                    [],
                    s=46,
                    color=PROVIDER_COLORS[provider],
                    marker=PROVIDER_MARKERS[provider],
                    alpha=0.95,
                )
            )
            legend_labels.append(PROVIDER_DISPLAY_NAMES[provider])

    if legend_handles:
        legend = ax.legend(
            legend_handles,
            legend_labels,
            loc="lower right",
            fontsize=8.5,
            frameon=True,
            ncol=2,
            columnspacing=1.0,
            handletextpad=0.5,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#cfcfcf")
        legend.get_frame().set_alpha(0.93)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Active parameter count (billions)", fontsize=13)
    ax.set_ylabel("Measured CC", fontsize=13)
    ax.set_title("CC vs Active Parameter Count", fontsize=15)
    ax.grid(alpha=0.18, which="both", zorder=0)
    ax.tick_params(axis="both", which="both", labelsize=11)
    set_spine_width(ax)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    points = build_points()
    make_figure(points)
    print(f"SI figure: {OUTPUT_PDF}")
    print(f"Points: {len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
