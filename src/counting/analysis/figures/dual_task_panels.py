#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

MUTED_BLUE = "#7f97b2"
MUTED_ORANGE = "#c6a78c"
MUTED_RED = "#b78888"
MUTED_GREEN = "#7b9e87"
MUTED_PURPLE = "#8f7aa8"
MUTED_GRAY = "#aeb3b8"
BASELINE_DARK = "#2f2f2f"
FOCUS_BLUE = "#5f84ad"
FOCUS_RED = "#c46a6a"


CONDITION_LABELS = {
    "plain_count": "Plain count",
    "real_task": "Real task",
    "irrelevant_code": "Code control",
    "dual_counting": "Dual count",
}
CONDITION_COLORS = {
    "plain_count": BASELINE_DARK,
    "real_task": FOCUS_BLUE,
    "irrelevant_code": MUTED_ORANGE,
    "dual_counting": FOCUS_RED,
}
PANEL_COLORS = {
    "plain_count": BASELINE_DARK,
    "bbh_reasoning": FOCUS_BLUE,
    "cruxeval_o": MUTED_BLUE,
    "math500": MUTED_PURPLE,
    "mmlu_pro": MUTED_GREEN,
    "irrelevant_code": MUTED_ORANGE,
    "dual_counting": FOCUS_RED,
}
PANEL_MARKERS = {
    "plain_count": "o",
    "bbh_reasoning": "s",
    "cruxeval_o": "D",
    "math500": "^",
    "mmlu_pro": "v",
    "irrelevant_code": "X",
    "dual_counting": "*",
}
CAT_LABELS = {
    "bbh_reasoning": "BBH",
    "cruxeval_o": "CRUXEval-O",
    "math500": "MATH-500",
    "mmlu_pro": "MMLU-Pro",
    "plain_count": "Plain count",
    "irrelevant_code": "Code control",
    "dual_counting": "Dual-count control",
}


def set_spine_width(ax: plt.Axes, width: float = 1.5) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot count scatter-control sweep.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_value(value: str) -> bool:
    return value == "True"


def panel_subset(rows: list[dict[str, str]], panel_key: str, condition: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["condition"] == condition
        and (
            row["category"] == panel_key
            if condition in {"real_task", "plain_count"}
            else row["category"] == "pooled_control"
        )
    ]


def save_all(fig: plt.Figure, output: Path, dpi: int = 240) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size < 3:
        return values
    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    pad = window // 2
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def plot_mean_count_error(run_dir: Path, output: Path) -> None:
    rows = load_rows(run_dir)
    panels = [
        ("plain_count", "plain_count"),
        ("bbh_reasoning", "real_task"),
        ("cruxeval_o", "real_task"),
        ("math500", "real_task"),
        ("mmlu_pro", "real_task"),
        ("irrelevant_code", "irrelevant_code"),
        ("dual_counting", "dual_counting"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 3.25), constrained_layout=True)
    for panel_key, condition in panels:
        subset = [
            row
            for row in panel_subset(rows, panel_key, condition)
            if row["count_error"] not in {"", "None"}
        ]
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in subset:
            grouped[int(row["target_count"])].append(float(row["count_error"]))
        xs = np.array(sorted(grouped), dtype=np.int32)
        ys = np.array([np.mean(grouped[int(x)]) for x in xs], dtype=np.float32)
        smoothed = smooth_series(ys, window=5)
        ax.plot(
            xs,
            smoothed,
            marker=PANEL_MARKERS[panel_key],
            markersize=3.8,
            linewidth=1.65,
            color=PANEL_COLORS[panel_key],
            label=CAT_LABELS.get(panel_key, panel_key),
            alpha=0.95,
        )

    ax.axhline(0.0, color=BASELINE_DARK, linewidth=1.1, linestyle="--", alpha=0.75)
    ax.set_title("Counting Error with Matched Controls", fontsize=16)
    ax.set_xlabel("Target count", fontsize=14)
    ax.set_ylabel("Mean parsed count error", fontsize=14)
    ax.grid(alpha=0.18, linewidth=0.9)
    ax.tick_params(axis="both", which="both", labelsize=12)
    set_spine_width(ax)
    legend = ax.legend(
        frameon=True,
        fontsize=8.5,
        loc="lower left",
        ncol=2,
        borderpad=0.35,
        handlelength=2.1,
        columnspacing=0.8,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cfcfcf")
    legend.get_frame().set_alpha(0.92)
    save_all(fig, output)
    plt.close(fig)


def plot(run_dir: Path, output: Path) -> None:
    rows = load_rows(run_dir)
    model = sorted({row["model"] for row in rows})[0]
    categories = ["bbh_reasoning", "cruxeval_o", "math500", "mmlu_pro"]
    conditions = ["real_task", "irrelevant_code", "dual_counting"]

    grouped: dict[tuple[str, str, int], list[bool]] = defaultdict(list)
    errors: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        category_key = row["category"]
        if category_key == "pooled_control":
            category_key = "pooled_control"
        key = (category_key, row["condition"], int(row["target_count"]))
        grouped[key].append(bool_value(row["count_correct"]))
        if row["count_abs_error"]:
            errors[key].append(float(row["count_abs_error"]))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True)
    axes_flat = list(axes.flatten())
    for axis, category in zip(axes_flat, categories):
        for condition in conditions:
            metric_category = category if condition == "real_task" else "pooled_control"
            xs = sorted({key[2] for key in grouped if key[0] == metric_category and key[1] == condition})
            ys = [
                sum(grouped[(metric_category, condition, x)]) / len(grouped[(metric_category, condition, x)])
                for x in xs
            ]
            axis.plot(
                xs,
                ys,
                marker="o",
                markersize=2.8,
                linewidth=1.2,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
                alpha=0.9,
            )
        axis.set_title(CAT_LABELS.get(category, category), fontsize=16)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.18, linewidth=0.9)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", which="both", labelsize=12)
        set_spine_width(axis)

    for axis in axes[:, 0]:
        axis.set_ylabel("Count exact-match accuracy", fontsize=14)
    for axis in axes[-1, :]:
        axis.set_xlabel("Target count", fontsize=14)

    handles, labels = axes_flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.015), fontsize=11)
    fig.suptitle(f"Counting accuracy sweep with matched controls: {model}", fontsize=16)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    save_all(fig, output)
    plt.close(fig)


def plot_count_values(run_dir: Path, output: Path) -> None:
    rows = load_rows(run_dir)
    model = sorted({row["model"] for row in rows})[0]
    panels = [
        ("plain_count", "plain_count"),
        ("bbh_reasoning", "real_task"),
        ("cruxeval_o", "real_task"),
        ("math500", "real_task"),
        ("mmlu_pro", "real_task"),
        ("irrelevant_code", "irrelevant_code"),
        ("dual_counting", "dual_counting"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.6), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = list(axes.reshape(-1))
    all_targets = [int(row["target_count"]) for row in rows]
    max_target = max(all_targets)
    min_target = min(all_targets)

    for axis, (panel_key, condition) in zip(axes_flat, panels):
        subset = [row for row in panel_subset(rows, panel_key, condition) if row["parsed_count"] not in {"", "None"}]
        xs = [int(row["target_count"]) for row in subset]
        ys = [int(float(row["parsed_count"])) for row in subset]
        axis.scatter(
            xs,
            ys,
            s=10,
            alpha=0.55,
            color=CONDITION_COLORS[condition],
            linewidths=0,
        )
        axis.plot([min_target, max_target], [min_target, max_target], color="#333333", linewidth=1.0, linestyle="--")
        axis.set_title(CAT_LABELS.get(panel_key, panel_key), fontsize=16)
        axis.grid(alpha=0.18, linewidth=0.9)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", which="both", labelsize=12)
        set_spine_width(axis)
    for axis in axes_flat[len(panels):]:
        axis.axis("off")

    axes_flat[0].set_ylabel("Parsed count", fontsize=14)
    axes_flat[4].set_ylabel("Parsed count", fontsize=14)
    for axis in axes_flat[:len(panels)]:
        axis.set_xlabel("Target count", fontsize=14)

    fig.suptitle(f"Parsed count vs target count with matched controls: {model}", fontsize=16)
    save_all(fig, output)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output = args.output or (args.run_dir / "count-scatter-values.png")
    plot_count_values(args.run_dir, output)
    accuracy_output = output.with_name("count-scatter-accuracy.png")
    plot(args.run_dir, accuracy_output)
    error_output = output.with_name("count-scatter-mean-error.png")
    plot_mean_count_error(args.run_dir, error_output)
    print(f"Wrote {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Wrote {output.with_suffix('.pdf')}")
    print(f"Wrote {accuracy_output}")
    print(f"Wrote {accuracy_output.with_suffix('.pdf')}")
    print(f"Wrote {error_output}")
    print(f"Wrote {error_output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
