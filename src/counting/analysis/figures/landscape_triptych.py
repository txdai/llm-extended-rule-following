#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import string
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

from counting.analysis.figures.benchmark_correlations import (
    BENCHMARK_DIR,
    build_alias_map,
    build_counting_models,
    draw_benchmark_scatter,
    load_benchmark_dataset,
)
from counting.analysis.figures.attractors import BASELINE_DARK, FOCUS_BLUE, MUTED_BLUE, MUTED_ORANGE, set_spine_width
from counting.analysis.reasoning_pairs import LABEL_OFFSETS, PLOT_LABELS, load_payloads, pair_records
from counting.analysis.figures.search_panels import PROVIDER_COLORS, PROVIDER_DISPLAY_NAMES, PROVIDER_MARKERS
from counting.analysis.figures.search_panels import current_run_dir, load_current_data, mean_or_none, merge_token_efficiency_sources
from counting.analysis.figures.search_panels import stable_count_trial_records


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

from counting.paths import REPO_ROOT as ROOT
DEFAULT_SCATTER_RUN = ROOT / "results" / "counting-dual-task" / "20260424T185546Z"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "figure2_full.png"
SMALL_TEXT_FONTSIZE = 10

FIGURE_WIDTH_IN = 12.5
FIGURE_HEIGHT_IN = 7.25
PLOT_LEFT_IN = 0.78
PLOT_WIDTH_IN = 11.47
PLOT_BOTTOM_IN = 0.58
PLOT_HEIGHT_IN = 2.5
PANEL_GAP_IN = 0.95
ROW_GAP_IN = 0.92

MUTED_RED = "#b78888"
MUTED_GREEN = "#7b9e87"
MUTED_PURPLE = "#8f7aa8"
CONDITION_COLORS = {
    "plain_count": BASELINE_DARK,
    "real_task": FOCUS_BLUE,
    "irrelevant_code": MUTED_ORANGE,
    "dual_counting": "#c46a6a",
}
PANEL_COLORS = {
    "plain_count": BASELINE_DARK,
    "bbh_reasoning": FOCUS_BLUE,
    "cruxeval_o": MUTED_BLUE,
    "math500": MUTED_PURPLE,
    "mmlu_pro": MUTED_GREEN,
    "irrelevant_code": MUTED_ORANGE,
    "dual_counting": "#c46a6a",
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
    "plain_count": "Plain count",
    "bbh_reasoning": "BBH",
    "cruxeval_o": "CRUXEval-O",
    "math500": "MATH-500",
    "mmlu_pro": "MMLU-Pro",
    "irrelevant_code": "Random code control",
    "dual_counting": "Dual-count control",
}
TASK_CATEGORY_LABELS = {
    "plain_count": "Plain count",
    "bbh_reasoning": "Reasoning",
    "cruxeval_o": "Coding",
    "math500": "Math",
    "mmlu_pro": "Knowledge",
    "irrelevant_code": "Random code",
    "dual_counting": "Dual count",
}
BENCHMARK_PANELS = (
    ("gpqa_diamond.csv", "Correlation with Knowledge"),
    ("swe_bench_verified.csv", "Correlation with Coding"),
    ("arc_agi_2_external.csv", "Correlation with Fluid Intelligence"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a one-row Figure 2 analysis composite.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--scatter-run-dir", type=Path, default=DEFAULT_SCATTER_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def save_all(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"))


def add_manual_grid_axes(fig: plt.Figure, *, nrows: int, ncols: int) -> list[plt.Axes]:
    panel_width = (PLOT_WIDTH_IN - PANEL_GAP_IN * (ncols - 1)) / ncols
    axes: list[plt.Axes] = []
    for row in range(nrows):
        bottom = PLOT_BOTTOM_IN + (nrows - 1 - row) * (PLOT_HEIGHT_IN + ROW_GAP_IN)
        for col in range(ncols):
            axes.append(
                fig.add_axes(
                    [
                        (PLOT_LEFT_IN + col * (panel_width + PANEL_GAP_IN)) / FIGURE_WIDTH_IN,
                        bottom / FIGURE_HEIGHT_IN,
                        panel_width / FIGURE_WIDTH_IN,
                        PLOT_HEIGHT_IN / FIGURE_HEIGHT_IN,
                    ]
                )
            )
    return axes


def add_manual_row_axes(fig: plt.Figure, panel_count: int) -> list[plt.Axes]:
    panel_width = (PLOT_WIDTH_IN - PANEL_GAP_IN * (panel_count - 1)) / panel_count
    return [
        fig.add_axes(
            [
                (PLOT_LEFT_IN + index * (panel_width + PANEL_GAP_IN)) / FIGURE_WIDTH_IN,
                PLOT_BOTTOM_IN / FIGURE_HEIGHT_IN,
                panel_width / FIGURE_WIDTH_IN,
                PLOT_HEIGHT_IN / FIGURE_HEIGHT_IN,
            ]
        )
        for index in range(panel_count)
    ]


def multiplier_formatter(value: float, _: object) -> str:
    if value < 1:
        return ""
    if abs(value - round(value)) < 1e-8:
        return f"{int(round(value))}x"
    return f"{value:g}x"


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


def token_efficiency_points(results_dir: Path) -> list[dict[str, object]]:
    run_dir = current_run_dir(results_dir)
    summary, model_payloads = load_current_data(results_dir)
    combined_summaries, combined_payloads = merge_token_efficiency_sources(run_dir, summary, model_payloads)
    points: list[dict[str, object]] = []
    for model, model_summary in sorted(combined_summaries.items()):
        stable_count = int(model_summary["stable_count"])
        if stable_count <= 0 or model not in combined_payloads:
            continue
        trials = stable_count_trial_records(combined_payloads[model])
        token_values = [
            float(trial["total_tokens"])
            for trial in trials
            if trial.get("total_tokens") is not None and float(trial["total_tokens"]) > 0
        ]
        avg_tokens = mean_or_none(token_values)
        if avg_tokens is None:
            continue
        provider = str(model_summary.get("provider") or "")
        if not provider:
            from counting.analysis.figures.search_panels import infer_provider_family

            provider = infer_provider_family(model)
        points.append(
            {
                "model": model,
                "stable_count": stable_count,
                "avg_tokens": avg_tokens,
                "provider": provider,
                "color": PROVIDER_COLORS.get(provider, PROVIDER_COLORS["other"]),
                "marker": PROVIDER_MARKERS.get(provider, PROVIDER_MARKERS["other"]),
            }
        )
    if not points:
        raise RuntimeError("No token-efficiency points found.")
    return points


def draw_token_efficiency(ax: plt.Axes, results_dir: Path) -> None:
    points = token_efficiency_points(results_dir)
    x_values = [float(point["stable_count"]) for point in points]
    y_values = [float(point["avg_tokens"]) for point in points]
    x_lower = max(1.0, min(x_values) * 0.8)
    x_upper = max(x_values) * 2.5
    y_lower = max(1.0, min(y_values) * 0.8)
    y_upper = max(y_values) * 2.5

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    for factor, color in ((0.5, "#b0b0b0"), (0.25, "#b8b8b8"), (1.0, "#c2c2c2")):
        xs = [x_lower, x_upper]
        ys = [value / factor for value in xs]
        ax.plot(xs, ys, linestyle="--", linewidth=1.0, color=color, alpha=0.9, zorder=1)
        label_y = x_upper / factor
        if y_lower <= label_y <= y_upper:
            ax.annotate(
                f"{factor:g}x",
                (x_upper, label_y),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=SMALL_TEXT_FONTSIZE,
                color="#000000",
            )

    for point in points:
        ax.scatter(
            point["stable_count"],
            point["avg_tokens"],
            s=34,
            color=point["color"],
            marker=point["marker"],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

    handles = []
    labels = []
    for provider in ("openai", "anthropic", "google", "deepseek", "qwen", "moonshot", "zai", "mistral"):
        if any(point["provider"] == provider for point in points):
            handles.append(
                ax.scatter([], [], s=32, color=PROVIDER_COLORS[provider], marker=PROVIDER_MARKERS[provider])
            )
            labels.append(PROVIDER_DISPLAY_NAMES[provider])
    legend = ax.legend(
        handles,
        labels,
        loc="lower right",
        frameon=True,
        fontsize=9,
        ncol=2,
        columnspacing=0.6,
        handletextpad=0.35,
        borderpad=0.3,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cfcfcf")
    legend.get_frame().set_alpha(0.92)

    ax.set_title("Total Token Efficiency", fontsize=16)
    ax.set_xlabel("Count Capacity", fontsize=13)
    ax.set_ylabel("Avg total tokens at CC", fontsize=13)
    ax.grid(alpha=0.2, which="both")
    ax.tick_params(axis="both", which="both", labelsize=10.5)
    set_spine_width(ax)


def draw_reasoning_comparison(ax: plt.Axes) -> None:
    records = pair_records(load_payloads())
    x_values = [int(record["delta_scc"]) for record in records]
    y_values = [float(record["token_multiplier"]) for record in records]
    x_min = min(-36, min(x_values) - 4)
    x_max = max(6, max(x_values) + 4)
    y_max = max(y_values) * 1.25

    ax.axvline(0, color="#b8b8b8", linewidth=1.2, linestyle="--", zorder=1)
    ax.axhline(1.0, color="#b8b8b8", linewidth=1.2, linestyle="--", zorder=1)
    ax.scatter([0], [1.0], s=34, color=BASELINE_DARK, edgecolors="white", linewidths=0.7, zorder=5)
    ax.annotate(
        "Non-reasoning",
        (0, 1.0),
        xytext=(-10, 8),
        textcoords="offset points",
        ha="right",
        fontsize=SMALL_TEXT_FONTSIZE,
    )

    for record in records:
        x1 = float(record["delta_scc"])
        y1 = float(record["token_multiplier"])
        color = str(record["color"])
        ax.scatter([x1], [y1], s=26, color=color, edgecolors="white", linewidths=0.6, zorder=4)
        # With 77 pairs the points are unlabeled; PLOT_LABELS names the few
        # worth calling out, and is currently empty by design.
        if str(record["label"]) in PLOT_LABELS:
            dx, dy = LABEL_OFFSETS.get(str(record["label"]), (6, 6))
            ax.annotate(
                f"{PLOT_LABELS[str(record['label'])]}\n{record['token_multiplier']:.1f}x",
                (x1, y1), xytext=(dx, dy), textcoords="offset points",
                fontsize=SMALL_TEXT_FONTSIZE, color=color,
                ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.9),
                zorder=6,
            )

    ax.set_xlim(x_min, x_max)
    ax.set_yscale("log")
    ax.set_ylim(0.85, y_max)
    ax.set_title("Reasoning Comparison", fontsize=16)
    ax.set_xlabel(r"$\Delta$CC", fontsize=13)
    ax.set_ylabel("Token multiplier at CC", fontsize=13)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(multiplier_formatter))
    ax.set_xticks([-32, -24, -16, -8, 0, 4])
    ax.tick_params(axis="both", which="both", labelsize=10.5)
    ax.grid(alpha=0.18, which="both", zorder=0)
    set_spine_width(ax)


def load_scatter_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def panel_subset(rows: list[dict[str, str]], panel_key: str, condition: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["condition"] == condition
        and (row["category"] == panel_key if condition in {"real_task", "plain_count"} else row["category"] == "pooled_control")
    ]


def draw_count_error(ax: plt.Axes, run_dir: Path) -> None:
    rows = load_scatter_rows(run_dir)
    panels = [
        ("plain_count", "plain_count"),
        ("bbh_reasoning", "real_task"),
        ("cruxeval_o", "real_task"),
        ("math500", "real_task"),
        ("mmlu_pro", "real_task"),
        ("irrelevant_code", "irrelevant_code"),
        ("dual_counting", "dual_counting"),
    ]
    for panel_key, condition in panels:
        subset = [
            row
            for row in panel_subset(rows, panel_key, condition)
            if row["count_abs_error"] not in {"", "None"}
        ]
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in subset:
            grouped[int(row["target_count"])].append(float(row["count_abs_error"]))
        xs = np.array(sorted(grouped), dtype=np.int32)
        ys = np.array([np.mean(grouped[int(x)]) for x in xs], dtype=np.float32)
        ax.plot(
            xs,
            smooth_series(ys, window=5),
            marker=PANEL_MARKERS[panel_key],
            markersize=3.2,
            linewidth=1.35,
            color=PANEL_COLORS[panel_key],
            label=TASK_CATEGORY_LABELS[panel_key],
            alpha=0.95,
        )

    ax.axhline(0.0, color=BASELINE_DARK, linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_title("Tasks Interfering Counting", fontsize=16)
    ax.set_xlabel("Target count", fontsize=13)
    ax.set_ylabel("MAE", fontsize=13)
    ax.grid(alpha=0.18, linewidth=0.9)
    ax.tick_params(axis="both", which="both", labelsize=10.5)
    set_spine_width(ax)
    legend = ax.legend(
        frameon=True,
        fontsize=SMALL_TEXT_FONTSIZE,
        loc="upper left",
        ncol=2,
        borderpad=0.3,
        handlelength=1.8,
        columnspacing=0.65,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cfcfcf")
    legend.get_frame().set_alpha(0.92)


def load_selected_benchmark_datasets() -> list[tuple[object, str]]:
    counting_models = build_counting_models()
    alias_map = build_alias_map(counting_models)
    selected: list[tuple[object, str]] = []
    for filename, title in BENCHMARK_PANELS:
        dataset = load_benchmark_dataset(BENCHMARK_DIR / filename, counting_models, alias_map)
        if dataset is None or not dataset.matched_records:
            raise RuntimeError(f"No matched benchmark records found for {filename}")
        selected.append((dataset, title))
    return selected


def draw_benchmark_row(axes: list[plt.Axes]) -> None:
    for ax, (dataset, title) in zip(axes, load_selected_benchmark_datasets()):
        draw_benchmark_scatter(
            ax,
            dataset,
            show_axis_labels=True,
            title_fontsize=16,
            label_fontsize=13,
            tick_fontsize=10.5,
            stats_fontsize=SMALL_TEXT_FONTSIZE,
            point_size=28.0,
            point_linewidth=0.65,
            point_alpha=0.9,
            use_house_spines=True,
            highlight_post_arc_agi2=dataset.path.name == "arc_agi_2_external.csv",
            split_post_arc_agi2_trendlines=dataset.path.name == "arc_agi_2_external.csv",
            legend_fontsize=SMALL_TEXT_FONTSIZE,
            title_override=title,
        )


def add_panel_labels(axes: list[plt.Axes]) -> None:
    for index, ax in enumerate(axes):
        ax.text(
            -0.16,
            1.16,
            string.ascii_uppercase[index],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
        )


def make_figure(results_dir: Path, scatter_run_dir: Path, output: Path) -> None:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    axes = add_manual_grid_axes(fig, nrows=2, ncols=3)
    draw_benchmark_row(axes[:3])
    draw_token_efficiency(axes[3], results_dir)
    draw_reasoning_comparison(axes[4])
    draw_count_error(axes[5], scatter_run_dir)
    # add_panel_labels(axes)
    save_all(fig, output)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    make_figure(args.results_dir, args.scatter_run_dir, args.output)
    print(f"Wrote {args.output}")
    if args.output.suffix.lower() != ".pdf":
        print(f"Wrote {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
