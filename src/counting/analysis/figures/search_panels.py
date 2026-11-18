#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

PERSISTENT_RUN_DIR_NAME = "counting-stable-search"
from counting.paths import REPO_ROOT as ROOT

DEFAULT_ARTICLE_FIGURE_DIR = ROOT / "paper" / "figures"

PHASE_COLORS = {
    "expansion": "#7f97b2",
    "refine": "#c6a78c",
}

PROVIDER_COLORS = {
    "openai": "#d09b52",
    "anthropic": "#8f7aa8",
    "google": "#7b9e87",
    "deepseek": "#b55d60",
    "zai": "#5b8c5a",
    "moonshot": "#4f8fba",
    "minimax": "#8c9a3c",
    "meta": "#6f8fa6",
    "mistral": "#a68a5b",
    "qwen": "#b28d9f",
    "microsoft": "#7f9c96",
    "xai": "#4d5358",
    "other": "#aeb3b8",
}

PROVIDER_MARKERS = {
    "openai": "o",
    "anthropic": "s",
    "google": "^",
    "deepseek": "D",
    "zai": "8",
    "moonshot": ">",
    "minimax": "H",
    "meta": "P",
    "mistral": "X",
    "qwen": "v",
    "microsoft": "<",
    "xai": "*",
    "other": "h",
}

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "deepseek": "DeepSeek",
    "zai": "Z.ai",
    "moonshot": "Moonshot",
    "minimax": "MiniMax",
    "meta": "Meta",
    "mistral": "Mistral",
    "qwen": "Qwen",
    "microsoft": "Microsoft",
    "xai": "xAI",
    "other": "Other",
}

CONTEXT_REFERENCE_TICKS = [4096, 8192, 16384, 32768, 128000, 1048576]
CONTEXT_REFERENCE_MILESTONES = {
    4096: ("gpt-3.5", -0.095),
    8192: ("gpt-4", -0.095),
    16384: ("gpt-3.5-16k", -0.095),
    32768: ("gpt-4-32k", -0.095),
    128000: ("gpt-4-turbo", -0.095),
    1048576: ("gemini-2.5-pro", -0.095),
}
CONTEXT_AXIS_COLOR = "#8c8c8c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot adaptive stable-count benchmark results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a run directory or manifest.json. Defaults to results/counting-stable-search.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory searched when --input is not provided.",
    )
    return parser.parse_args()


def format_context_reference_tick(value: int) -> str:
    if value >= 1_000_000:
        return f"{value // 1_000_000}M"
    if value >= 1000:
        return f"{value // 1000}k"
    return str(value)


def infer_level(model_id: str) -> str:
    if "pro" in model_id:
        return "pro"
    if "nano" in model_id:
        return "nano"
    if "mini" in model_id:
        return "mini"
    return "base"


def exclude_from_number_line(model_id: str) -> bool:
    label = str(model_id).lower()
    return re.search(r"(^|[-_/])(mini|nano|haiku|lite)(?=($|[-_/]))", label) is not None


def infer_provider_family(model_id: str) -> str:
    label = str(model_id).lower()
    if label.startswith("anthropic/") or label.startswith("claude-"):
        return "anthropic"
    if label.startswith("google/") or label.startswith("gemini-") or label.startswith("gemma-"):
        return "google"
    if label.startswith("deepseek/"):
        return "deepseek"
    if label.startswith("z-ai/") or label.startswith("zai-org/") or label.startswith("glm-") or label.startswith("chatglm"):
        return "zai"
    if label.startswith("moonshotai/") or label.startswith("kimi-"):
        return "moonshot"
    if label.startswith("minimax/") or label.startswith("minimax-"):
        return "minimax"
    if label.startswith("meta-llama/") or label.startswith("meta/"):
        return "meta"
    if label.startswith("mistralai/"):
        return "mistral"
    if label.startswith("qwen/"):
        return "qwen"
    if label.startswith("microsoft/"):
        return "microsoft"
    if label.startswith("x-ai/") or label.startswith("grok-"):
        return "xai"
    if (
        label.startswith("openai/")
        or label.startswith("gpt-")
        or label.startswith("o1")
        or label.startswith("o3")
        or label.startswith("o4")
    ):
        return "openai"
    return "other"


def display_model_label(model_id: str) -> str:
    label = str(model_id)
    if label.startswith("anthropic/"):
        label = label[len("anthropic/") :]
    if label.startswith("google/"):
        label = label[len("google/") :]
    if label.startswith("deepseek/"):
        label = label[len("deepseek/") :]
    if label.startswith("z-ai/"):
        label = label[len("z-ai/") :]
    if label.startswith("zai-org/"):
        label = label[len("zai-org/") :]
    if label.startswith("moonshotai/"):
        label = label[len("moonshotai/") :]
    if label.startswith("minimax/"):
        label = label[len("minimax/") :]
    if label.startswith("meta-llama/"):
        label = label[len("meta-llama/") :]
    if label.startswith("mistralai/"):
        label = label[len("mistralai/") :]
    if label.startswith("qwen/"):
        label = label[len("qwen/") :]
    if label.startswith("microsoft/"):
        label = label[len("microsoft/") :]
    if label.startswith("claude-"):
        if label.endswith(":thinking"):
            base = display_model_label(label[: -len(":thinking")])
            return f"{base} (thinking)"

        label = re.sub(r"-\d{8}$", "", label)
        alias_map = {
            "claude-3-haiku": "claude-3-haiku",
            "claude-3.5-haiku": "claude-3.5-haiku",
            "claude-3.7-sonnet": "claude-3.7-sonnet",
            "claude-haiku-4-5": "claude-4.5-haiku",
            "claude-opus-4": "claude-4-opus",
            "claude-opus-4-1": "claude-4.1-opus",
            "claude-opus-4-5": "claude-4.5-opus",
            "claude-opus-4-6": "claude-4.6-opus",
            "claude-sonnet-4": "claude-4-sonnet",
            "claude-sonnet-4-5": "claude-4.5-sonnet",
            "claude-sonnet-4-6": "claude-4.6-sonnet",
        }
        return alias_map.get(label, label)
    return label


def number_line_anchor_and_error(model_summary: dict[str, object]) -> tuple[float, float, str]:
    raw_stable = int(model_summary["stable_count"])
    upper_bound = int(model_summary.get("unstable_upper_bound") or 1)
    anchor = float(max(1, raw_stable if raw_stable > 0 else upper_bound))
    display_label = display_model_label(str(model_summary["model"]))
    error_width = max(1.0, anchor * 0.1)
    label = (
        display_label
        if raw_stable > 0
        else f"{display_label} (<= {upper_bound})"
    )
    return anchor, error_width, label


def run_kind_label(summary: dict[str, object], output_dir: Path) -> str:
    config = summary.get("config", {})
    if isinstance(config, dict):
        run_subdir = str(config.get("run_subdir") or "").lower()
        if "open-source" in run_subdir:
            return "open"
        if run_subdir:
            return "closed"
    if "open-source" in str(output_dir).lower():
        return "open"
    return "closed"


def label_span_log(center: float, lower: float, upper: float, label: str) -> tuple[float, float]:
    del center
    lower_log = math.log10(lower)
    upper_log = math.log10(upper)
    text_width = 0.024 * len(label) + 0.06
    text_gap = 0.035
    return lower_log - 0.02, upper_log + text_gap + text_width


def assign_number_line_levels(items: list[dict[str, object]]) -> list[float]:
    marker_base_level = 0.48
    lane_step = 0.35
    max_lanes_per_side = 120

    def boxes_for(span: tuple[float, float], level: float) -> list[tuple[float, float, float, float]]:
        label_height = 0.22
        label_x_pad = 0.016

        return [(
            span[0] - label_x_pad,
            span[1] + label_x_pad,
            level - label_height / 2,
            level + label_height / 2,
        )]

    def boxes_collide(
        candidate_boxes: list[tuple[float, float, float, float]],
        placed_boxes: list[tuple[float, float, float, float]],
    ) -> bool:
        for cand_x0, cand_x1, cand_y0, cand_y1 in candidate_boxes:
            for placed_x0, placed_x1, placed_y0, placed_y1 in placed_boxes:
                if (
                    cand_x0 <= placed_x1
                    and cand_x1 >= placed_x0
                    and cand_y0 <= placed_y1
                    and cand_y1 >= placed_y0
                ):
                    return True
        return False

    # Place broad labels first, then greedily choose the nearest non-colliding lane.
    indexed_items = list(enumerate(items))
    indexed_items.sort(key=lambda x: (x[1]["label_span"][1] - x[1]["label_span"][0]), reverse=True)

    assigned_boxes: list[tuple[float, float, float, float]] = []
    levels_out = [0.0] * len(items)

    for orig_idx, item in indexed_items:
        span_i = item["label_span"]
        sign_order = [1, -1] if orig_idx % 2 == 0 else [-1, 1]

        best_y: float | None = None
        best_boxes: list[tuple[float, float, float, float]] = []
        for lane in range(max_lanes_per_side):
            for sign in sign_order:
                y_cand = sign * (marker_base_level + lane * lane_step)
                candidate_boxes = boxes_for(span_i, y_cand)
                if not boxes_collide(candidate_boxes, assigned_boxes):
                    best_y = y_cand
                    best_boxes = candidate_boxes
                    break

            if best_y is not None:
                break

        if best_y is None:
            best_y = sign_order[0] * (marker_base_level + max_lanes_per_side * lane_step)
            best_boxes = boxes_for(span_i, best_y)

        assigned_boxes.extend(best_boxes)
        levels_out[orig_idx] = best_y

    return levels_out


def resolve_run_dir(path: Path) -> Path:
    if path.is_dir():
        if not (path / "manifest.json").exists():
            raise ValueError(f"Run directory {path} is missing manifest.json")
        return path
    if path.name == "manifest.json":
        return path.parent
    raise ValueError(f"Unsupported input path {path}")


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_data(run_dir: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    manifest = load_summary(run_dir / "manifest.json")
    model_payloads: dict[str, dict[str, object]] = {}
    for model, rel_path in manifest["model_files"].items():
        model_payloads[model] = load_summary(run_dir / str(rel_path))
    return manifest, model_payloads


def current_run_dir(results_dir: Path) -> Path:
    return results_dir / PERSISTENT_RUN_DIR_NAME


def load_current_data(results_dir: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    run_dir = current_run_dir(results_dir)
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"No persistent benchmark directory found at {run_dir}")
    return load_run_data(run_dir)


def merge_token_efficiency_sources(
    run_dir: Path,
    summary: dict[str, object],
    model_payloads: dict[str, dict[str, object]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    combined_summary_models = dict(summary.get("models", {}))
    combined_model_payloads = dict(model_payloads)
    results_dir = run_dir.parent
    for extra_name in (PERSISTENT_RUN_DIR_NAME, "counting-stable-search-open-source"):
        extra_dir = results_dir / extra_name
        if extra_dir == run_dir or not (extra_dir / "manifest.json").exists():
            continue
        extra_summary, extra_payloads = load_run_data(extra_dir)
        for model, model_summary in extra_summary.get("models", {}).items():
            combined_summary_models.setdefault(model, model_summary)
        for model, payload in extra_payloads.items():
            combined_model_payloads.setdefault(model, payload)
    return combined_summary_models, combined_model_payloads


def load_pyplot():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    return plt


def positive_values(records: list[dict[str, object]]) -> list[float]:
    values: list[float] = []
    for record in records:
        expected = float(record["expected"])
        if expected > 0:
            values.append(expected)
        parsed = record.get("parsed")
        if parsed is not None and float(parsed) > 0:
            values.append(float(parsed))
    return values


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def scatter_limits(model_records: list[dict[str, object]], stable_count: int) -> tuple[float, float]:
    values = positive_values(model_records)
    if stable_count > 0:
        values.append(float(stable_count))
    lower = max(1.0, min(values) * 0.9)
    upper = max(values) * 1.1
    return lower, upper


def stable_trial_records(model_payload: dict[str, object]) -> list[dict[str, object]]:
    summary = dict(model_payload.get("summary", {}))
    passed_keys = {
        (str(item["phase"]), int(item["evaluation_index"]))
        for item in summary.get("evaluations", [])
        if item.get("passed")
    }
    return [
        trial
        for trial in model_payload.get("trials", [])
        if (str(trial.get("phase")), int(trial.get("evaluation_index", -999))) in passed_keys
    ]


def stable_count_trial_records(model_payload: dict[str, object]) -> list[dict[str, object]]:
    summary = dict(model_payload.get("summary", {}))
    stable_count = int(summary.get("stable_count") or 0)
    if stable_count <= 0:
        return []
    matching_evaluations = [
        item
        for item in summary.get("evaluations", [])
        if item.get("passed")
        and int(item.get("center_length") or 0) == stable_count
    ]
    if not matching_evaluations:
        return []
    target = max(matching_evaluations, key=lambda item: int(item.get("evaluation_index") or -1))
    target_key = (str(target["phase"]), int(target["evaluation_index"]))
    return [
        trial
        for trial in model_payload.get("trials", [])
        if (str(trial.get("phase")), int(trial.get("evaluation_index", -999))) == target_key
    ]


def is_token_limit_failure(trial: dict[str, object]) -> bool:
    detail = str(trial.get("response_detail") or "")
    return trial.get("response_status") == "incomplete" and "max_output_tokens" in detail


def is_null_token_limit_failure(trial: dict[str, object]) -> bool:
    raw_response = str(trial.get("raw_response") or "")
    return is_token_limit_failure(trial) and trial.get("parsed") is None and not raw_response.strip()


def summarize_token_limit_center(center_trials: list[dict[str, object]]) -> dict[str, object]:
    failures = [trial for trial in center_trials if is_token_limit_failure(trial)]
    null_failures = [trial for trial in failures if is_null_token_limit_failure(trial)]
    sample = center_trials[0]
    return {
        "phase": sample["phase"],
        "evaluation_index": sample["evaluation_index"],
        "center_length": sample["center_length"],
        "trial_count": len(center_trials),
        "token_limit_failures": len(failures),
        "null_token_limit_failures": len(null_failures),
        "failure_rate": len(failures) / len(center_trials),
        "null_failure_rate": len(null_failures) / len(center_trials),
    }


def summarize_token_limit_model(model_payload: dict[str, object]) -> dict[str, object]:
    model = str(model_payload["model"])
    trials = list(model_payload.get("trials", []))
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for trial in trials:
        key = (str(trial["phase"]), int(trial["center_length"]))
        grouped.setdefault(key, []).append(trial)

    center_summaries = [
        summarize_token_limit_center(grouped[key])
        for key in sorted(grouped, key=lambda item: (item[0], item[1]))
    ]
    failure_centers = [item for item in center_summaries if item["token_limit_failures"] > 0]
    null_failure_centers = [item for item in center_summaries if item["null_token_limit_failures"] > 0]
    expansion_centers = [item for item in center_summaries if item["phase"] == "expansion"]
    expansion_failures = [item for item in expansion_centers if item["token_limit_failures"] > 0]
    first_expansion_center = expansion_centers[0]["center_length"] if expansion_centers else None
    first_expansion_failure_center = (
        expansion_failures[0]["center_length"] if expansion_failures else None
    )
    ladder_triggered = (
        first_expansion_center is not None
        and first_expansion_failure_center is not None
        and first_expansion_failure_center > first_expansion_center
    )
    total_trials = len(trials)
    total_failures = sum(1 for trial in trials if is_token_limit_failure(trial))
    total_null_failures = sum(1 for trial in trials if is_null_token_limit_failure(trial))

    return {
        "model": model,
        "trial_count": total_trials,
        "token_limit_failures": total_failures,
        "null_token_limit_failures": total_null_failures,
        "token_limit_failure_rate": total_failures / total_trials if total_trials else 0.0,
        "null_token_limit_failure_rate": total_null_failures / total_trials if total_trials else 0.0,
        "has_issue": total_failures > 0,
        "has_null_issue": total_null_failures > 0,
        "ladder_triggered": ladder_triggered,
        "first_expansion_center": first_expansion_center,
        "first_expansion_failure_center": first_expansion_failure_center,
        "first_failure_center": failure_centers[0]["center_length"] if failure_centers else None,
        "first_null_failure_center": (
            null_failure_centers[0]["center_length"] if null_failure_centers else None
        ),
        "center_summaries": center_summaries,
    }


def build_token_limit_report(
    summary: dict[str, object],
    model_payloads: dict[str, dict[str, object]],
) -> dict[str, object]:
    models = [summarize_token_limit_model(model_payload) for model_payload in model_payloads.values()]
    models_with_issue = [item for item in models if item["has_issue"]]
    ladder_models = [item for item in models_with_issue if item["ladder_triggered"]]
    immediate_models = [item for item in models_with_issue if not item["ladder_triggered"]]

    models_with_issue.sort(
        key=lambda item: (
            -int(item["null_token_limit_failures"]),
            -int(item["token_limit_failures"]),
            int(item["first_failure_center"] or 0),
            str(item["model"]),
        )
    )
    ladder_models.sort(
        key=lambda item: (
            int(item["first_expansion_failure_center"] or 0),
            -int(item["null_token_limit_failures"]),
            str(item["model"]),
        )
    )
    immediate_models.sort(
        key=lambda item: (-int(item["null_token_limit_failures"]), str(item["model"]))
    )

    return {
        "generated_at_utc": summary.get("generated_at_utc"),
        "source": summary.get("source", "run"),
        "config": summary.get("config", {}),
        "model_count": len(models),
        "models_with_issue": models_with_issue,
        "ladder_triggered_models": ladder_models,
        "immediate_issue_models": immediate_models,
    }


def token_limit_report_lines(
    summary: dict[str, object],
    model_payloads: dict[str, dict[str, object]],
) -> list[str]:
    report = build_token_limit_report(summary, model_payloads)
    models_with_issue = report["models_with_issue"]
    if not models_with_issue:
        return ["Token-limit failures: none"]

    lines = [
        "Token-limit failures:",
        (
            f"  models_with_issue={len(models_with_issue)} "
            f"ladder_triggered={len(report['ladder_triggered_models'])}"
        ),
    ]
    for item in report["ladder_triggered_models"]:
        lines.append(
            "  "
            f"{item['model']}: ladder-triggered at expansion@{item['first_expansion_failure_center']} "
            f"null_failures={item['null_token_limit_failures']}/{item['trial_count']}"
        )
    for item in report["immediate_issue_models"]:
        lines.append(
            "  "
            f"{item['model']}: immediate token-limit issue at expansion@{item['first_expansion_failure_center']} "
            f"null_failures={item['null_token_limit_failures']}/{item['trial_count']}"
        )
    return lines


def assign_scatter_label_offsets(ax, points: list[dict[str, object]]) -> None:
    if not points:
        return

    ax.figure.canvas.draw()
    sorted_points = sorted(points, key=lambda item: (float(item["x"]), float(item["y"]), item["model"]))
    candidate_offsets = [
        (10, 0),
        (10, 12),
        (10, -12),
        (14, 24),
        (14, -24),
    ]
    occupied: list[tuple[float, float, float, float]] = []
    padding_x = 22.0
    padding_y = 12.0

    for point in sorted_points:
        x_disp, y_disp = ax.transData.transform((float(point["x"]), float(point["y"])))
        label_width = max(42.0, 6.8 * len(str(point["label"])))
        label_height = 17.0
        chosen = None
        for dx, dy in candidate_offsets:
            bbox = (
                x_disp + dx - padding_x,
                x_disp + dx + label_width + padding_x,
                y_disp + dy - label_height / 2 - padding_y,
                y_disp + dy + label_height / 2 + padding_y,
            )
            if all(
                bbox[1] < other[0] or bbox[0] > other[1] or bbox[3] < other[2] or bbox[2] > other[3]
                for other in occupied
            ):
                chosen = (dx, dy)
                occupied.append(bbox)
                break
        point["offset"] = chosen or candidate_offsets[0]
        point["show_label"] = chosen is not None


def save_model_plots(
    summary: dict[str, object],
    model_payloads: dict[str, dict[str, object]],
    output_dir: Path,
    output_name: str,
) -> list[Path]:
    plt = load_pyplot()
    tolerance_fraction = float(summary["config"]["search"]["tolerance_fraction"])
    models = list(summary["models"].items())
    ncols = 4
    max_rows_per_figure = 5
    models_per_figure = ncols * max_rows_per_figure
    output_spec = Path(output_name)
    stem = output_spec.stem
    suffix = output_spec.suffix or ".pdf"
    run_kind = run_kind_label(summary, output_dir)

    for old_path in output_dir.glob(f"{stem}*{suffix}"):
        old_path.unlink()

    output_paths: list[Path] = []
    model_chunks = [
        models[index : index + models_per_figure]
        for index in range(0, len(models), models_per_figure)
    ] or [[]]

    for page_index, model_chunk in enumerate(model_chunks, start=1):
        nrows = max(1, math.ceil(len(model_chunk) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(10, max(3.0, 2.6 * nrows)),
            constrained_layout=True,
            squeeze=False,
        )

        for axis in axes.flat:
            axis.set_visible(False)

        for axis, (model, model_summary) in zip(axes.flat, model_chunk):
            axis.set_visible(True)
            model_records = model_payloads[model]["trials"]
            parsed_records = [
                record
                for record in model_records
                if record.get("parsed") is not None and float(record["parsed"]) > 0
            ]
            stable_count = int(model_summary["stable_count"])
            if parsed_records:
                lower, upper = scatter_limits(parsed_records, stable_count)
            else:
                lower, upper = 1.0, max(10.0, float(model_summary.get("unstable_upper_bound") or 10))
            xs = [lower, upper]
            ax = axis

            for phase in sorted({str(record["phase"]) for record in parsed_records}):
                phase_records = [record for record in parsed_records if record["phase"] == phase]
                x_values = [float(record["expected"]) for record in phase_records]
                y_values = [float(record["parsed"]) for record in phase_records]
                ax.scatter(
                    x_values,
                    y_values,
                    s=12,
                    alpha=0.85,
                    color=PHASE_COLORS.get(phase, "#999999"),
                    label=phase,
                )

            ax.plot(xs, xs, color="#444444", linewidth=1.1, label="ideal")
            ax.plot(xs, [value * (1 + tolerance_fraction) for value in xs], color="#999999", linewidth=1.0, linestyle="--")
            ax.plot(xs, [value * (1 - tolerance_fraction) for value in xs], color="#999999", linewidth=1.0, linestyle="--")
            if stable_count > 0:
                ax.axvline(stable_count, color="#e45756", linewidth=1.2, linestyle=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_title(display_model_label(model))
            ax.set_xlabel("Actual Count")
            ax.set_ylabel("Predicted Count")
            ax.grid(alpha=0.2)
            if parsed_records:
                ax.legend(loc="lower right", fontsize=7)
            ax.text(
                0.03,
                0.97,
                (
                    f"CC={stable_count}\n"
                    f"upper={model_summary['unstable_upper_bound']}\n"
                    f"trials={model_summary['trial_count']}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )

        output_path = output_dir / f"{stem}-{run_kind}-{page_index}{suffix}"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def save_number_line_plot(summary: dict[str, object], output_dir: Path, output_name: str) -> Path:
    plt = load_pyplot()
    import matplotlib.ticker as ticker
    
    items: list[dict[str, object]] = []
    for model, raw_summary in summary["models"].items():
        if exclude_from_number_line(model):
            continue
        model_summary = dict(raw_summary)
        model_summary["model"] = model
        center, error_width, label = number_line_anchor_and_error(model_summary)
        lower = max(1.0, center - error_width)
        upper = center + error_width
        items.append(
            {
                "model": model,
                "summary": model_summary,
                "provider": infer_provider_family(model),
                "color": PROVIDER_COLORS[infer_provider_family(model)],
                "center": center,
                "error_width": error_width,
                "lower": lower,
                "upper": upper,
                "label": label,
                "label_span": label_span_log(center, lower, upper, label),
            }
        )

    # Pre-Sorting the initial list ensures the horizontal progression translates logically 
    items.sort(key=lambda item: item["center"])
    levels = assign_number_line_levels(items)
    for item, level in zip(items, levels):
        item["level"] = level

    max_level = max(abs(float(item["level"])) for item in items) if items else 1.0
    # Because dense packing drops max_level significantly, fig_height drastically minimizes dynamically
    fig_height = max(2.35, 1.18 + max_level * 0.86)
    fig, ax = plt.subplots(figsize=(10, fig_height), constrained_layout=True)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    ax.set_xscale("log")
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["bottom"].set_linewidth(2.4)
    ax.spines["bottom"].set_color("#2f2f2f")
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)

    for item in items:
        level = float(item["level"])
        color = str(item["color"])
        
        ax.plot(
            [item["center"], item["center"]],
            [0, level],
            color=color,
            linewidth=1.0,
            alpha=0.28,
            solid_capstyle="round",
            zorder=1,
        )
        ax.errorbar(
            item["center"],
            level,
            xerr=[
                [item["center"] - item["lower"]],
                [item["upper"] - item["center"]],
            ],
            fmt=PROVIDER_MARKERS[item["provider"]],
            color=color,
            ecolor=color,
            elinewidth=1.7,
            capsize=6,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
            zorder=3,
        )
        ax.annotate(
            item["label"],
            xy=(item["upper"], level),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=11,
            color=color,
            zorder=4,
        )

    legend_providers = []
    for provider in ("openai", "google", "anthropic"):
        if any(item["provider"] == provider for item in items):
            legend_providers.append(provider)
    if legend_providers:
        legend_handles = [
            ax.plot(
                [],
                [],
                linestyle="None",
                marker=PROVIDER_MARKERS[provider],
                markersize=6.5,
                markerfacecolor="white",
                markeredgecolor=PROVIDER_COLORS[provider],
                markeredgewidth=1.6,
                color=PROVIDER_COLORS[provider],
            )[0]
            for provider in legend_providers
        ]
        legend = ax.legend(
            legend_handles,
            [PROVIDER_DISPLAY_NAMES[provider] for provider in legend_providers],
            loc="upper left",
            bbox_to_anchor=(1.01, 0.995),
            frameon=True,
            fontsize=8,
            ncol=1,
            borderpad=0.35,
            handletextpad=0.4,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#d7d7d7")
        legend.get_frame().set_alpha(0.94)

    lower_limit = min(item["lower"] for item in items) if items else 1.0
    upper_limit = max(item["upper"] for item in items) if items else 10.0
    ax.set_xlim(max(1.0, lower_limit / 1.15), upper_limit * 1.15)
    ax.set_ylim(-max_level - 0.24, max_level + 0.28)
    ax.set_yticks([])
    ax.set_xlabel("")
    
    # Render explicitly stepped visual interval ticks
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20))
    formatter = ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
    ax.xaxis.set_major_formatter(formatter)

    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(1, 10), numticks=100))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.tick_params(axis="x", which="major", length=0, width=0.0, pad=10, labelbottom=False)
    ax.tick_params(axis="x", which="minor", length=5, width=1.0, color="#2f2f2f")

    major_ticks = [
        tick for tick in ax.get_xticks()
        if math.isfinite(float(tick)) and ax.get_xlim()[0] <= float(tick) <= ax.get_xlim()[1]
    ]
    for tick in major_ticks:
        ax.text(
            float(tick),
            0,
            formatter(float(tick), None),
            ha="center",
            va="center",
            fontsize=9,
            color="#2f2f2f",
            fontweight="bold",
            bbox=dict(
                boxstyle="circle,pad=0.28",
                facecolor="white",
                edgecolor="#2f2f2f",
                linewidth=2.0,
            ),
            zorder=11,
            clip_on=False,
        )
    
    # Plot robust arrowhead directly affixed to the right bounding tip of the main spine axis 
    ax.plot(
        1.0, 0,
        transform=ax.get_yaxis_transform(),
        marker=">",
        markersize=8,
        color="#2f2f2f",
        clip_on=False,
        zorder=10,
    )
    
    # Add adjacent floating text directly to the right pointing forwards organically 
    ax.text(
        1.015, 0,
        "Count Capacity",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=11,
        color="#2f2f2f",
        clip_on=False,
    )

    context_axis_y = -0.17
    context_log_min = 3.5
    context_log_max = 6.5
    ax.plot(
        [0.0, 1.0],
        [context_axis_y, context_axis_y],
        transform=ax.transAxes,
        color=CONTEXT_AXIS_COLOR,
        linewidth=2.4,
        clip_on=False,
        zorder=10,
    )
    for tick in CONTEXT_REFERENCE_TICKS:
        x_pos = (math.log10(float(tick)) - context_log_min) / (context_log_max - context_log_min)
        ax.text(
            x_pos,
            context_axis_y,
            format_context_reference_tick(tick),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=CONTEXT_AXIS_COLOR,
            fontweight="bold",
            bbox=dict(
                boxstyle="circle,pad=0.28",
                facecolor="white",
                edgecolor=CONTEXT_AXIS_COLOR,
                linewidth=2.0,
            ),
            zorder=11,
            clip_on=False,
        )
        milestone_label, milestone_offset = CONTEXT_REFERENCE_MILESTONES[tick]
        ax.text(
            x_pos,
            context_axis_y + milestone_offset,
            milestone_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            color=CONTEXT_AXIS_COLOR,
            clip_on=False,
        )
    ax.plot(
        1.0,
        context_axis_y,
        transform=ax.transAxes,
        marker=">",
        markersize=8,
        color=CONTEXT_AXIS_COLOR,
        clip_on=False,
        zorder=10,
    )
    ax.text(
        1.015,
        context_axis_y,
        "Context Length",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        color=CONTEXT_AXIS_COLOR,
        clip_on=False,
    )
    ax.grid(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output_path


def save_token_efficiency_plot(
    summary: dict[str, object],
    model_payloads: dict[str, dict[str, object]],
    source_run_dir: Path,
    output_dir: Path,
    output_name: str,
) -> Path:
    plt = load_pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4), constrained_layout=True)
    combined_summaries, combined_model_payloads = merge_token_efficiency_sources(
        source_run_dir,
        summary,
        model_payloads,
    )

    points: list[dict[str, object]] = []
    for model, model_summary in sorted(combined_summaries.items()):
        stable_count_trials = stable_count_trial_records(combined_model_payloads[model])
        total_token_values = [
            float(trial["total_tokens"])
            for trial in stable_count_trials
            if trial.get("total_tokens") is not None and float(trial["total_tokens"]) > 0
        ]
        stable_count_total_tokens = mean_or_none(total_token_values)
        stable_count = int(model_summary["stable_count"])
        if stable_count <= 0 or stable_count_total_tokens is None:
            continue
        points.append(
            {
                "model": model,
                "stable_count": stable_count,
                "stable_count_total_tokens": stable_count_total_tokens,
                "provider": infer_provider_family(model),
                "color": PROVIDER_COLORS[infer_provider_family(model)],
                "marker": PROVIDER_MARKERS[infer_provider_family(model)],
                "x": float(stable_count),
                "y": stable_count_total_tokens,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    if not points:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No token-usage data captured yet.\nRerun one or more models with the updated benchmark.",
            ha="center",
            va="center",
            fontsize=14,
            color="#555555",
            transform=ax.transAxes,
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    x_values = [float(point["stable_count"]) for point in points]
    y_values = [float(point["stable_count_total_tokens"]) for point in points]
    x_lower = max(1.0, min(x_values) * 0.8)
    x_upper = max(x_values) * 2.5
    y_lower = max(1.0, min(y_values) * 0.8)
    y_upper = max(y_values) * 2.5

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)

    guide_specs = [
        (0.5, "#b0b0b0"),
        (0.25, "#b8b8b8"),
        (1.0, "#c2c2c2"),
    ]
    xs = [x_lower, x_upper]
    for factor, color in guide_specs:
        ys = [value / factor for value in xs]
        ax.plot(xs, ys, linestyle="--", linewidth=1.0, color=color, alpha=0.9, zorder=1)
        label_y = x_upper / factor
        if y_lower <= label_y <= y_upper:
            ax.annotate(
                f"{factor:g}x",
                (x_upper, label_y),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=12,
                color="#000000",
                zorder=2,
            )

    for point in points:
        ax.scatter(
            point["stable_count"],
            point["stable_count_total_tokens"],
            s=58,
            color=point["color"],
            marker=point["marker"],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    legend_handles = []
    legend_labels = []
    for provider in (
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
    ):
        if any(point["provider"] == provider for point in points):
            legend_handles.append(
                ax.scatter(
                    [],
                    [],
                    s=58,
                    color=PROVIDER_COLORS[provider],
                    marker=PROVIDER_MARKERS[provider],
                    alpha=0.9,
                )
            )
            legend_labels.append(PROVIDER_DISPLAY_NAMES[provider])
    if legend_handles:
        legend = ax.legend(
            legend_handles,
            legend_labels,
            loc="lower right",
            frameon=True,
            fontsize=9,
            ncol=2,
            columnspacing=1.1,
            handletextpad=0.5,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#cfcfcf")
        legend.get_frame().set_alpha(0.92)

    ax.set_xlabel("Count Capacity", fontsize=14)
    ax.set_ylabel("Avg total tokens at stable count", fontsize=14)
    ax.set_title("Total Token Efficiency", fontsize=16)
    ax.tick_params(axis="both", which="both", labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(alpha=0.2, which="both")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    if args.input:
        run_dir = resolve_run_dir(args.input)
        summary, model_payloads = load_run_data(run_dir)
        output_dir = run_dir
        model_plot_paths = save_model_plots(summary, model_payloads, output_dir, "search-panels.pdf")
        number_line_path = save_number_line_plot(summary, output_dir, "stable-count-number-line.pdf")
        token_efficiency_path = save_token_efficiency_plot(
            summary,
            model_payloads,
            run_dir,
            output_dir,
            "token-efficiency-scatter.pdf",
        )
        print(f"Run directory: {run_dir}")
        print(f"Manifest: {run_dir / 'manifest.json'}")
    else:
        run_dir = current_run_dir(args.results_dir)
        summary, model_payloads = load_current_data(args.results_dir)
        output_dir = run_dir
        article_figure_dir = DEFAULT_ARTICLE_FIGURE_DIR
        model_plot_paths = save_model_plots(summary, model_payloads, output_dir, "search-panels.pdf")
        number_line_path = save_number_line_plot(summary, article_figure_dir, "stable-count-number-line.pdf")
        token_efficiency_path = save_token_efficiency_plot(
            summary,
            model_payloads,
            run_dir,
            article_figure_dir,
            "token-efficiency-scatter.pdf",
        )
        print(f"Run directory: {run_dir}")
        print(f"Manifest: {run_dir / 'manifest.json'}")
    for model_plot_path in model_plot_paths:
        print(f"Search panels: {model_plot_path}")
    print(f"Number line: {number_line_path}")
    print(f"Token efficiency: {token_efficiency_path}")
    for line in token_limit_report_lines(summary, model_payloads):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
