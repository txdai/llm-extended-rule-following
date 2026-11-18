#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from counting.analysis.figures.attractors import set_spine_width
from counting.analysis.figures.search_panels import (
    PROVIDER_COLORS,
    PROVIDER_MARKERS,
    display_model_label,
    infer_provider_family,
)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

from counting.datastore import load_registry
from counting.paths import REPO_ROOT as ROOT
BENCHMARK_DIR = ROOT / "results" / "benchmark_data"
COUNTING_RUN = ROOT / "results" / "counting-data"
# CC is the fitted transition of the SCC scan, which lives in scc.csv. The
# registry's own `cc` column is the older first-mistake reading and is kept only
# for comparison; correlating against it would not be the quantity the paper
# defines.
SCC_TABLE = COUNTING_RUN / "scc.csv"
ARTICLE_FIGURE_DIR = ROOT / "paper" / "figures"

ROW_FIGURE_WIDTH_IN = 12.5
ROW_PLOT_LEFT_IN = 0.78
ROW_PLOT_WIDTH_IN = 11.2

MAPPING_TABLE_PATH = BENCHMARK_DIR / "counting_benchmark_model_mapping.csv"
MERGED_TABLE_PATH = BENCHMARK_DIR / "counting_benchmark_merged.csv"
SUMMARY_PATH = BENCHMARK_DIR / "counting_benchmark_summary.json"
PAIRWISE_RHO_PATH = BENCHMARK_DIR / "counting_benchmark_pairwise_rho.csv"

AUTHORITATIVE_BENCHMARKS = (
    "gpqa_diamond.csv",
    "swe_bench_verified.csv",
    "arc_agi_2_external.csv",
)

AUTHORITATIVE_FIGURE_PDF = ARTICLE_FIGURE_DIR / "benchmark-correlation-authoritative.pdf"
AUTHORITATIVE_FIGURE_PNG = ARTICLE_FIGURE_DIR / "benchmark-correlation-authoritative.png"
AUTHORITATIVE_FIGURE_SVG = ARTICLE_FIGURE_DIR / "benchmark-correlation-authoritative.svg"
SECONDARY_FIGURE_PDF = ARTICLE_FIGURE_DIR / "benchmark-correlation-secondary.pdf"
SECONDARY_FIGURE_PNG = ARTICLE_FIGURE_DIR / "benchmark-correlation-secondary.png"
PRICE_SNAPSHOT = ROOT / "results" / "model_catalogue" / "openrouter_prices_2026-08-25.json"
SECONDARY_BENCHMARKS = ("epoch_capabilities_index.csv", "metr_time_horizons_external.csv")
RHO_DISTRIBUTION_FIGURE_PDF = ARTICLE_FIGURE_DIR / "benchmark-correlation-rho-distribution.pdf"
RHO_DISTRIBUTION_FIGURE_PNG = ARTICLE_FIGURE_DIR / "benchmark-correlation-rho-distribution.png"
FULL_PANEL_STEM = "benchmark-correlation-panels"
ARC_AGI_2_RELEASE_DATE = date(2025, 3, 24)
ARC_AGI_2_POST_TRAINING_BUFFER_DATE = date(2025, 11, 24)
POST_ARC_AGI_2_HIGHLIGHT = "#d95f02"
ARC_AGI_2_POST_RELEASE_MODELS = {"claude-4.6-opus", "claude-4.6-sonnet", "gemini-3-pro-preview"}
# Released 23 days past the buffer date, but its training predates ARC-format
# saturation (0.336 even at High effort against 0.6-0.8 for capacity peers);
# treated as pre-release rather than as a post-cohort residual.
ARC_AGI_2_POST_RELEASE_EXCLUDED = {"google/gemini-3-flash-preview"}

METADATA_COLUMNS = {
    "Model version",
    "Release date",
    "Organization",
    "Country",
    "Training compute (FLOP)",
    "Training compute notes",
    "stderr",
    "Log viewer",
    "Logs",
    "Started at",
    "id",
    "Best score (across scorers)",
    "Name",
    "Notes",
    "Source",
    "Source link",
}
NON_SCORE_COLUMNS = {"Shots", "Scaffold", "Tools", "Agent"}
SCORE_COLUMN_PREFERENCES = (
    "ECI Score",
    "mean_score",
    "EM",
    "Accuracy mean",
    "Score OPT@1",
    "ACW Avg Score",
    "Overall pass (%)",
    "Pass@1 score",
    "Arena Score",
    "% Score",
    "% Resolved",
    "Average (%)",
    "Average progress",
    "Global average",
    "Average score",
    "Accuracy",
    "Score (AVG@5)",
    "Correct",
    "Time horizon",
    "Win Rate (%)",
    "Challenge score",
    "Overall (no subtitles)",
    "120k token score",
    "Unguided % Solved",
    "Mean score",
    "Overall accuracy",
    "Percent correct",
    "Score",
)
BENCHMARK_TITLE_OVERRIDES = {
    "arc_agi_2": "ARC-AGI-2",
    "arc_agi": "ARC-AGI",
    "arc_ai2": "ARC AI2",
    "bool_q": "BoolQ",
    "epoch_capabilities_index": "Epoch Capabilities Index",
    "frontiermath": "FrontierMath",
    "frontiermath_tier_4": "FrontierMath Tier 4",
    "gpqa_diamond": "GPQA Diamond",
    "gsm8k": "GSM8K",
    "hle": "HLE",
    "math_level_5": "Math Level 5",
    "mmlu": "MMLU",
    "os_world": "OSWorld",
    "otis_mock_aime_2024_2025": "OTIS Mock AIME",
    "piqa": "PIQA",
    "simpleqa_verified": "SimpleQA Verified",
    "swe_bench_bash": "SWE-Bench Bash",
    "swe_bench_verified": "SWE-Bench Verified",
    "video_mme": "Video-MME",
    "vpct": "VPCT",
    "webdev_arena": "WebDev Arena",
    "weirdml": "WeirdML",
    "wino_grande": "WinoGrande",
}


def add_manual_row_axes(
    fig: plt.Figure,
    *,
    panel_count: int,
    figure_width_in: float,
    figure_height_in: float,
    plot_left_in: float,
    plot_bottom_in: float,
    plot_width_in: float,
    plot_height_in: float,
    panel_gap_in: float,
) -> list[plt.Axes]:
    panel_width_in = (plot_width_in - panel_gap_in * (panel_count - 1)) / panel_count
    return [
        fig.add_axes(
            [
                (plot_left_in + index * (panel_width_in + panel_gap_in)) / figure_width_in,
                plot_bottom_in / figure_height_in,
                panel_width_in / figure_width_in,
                plot_height_in / figure_height_in,
            ]
        )
        for index in range(panel_count)
    ]


@dataclass(frozen=True)
class CountingModel:
    model: str
    stable_count: float
    provider: str
    display_label: str
    release_label: str
    alias_labels: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark_file: str
    benchmark_name: str
    score_column: str
    row_index: int
    benchmark_model_version: str
    benchmark_name_fallback: str
    matched_input: str
    matched_input_column: str
    normalized_key: str
    reasoning_effort: str
    benchmark_release_date: date | None
    benchmark_release_tag: str
    counting_release_tag: str
    date_match_status: str
    counting_model: str
    counting_model_label: str
    provider: str
    benchmark_score: float
    stable_count: float


@dataclass(frozen=True)
class BenchmarkDataset:
    path: Path
    display_name: str
    score_column: str
    matched_records: list[BenchmarkRecord]
    matched_model_count: int
    plotted_point_count: int
    zero_scc_count: int
    spearman_rho: float | None
    pearson_log_scc: float | None


def normalized_model_key(label: str) -> str:
    key = label.strip().lower()
    key = key.replace("：", ":")
    key = key.replace(":thinking", "-thinking")
    key = re.sub(r"^(chutes|fireworks|moonshotai|minimax|minimaxai|z-ai|zai-org)/", "", key)
    key = re.sub(r"\bglm[\s_]+", "glm-", key)
    # Some tables write a decimal point as "p": glm-4p5, deepseek-v3p2. The
    # rule was per-model; it is the convention, not the model, so apply it once.
    key = re.sub(r"(\d)p(\d)", r"\1.\2", key)
    key = key.replace("kimi-k2-thinking-turbo", "kimi-k2-thinking")
    key = key.replace("kimi-k2-instruct-0905", "kimi-k2-0905")
    key = key.replace("kimi-k2-instruct", "kimi-k2")
    key = key.replace("minimax-2.5", "minimax-m2.5")
    key = key.replace("minimax-2.1", "minimax-m2.1")
    key = key.replace("minimax-2", "minimax-m2")
    key = re.sub(r"_(\d+[kK]|max)$", "", key)
    key = re.sub(r"-(20\d{2}-\d{2}-\d{2}|\d{8})", "", key)
    key = re.sub(r"_(low|medium|high|xhigh|unknown)$", "", key)
    # Anthropic minor versions are dotted in the model id and dashed in several
    # benchmark tables: claude-opus-4-5 is our claude-opus-4.5. Without this,
    # five Claude models drop out of every benchmark comparison, including two
    # of the highest capacities measured. Scoped to claude because a dash
    # between digits is meaningful in other vendors' names.
    if key.startswith("claude-"):
        key = re.sub(r"-(\d)-(\d)(?=-|$)", r"-\1.\2", key)
    # A space in a model name stands where a dash belongs -- one table writes
    # "Llama-4-Scout-17B-16E Instruct" -- so collapse to a dash rather than
    # deleting, which would run the two parts together into a new token.
    key = re.sub(r"\s+", "-", key).strip("-")
    # Quantisation is a serving precision, not a different model, and is treated
    # like the context-window and effort suffixes already stripped above.
    key = re.sub(r"[-_]fp(8|16)$", "", key)
    key = re.sub(r"-{2,}", "-", key)
    return key


def extract_release_tag(label: str) -> str:
    if not label:
        return ""
    match = re.search(r"(20\d{2}-\d{2}-\d{2}|\d{8})", label)
    if match:
        return match.group(1)
    month_day_match = re.search(r"(?<!\d)(\d{4})(?!\d)", label)
    return month_day_match.group(1) if month_day_match else ""


def release_tags_match(benchmark_tag: str, counting_tag: str) -> bool:
    if not benchmark_tag or not counting_tag:
        return False
    if benchmark_tag == counting_tag:
        return True
    normalized_benchmark = benchmark_tag.replace("-", "")
    normalized_counting = counting_tag.replace("-", "")
    if len(normalized_benchmark) == 8 and len(normalized_counting) == 4:
        return normalized_benchmark.endswith(normalized_counting)
    if len(normalized_benchmark) == 4 and len(normalized_counting) == 8:
        return normalized_counting.endswith(normalized_benchmark)
    if len(normalized_benchmark) == 8 and len(normalized_counting) == 8:
        try:
            benchmark_date = datetime.strptime(normalized_benchmark, "%Y%m%d").date()
            counting_date = datetime.strptime(normalized_counting, "%Y%m%d").date()
        except ValueError:
            return False
        return abs((benchmark_date - counting_date).days) <= 1
    return False


def infer_date_match_status(
    benchmark_label: str,
    benchmark_release_date: str,
    counting_label: str,
) -> tuple[str, str, str]:
    benchmark_tag = extract_release_tag(benchmark_label) or extract_release_tag(benchmark_release_date)
    counting_tag = extract_release_tag(counting_label)
    if benchmark_tag and counting_tag:
        status = "exact" if release_tags_match(benchmark_tag, counting_tag) else "mismatch"
    elif benchmark_tag and not counting_tag:
        status = "benchmark_dated_counting_undated"
    elif counting_tag and not benchmark_tag:
        status = "counting_dated_benchmark_undated"
    else:
        status = "no_dates"
    return benchmark_tag, counting_tag, status


def parse_release_date(value: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    for pattern in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            pass
    return None


def load_run_manifest_metadata(run_path: Path) -> dict[str, dict[str, str]]:
    manifest_path = run_path.with_name("manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    models = payload.get("models")
    return models if isinstance(models, dict) else {}


def counting_alias_labels(model: str, manifest_entry: dict[str, object] | None) -> tuple[str, ...]:
    labels = [model]
    if "/" in model:
        labels.append(model.split("/", 1)[1])
    metadata = manifest_entry.get("model_metadata") if isinstance(manifest_entry, dict) else None
    if isinstance(metadata, dict):
        for key in ("canonical_slug", "hugging_face_id", "id", "name"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                labels.append(value.strip())
                if "/" in value:
                    labels.append(value.split("/", 1)[1].strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return tuple(deduped)


def build_counting_models() -> dict[str, CountingModel]:
    """Measured capacities, keyed by model.

    CC comes from scc.csv (the fitted transition); the registry supplies the
    metadata and the alias labels used to match benchmark tables, which name
    models inconsistently. A model measured both with and without reasoning
    contributes one entry, the configuration that counted furthest, because the
    benchmark tables report a single score per model.
    """
    fitted: dict[str, float] = {}
    with SCC_TABLE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = float(row["scc"])
            model = row["model"]
            if model not in fitted or value > fitted[model]:
                fitted[model] = value

    models: dict[str, CountingModel] = {}
    for row in load_registry(COUNTING_RUN):
        model = row["model"]
        if model not in fitted or model in models:
            continue
        canonical_slug = str(row.get("canonical_slug") or "").strip()
        aliases = [model]
        if "/" in model:
            aliases.append(model.split("/", 1)[1])
        for key in ("canonical_slug", "hugging_face_id", "display_name"):
            value = str(row.get(key) or "").strip()
            if value:
                aliases.append(value)
                if "/" in value:
                    aliases.append(value.split("/", 1)[1])
        models[model] = CountingModel(
            model=model,
            stable_count=fitted[model],
            provider=infer_provider_family(model),
            display_label=display_model_label(model),
            release_label=canonical_slug or model,
            alias_labels=tuple(dict.fromkeys(aliases)),
        )
    return models


def counting_aliases(counting_model: CountingModel) -> set[str]:
    aliases = {normalized_model_key(label) for label in counting_model.alias_labels}
    # The raw slug tail as well (gpt-4o-2024-08-06), so a dated leaderboard
    # label resolves to the dated checkpoint before date stripping collapses
    # every gpt-4o release onto one key.
    aliases.update(f"raw:{label.strip().lower()}" for label in counting_model.alias_labels)
    normalized = normalized_model_key(counting_model.model)
    if normalized == "kimi-k2":
        aliases.add("kimi-k2-instruct")
    if normalized == "kimi-k2-0905":
        aliases.add("kimi-k2-instruct-0905")
    if normalized == "kimi-k2-thinking":
        aliases.add("kimi-k2-thinking-turbo")
    if normalized == "kimi-k2.5":
        aliases.add("kimi-k2p5")
    if normalized == "glm-4.5":
        aliases.add("glm-4p5")
    if normalized == "glm-4.5-air":
        aliases.add("glm-4p5-air")
    return aliases


def build_alias_map(counting_models: dict[str, CountingModel]) -> dict[str, list[str]]:
    alias_map: dict[str, list[str]] = defaultdict(list)
    for model, record in sorted(counting_models.items()):
        for alias in counting_aliases(record):
            alias_map[alias].append(model)
    return alias_map


def choose_score_column(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    fieldnames = list(rows[0].keys())
    for preferred in SCORE_COLUMN_PREFERENCES:
        if preferred in fieldnames:
            return preferred

    candidates: list[tuple[int, str]] = []
    for column in fieldnames:
        if column in METADATA_COLUMNS or column in NON_SCORE_COLUMNS:
            continue
        numeric_count = 0
        for row in rows[:100]:
            value = (row.get(column) or "").strip()
            if not value:
                continue
            try:
                float(value)
            except ValueError:
                continue
            numeric_count += 1
        if numeric_count > 0:
            candidates.append((numeric_count, column))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def benchmark_display_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_external"):
        stem = stem[: -len("_external")]
    return BENCHMARK_TITLE_OVERRIDES.get(
        stem,
        stem.replace("_", " ").title(),
    )


def select_counting_model(
    aliases: list[tuple[str, str]],
    alias_map: dict[str, list[str]],
) -> tuple[str, str, str, str] | None:
    expanded: list[tuple[str, str]] = []
    for source_column, label in aliases:
        raw_matches = alias_map.get(f"raw:{label.strip().lower()}", [])
        if len(raw_matches) == 1:
            return raw_matches[0], label, source_column, normalized_model_key(label)
        expanded.append((source_column, label))
        stripped = re.sub(r"[-_]\(?20\d{2}-\d{2}-\d{2}\)?$|[-_]20\d{6}$", "", label)
        if stripped != label:
            expanded.append((source_column, stripped))
    for source_column, label in expanded:
        normalized_key = normalized_model_key(label)
        direct_matches = alias_map.get(normalized_key, [])
        if len(direct_matches) == 1:
            model = direct_matches[0]
            return model, label, source_column, normalized_key
        if len(direct_matches) > 1:
            thinking_matches = [item for item in direct_matches if "thinking" in item]
            plain_matches = [item for item in direct_matches if "thinking" not in item]
            if "thinking" in normalized_key and len(thinking_matches) == 1:
                return thinking_matches[0], label, source_column, normalized_key
            if "thinking" not in normalized_key and len(plain_matches) == 1:
                return plain_matches[0], label, source_column, normalized_key
    return None


def infer_reasoning_effort(*labels: str) -> str:
    combined = " ".join(label for label in labels if label).lower()
    if re.search(r"(?<![a-z])xhigh(?![a-z])", combined):
        return "xhigh"
    if re.search(r"(?<![a-z])high(?![a-z])", combined):
        return "high"
    if re.search(r"(?<![a-z])medium(?![a-z])", combined):
        return "medium"
    if re.search(r"(?<![a-z])low(?![a-z])", combined):
        return "low"
    if re.search(r"(^|[_\s(])max($|[_\s,)])", combined):
        return "max"
    if "thinking" in combined:
        return "other"
    if re.search(r"(^|[_\s\-(])\d+k($|[_\s,\-)])", combined):
        return "other"
    return "default"


def context_variant_rank(label: str) -> int:
    normalized = label.lower()
    if "_max" in normalized:
        return 4
    if re.search(r"_\d+k\b", normalized):
        return 3
    if "-thinking" in normalized or ":thinking" in normalized:
        return 2
    return 0


def record_preference_key(record: BenchmarkRecord) -> tuple[int, int, int, int, float]:
    effort_rank = {"medium": 0, "default": 1}.get(record.reasoning_effort, 9)
    source_rank = 0 if record.matched_input_column == "Model version" else 1
    model_version_rank = context_variant_rank(record.benchmark_model_version)
    matched_input_rank = context_variant_rank(record.matched_input)
    # Prefer the best-scoring retained row when everything else ties.
    return (
        effort_rank,
        source_rank,
        model_version_rank,
        matched_input_rank,
        -record.benchmark_score,
    )


def keep_reasoning_effort(benchmark_file: str, reasoning_effort: str) -> bool:
    # "max" configurations are excluded everywhere: they are marketed showcase
    # settings with no counterpart in our measurement configuration.
    if reasoning_effort in {"medium", "default"}:
        return True
    if benchmark_file == "arc_agi_2_external.csv" and reasoning_effort in {"high", "other"}:
        return True
    # METR lists one row per served configuration (a 16K thinking budget, or an
    # effort level). Keep them all except "max": the counting side already uses
    # the larger of a model's two configurations, and dedupe prefers the medium
    # or default row when the leaderboard lists several for one model.
    if benchmark_file == "metr_time_horizons_external.csv" and reasoning_effort in {"high", "xhigh", "other", "low"}:
        return True
    return False


def deduplicate_records(records: list[BenchmarkRecord]) -> list[BenchmarkRecord]:
    selected: dict[tuple[str, str], BenchmarkRecord] = {}
    for record in records:
        key = (record.benchmark_file, record.counting_model)
        incumbent = selected.get(key)
        if incumbent is None or record_preference_key(record) < record_preference_key(incumbent):
            selected[key] = record
    return sorted(
        selected.values(),
        key=lambda item: (item.benchmark_name.lower(), item.counting_model.lower(), item.row_index),
    )


def rankdata_average(values: list[float]) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    order = np.argsort(values)
    sorted_values = np.array(values, dtype=float)[order]
    ranks = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_ranks = rankdata_average(xs)
    y_ranks = rankdata_average(ys)
    return pearson_correlation(list(x_ranks), list(y_ranks))


def load_benchmark_dataset(
    path: Path,
    counting_models: dict[str, CountingModel],
    alias_map: dict[str, list[str]],
) -> BenchmarkDataset | None:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    score_column = choose_score_column(rows)
    if not score_column:
        return None

    matched_records: list[BenchmarkRecord] = []
    unique_counting_models: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        score_raw = (row.get(score_column) or "").strip()
        if not score_raw:
            continue
        try:
            score = float(score_raw)
        except ValueError:
            continue

        candidate_aliases: list[tuple[str, str]] = []
        model_version = (row.get("Model version") or "").strip()
        if model_version:
            candidate_aliases.append(("Model version", model_version))
        name_fallback = (row.get("Name") or "").strip()
        if name_fallback:
            candidate_aliases.append(("Name", name_fallback))
        release_date = (row.get("Release date") or "").strip()

        selection = select_counting_model(candidate_aliases, alias_map)
        if selection is None:
            continue
        counting_model_name, matched_input, matched_input_column, normalized_key = selection
        reasoning_effort = infer_reasoning_effort(model_version, name_fallback)
        if not keep_reasoning_effort(path.name, reasoning_effort):
            continue
        counting_model = counting_models[counting_model_name]
        parsed_release_date = parse_release_date(release_date)
        benchmark_release_tag, counting_release_tag, date_match_status = infer_date_match_status(
            model_version or name_fallback,
            release_date,
            counting_model.release_label,
        )
        matched_records.append(
            BenchmarkRecord(
                benchmark_file=path.name,
                benchmark_name=benchmark_display_name(path),
                score_column=score_column,
                row_index=row_index,
                benchmark_model_version=model_version,
                benchmark_name_fallback=name_fallback,
                matched_input=matched_input,
                matched_input_column=matched_input_column,
                normalized_key=normalized_key,
                reasoning_effort=reasoning_effort,
                benchmark_release_date=parsed_release_date,
                benchmark_release_tag=benchmark_release_tag,
                counting_release_tag=counting_release_tag,
                date_match_status=date_match_status,
                counting_model=counting_model_name,
                counting_model_label=counting_model.display_label,
                provider=counting_model.provider,
                benchmark_score=score,
                stable_count=counting_model.stable_count,
            )
        )

    matched_records = deduplicate_records(matched_records)
    unique_counting_models = {record.counting_model for record in matched_records}

    plotted_records = [record for record in matched_records if record.stable_count > 0]
    zero_scc_count = len(matched_records) - len(plotted_records)
    score_values = [record.benchmark_score for record in plotted_records]
    log_scc_values = [math.log10(record.stable_count) for record in plotted_records]

    return BenchmarkDataset(
        path=path,
        display_name=benchmark_display_name(path),
        score_column=score_column,
        matched_records=matched_records,
        matched_model_count=len(unique_counting_models),
        plotted_point_count=len(plotted_records),
        zero_scc_count=zero_scc_count,
        spearman_rho=spearman_correlation(score_values, log_scc_values),
        pearson_log_scc=pearson_correlation(score_values, log_scc_values),
    )


def benchmark_rows_for_plot(dataset: BenchmarkDataset) -> list[BenchmarkRecord]:
    return [record for record in dataset.matched_records if record.stable_count > 0]


def x_limits(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        pad = max(0.05, abs(low) * 0.05)
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def y_limits(values: list[float]) -> tuple[float, float]:
    if not values:
        return 1.0, 10.0
    low = min(values)
    high = max(values)
    return max(1.0, low / 1.25), high * 1.25


LOG_X_BENCHMARKS = {"metr_time_horizons_external.csv": "Time horizon (min)"}


def fit_log_trendline(records: list[BenchmarkRecord], log_x: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    if len(records) < 2:
        return None
    raw = np.array([record.benchmark_score for record in records], dtype=float)
    if log_x:
        keep = raw > 0
        raw = raw[keep]
        ys = np.array([math.log10(record.stable_count) for record, k in zip(records, keep) if k], dtype=float)
        xs = np.log10(raw)
    else:
        xs = raw
        ys = np.array([math.log10(record.stable_count) for record in records], dtype=float)
    if len(xs) < 2 or np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    x_fit = np.linspace(xs.min(), xs.max(), 200)
    y_line = np.power(10.0, intercept + slope * x_fit)
    x_line = np.power(10.0, x_fit) if log_x else x_fit
    return x_line, y_line


def log_trendline_stats(records: list[BenchmarkRecord]) -> tuple[float | None, float | None]:
    if len(records) < 2:
        return None, None
    xs = [record.benchmark_score for record in records]
    ys = [math.log10(record.stable_count) for record in records]
    return spearman_correlation(xs, ys), pearson_correlation(xs, ys)


def is_after_arc_agi2_buffer(record: BenchmarkRecord) -> bool:
    return (
        record.benchmark_release_date is not None
        and record.benchmark_release_date > ARC_AGI_2_POST_TRAINING_BUFFER_DATE
    )


def is_labeled_post_arc_agi2_model(record: BenchmarkRecord) -> bool:
    return record.counting_model_label in ARC_AGI_2_POST_RELEASE_MODELS


def is_post_arc_agi2_generation(record: BenchmarkRecord) -> bool:
    if record.counting_model in ARC_AGI_2_POST_RELEASE_EXCLUDED:
        return False
    return is_after_arc_agi2_buffer(record) or is_labeled_post_arc_agi2_model(record)


def draw_benchmark_scatter(
    ax: plt.Axes,
    dataset: BenchmarkDataset,
    *,
    show_axis_labels: bool,
    title_fontsize: float | None = None,
    label_fontsize: float | None = None,
    tick_fontsize: float | None = None,
    stats_fontsize: float = 8.0,
    point_size: float = 32.0,
    point_linewidth: float = 0.7,
    point_alpha: float = 0.9,
    use_house_spines: bool = False,
    highlight_post_arc_agi2: bool = False,
    split_post_arc_agi2_trendlines: bool = False,
    legend_fontsize: float = 8.0,
    title_override: str | None = None,
) -> None:
    records = benchmark_rows_for_plot(dataset)
    if not records:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No positive-CC overlap",
            ha="center",
            va="center",
            fontsize=stats_fontsize,
            color="#555555",
            transform=ax.transAxes,
        )
        return

    x_values = [record.benchmark_score for record in records]
    y_values = [record.stable_count for record in records]
    for record in records:
        is_post_arc_agi2 = highlight_post_arc_agi2 and is_post_arc_agi2_generation(record)
        ax.scatter(
            record.benchmark_score,
            record.stable_count,
            s=point_size * (0.8 if is_post_arc_agi2 else 1.0),
            color=PROVIDER_COLORS[record.provider],
            marker=PROVIDER_MARKERS[record.provider],
            alpha=point_alpha,
            edgecolors=POST_ARC_AGI_2_HIGHLIGHT if is_post_arc_agi2 else "white",
            linewidths=point_linewidth * (2.6 if is_post_arc_agi2 else 1.0),
            zorder=4 if is_post_arc_agi2 else 3,
        )

    split_trends = split_post_arc_agi2_trendlines and highlight_post_arc_agi2
    split_stats: dict[str, tuple[float | None, float | None]] = {}
    if split_trends:
        trendline_groups = (
            (
                "Older fit",
                [record for record in records if not is_post_arc_agi2_generation(record)],
                "#666666",
                "--",
            ),
            (
                "Newer fit",
                [record for record in records if is_post_arc_agi2_generation(record)],
                POST_ARC_AGI_2_HIGHLIGHT,
                "-",
            ),
        )
        for label, group_records, color, linestyle in trendline_groups:
            trendline = fit_log_trendline(group_records)
            if trendline is None:
                continue
            split_stats[label] = log_trendline_stats(group_records)
            x_line, y_line = trendline
            is_newer_fit = label == "Newer fit"
            ax.plot(
                x_line,
                y_line,
                color=color,
                linewidth=1.8 if is_newer_fit else 1.45,
                linestyle=linestyle,
                alpha=0.95,
                zorder=5 if is_newer_fit else 4.5,
            )
    else:
        log_x = dataset.path.name in LOG_X_BENCHMARKS
        trendline = fit_log_trendline(records, log_x=log_x)
        if trendline is not None:
            x_line, y_line = trendline
            ax.plot(
                x_line,
                y_line,
                color="#666666",
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
                zorder=2,
            )

    ax.set_yscale("log")
    # Plain numbers on the CC axis. The default log formatter writes 6 x 10^1
    # and 2 x 10^2, which is unreadable at panel size and, on an axis whose
    # values are item counts, says nothing a bare integer does not.
    lo, hi = y_limits(y_values)
    ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    if dataset.path.name in LOG_X_BENCHMARKS:
        ax.set_xscale("log")
        pos = [v for v in x_values if v > 0]
        ax.set_xlim(min(pos) / 1.6, max(pos) * 1.6)
        xt = [t for t in (1, 10, 100, 1000) if min(pos) / 1.6 <= t <= max(pos) * 1.6]
        ax.set_xticks(xt)
        ax.set_xticklabels([str(t) for t in xt])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.set_xlim(*x_limits(x_values))
    if split_trends:
        # The two stats boxes sit in the top corners, which is exactly where the
        # highest-capacity models plot. Open headroom above the data so the
        # boxes annotate empty space instead of covering points.
        hi = hi * 2.6
    ax.set_ylim(lo, hi)
    ax.set_title(title_override or dataset.display_name, fontsize=title_fontsize)
    if show_axis_labels:
        ax.set_xlabel(LOG_X_BENCHMARKS.get(dataset.path.name, "Mean score"), fontsize=label_fontsize)
        ax.set_ylabel("CC", fontsize=label_fontsize)
    ax.grid(alpha=0.18 if use_house_spines else 0.2, which="both", zorder=0)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    if use_house_spines:
        set_spine_width(ax)

    if highlight_post_arc_agi2 and any(is_post_arc_agi2_generation(record) for record in records):
        handle = ax.scatter(
            [],
            [],
            s=point_size * 1.35,
            facecolors="white",
            edgecolors=POST_ARC_AGI_2_HIGHLIGHT,
            linewidths=point_linewidth * 2.6,
        )
        handles: list[object] = [handle]
        labels = ["Model trained after release"]
        if split_trends:
            older_line = plt.Line2D([], [], color="#666666", linewidth=1.2, linestyle="--")
            newer_line = plt.Line2D([], [], color=POST_ARC_AGI_2_HIGHLIGHT, linewidth=1.2, linestyle="-")
            handles.extend([older_line, newer_line])
            labels.extend(["Older log-CC fit", "Newer log-CC fit"])
        legend = ax.legend(
            handles,
            labels,
            loc="lower right",
            frameon=True,
            fontsize=legend_fontsize,
            borderpad=0.35,
            handletextpad=0.45,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#cfcfcf")
        legend.get_frame().set_alpha(0.92)

    if split_trends and split_stats:
        if "Older fit" in split_stats:
            older_rho, older_r = split_stats["Older fit"]
            ax.text(
                0.03,
                0.97,
                # No "older"/"newer" caption: the colour already carries it, and
                # the words made the box wider than the space it has to sit in.
                "\n".join(
                    line
                    for line in (
                        f"rho={older_rho:.2f}" if older_rho is not None else "",
                        f"r(log CC)={older_r:.2f}" if older_r is not None else "",
                    )
                    if line
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=stats_fontsize,
                color="#666666",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5),
                zorder=8,
            )
        if "Newer fit" in split_stats:
            newer_rho, newer_r = split_stats["Newer fit"]
            ax.text(
                0.03,
                0.79,
                # No "older"/"newer" caption: the colour already carries it, and
                # the words made the box wider than the space it has to sit in.
                "\n".join(
                    line
                    for line in (
                        f"rho={newer_rho:.2f}" if newer_rho is not None else "",
                        f"r(log CC)={newer_r:.2f}" if newer_r is not None else "",
                    )
                    if line
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=stats_fontsize,
                color=POST_ARC_AGI_2_HIGHLIGHT,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5),
                zorder=8,
            )
    else:
        stats_lines: list[str] = []
        if dataset.spearman_rho is not None:
            stats_lines.append(f"rho={dataset.spearman_rho:.2f} (n={dataset.matched_model_count})")
        if dataset.pearson_log_scc is not None:
            if dataset.path.name in LOG_X_BENCHMARKS:
                pos = [r for r in records if r.benchmark_score > 0]
                r_ll = pearson_correlation([math.log10(r.benchmark_score) for r in pos], [math.log10(r.stable_count) for r in pos])
                if r_ll is not None:
                    stats_lines.append(f"r(log-log)={r_ll:.2f}")
            else:
                stats_lines.append(f"r(log CC)={dataset.pearson_log_scc:.2f}")
        if stats_lines:
            ax.text(
                0.03,
                0.97,
                "\n".join(stats_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=stats_fontsize,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5),
                zorder=8,
            )


def write_mapping_table(datasets: list[BenchmarkDataset]) -> None:
    mapping_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for dataset in datasets:
        for record in dataset.matched_records:
            key = (record.benchmark_file, record.matched_input, record.counting_model)
            if key in seen:
                continue
            seen.add(key)
            mapping_rows.append(
                {
                    "benchmark_file": record.benchmark_file,
                    "benchmark_name": record.benchmark_name,
                    "matched_input_column": record.matched_input_column,
                    "benchmark_input": record.matched_input,
                    "normalized_key": record.normalized_key,
                    "reasoning_effort": record.reasoning_effort,
                    "benchmark_release_date": (
                        record.benchmark_release_date.isoformat() if record.benchmark_release_date else ""
                    ),
                    "benchmark_release_tag": record.benchmark_release_tag,
                    "counting_release_tag": record.counting_release_tag,
                    "date_match_status": record.date_match_status,
                    "counting_model": record.counting_model,
                    "counting_model_label": record.counting_model_label,
                    "provider": record.provider,
                }
            )

    with MAPPING_TABLE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark_file",
                "benchmark_name",
                "matched_input_column",
                "benchmark_input",
                "normalized_key",
                "reasoning_effort",
                "benchmark_release_date",
                "benchmark_release_tag",
                "counting_release_tag",
                "date_match_status",
                "counting_model",
                "counting_model_label",
                "provider",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(mapping_rows, key=lambda row: (row["benchmark_file"], row["benchmark_input"])))


def write_merged_table(datasets: list[BenchmarkDataset]) -> None:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for record in dataset.matched_records:
            rows.append(
                {
                    "benchmark_file": record.benchmark_file,
                    "benchmark_name": record.benchmark_name,
                    "score_column": record.score_column,
                    "row_index": record.row_index,
                    "benchmark_model_version": record.benchmark_model_version,
                    "benchmark_name_fallback": record.benchmark_name_fallback,
                    "matched_input": record.matched_input,
                    "matched_input_column": record.matched_input_column,
                    "normalized_key": record.normalized_key,
                    "reasoning_effort": record.reasoning_effort,
                    "benchmark_release_date": (
                        record.benchmark_release_date.isoformat() if record.benchmark_release_date else ""
                    ),
                    "benchmark_release_tag": record.benchmark_release_tag,
                    "counting_release_tag": record.counting_release_tag,
                    "date_match_status": record.date_match_status,
                    "counting_model": record.counting_model,
                    "counting_model_label": record.counting_model_label,
                    "provider": record.provider,
                    "benchmark_score": f"{record.benchmark_score:.12g}",
                    "stable_count": f"{record.stable_count:.12g}",
                    "log10_stable_count": (
                        f"{math.log10(record.stable_count):.12g}" if record.stable_count > 0 else ""
                    ),
                }
            )

    with MERGED_TABLE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_summary(
    datasets: list[BenchmarkDataset],
    pairwise_rows: list[dict[str, object]],
) -> dict[str, object]:
    scc_rhos = [
        dataset.spearman_rho
        for dataset in datasets
        if dataset.spearman_rho is not None
    ]
    pairwise_rhos = [
        row["spearman_rho"]
        for row in pairwise_rows
        if row["spearman_rho"] is not None
    ]
    payload = {
        "authoritative_benchmarks": list(AUTHORITATIVE_BENCHMARKS),
        "date_match_status_counts": {
            status: sum(
                1
                for dataset in datasets
                for record in dataset.matched_records
                if record.date_match_status == status
            )
            for status in (
                "exact",
                "benchmark_dated_counting_undated",
                "counting_dated_benchmark_undated",
                "no_dates",
                "mismatch",
            )
        },
        "scc_vs_benchmark_spearman_summary": summarize_distribution(scc_rhos),
        "benchmark_vs_benchmark_spearman_summary": summarize_distribution(pairwise_rhos),
        "scc_vs_benchmark_percentile_within_pairwise": (
            percentile_rank(pairwise_rhos, float(np.median(scc_rhos)))
            if scc_rhos and pairwise_rhos
            else None
        ),
        "benchmarks": [
            {
                "benchmark_file": dataset.path.name,
                "benchmark_name": dataset.display_name,
                "score_column": dataset.score_column,
                "matched_rows": len(dataset.matched_records),
                "matched_models": dataset.matched_model_count,
                "plotted_rows": dataset.plotted_point_count,
                "zero_scc_rows": dataset.zero_scc_count,
                "spearman_rho": dataset.spearman_rho,
                "pearson_log_scc": dataset.pearson_log_scc,
            }
            for dataset in datasets
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize_distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    sorted_values = sorted(values)
    return {
        "count": float(len(sorted_values)),
        "min": float(sorted_values[0]),
        "p25": float(np.quantile(sorted_values, 0.25)),
        "median": float(np.quantile(sorted_values, 0.5)),
        "p75": float(np.quantile(sorted_values, 0.75)),
        "max": float(sorted_values[-1]),
        "mean": float(np.mean(sorted_values)),
    }


def percentile_rank(reference_values: list[float], value: float) -> float | None:
    if not reference_values:
        return None
    count = sum(1 for item in reference_values if item <= value)
    return 100.0 * count / len(reference_values)


def dataset_score_map(dataset: BenchmarkDataset) -> dict[str, float]:
    return {
        record.counting_model: record.benchmark_score
        for record in dataset.matched_records
    }


def compute_pairwise_rho_rows(datasets: list[BenchmarkDataset]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    score_maps = {dataset.path.name: dataset_score_map(dataset) for dataset in datasets}
    for index, left in enumerate(datasets):
        left_scores = score_maps[left.path.name]
        for right in datasets[index + 1 :]:
            right_scores = score_maps[right.path.name]
            overlap_models = sorted(set(left_scores) & set(right_scores))
            xs = [left_scores[model] for model in overlap_models]
            ys = [right_scores[model] for model in overlap_models]
            rows.append(
                {
                    "left_file": left.path.name,
                    "left_name": left.display_name,
                    "right_file": right.path.name,
                    "right_name": right.display_name,
                    "overlap_models": len(overlap_models),
                    "spearman_rho": spearman_correlation(xs, ys) if len(overlap_models) >= 2 else None,
                }
            )
    rows.sort(
        key=lambda row: (
            -(row["spearman_rho"] if row["spearman_rho"] is not None else -999.0),
            -int(row["overlap_models"]),
            str(row["left_name"]),
            str(row["right_name"]),
        )
    )
    return rows


def write_pairwise_rho_table(rows: list[dict[str, object]]) -> None:
    with PAIRWISE_RHO_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "left_file",
                "left_name",
                "right_file",
                "right_name",
                "overlap_models",
                "spearman_rho",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def diagnosis_lines(
    datasets: list[BenchmarkDataset],
    pairwise_rows: list[dict[str, object]],
    summary_payload: dict[str, object],
) -> list[str]:
    lines: list[str] = ["Diagnosis:"]

    scc_summary = summary_payload.get("scc_vs_benchmark_spearman_summary")
    pairwise_summary = summary_payload.get("benchmark_vs_benchmark_spearman_summary")
    if isinstance(scc_summary, dict) and isinstance(pairwise_summary, dict):
        lines.append(
            "  "
            f"all overlaps: median rho(CC, benchmark)={scc_summary['median']:.3f} "
            f"across {int(scc_summary['count'])} benchmarks; "
            f"median rho(benchmark, benchmark)={pairwise_summary['median']:.3f} "
            f"across {int(pairwise_summary['count'])} pairs"
        )

    scc_overlap_filtered = [
        dataset
        for dataset in datasets
        if dataset.spearman_rho is not None and dataset.matched_model_count >= 10
    ]
    pairwise_overlap_filtered = [
        row
        for row in pairwise_rows
        if row["spearman_rho"] is not None and int(row["overlap_models"]) >= 10
    ]
    if scc_overlap_filtered and pairwise_overlap_filtered:
        scc_values = [float(dataset.spearman_rho) for dataset in scc_overlap_filtered]
        pairwise_values = [float(row["spearman_rho"]) for row in pairwise_overlap_filtered]
        scc_median = float(np.median(scc_values))
        pairwise_median = float(np.median(pairwise_values))
        percentile = percentile_rank(pairwise_values, scc_median)
        lines.append(
            "  "
            f"overlap>=10: median rho(CC, benchmark)={scc_median:.3f} "
            f"across {len(scc_values)} benchmarks; "
            f"median rho(benchmark, benchmark)={pairwise_median:.3f} "
            f"across {len(pairwise_values)} pairs; "
            f"CC median sits at the {percentile:.1f} percentile of benchmark-benchmark rho"
        )

    if AUTHORITATIVE_BENCHMARKS:
        lines.append("  authoritative panels:")
        for benchmark_file in AUTHORITATIVE_BENCHMARKS:
            dataset = next((item for item in datasets if item.path.name == benchmark_file), None)
            if dataset is None or dataset.spearman_rho is None:
                continue
            related_pairs = [
                row
                for row in pairwise_rows
                if row["spearman_rho"] is not None
                and int(row["overlap_models"]) >= 10
                and (
                    row["left_file"] == benchmark_file or row["right_file"] == benchmark_file
                )
            ]
            pairwise_median = (
                float(np.median([float(row["spearman_rho"]) for row in related_pairs]))
                if related_pairs
                else None
            )
            if pairwise_median is None:
                lines.append(
                    "    "
                    f"{dataset.display_name}: rho(CC)={dataset.spearman_rho:.3f}, "
                    f"matched_models={dataset.matched_model_count}"
                )
            else:
                lines.append(
                    "    "
                    f"{dataset.display_name}: rho(CC)={dataset.spearman_rho:.3f}, "
                    f"benchmark-pair median rho={pairwise_median:.3f}, "
                    f"matched_models={dataset.matched_model_count}"
                )

    top_pairs = [
        row
        for row in pairwise_rows
        if row["spearman_rho"] is not None and int(row["overlap_models"]) >= 10
    ][:5]
    if top_pairs:
        lines.append("  strongest benchmark-benchmark pairs (overlap>=10):")
        for row in top_pairs:
            lines.append(
                "    "
                f"{row['left_name']} <-> {row['right_name']}: "
                f"rho={float(row['spearman_rho']):.3f}, overlap={int(row['overlap_models'])}"
            )

    return lines


LADDER_FITS_CSV = ROOT / "results" / "ladder-full" / "fits.csv"
LADDER_CENSORED_ANALYSIS_JSON = (
    ROOT / "results" / "ladder-full" / "censored_analysis.json"
)
RUNG_COLORS = {"r2": "#5f84ad", "r3": "#a47f6b", "r4": "#e07b39"}
LADDER_PROTOCOL_EXCLUDED_MODELS = {"claude-fable-5"}


def draw_ratio_law(ax) -> None:
    # Proportional-scaling test for the counting-to-ARC ladder: measured
    # capacity on each ladder task against CC scaled by the task's fitted
    # factor (the censored regression of Appendix H).  The identity line is
    # proportional scaling.  Circles are resolved boundaries, down and up
    # triangles are upper and lower bounds, and vertical bars are the tested
    # ranges of the mixed scans.  Colours run from the counting blue to ARC's
    # orange with the level of the task.
    import csv as _csv
    import json as _json
    import math as _math

    rows = [
        r for r in _csv.DictReader(LADDER_FITS_CSV.open())
        if r["predicted"]
        and r.get("censoring") in {"exact", "left", "right", "interval"}
        and r["model"] not in LADDER_PROTOCOL_EXCLUDED_MODELS
        and r.get("analysis_included", "True").lower() != "false"
    ]
    plot_ys = []
    for r in rows:
        x = float(r["predicted"])
        rung = r["scan"][:2]
        color = RUNG_COLORS[rung]
        censoring = r["censoring"]
        if censoring == "exact":
            y = float(r["boundary"])
            ax.scatter([x], [y], s=28.0, marker="o", color=color,
                       edgecolors="white", linewidths=0.55, alpha=0.86, zorder=4)
            plot_ys.append(y)
        elif censoring == "left":
            y = float(r["bound_high"])
            ax.scatter([x], [y], s=27.0, marker="v", color=color,
                       edgecolors="white", linewidths=0.45, alpha=0.78, zorder=3)
            plot_ys.append(y)
        elif censoring == "right":
            y = float(r["bound_low"])
            ax.scatter([x], [y], s=27.0, marker="^", color=color,
                       edgecolors="white", linewidths=0.45, alpha=0.78, zorder=3)
            plot_ys.append(y)
        else:
            low, high = float(r["bound_low"]), float(r["bound_high"])
            mid = _math.sqrt(low * high)
            ax.plot([x, x], [low, high], color=color, lw=1.0, alpha=0.58, zorder=2)
            ax.scatter([x], [mid], s=18.0, marker="x", color=color,
                       linewidths=0.8, alpha=0.72, zorder=3)
            plot_ys.extend((low, high))
    xs = [float(r["predicted"]) for r in rows]
    lo = min(min(xs), min(plot_ys)) * 0.75
    hi = max(max(xs), max(plot_ys)) * 1.3
    ax.plot([lo, hi], [lo, hi], color="#666666", lw=1.4, ls="--", zorder=2)
    analysis = _json.loads(LADDER_CENSORED_ANALYSIS_JSON.read_text())
    resolved = analysis["resolved_only"]
    imputed = analysis["censored_multiple_imputation"]
    rho = float(resolved["spearman_rho"])
    all_rho = float(imputed["spearman_rho_median"])
    print(f"ladder panel: resolved n={resolved['n']} rho={rho:.2f}; bound-aware rho={all_rho:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000) if lo <= t <= hi]
    for setter, labeler, axis in ((ax.set_xticks, ax.set_xticklabels, ax.xaxis),
                                  (ax.set_yticks, ax.set_yticklabels, ax.yaxis)):
        setter(ticks)
        labeler([str(t) for t in ticks])
        axis.set_minor_formatter(ticker.NullFormatter())
    ax.text(0.03, 0.97,
            f"resolved rho={rho:.2f}\nbound-aware rho={all_rho:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5))
    labels = {"r2": "count cells", "r3": "row rule", "r4": "grid rule"}
    handles = [plt.Line2D([], [], marker="o", ls="", color=RUNG_COLORS[k], markersize=6,
                          label=labels[k]) for k in ("r2", "r3", "r4")]
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="lower right",
              handlelength=1.0, borderaxespad=0.2, handletextpad=0.4)
    title_fs = AUTHORITATIVE_TITLE_FS if "AUTHORITATIVE_TITLE_FS" in globals() else 15
    label_fs = AUTHORITATIVE_LABEL_FS if "AUTHORITATIVE_LABEL_FS" in globals() else 12.5
    ax.set_title("Ladder Scaling with CC", fontsize=title_fs)
    ax.set_xlabel("CC × task factor", fontsize=label_fs)
    ax.set_ylabel("Ladder task capacity", fontsize=label_fs)
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.grid(alpha=0.2, linewidth=0.9)
    ax.set_box_aspect(1)
    set_spine_width(ax)

def save_authoritative_figure(datasets_by_name: dict[str, BenchmarkDataset]) -> None:
    # Two rows of four square panels at the standard row width. Top row: the
    # axes CC is NOT reducible to (knowledge, coding, scale). Bottom row: the
    # quantities it is intrinsically tied to (task horizon, fluid intelligence
    # with the familiarity split, its own execution ladder, and market price).
    counting_models = build_counting_models()
    plan = [
        ("dataset", "hle_external.csv", "HLE"),
        ("dataset", "gpqa_diamond.csv", "GPQA Diamond"),
        ("dataset", "swe_bench_verified.csv", "SWE-bench Verified"),
        ("params", None, None),
        ("dataset", "metr_time_horizons_external.csv", "METR Time Horizons"),
        ("dataset", "arc_agi_2_external.csv", "ARC-AGI-2"),
        ("ratio", None, None),
        ("tracking", None, None),
    ]
    panel_h = 2.30
    bottom = 0.54
    row_gap = 1.02
    top_pad = 0.45
    figure_height_in = bottom + 2 * panel_h + row_gap + top_pad
    fig = plt.figure(figsize=(ROW_FIGURE_WIDTH_IN, figure_height_in))
    rows = [
        add_manual_row_axes(
            fig, panel_count=4,
            figure_width_in=ROW_FIGURE_WIDTH_IN, figure_height_in=figure_height_in,
            plot_left_in=0.72,
            plot_bottom_in=bottom + (panel_h + row_gap if row == 0 else 0),
            plot_width_in=11.43, plot_height_in=panel_h,
            panel_gap_in=0.72,
        )
        for row in (0, 1)
    ]
    axes = rows[0] + rows[1]
    for ax, (kind, name, title) in zip(axes, plan):
        if kind == "dataset":
            dataset = datasets_by_name.get(name)
            if dataset is None:
                ax.axis("off")
                continue
            draw_benchmark_scatter(
                ax, dataset, show_axis_labels=True, title_fontsize=16,
                label_fontsize=14, tick_fontsize=12, stats_fontsize=10,
                point_size=36.0, point_linewidth=0.7, point_alpha=0.9,
                use_house_spines=True,
                highlight_post_arc_agi2=name == "arc_agi_2_external.csv",
                split_post_arc_agi2_trendlines=name == "arc_agi_2_external.csv",
                title_override=title,
            )
        elif kind == "params":
            draw_params_scatter(ax, counting_models)
        elif kind == "ratio":
            draw_ratio_law(ax)
        elif kind == "price":
            draw_price_scatter(ax, counting_models)
        else:
            draw_tracking_scatter(ax)
        ax.set_box_aspect(1)

    ARTICLE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(AUTHORITATIVE_FIGURE_PDF, dpi=200)
    fig.savefig(AUTHORITATIVE_FIGURE_PNG, dpi=200)
    fig.savefig(AUTHORITATIVE_FIGURE_SVG)
    plt.close(fig)


def draw_tracking_scatter(ax) -> None:
    """State-tracking capacity (Appendix K) against CC, one point per configuration:
    filled markers reasoning off, open markers reasoning on, provider colours and
    markers, log-log, house spines. Reads the scan summaries and the catalogue."""
    import json as _json, glob as _glob, csv as _csv
    track = ROOT / "results" / "tracking"
    stats = _json.loads((ROOT / "paper" / "generated" / "tracking_stats.json").read_text())
    runs = {}
    for f in _glob.glob(str(track / "*.summary.json")):
        s = _json.loads(Path(f).read_text())
        if "failed" in s:
            continue
        rows = [_json.loads(l) for l in Path(f.replace(".summary.json", ".jsonl")).read_text().splitlines() if l.strip()]
        rt = [r.get("reasoning_tokens") or 0 for r in rows if r["kind"] == "scan" and r["status"] != "error"]
        s["on"] = (sum(x > 0 for x in rt) / len(rt)) > 0.5 if rt else False
        runs[s["key"]] = s
    cc = {}
    for r in _csv.DictReader((ROOT / "results" / "counting-data" / "scc.csv").open()):
        try:
            v = float(r["scc"])
        except ValueError:
            continue
        cc[r["model"] + (f"#{r['variant']}" if r["variant"] else "")] = v
    vendor = {r["counting_model"]: r["provider"] for r in _csv.DictReader((ROOT / "results" / "benchmark_data" / "counting_benchmark_merged.csv").open())}
    for key, s in runs.items():
        if key not in cc:
            continue
        tc = 1 if s["sufficiency"] == "never_stable" else max(1, s["tc"])
        model = key.split("#")[0]
        prov = vendor.get(model, model.split("/")[0])
        col = PROVIDER_COLORS.get(prov, "#888888"); mk = PROVIDER_MARKERS.get(prov, "o")
        if s["on"]:
            ax.scatter([tc], [cc[key]], s=36.0, marker=mk, facecolors="white", edgecolors=col, linewidths=1.3, alpha=0.95, zorder=3)
        else:
            ax.scatter([tc], [cc[key]], s=36.0, marker=mk, color=col, edgecolors="white", linewidths=0.7, alpha=0.9, zorder=4)
    # dashed log-log fit through the reasoning-off points, the coupled set
    off_pts = [(1 if s["sufficiency"] == "never_stable" else max(1, s["tc"]), cc[k]) for k, s in runs.items() if k in cc and not s["on"]]
    if len(off_pts) >= 3:
        xs = np.log10([x for x, _ in off_pts]); ys = np.log10([y for _, y in off_pts])
        slope, intercept = np.polyfit(xs, ys, 1)
        x_fit = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(np.power(10.0, x_fit), np.power(10.0, intercept + slope * x_fit), color="#666666", linewidth=1.2, linestyle="--", alpha=0.9, zorder=2)
    p1 = stats["P1"]; on = p1.get("reasoning_on", {})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.7, 2000); ax.set_ylim(8, 3000)
    for axis, ticks in ((ax.xaxis, [1, 10, 100, 1000]), (ax.yaxis, [10, 20, 50, 100, 200, 500])):
        axis.set_major_locator(ticker.FixedLocator(ticks))
        axis.set_major_formatter(ticker.FixedFormatter([str(t) for t in ticks]))
        axis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Tracking capacity (swaps)", fontsize=14)
    ax.set_ylabel("CC", fontsize=14)
    ax.set_title("State Tracking", fontsize=16)
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.grid(alpha=0.18, which="both", zorder=0)
    set_spine_width(ax)
    ax.text(0.03, 0.97, f"reasoning off: rho={p1['rho']:.2f} (n={p1['n']})\nreasoning on: rho={on.get('rho', float('nan')):.2f} (n={on.get('n', 0)})",
            transform=ax.transAxes, fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cfcfcf", alpha=0.92))
    h_off = ax.scatter([], [], s=36.0, color="#666666", edgecolors="white")
    h_on = ax.scatter([], [], s=36.0, facecolors="white", edgecolors="#666666", linewidths=1.3)
    legend = ax.legend([h_off, h_on], ["reasoning off", "reasoning on"], loc="lower right", fontsize=9, frameon=True, borderpad=0.35, handletextpad=0.45)
    legend.get_frame().set_facecolor("white"); legend.get_frame().set_edgecolor("#cfcfcf"); legend.get_frame().set_alpha(0.92)


def draw_price_scatter(ax, counting_models) -> None:
    # Serving price against CC in the same visual grammar as the benchmark
    # panels: provider colours and markers, log CC on y, stats box, house
    # spines. The x axis is price, which unlike a score is logarithmic.
    import json as _json
    import math as _math
    import statistics as _stats

    live = _json.loads(PRICE_SNAPSHOT.read_text())
    best = {}
    for cm in counting_models.values():
        if cm.model not in best or cm.stable_count > best[cm.model].stable_count:
            best[cm.model] = cm
    xs, ys, colours, marks = [], [], [], []
    for cm in best.values():
        price = live.get(cm.model, (0, 0))[1]
        if price <= 0 or cm.stable_count <= 0:
            continue
        xs.append(price)
        ys.append(cm.stable_count)
        colours.append(PROVIDER_COLORS.get(cm.provider, "#888888"))
        marks.append(PROVIDER_MARKERS.get(cm.provider, "o"))
    for x, y, c, m in zip(xs, ys, colours, marks):
        ax.scatter([x], [y], s=36.0, marker=m, color=c, edgecolors="white",
                   linewidths=0.7, alpha=0.9, zorder=3)
    lx = [_math.log10(x) for x in xs]
    ly = [_math.log10(y) for y in ys]

    def _pear(a, b):
        ma, mb = _stats.mean(a), _stats.mean(b)
        num = sum((u - ma) * (v - mb) for u, v in zip(a, b))
        den = (sum((u - ma) ** 2 for u in a) * sum((v - mb) ** 2 for v in b)) ** 0.5
        return num / den if den else float("nan")

    rho = spearman_correlation(lx, ly)
    slope = _pear(lx, ly) * _stats.stdev(ly) / _stats.stdev(lx)
    intercept = _stats.mean(ly) - slope * _stats.mean(lx)
    grid = (min(xs), max(xs))
    ax.plot(list(grid), [10 ** (slope * _math.log10(g) + intercept) for g in grid],
            color="#666666", lw=1.4, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = min(ys) * 0.8, max(ys) * 1.3
    ticks = [t for t in (2, 5, 10, 20, 50, 100, 200, 500, 1000) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_ylim(lo, hi)
    ax.text(0.03, 0.97, f"rho={rho:.2f}", transform=ax.transAxes, ha="left",
            va="top", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5))
    ax.set_title("Serving Price", fontsize=16)
    ax.set_xlabel("Completion USD/Mtok", fontsize=14)
    ax.set_ylabel("CC", fontsize=14)
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.grid(alpha=0.2, linewidth=0.9)
    ax.set_box_aspect(1)
    set_spine_width(ax)


def draw_params_scatter(ax, counting_models) -> None:
    # Total parameter count against CC in the same visual grammar as the
    # price panel. Parameter metadata comes from the released registry. Total
    # predicts CC far better than active (rho 0.65 vs 0.29 on the models with
    # both counts), and active adds nothing given total; see Appendix J.
    import math as _math

    params_by_model: dict[str, float] = {}
    for row in load_registry(COUNTING_RUN):
        model = row.get("model") or ""
        try:
            total = float(row.get("total_params_b") or 0)
        except (TypeError, ValueError):
            total = 0.0
        try:
            active = float(row.get("active_params_b") or 0)
        except (TypeError, ValueError):
            active = 0.0
        value = total or active
        if model and value > 0:
            params_by_model[model] = value
    best = {}
    for cm in counting_models.values():
        if cm.model not in best or cm.stable_count > best[cm.model].stable_count:
            best[cm.model] = cm
    xs, ys, colours, marks = [], [], [], []
    for cm in best.values():
        value = params_by_model.get(cm.model)
        if not value or cm.stable_count <= 0:
            continue
        xs.append(value)
        ys.append(cm.stable_count)
        colours.append(PROVIDER_COLORS.get(cm.provider, "#888888"))
        marks.append(PROVIDER_MARKERS.get(cm.provider, "o"))
    for x, y, c, m in zip(xs, ys, colours, marks):
        ax.scatter([x], [y], s=36.0, marker=m, color=c, edgecolors="white",
                   linewidths=0.7, alpha=0.9, zorder=3)
    rho = spearman_correlation([_math.log10(x) for x in xs],
                               [_math.log10(y) for y in ys])
    print(f"params panel: n={len(xs)} spearman rho={rho:.2f}")
    import statistics as _stats
    lx = [_math.log10(x) for x in xs]
    ly = [_math.log10(y) for y in ys]
    mx, my = _stats.mean(lx), _stats.mean(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    slope = num / den
    intercept = my - slope * mx
    grid = (min(xs), max(xs))
    ax.plot(list(grid),
            [10 ** (slope * _math.log10(g) + intercept) for g in grid],
            color="#666666", lw=1.4, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = min(ys) * 0.8, max(ys) * 1.3
    ticks = [t for t in (2, 5, 10, 20, 50, 100, 200, 500, 1000) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_ylim(lo, hi)
    ax.text(0.03, 0.97, f"rho={rho:.2f}", transform=ax.transAxes, ha="left",
            va="top", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5))
    ax.set_title("Total Parameters", fontsize=16)
    ax.set_xlabel("Total parameters (B)", fontsize=14)
    ax.set_ylabel("CC", fontsize=14)
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.grid(alpha=0.2, linewidth=0.9)
    ax.set_box_aspect(1)
    set_spine_width(ax)


def save_secondary_figure(datasets_by_name, counting_models) -> None:
    # ECI, METR, serving price, and parameter count as a 2x2 grid in the
    # authoritative-row style.
    selected = [datasets_by_name[n] for n in SECONDARY_BENCHMARKS if n in datasets_by_name]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.15), constrained_layout=True)
    axes = axes.ravel()
    titles = {"epoch_capabilities_index.csv": "Epoch Capabilities Index",
              "metr_time_horizons_external.csv": "METR Time Horizons"}
    for ax, dataset in zip(axes, selected):
        draw_benchmark_scatter(
            ax, dataset, show_axis_labels=True, title_fontsize=16,
            label_fontsize=14, tick_fontsize=12, stats_fontsize=10,
            point_size=36.0, point_linewidth=0.7, point_alpha=0.9,
            use_house_spines=True,
            title_override=titles.get(dataset.path.name),
        )
    draw_price_scatter(axes[len(selected)], counting_models)
    draw_params_scatter(axes[len(selected) + 1], counting_models)
    for ax in axes:
        ax.set_box_aspect(0.58)
    ARTICLE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SECONDARY_FIGURE_PDF, dpi=200)
    fig.savefig(SECONDARY_FIGURE_PNG, dpi=200)
    plt.close(fig)


def save_rho_distribution_figure(
    datasets: list[BenchmarkDataset],
    pairwise_rows: list[dict[str, object]],
) -> None:
    scc_datasets = [
        dataset
        for dataset in datasets
        if dataset.spearman_rho is not None and dataset.matched_model_count >= 10
    ]
    pairwise_filtered = [
        row
        for row in pairwise_rows
        if row["spearman_rho"] is not None and int(row["overlap_models"]) >= 10
    ]
    if not scc_datasets or not pairwise_filtered:
        return

    scc_values = [float(dataset.spearman_rho) for dataset in scc_datasets]
    pairwise_values = [float(row["spearman_rho"]) for row in pairwise_filtered]
    scc_median = float(np.median(scc_values))
    pairwise_median = float(np.median(pairwise_values))
    scc_percentile = percentile_rank(pairwise_values, scc_median)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bins = np.linspace(-1.0, 1.0, 21)
    ax.hist(
        pairwise_values,
        bins=bins,
        color="#aeb3b8",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.95,
    )
    ax.axvline(
        pairwise_median,
        color="#666666",
        linewidth=1.6,
        linestyle="--",
        zorder=3,
    )
    ax.axvline(
        scc_median,
        color="#d09b52",
        linewidth=2.2,
        zorder=4,
    )
    # ax.text(
    #     pairwise_median,
    #     ax.get_ylim()[1] * 0.97,
    #     f"pair median = {pairwise_median:.2f}",
    #     ha="right",
    #     va="top",
    #     fontsize=10,
    #     color="#555555",
    #     bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    # )
    ax.text(
        scc_median-0.65,
        ax.get_ylim()[1] * 0.64,
        (
            f"CC median = {scc_median:.2f}\n"
            f"{scc_percentile:.1f} percentile"
        ),
        ha="left",
        va="top",
        fontsize=14,
        color="#a15f1c",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Spearman rho", fontsize=14)
    ax.set_ylabel("Benchmark-pair count", fontsize=14)
    ax.set_title("Benchmark Correlation Distribution", fontsize=16, y=0.99)
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.grid(alpha=0.18, axis="y", zorder=0)
    set_spine_width(ax)

    fig.savefig(RHO_DISTRIBUTION_FIGURE_PDF, dpi=200, bbox_inches="tight")
    fig.savefig(RHO_DISTRIBUTION_FIGURE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def clear_old_full_panel_outputs() -> None:
    for path in ARTICLE_FIGURE_DIR.glob(f"{FULL_PANEL_STEM}-*.pdf"):
        path.unlink()


def save_full_panel_pages(datasets: list[BenchmarkDataset]) -> list[Path]:
    # Panels are drawn only for benchmarks sharing at least five models with the
    # evaluation set; a two- or three-point panel reports a spurious +/-1.
    panel_datasets = [dataset for dataset in datasets
                      if dataset.matched_records and dataset.matched_model_count >= 5]
    ncols = 4
    max_rows_per_page = 5
    panels_per_page = ncols * max_rows_per_page
    pages: list[Path] = []
    clear_old_full_panel_outputs()

    for page_index, start in enumerate(range(0, len(panel_datasets), panels_per_page), start=1):
        chunk = panel_datasets[start : start + panels_per_page]
        nrows = max(1, math.ceil(len(chunk) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(10, max(3.0, 2.6 * nrows)),
            constrained_layout=True,
            squeeze=False,
        )
        for axis in axes.flat:
            axis.set_visible(False)

        for axis, dataset in zip(axes.flat, chunk):
            axis.set_visible(True)
            draw_benchmark_scatter(
                axis,
                dataset,
                show_axis_labels=True,
                title_fontsize=None,
                label_fontsize=None,
                tick_fontsize=None,
                stats_fontsize=8.0,
                point_size=14.0,
                point_linewidth=0.45,
                point_alpha=0.85,
                use_house_spines=False,
                highlight_post_arc_agi2=dataset.path.name == "arc_agi_2_external.csv",
            )

        output_path = ARTICLE_FIGURE_DIR / f"{FULL_PANEL_STEM}-{page_index}.pdf"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        pages.append(output_path)
    return pages


def main() -> int:
    counting_models = build_counting_models()
    alias_map = build_alias_map(counting_models)
    datasets: list[BenchmarkDataset] = []
    for path in sorted(BENCHMARK_DIR.glob("*.csv")):
        dataset = load_benchmark_dataset(path, counting_models, alias_map)
        if dataset is not None and dataset.matched_records:
            datasets.append(dataset)

    datasets.sort(key=lambda dataset: dataset.display_name.lower())
    datasets_by_name = {dataset.path.name: dataset for dataset in datasets}

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    ARTICLE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pairwise_rows = compute_pairwise_rho_rows(datasets)
    write_mapping_table(datasets)
    write_merged_table(datasets)
    write_pairwise_rho_table(pairwise_rows)
    summary_payload = write_summary(datasets, pairwise_rows)
    save_authoritative_figure(datasets_by_name)
    save_secondary_figure(datasets_by_name, counting_models)
    save_rho_distribution_figure(datasets, pairwise_rows)
    full_panel_paths = save_full_panel_pages(datasets)

    print(f"Mapping table: {MAPPING_TABLE_PATH}")
    print(f"Merged table: {MERGED_TABLE_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Pairwise rho table: {PAIRWISE_RHO_PATH}")
    print(f"Authoritative figure: {AUTHORITATIVE_FIGURE_PDF}")
    print(f"Authoritative preview: {AUTHORITATIVE_FIGURE_PNG}")
    print(f"Authoritative PowerPoint asset: {AUTHORITATIVE_FIGURE_SVG}")
    print(f"Rho distribution figure: {RHO_DISTRIBUTION_FIGURE_PDF}")
    print(f"Rho distribution preview: {RHO_DISTRIBUTION_FIGURE_PNG}")
    for path in full_panel_paths:
        print(f"Full panels: {path}")
    for line in diagnosis_lines(datasets, pairwise_rows, summary_payload):
        print(line)
    for dataset in sorted(datasets, key=lambda item: (-len(item.matched_records), item.display_name)):
        rho = "nan" if dataset.spearman_rho is None else f"{dataset.spearman_rho:.3f}"
        pearson = "nan" if dataset.pearson_log_scc is None else f"{dataset.pearson_log_scc:.3f}"
        print(
            f"{dataset.path.name}: rows={len(dataset.matched_records)} "
            f"models={dataset.matched_model_count} zero={dataset.zero_scc_count} "
            f"rho={rho} r(log)={pearson}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
