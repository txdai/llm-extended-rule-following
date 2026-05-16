#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

from run_scc_benchmark import collect_existing_models, init_client, model_filename
from count_scatter_control_common import (
    extract_json_object,
    generate_response,
    selected_cases,
)


DEFAULT_CONFIG_PATH = Path("configs/benchmarks/agent_count_disruption.json")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_RUN_SUBDIR = "agent_count_disruption_runs"
DEFAULT_SUMMARY_CSV_NAME = "agent_count_disruption_summary.csv"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_RETRY_MAX_OUTPUT_TOKENS = 32768
DEFAULT_PARALLEL_REQUESTS = 1
DEFAULT_TASK_SAMPLES = 6
DEFAULT_COUNT_SAMPLE_MIN = 32
DEFAULT_COUNT_SAMPLE_MAX_CAPACITY_FRACTION = 0.6
DEFAULT_AGENT_COUNT_SAMPLES_PER_TASK = 3
DEFAULT_COPY_COUNT_SAMPLES = 2
DEFAULT_COUNT_LADDER_START = 50
DEFAULT_COUNT_LADDER_STOP = 1000
DEFAULT_COUNT_LADDER_STEP = 50
DEFAULT_COUNT_LADDER_JITTER = 10
DEFAULT_MARKER = "a"
EXPECTED_COPY_ACTION = "write_file"
EXPECTED_COPY_PATH = "scratch/marker_sequence.txt"


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    task_index: int
    prompt: str
    expected: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSpec:
    spec_id: str
    condition: str
    model: str
    task_index: int | None
    marker_count: int
    task: AgentTask | None
    capacity_center: int | None
    count_sample_min: int | None
    count_sample_max: int | None
    schedule_kind: str


@dataclass
class DisruptionRecord:
    run_id: str
    model: str
    api: str
    condition: str
    spec_id: str
    task_id: str | None
    task_index: int | None
    marker_count: int
    capacity_center: int | None
    count_sample_min: int | None
    count_sample_max: int | None
    schedule_kind: str
    prompt_text: str
    required_keys: list[str]
    expected_payload: dict[str, Any]
    expected_marker_sequence: str | None
    parsed_json: dict[str, Any] | None
    parse_error: str | None
    json_candidate_texts: list[str]
    json_candidate_parse_errors: list[str]
    json_object_parseable: bool
    strict_json_only: bool
    required_keys_present: bool
    extra_keys_absent: bool
    schema_exact: bool
    format_contract_success: bool
    missing_keys: list[str]
    extra_keys: list[str]
    parsed_marker_count: int | None
    marker_count_correct: bool | None
    copy_action_correct: bool | None
    copy_path_correct: bool | None
    copy_exact: bool | None
    copied_marker_count: int | None
    copy_marker_count_correct: bool | None
    reported_marker_count_correct: bool | None
    externalization_success: bool | None
    final_balances_correct: bool | None
    review_ids_correct: bool | None
    net_total_correct: bool | None
    next_action_correct: bool | None
    primary_task_correct: bool | None
    task_field_score: float | None
    raw_response: str
    response_status: str | None
    response_detail: str | None
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    requested_max_output_tokens: int
    final_max_output_tokens: int
    retry_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark whether count-heavy prompts destabilize externalization and agent-like state tracking."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", help="Optional model filter.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "cases" not in payload or not isinstance(payload["cases"], list):
        raise ValueError(f"Config {path} must contain a 'cases' list")
    if "seed" not in payload:
        payload["seed"] = 17
    if "run_subdir" not in payload or not str(payload["run_subdir"]).strip():
        payload["run_subdir"] = DEFAULT_RUN_SUBDIR
    if "max_output_tokens" not in payload:
        payload["max_output_tokens"] = DEFAULT_MAX_OUTPUT_TOKENS
    if "retry_max_output_tokens" not in payload:
        payload["retry_max_output_tokens"] = DEFAULT_RETRY_MAX_OUTPUT_TOKENS
    if "parallel_requests" not in payload:
        payload["parallel_requests"] = DEFAULT_PARALLEL_REQUESTS
    if "task_samples" not in payload:
        payload["task_samples"] = DEFAULT_TASK_SAMPLES
    if "include_copy_then_agent" not in payload:
        payload["include_copy_then_agent"] = True
    return payload


def make_marker_sequence(count: int, marker: str = DEFAULT_MARKER) -> str:
    return ", ".join(marker for _ in range(max(0, count)))


def adjust_away_from_round_number(value: int, minimum: int, maximum: int) -> int:
    candidate = max(minimum, min(maximum, value))
    if candidate % 50 != 0 and candidate % 100 != 0:
        return candidate
    for delta in range(1, 12):
        for sign in (1, -1):
            shifted = candidate + (sign * delta)
            if minimum <= shifted <= maximum and shifted % 50 != 0 and shifted % 100 != 0:
                return shifted
    return candidate


def build_count_ladder(
    seed: int,
    start: int = DEFAULT_COUNT_LADDER_START,
    stop: int = DEFAULT_COUNT_LADDER_STOP,
    step: int = DEFAULT_COUNT_LADDER_STEP,
    jitter: int = DEFAULT_COUNT_LADDER_JITTER,
    salt: str = "ladder",
) -> list[int]:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be >= start")
    rng = random.Random(f"{seed}:{salt}:{start}:{stop}:{step}:{jitter}")
    counts: list[int] = []
    previous = 0
    for base in range(start, stop + 1, step):
        candidate = base + rng.randint(-abs(jitter), abs(jitter))
        candidate = adjust_away_from_round_number(candidate, start, stop)
        if candidate <= previous:
            candidate = min(stop, previous + 1)
            candidate = adjust_away_from_round_number(candidate, start, stop)
        if counts and candidate == counts[-1]:
            continue
        counts.append(candidate)
        previous = candidate
    return counts


def sample_count_range(
    seed: int,
    model: str,
    condition: str,
    minimum: int,
    maximum: int,
    sample_count: int,
    task_index: int | None,
) -> list[int]:
    if maximum < minimum:
        raise ValueError("maximum must be >= minimum")
    candidates = [
        value
        for value in range(minimum, maximum + 1)
        if value % 50 != 0 and value % 100 != 0
    ]
    if not candidates:
        candidates = list(range(minimum, maximum + 1))
    rng = random.Random(f"{seed}:{model}:{condition}:{task_index}:{minimum}:{maximum}:{sample_count}")
    if sample_count <= len(candidates):
        return sorted(rng.sample(candidates, sample_count))
    return sorted(rng.choice(candidates) for _ in range(sample_count))


def build_agent_task(seed: int, task_index: int) -> AgentTask:
    rng = random.Random(f"{seed}:agent-task:{task_index}")
    accounts = ["A", "B", "C"]
    starting_balances = {account: rng.randint(40, 120) for account in accounts}
    review_threshold = rng.choice([25, 30, 35, 40])
    transaction_count = 6

    balances = dict(starting_balances)
    review_ids: list[str] = []
    transaction_lines: list[str] = []
    transactions: list[dict[str, Any]] = []

    for index in range(transaction_count):
        account = rng.choice(accounts)
        delta = 0
        while delta == 0:
            delta = rng.randint(-45, 45)
        flagged = rng.random() < 0.25
        transaction_id = f"T{task_index + 1:02d}-{index + 1:02d}"
        balances[account] += delta
        if flagged or abs(delta) >= review_threshold:
            review_ids.append(transaction_id)
        transaction = {
            "transaction_id": transaction_id,
            "account": account,
            "delta": delta,
            "flagged": flagged,
        }
        transactions.append(transaction)
        transaction_lines.append(
            f"- {transaction_id}: account={account}, delta={delta:+d}, flagged={'yes' if flagged else 'no'}"
        )

    net_total = sum(int(value) for value in balances.values())
    if any(value < 0 for value in balances.values()):
        next_action = "escalate"
    elif review_ids:
        next_action = "manual_review"
    else:
        next_action = "accept"

    prompt = "\n".join(
        [
            "Simulate a lightweight operations agent.",
            "Apply each signed delta in order to the referenced account.",
            f"Add a transaction id to review_ids if flagged=yes or abs(delta) >= {review_threshold}.",
            "Keep review_ids in encounter order.",
            "net_total is the sum of the final balances.",
            'next_action is "escalate" if any final balance is negative, otherwise "manual_review" if review_ids is non-empty, otherwise "accept".',
            "",
            "Return JSON with exactly this schema:",
            '{"final_balances": {"A": <int>, "B": <int>, "C": <int>}, "review_ids": [<transaction ids>], "net_total": <int>, "next_action": <string>}',
            "",
            f"Starting balances: A={starting_balances['A']}, B={starting_balances['B']}, C={starting_balances['C']}",
            "Transactions:",
            *transaction_lines,
        ]
    )
    expected = {
        "final_balances": {account: int(balance) for account, balance in balances.items()},
        "review_ids": list(review_ids),
        "net_total": int(net_total),
        "next_action": next_action,
    }
    return AgentTask(
        task_id=f"agent_task_{task_index:03d}",
        task_index=task_index,
        prompt=prompt,
        expected=expected,
        metadata={
            "review_threshold": review_threshold,
            "starting_balances": starting_balances,
            "transactions": transactions,
        },
    )


def build_agent_prompt(task: AgentTask) -> str:
    return task.prompt


def build_agent_count_prompt(task: AgentTask, marker_count: int) -> str:
    return "\n".join(
        [
            "You must complete the task and count the marker sequence exactly.",
            "Return JSON with exactly this schema:",
            '{"final_balances": {"A": <int>, "B": <int>, "C": <int>}, "review_ids": [<transaction ids>], "net_total": <int>, "next_action": <string>, "marker_count": <int>}',
            "",
            "Marker sequence:",
            make_marker_sequence(marker_count, DEFAULT_MARKER),
            "",
            "Task:",
            task.prompt,
        ]
    )


def build_copy_prompt(marker_count: int) -> str:
    return "\n".join(
        [
            "Simulate an external tool call that writes the marker sequence to a file.",
            "Return JSON with exactly this schema:",
            '{"action": "write_file", "path": "scratch/marker_sequence.txt", "file_content": <exact marker sequence copied verbatim>, "marker_count": <int>}',
            "Copy the marker sequence into file_content exactly. Do not abbreviate, summarize, or normalize it.",
            "",
            "Marker sequence:",
            make_marker_sequence(marker_count, DEFAULT_MARKER),
        ]
    )


def build_copy_then_agent_prompt(task: AgentTask, marker_count: int) -> str:
    return "\n".join(
        [
            "Simulate an agent that first externalizes the marker sequence, then completes the task.",
            "Return JSON with exactly this schema:",
            '{"action": "write_file", "path": "scratch/marker_sequence.txt", "file_content": <exact marker sequence copied verbatim>, "marker_count": <int>, "final_balances": {"A": <int>, "B": <int>, "C": <int>}, "review_ids": [<transaction ids>], "net_total": <int>, "next_action": <string>}',
            "Copy the marker sequence into file_content exactly. Do not abbreviate, summarize, or normalize it.",
            "",
            "Marker sequence:",
            make_marker_sequence(marker_count, DEFAULT_MARKER),
            "",
            "Task:",
            task.prompt,
        ]
    )


def count_markers_in_text(text: str, marker: str = DEFAULT_MARKER) -> int:
    if not text.strip():
        return 0
    return sum(1 for part in text.split(",") if part.strip() == marker)


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def strict_json_diagnostics(
    raw_response: str,
    parsed: dict[str, Any] | None,
    required_keys: list[str],
) -> dict[str, Any]:
    stripped = raw_response.strip()
    strict_json_only = False
    if parsed is not None and stripped.startswith("{") and stripped.endswith("}"):
        try:
            strict_json_only = isinstance(json.loads(stripped), dict)
        except json.JSONDecodeError:
            strict_json_only = False
    parsed_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
    required = set(required_keys)
    missing_keys = sorted(required - parsed_keys)
    extra_keys = sorted(parsed_keys - required)
    required_keys_present = not missing_keys
    extra_keys_absent = not extra_keys
    schema_exact = required_keys_present and extra_keys_absent
    return {
        "json_object_parseable": parsed is not None,
        "strict_json_only": strict_json_only,
        "required_keys_present": required_keys_present,
        "extra_keys_absent": extra_keys_absent,
        "schema_exact": schema_exact,
        "format_contract_success": parsed is not None and strict_json_only and schema_exact,
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
    }


def json_candidate_texts(raw_response: str) -> list[str]:
    stripped = raw_response.strip()
    if not stripped:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(candidate: str) -> None:
        if candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    if stripped.startswith("{") and stripped.endswith("}"):
        add_candidate(stripped)

    fence_start = stripped.find("```")
    while fence_start != -1:
        fence_end = stripped.find("```", fence_start + 3)
        if fence_end == -1:
            break
        fenced = stripped[fence_start + 3 : fence_end].strip()
        if fenced.lower().startswith("json"):
            fenced = fenced[4:].strip()
        if fenced.startswith("{") and fenced.endswith("}"):
            add_candidate(fenced)
        fence_start = stripped.find("```", fence_end + 3)

    for start, char in enumerate(stripped):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for index, inner_char in enumerate(stripped[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif inner_char == "\\":
                    escape = True
                elif inner_char == '"':
                    in_string = False
                continue
            if inner_char == '"':
                in_string = True
            elif inner_char == "{":
                depth += 1
            elif inner_char == "}":
                depth -= 1
                if depth == 0:
                    add_candidate(stripped[start : index + 1])
                    break
    return candidates


def json_candidate_parse_errors(candidates: list[str]) -> list[str]:
    errors: list[str] = []
    for index, candidate in enumerate(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"candidate_{index}:json:{exc}")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"candidate_{index}:json_root:{type(parsed).__name__}")
    return errors


def normalize_balances(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, int] = {}
    for key in ("A", "B", "C"):
        coerced = coerce_int(value.get(key))
        if coerced is None:
            return None
        normalized[key] = coerced
    return normalized


def normalize_review_ids(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        normalized.append(item)
    return normalized


def score_task(parsed: dict[str, Any] | None, task: AgentTask) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {
            "final_balances_correct": False,
            "review_ids_correct": False,
            "net_total_correct": False,
            "next_action_correct": False,
            "primary_task_correct": False,
            "task_field_score": 0.0,
        }
    predicted_balances = normalize_balances(parsed.get("final_balances"))
    predicted_review_ids = normalize_review_ids(parsed.get("review_ids"))
    predicted_net_total = coerce_int(parsed.get("net_total"))
    predicted_next_action = parsed.get("next_action")

    final_balances_correct = predicted_balances == task.expected["final_balances"]
    review_ids_correct = predicted_review_ids == task.expected["review_ids"]
    net_total_correct = predicted_net_total == task.expected["net_total"]
    next_action_correct = predicted_next_action == task.expected["next_action"]

    correct_fields = [
        final_balances_correct,
        review_ids_correct,
        net_total_correct,
        next_action_correct,
    ]
    task_field_score = sum(1.0 for item in correct_fields if item) / len(correct_fields)
    return {
        "final_balances_correct": final_balances_correct,
        "review_ids_correct": review_ids_correct,
        "net_total_correct": net_total_correct,
        "next_action_correct": next_action_correct,
        "primary_task_correct": all(correct_fields),
        "task_field_score": task_field_score,
    }


def score_copy(parsed: dict[str, Any] | None, marker_count: int, condition: str) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {
            "copy_action_correct": False,
            "copy_path_correct": False,
            "copy_exact": False,
            "copied_marker_count": None,
            "copy_marker_count_correct": False,
            "reported_marker_count_correct": False,
            "externalization_success": False,
        }
    expected_content = make_marker_sequence(marker_count, DEFAULT_MARKER)
    action_correct = parsed.get("action") == EXPECTED_COPY_ACTION
    path_correct = parsed.get("path") == EXPECTED_COPY_PATH
    file_content = parsed.get("file_content")
    file_content_text = file_content if isinstance(file_content, str) else ""
    copied_marker_count = count_markers_in_text(file_content_text, DEFAULT_MARKER)
    copy_exact = file_content_text == expected_content
    copy_marker_count_correct = copied_marker_count == marker_count
    reported_marker_count_correct = coerce_int(parsed.get("marker_count")) == marker_count
    return {
        "copy_action_correct": action_correct,
        "copy_path_correct": path_correct,
        "copy_exact": copy_exact,
        "copied_marker_count": copied_marker_count,
        "copy_marker_count_correct": copy_marker_count_correct,
        "reported_marker_count_correct": reported_marker_count_correct,
        "externalization_success": (
            action_correct
            and path_correct
            and copy_exact
            and copy_marker_count_correct
            and reported_marker_count_correct
        ),
        "condition": condition,
    }


def expected_payload_for_spec(spec: ExperimentSpec) -> dict[str, Any]:
    if spec.condition == "agent_baseline":
        assert spec.task is not None
        return dict(spec.task.expected)
    if spec.condition == "agent_count_insert":
        assert spec.task is not None
        payload = dict(spec.task.expected)
        payload["marker_count"] = spec.marker_count
        return payload
    if spec.condition == "copy_externalize":
        return {
            "action": EXPECTED_COPY_ACTION,
            "path": EXPECTED_COPY_PATH,
            "file_content": make_marker_sequence(spec.marker_count, DEFAULT_MARKER),
            "marker_count": spec.marker_count,
        }
    if spec.condition == "copy_then_agent":
        assert spec.task is not None
        payload = {
            "action": EXPECTED_COPY_ACTION,
            "path": EXPECTED_COPY_PATH,
            "file_content": make_marker_sequence(spec.marker_count, DEFAULT_MARKER),
            "marker_count": spec.marker_count,
        }
        payload.update(spec.task.expected)
        return payload
    raise ValueError(f"Unsupported condition: {spec.condition}")


def resolve_marker_counts(
    config: dict[str, Any],
    model: str,
    condition_group: str,
) -> tuple[list[int], str, int | None, int | None, int | None]:
    explicit_key = f"{condition_group}_marker_counts"
    if explicit_key in config:
        counts = [int(value) for value in config.get(explicit_key, [])]
        return counts, "explicit", config.get("model_count_capacities", {}).get(model), (
            min(counts) if counts else None
        ), (max(counts) if counts else None)

    ladder_key = f"use_count_ladder_for_{condition_group}"
    if bool(config.get(ladder_key)):
        counts = build_count_ladder(
            seed=int(config["seed"]),
            start=int(config.get("count_ladder_start", DEFAULT_COUNT_LADDER_START)),
            stop=int(config.get("count_ladder_stop", DEFAULT_COUNT_LADDER_STOP)),
            step=int(config.get("count_ladder_step", DEFAULT_COUNT_LADDER_STEP)),
            jitter=int(config.get("count_ladder_jitter", DEFAULT_COUNT_LADDER_JITTER)),
            salt=f"{model}:{condition_group}",
        )
        return counts, "ladder", config.get("model_count_capacities", {}).get(model), (
            min(counts) if counts else None
        ), (max(counts) if counts else None)

    capacities = dict(config.get("model_count_capacities") or {})
    minimum = int(config.get("count_sample_min", DEFAULT_COUNT_SAMPLE_MIN))
    capacity_center = capacities.get(model)
    if capacity_center is not None:
        capacity_center = int(capacity_center)
    if capacity_center is None:
        capacity_center = int(config.get("default_count_capacity", DEFAULT_COUNT_LADDER_STOP))
    maximum = max(
        minimum,
        int(
            math.ceil(
                capacity_center
                * float(
                    config.get(
                        "count_sample_max_capacity_fraction",
                        DEFAULT_COUNT_SAMPLE_MAX_CAPACITY_FRACTION,
                    )
                )
            )
        ),
    )
    sample_key = "agent_count_samples_per_task" if condition_group == "agent" else "copy_count_samples"
    task_index = None if condition_group == "copy" else 0
    counts = sample_count_range(
        seed=int(config["seed"]),
        model=model,
        condition=f"{condition_group}_count_insert",
        minimum=minimum,
        maximum=maximum,
        sample_count=int(
            config.get(
                sample_key,
                DEFAULT_AGENT_COUNT_SAMPLES_PER_TASK
                if condition_group == "agent"
                else DEFAULT_COPY_COUNT_SAMPLES,
            )
        ),
        task_index=task_index,
    )
    return counts, "capacity_sample", capacity_center, minimum, maximum


def build_specs_for_model(config: dict[str, Any], model: str) -> list[ExperimentSpec]:
    seed = int(config["seed"])
    task_samples = int(config.get("task_samples", DEFAULT_TASK_SAMPLES))
    include_copy_then_agent = bool(config.get("include_copy_then_agent", True))
    tasks = [build_agent_task(seed=seed, task_index=index) for index in range(task_samples)]

    agent_counts, agent_schedule_kind, agent_capacity_center, agent_min, agent_max = resolve_marker_counts(
        config,
        model,
        "agent",
    )
    copy_counts, copy_schedule_kind, copy_capacity_center, copy_min, copy_max = resolve_marker_counts(
        config,
        model,
        "copy",
    )

    specs: list[ExperimentSpec] = []
    for task in tasks:
        specs.append(
            ExperimentSpec(
                spec_id=f"{model}__agent_baseline__{task.task_id}",
                condition="agent_baseline",
                model=model,
                task_index=task.task_index,
                marker_count=0,
                task=task,
                capacity_center=agent_capacity_center,
                count_sample_min=agent_min,
                count_sample_max=agent_max,
                schedule_kind=agent_schedule_kind,
            )
        )
        for marker_count in agent_counts:
            specs.append(
                ExperimentSpec(
                    spec_id=f"{model}__agent_count_insert__{task.task_id}__{marker_count}",
                    condition="agent_count_insert",
                    model=model,
                    task_index=task.task_index,
                    marker_count=int(marker_count),
                    task=task,
                    capacity_center=agent_capacity_center,
                    count_sample_min=agent_min,
                    count_sample_max=agent_max,
                    schedule_kind=agent_schedule_kind,
                )
            )
        if include_copy_then_agent:
            for marker_count in copy_counts:
                specs.append(
                    ExperimentSpec(
                        spec_id=f"{model}__copy_then_agent__{task.task_id}__{marker_count}",
                        condition="copy_then_agent",
                        model=model,
                        task_index=task.task_index,
                        marker_count=int(marker_count),
                        task=task,
                        capacity_center=copy_capacity_center,
                        count_sample_min=copy_min,
                        count_sample_max=copy_max,
                        schedule_kind=copy_schedule_kind,
                    )
                )

    for marker_count in copy_counts:
        specs.append(
            ExperimentSpec(
                spec_id=f"{model}__copy_externalize__{marker_count}",
                condition="copy_externalize",
                model=model,
                task_index=None,
                marker_count=int(marker_count),
                task=None,
                capacity_center=copy_capacity_center,
                count_sample_min=copy_min,
                count_sample_max=copy_max,
                schedule_kind=copy_schedule_kind,
            )
        )
    return specs


def prompt_for_spec(spec: ExperimentSpec) -> tuple[str, list[str]]:
    if spec.condition == "agent_baseline":
        assert spec.task is not None
        return build_agent_prompt(spec.task), [
            "final_balances",
            "review_ids",
            "net_total",
            "next_action",
        ]
    if spec.condition == "agent_count_insert":
        assert spec.task is not None
        return build_agent_count_prompt(spec.task, spec.marker_count), [
            "final_balances",
            "review_ids",
            "net_total",
            "next_action",
            "marker_count",
        ]
    if spec.condition == "copy_externalize":
        return build_copy_prompt(spec.marker_count), [
            "action",
            "path",
            "file_content",
            "marker_count",
        ]
    if spec.condition == "copy_then_agent":
        assert spec.task is not None
        return build_copy_then_agent_prompt(spec.task, spec.marker_count), [
            "action",
            "path",
            "file_content",
            "marker_count",
            "final_balances",
            "review_ids",
            "net_total",
            "next_action",
        ]
    raise ValueError(f"Unsupported condition: {spec.condition}")


def mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1.0 for value in values if value) / len(values)


def summarize_records(records: list[DisruptionRecord]) -> dict[str, Any]:
    by_condition: dict[str, list[DisruptionRecord]] = {}
    by_condition_and_count: dict[tuple[str, int], list[DisruptionRecord]] = {}
    for record in records:
        by_condition.setdefault(record.condition, []).append(record)
        by_condition_and_count.setdefault((record.condition, record.marker_count), []).append(record)

    def summarize_slice(slice_records: list[DisruptionRecord]) -> dict[str, Any]:
        return {
            "n": len(slice_records),
            "parse_rate": mean_bool([record.json_object_parseable for record in slice_records]),
            "format_contract_rate": mean_bool(
                [record.format_contract_success for record in slice_records]
            ),
            "count_correct_rate": mean_bool(
                [
                    record.marker_count_correct
                    for record in slice_records
                    if record.marker_count_correct is not None
                ]
            ),
            "task_success_rate": mean_bool(
                [
                    record.primary_task_correct
                    for record in slice_records
                    if record.primary_task_correct is not None
                ]
            ),
            "externalization_success_rate": mean_bool(
                [
                    record.externalization_success
                    for record in slice_records
                    if record.externalization_success is not None
                ]
            ),
            "copy_exact_rate": mean_bool(
                [record.copy_exact for record in slice_records if record.copy_exact is not None]
            ),
            "mean_task_field_score": (
                sum(
                    float(record.task_field_score)
                    for record in slice_records
                    if record.task_field_score is not None
                )
                / max(
                    1,
                    sum(1 for record in slice_records if record.task_field_score is not None),
                )
            )
            if any(record.task_field_score is not None for record in slice_records)
            else None,
        }

    return {
        "overall": summarize_slice(records),
        "by_condition": {
            condition: summarize_slice(condition_records)
            for condition, condition_records in sorted(by_condition.items())
        },
        "by_condition_and_count": {
            f"{condition}::{marker_count}": summarize_slice(group_records)
            for (condition, marker_count), group_records in sorted(by_condition_and_count.items())
        },
    }


def build_record(
    run_id: str,
    case: dict[str, Any],
    spec: ExperimentSpec,
    raw_response: str,
    response_status: str | None,
    response_detail: str | None,
    usage: dict[str, int | None],
    final_max_output_tokens: int,
    retry_count: int,
    latency_seconds: float,
) -> DisruptionRecord:
    parsed_json, parse_error = extract_json_object(raw_response)
    prompt, required_keys = prompt_for_spec(spec)
    expected_payload = expected_payload_for_spec(spec)
    expected_marker_sequence = (
        make_marker_sequence(spec.marker_count, DEFAULT_MARKER)
        if spec.condition in {"agent_count_insert", "copy_externalize", "copy_then_agent"}
        else None
    )
    candidate_texts = json_candidate_texts(raw_response)
    candidate_parse_errors = json_candidate_parse_errors(candidate_texts)
    diagnostics = strict_json_diagnostics(raw_response, parsed_json, required_keys)
    task_scores: dict[str, Any] = {}
    copy_scores: dict[str, Any] = {}

    parsed_marker_count = None
    marker_count_correct: bool | None = None
    if isinstance(parsed_json, dict) and "marker_count" in parsed_json:
        parsed_marker_count = coerce_int(parsed_json.get("marker_count"))
        marker_count_correct = parsed_marker_count == spec.marker_count
    elif spec.condition in {"agent_count_insert", "copy_externalize", "copy_then_agent"}:
        marker_count_correct = False

    if spec.condition in {"agent_baseline", "agent_count_insert", "copy_then_agent"} and spec.task is not None:
        task_scores = score_task(parsed_json, spec.task)
    if spec.condition in {"copy_externalize", "copy_then_agent"}:
        copy_scores = score_copy(parsed_json, spec.marker_count, spec.condition)

    return DisruptionRecord(
        run_id=run_id,
        model=str(case["model"]),
        api=str(case["api"]),
        condition=spec.condition,
        spec_id=spec.spec_id,
        task_id=spec.task.task_id if spec.task is not None else None,
        task_index=spec.task_index,
        marker_count=spec.marker_count,
        capacity_center=spec.capacity_center,
        count_sample_min=spec.count_sample_min,
        count_sample_max=spec.count_sample_max,
        schedule_kind=spec.schedule_kind,
        prompt_text=prompt,
        required_keys=list(required_keys),
        expected_payload=expected_payload,
        expected_marker_sequence=expected_marker_sequence,
        parsed_json=parsed_json,
        parse_error=parse_error,
        json_candidate_texts=candidate_texts,
        json_candidate_parse_errors=candidate_parse_errors,
        json_object_parseable=bool(diagnostics["json_object_parseable"]),
        strict_json_only=bool(diagnostics["strict_json_only"]),
        required_keys_present=bool(diagnostics["required_keys_present"]),
        extra_keys_absent=bool(diagnostics["extra_keys_absent"]),
        schema_exact=bool(diagnostics["schema_exact"]),
        format_contract_success=bool(diagnostics["format_contract_success"]),
        missing_keys=list(diagnostics["missing_keys"]),
        extra_keys=list(diagnostics["extra_keys"]),
        parsed_marker_count=parsed_marker_count,
        marker_count_correct=marker_count_correct,
        copy_action_correct=copy_scores.get("copy_action_correct"),
        copy_path_correct=copy_scores.get("copy_path_correct"),
        copy_exact=copy_scores.get("copy_exact"),
        copied_marker_count=copy_scores.get("copied_marker_count"),
        copy_marker_count_correct=copy_scores.get("copy_marker_count_correct"),
        reported_marker_count_correct=copy_scores.get("reported_marker_count_correct"),
        externalization_success=copy_scores.get("externalization_success"),
        final_balances_correct=task_scores.get("final_balances_correct"),
        review_ids_correct=task_scores.get("review_ids_correct"),
        net_total_correct=task_scores.get("net_total_correct"),
        next_action_correct=task_scores.get("next_action_correct"),
        primary_task_correct=task_scores.get("primary_task_correct"),
        task_field_score=task_scores.get("task_field_score"),
        raw_response=raw_response,
        response_status=response_status,
        response_detail=response_detail,
        latency_seconds=latency_seconds,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        reasoning_tokens=usage.get("reasoning_tokens"),
        cached_input_tokens=usage.get("cached_input_tokens"),
        requested_max_output_tokens=int(case.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
        final_max_output_tokens=final_max_output_tokens,
        retry_count=retry_count,
    )


def run_spec(
    client: Any,
    case: dict[str, Any],
    config: dict[str, Any],
    run_id: str,
    spec: ExperimentSpec,
) -> DisruptionRecord:
    prompt, _ = prompt_for_spec(spec)
    requested_max_output_tokens = int(config.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
    started = time.perf_counter()
    try:
        raw_response, response_status, response_detail, usage, final_max_output_tokens, retry_count = (
            generate_response(
                client=client,
                api=str(case["api"]),
                model=str(case["model"]),
                prompt=prompt,
                max_output_tokens=requested_max_output_tokens,
                retry_max_output_tokens=int(
                    config.get("retry_max_output_tokens", DEFAULT_RETRY_MAX_OUTPUT_TOKENS)
                ),
                config=config,
                reasoning_effort=str(case.get("reasoning_effort") or "").strip() or None,
            )
        )
    except Exception as exc:
        raw_response = f"ERROR: {type(exc).__name__}: {exc}"
        response_status = "error"
        response_detail = type(exc).__name__
        usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "cached_input_tokens": None,
        }
        final_max_output_tokens = requested_max_output_tokens
        retry_count = 0
    latency_seconds = time.perf_counter() - started
    case_with_budget = dict(case)
    case_with_budget["max_output_tokens"] = requested_max_output_tokens
    return build_record(
        run_id=run_id,
        case=case_with_budget,
        spec=spec,
        raw_response=raw_response,
        response_status=response_status,
        response_detail=response_detail,
        usage=usage,
        final_max_output_tokens=final_max_output_tokens,
        retry_count=retry_count,
        latency_seconds=latency_seconds,
    )


def run_case_specs(
    run_id: str,
    client: Any,
    case: dict[str, Any],
    specs: list[ExperimentSpec],
    config: dict[str, Any],
) -> list[DisruptionRecord]:
    max_workers = max(1, int(config.get("parallel_requests", DEFAULT_PARALLEL_REQUESTS)))
    ordered_records: list[DisruptionRecord | None] = [None] * len(specs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_spec, client, case, config, run_id, spec): (index, spec)
            for index, spec in enumerate(specs)
        }
        completed = 0
        total = len(specs)
        for future in as_completed(futures):
            index, spec = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                fallback_case = dict(case)
                fallback_case["max_output_tokens"] = int(
                    config.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
                )
                record = build_record(
                    run_id=run_id,
                    case=fallback_case,
                    spec=spec,
                    raw_response=f"ERROR: {type(exc).__name__}: {exc}",
                    response_status="error",
                    response_detail=type(exc).__name__,
                    usage={
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "reasoning_tokens": None,
                        "cached_input_tokens": None,
                    },
                    final_max_output_tokens=int(
                        config.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
                    ),
                    retry_count=0,
                    latency_seconds=0.0,
                )
            ordered_records[index] = record
            completed += 1
            print(
                f"  [{completed}/{total}] {spec.condition} count={spec.marker_count}"
                f" parse={int(record.json_object_parseable)}"
                f" task={record.primary_task_correct if record.primary_task_correct is not None else 'n/a'}"
                f" copy={record.externalization_success if record.externalization_success is not None else 'n/a'}"
            )

    return [record for record in ordered_records if record is not None]


def run_dir_for_config(output_dir: Path, config: dict[str, Any]) -> Path:
    return output_dir / str(config.get("run_subdir", DEFAULT_RUN_SUBDIR))


def write_manifest(
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    requested_cases: list[dict[str, Any]],
    selected_model_names: list[str],
    existing_model_names: list[str],
) -> None:
    payload = {
        "generated_at_utc": utc_timestamp(),
        "run_id": str(config.get("run_subdir", DEFAULT_RUN_SUBDIR)),
        "config_path": str(config_path),
        "config": config,
        "requested_cases": requested_cases,
        "selected_models": selected_model_names,
        "already_completed_models": existing_model_names,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_model_payload(
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    case: dict[str, Any],
    specs: list[ExperimentSpec],
    records: list[DisruptionRecord],
) -> Path:
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / model_filename(str(case["model"]))
    payload = {
        "generated_at_utc": utc_timestamp(),
        "run_id": str(config.get("run_subdir", DEFAULT_RUN_SUBDIR)),
        "config_path": str(config_path),
        "config": config,
        "model": str(case["model"]),
        "api": str(case["api"]),
        "case": case,
        "specs": [asdict(spec) for spec in specs],
        "summary": summarize_records(records),
        "records": [asdict(record) for record in records],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_summary_csv(run_dir: Path) -> Path:
    models_dir = run_dir / "models"
    output_path = run_dir / DEFAULT_SUMMARY_CSV_NAME
    rows: list[dict[str, Any]] = []
    for model_path in sorted(models_dir.glob("*.json")):
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        summary = dict(payload.get("summary") or {})
        by_condition = dict(summary.get("by_condition") or {})
        by_condition_and_count = dict(summary.get("by_condition_and_count") or {})
        for condition, condition_summary in sorted(by_condition.items()):
            rows.append(
                {
                    "model": payload.get("model"),
                    "api": payload.get("api"),
                    "slice_type": "condition",
                    "slice_key": condition,
                    "condition": condition,
                    "marker_count": "",
                    "n": condition_summary.get("n"),
                    "parse_rate": condition_summary.get("parse_rate"),
                    "format_contract_rate": condition_summary.get("format_contract_rate"),
                    "count_correct_rate": condition_summary.get("count_correct_rate"),
                    "task_success_rate": condition_summary.get("task_success_rate"),
                    "externalization_success_rate": condition_summary.get(
                        "externalization_success_rate"
                    ),
                    "copy_exact_rate": condition_summary.get("copy_exact_rate"),
                    "mean_task_field_score": condition_summary.get("mean_task_field_score"),
                }
            )
        for slice_key, slice_summary in sorted(by_condition_and_count.items()):
            condition, marker_count = slice_key.split("::", 1)
            rows.append(
                {
                    "model": payload.get("model"),
                    "api": payload.get("api"),
                    "slice_type": "condition_and_count",
                    "slice_key": slice_key,
                    "condition": condition,
                    "marker_count": marker_count,
                    "n": slice_summary.get("n"),
                    "parse_rate": slice_summary.get("parse_rate"),
                    "format_contract_rate": slice_summary.get("format_contract_rate"),
                    "count_correct_rate": slice_summary.get("count_correct_rate"),
                    "task_success_rate": slice_summary.get("task_success_rate"),
                    "externalization_success_rate": slice_summary.get(
                        "externalization_success_rate"
                    ),
                    "copy_exact_rate": slice_summary.get("copy_exact_rate"),
                    "mean_task_field_score": slice_summary.get("mean_task_field_score"),
                }
            )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "api",
                "slice_type",
                "slice_key",
                "condition",
                "marker_count",
                "n",
                "parse_rate",
                "format_contract_rate",
                "count_correct_rate",
                "task_success_rate",
                "externalization_success_rate",
                "copy_exact_rate",
                "mean_task_field_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 1

    model_filter = set(args.models or [])
    try:
        cases = selected_cases(config, model_filter or None)
    except Exception as exc:
        print(f"Failed to select cases: {exc}", file=sys.stderr)
        return 1

    if args.list_cases:
        for case in cases:
            print(
                f"{case['model']}\tapi={case['api']}\treasoning={case.get('reasoning_effort', '')}"
            )
        return 0

    run_dir = run_dir_for_config(args.output_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_models = collect_existing_models(args.output_dir, config)
    pending_cases = [case for case in cases if str(case["model"]) not in existing_models]

    write_manifest(
        run_dir=run_dir,
        config_path=args.config,
        config=config,
        requested_cases=cases,
        selected_model_names=[str(case["model"]) for case in pending_cases],
        existing_model_names=sorted(existing_models.keys()),
    )

    if args.prepare_only:
        for case in pending_cases:
            specs = build_specs_for_model(config, str(case["model"]))
            print(f"{case['model']}\t{len(specs)} specs")
            for spec in specs[:8]:
                print(
                    f"  {spec.condition}\tcount={spec.marker_count}\ttask={spec.task_index}\tschedule={spec.schedule_kind}"
                )
        return 0

    if not pending_cases:
        print("No pending disruption cases to run.")
        return 0

    clients: dict[str, Any] = {}
    try:
        for case in pending_cases:
            api = str(case["api"])
            if api not in clients:
                clients[api] = init_client(api)
    except Exception as exc:
        print(f"Failed to initialize client: {exc}", file=sys.stderr)
        return 1

    run_id = str(config.get("run_subdir", DEFAULT_RUN_SUBDIR))
    for case in pending_cases:
        model = str(case["model"])
        specs = build_specs_for_model(config, model)
        print(
            f"Running {model} with {len(specs)} disruption specs"
            f" using parallel_requests={max(1, int(config.get('parallel_requests', DEFAULT_PARALLEL_REQUESTS)))}..."
        )
        records = run_case_specs(
            run_id=run_id,
            client=clients[str(case["api"])],
            case=case,
            specs=specs,
            config=config,
        )
        write_model_payload(
            run_dir=run_dir,
            config_path=args.config,
            config=config,
            case=case,
            specs=specs,
            records=records,
        )
        write_summary_csv(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
