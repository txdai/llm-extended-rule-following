#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from openai import OpenAI


DEFAULT_CONFIG_PATH = Path("configs/benchmarks/scc_closed_openai.json")
DEFAULT_MODEL_LIST_PATH = Path("../model_lists/openai_models.json")
DEFAULT_RETRY_MAX_OUTPUT_TOKENS = 32768
DEFAULT_PARALLEL_TRIALS_PER_BATCH = 15
DEFAULT_GOOGLE_UNAVAILABLE_RETRIES = 20
DEFAULT_GOOGLE_UNAVAILABLE_RETRY_DELAY_SECONDS = 2.0
DEFAULT_ANTHROPIC_TRANSIENT_RETRIES = 8
DEFAULT_ANTHROPIC_TRANSIENT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_ANTHROPIC_API_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_OPENROUTER_TRANSIENT_RETRIES = 8
DEFAULT_OPENROUTER_TRANSIENT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PERSISTENT_RUN_DIR_NAME = "scc_closed_model_runs"
DEFAULT_OPEN_SOURCE_RUN_SUBDIR = "scc_open_model_runs"

INTEGER_PATTERN = re.compile(r"-?\d+")


@dataclass
class TrialResult:
    api: str
    model: str
    phase: str
    evaluation_index: int
    center_length: int
    trial_index: int
    sample_low: int
    sample_high: int
    expected: int
    raw_response: str
    response_status: str | None
    response_detail: str | None
    parsed: int | None
    exact_match: bool
    absolute_error: float | None
    squared_error: float | None
    relative_error: float | None
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    requested_max_output_tokens: int
    final_max_output_tokens: int
    retry_count: int


@dataclass
class RunPaths:
    run_id: str
    timestamp: str
    run_dir: Path
    models_dir: Path
    manifest_path: Path
    cc_json_path: Path
    cc_csv_path: Path


@dataclass(frozen=True)
class SequenceFormatSpec:
    item: str = "a"
    delimiter: str = ", "
    item_label: str | None = None


@dataclass
class AnthropicClient:
    api_key: str
    api_version: str = DEFAULT_ANTHROPIC_API_VERSION
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL


class AnthropicAPIError(RuntimeError):
    def __init__(self, status_code: int, body: str, retry_after: str | None = None):
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        super().__init__(f"HTTP {status_code}: {body}")


@dataclass
class OpenRouterClient:
    api_key: str
    base_url: str = DEFAULT_OPENROUTER_BASE_URL


class OpenRouterAPIError(RuntimeError):
    def __init__(self, status_code: int, body: str, retry_after: str | None = None):
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        super().__init__(f"HTTP {status_code}: {body}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCC benchmark for measuring counting capacity (CC) in text models."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the benchmark config JSON file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional explicit model override. If omitted, models come from the model list file.",
    )
    parser.add_argument(
        "--model-list",
        type=Path,
        help="Optional override for the model list JSON file.",
    )
    parser.add_argument(
        "--api",
        choices=("responses", "chat", "google_genai", "anthropic", "openrouter"),
        help="Optional backend override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the benchmark result folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional RNG seed override.",
    )
    parser.add_argument(
        "--max-output-tokens",
        "--max-completion-tokens",
        dest="max_output_tokens",
        type=int,
        help="Optional output token cap override.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the selected models and exit.",
    )
    return parser.parse_args()


def parse_integer(text: str) -> int | None:
    matches = INTEGER_PATTERN.findall(text.strip())
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def make_sequence(length: int, item: str = "a", delimiter: str = ", ") -> str:
    return delimiter.join(item for _ in range(length))


def sequence_item_label(item: str, item_label: str | None = None) -> str:
    return item if item_label is None else item_label


def build_prompt(sequence: str, item_label: str = "a") -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You count items exactly. Return only one integer with no words, "
                "punctuation, or explanation."
            ),
        },
        {
            "role": "user",
            "content": f'How many occurrences of "{item_label}" are in this sequence?\n\n{sequence}',
        },
    ]


def build_custom_prompt(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You count items exactly. Return only one integer with no words, "
                "punctuation, or explanation."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def system_instruction() -> str:
    return (
        "You count items exactly. Return only one integer with no words, "
        "punctuation, or explanation."
    )


def reasoning_config(reasoning_effort: str | None) -> dict[str, str] | None:
    if reasoning_effort is None:
        return None
    effort = str(reasoning_effort).strip().lower()
    if not effort:
        return None
    return {"effort": effort}


def user_prompt(sequence: str, item_label: str = "a") -> str:
    return f'How many occurrences of "{item_label}" are in this sequence?\n\n{sequence}'


def inline_google_prompt(sequence: str, item_label: str = "a") -> str:
    return f"{system_instruction()}\n\n{user_prompt(sequence, item_label=item_label)}"


def inline_google_custom_prompt(prompt: str) -> str:
    return f"{system_instruction()}\n\n{prompt}"


def safe_usage_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mapping_or_attr(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def extract_usage(response: Any, api: str) -> dict[str, int | None]:
    if api == "openrouter":
        usage = mapping_or_attr(response, "usage")
        if usage is None:
            return {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "reasoning_tokens": None,
                "cached_input_tokens": None,
            }
        return {
            "input_tokens": safe_usage_int(mapping_or_attr(usage, "prompt_tokens")),
            "output_tokens": safe_usage_int(mapping_or_attr(usage, "completion_tokens")),
            "total_tokens": safe_usage_int(mapping_or_attr(usage, "total_tokens")),
            "reasoning_tokens": None,
            "cached_input_tokens": None,
        }

    if api == "anthropic":
        usage = mapping_or_attr(response, "usage")
        if usage is None:
            return {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "reasoning_tokens": None,
                "cached_input_tokens": None,
            }
        input_tokens = safe_usage_int(mapping_or_attr(usage, "input_tokens"))
        output_tokens = safe_usage_int(mapping_or_attr(usage, "output_tokens"))
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": None,
            "cached_input_tokens": None,
        }

    if api == "google_genai":
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "reasoning_tokens": None,
                "cached_input_tokens": None,
            }
        return {
            "input_tokens": safe_usage_int(getattr(usage, "prompt_token_count", None)),
            "output_tokens": safe_usage_int(getattr(usage, "candidates_token_count", None)),
            "total_tokens": safe_usage_int(getattr(usage, "total_token_count", None)),
            "reasoning_tokens": safe_usage_int(getattr(usage, "thoughts_token_count", None)),
            "cached_input_tokens": safe_usage_int(getattr(usage, "cached_content_token_count", None)),
        }

    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "cached_input_tokens": None,
        }

    if api == "chat":
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        completion_tokens_details = getattr(usage, "completion_tokens_details", None)
        return {
            "input_tokens": safe_usage_int(getattr(usage, "prompt_tokens", None)),
            "output_tokens": safe_usage_int(getattr(usage, "completion_tokens", None)),
            "total_tokens": safe_usage_int(getattr(usage, "total_tokens", None)),
            "reasoning_tokens": safe_usage_int(
                getattr(completion_tokens_details, "reasoning_tokens", None)
                if completion_tokens_details is not None
                else None
            ),
            "cached_input_tokens": safe_usage_int(
                getattr(prompt_tokens_details, "cached_tokens", None)
                if prompt_tokens_details is not None
                else None
            ),
        }

    input_tokens_details = getattr(usage, "input_tokens_details", None)
    output_tokens_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": safe_usage_int(getattr(usage, "input_tokens", None)),
        "output_tokens": safe_usage_int(getattr(usage, "output_tokens", None)),
        "total_tokens": safe_usage_int(getattr(usage, "total_tokens", None)),
        "reasoning_tokens": safe_usage_int(
            getattr(output_tokens_details, "reasoning_tokens", None)
            if output_tokens_details is not None
            else None
        ),
        "cached_input_tokens": safe_usage_int(
            getattr(input_tokens_details, "cached_tokens", None)
            if input_tokens_details is not None
            else None
        ),
    }


def parse_billion_value(text: str) -> float | None:
    normalized = text.strip().lower().replace("billion", "b")
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", normalized)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def infer_open_source_parameter_metadata(model: str, model_metadata: dict[str, Any] | None) -> dict[str, Any]:
    model_metadata = model_metadata or {}
    candidates = [
        str(model),
        str(model_metadata.get("id", "") or ""),
        str(model_metadata.get("canonical_slug", "") or ""),
        str(model_metadata.get("name", "") or ""),
        str(model_metadata.get("display_name", "") or ""),
        str(model_metadata.get("hugging_face_id", "") or ""),
        str(model_metadata.get("description", "") or ""),
    ]
    lower_candidates = [value.lower() for value in candidates if value]
    description = str(model_metadata.get("description", "") or "")
    description_lower = description.lower()

    total_params_billion: float | None = None
    active_params_billion: float | None = None
    source = "unavailable"

    explicit_total = re.search(
        r"(\d+(?:\.\d+)?)\s*b(?:illion)?\s+(?:parameter|parameters|params?)",
        description_lower,
    )
    explicit_active = re.search(
        r"(\d+(?:\.\d+)?)\s*b(?:illion)?\s+(?:active|activated)",
        description_lower,
    )
    if explicit_total:
        total_params_billion = float(explicit_total.group(1))
        source = "description"
    if explicit_active:
        active_params_billion = float(explicit_active.group(1))
        source = "description"

    if total_params_billion is None or active_params_billion is None:
        moa_match = re.search(r"(\d+(?:\.\d+)?)b-a(\d+(?:\.\d+)?)b", " ".join(lower_candidates))
        if moa_match:
            total_params_billion = float(moa_match.group(1))
            active_params_billion = float(moa_match.group(2))
            source = "id_pattern"

    if total_params_billion is None or active_params_billion is None:
        expert_match = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", " ".join(lower_candidates))
        if expert_match:
            expert_count = float(expert_match.group(1))
            expert_size = float(expert_match.group(2))
            total_params_billion = expert_count * expert_size
            if active_params_billion is None:
                active_params_billion = 2 * expert_size
            source = "expert_pattern"

    if total_params_billion is None:
        generic_match = re.search(r"(\d+(?:\.\d+)?)b", " ".join(lower_candidates))
        if generic_match:
            total_params_billion = float(generic_match.group(1))
            source = "id_pattern"
    if active_params_billion is None and total_params_billion is not None:
        active_params_billion = total_params_billion

    if "phi-4" in " ".join(lower_candidates):
        total_params_billion = total_params_billion or 14.0
        active_params_billion = active_params_billion or 14.0
        source = "known_override"

    if "llama-4-maverick" in " ".join(lower_candidates):
        active_match = re.search(r"(\d+(?:\.\d+)?)\s*b(?:illion)?\s+active", description_lower)
        if active_match:
            active_params_billion = float(active_match.group(1))
            total_params_billion = total_params_billion
            source = "description"

    if total_params_billion is not None:
        total_params_billion = round(total_params_billion, 4)
    if active_params_billion is not None:
        active_params_billion = round(active_params_billion, 4)

    return {
        "total_params_billion": total_params_billion,
        "active_params_billion": active_params_billion,
        "parameter_source": source,
    }


def load_model_list(path: Path) -> list[str]:
    payload = load_model_catalog(path)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError(f"Model list file {path} must contain a JSON list or an object with a 'models' list")

    models: list[str] = []
    seen: set[str] = set()
    for item in raw_models:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Model list file {path} contains a non-string model entry: {item!r}")
        model = item.strip()
        if model not in seen:
            seen.add(model)
            models.append(model)
    return models


def load_model_catalog(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    else:
        return {"models": payload}


def load_model_catalog_metadata(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_model_catalog(path)
    raw_metadata = payload.get("model_metadata")
    if not isinstance(raw_metadata, dict):
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    for model, item in raw_metadata.items():
        if isinstance(model, str) and isinstance(item, dict):
            metadata[model] = item
    return metadata


def run_subdir(config: dict[str, Any] | None = None) -> str:
    if config is None:
        return PERSISTENT_RUN_DIR_NAME
    value = str(config.get("run_subdir", PERSISTENT_RUN_DIR_NAME)).strip()
    return value or PERSISTENT_RUN_DIR_NAME


def persistent_run_dir(results_dir: Path, config: dict[str, Any] | None = None) -> Path:
    return results_dir / run_subdir(config)


def collect_existing_models(results_dir: Path, config: dict[str, Any] | None = None) -> dict[str, Path]:
    existing: dict[str, Path] = {}
    models_dir = persistent_run_dir(results_dir, config) / "model_runs"
    if not models_dir.exists():
        return existing
    for model_path in sorted(models_dir.glob("*.json")):
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model = payload.get("model")
        if isinstance(model, str) and model.strip():
            existing[model] = model_path
    return existing


def normalize_cc_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return summary


def load_existing_model_summaries(existing_model_paths: dict[str, Path]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for model, model_path in sorted(existing_model_paths.items()):
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = payload.get("summary")
        if isinstance(summary, dict):
            summaries[model] = normalize_cc_summary(summary)
    return summaries


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return statistics.fmean(items)


def safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def format_metric(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def load_benchmark_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "search" not in payload:
        raise ValueError(f"Config file {path} is missing the 'search' section")
    if "model_list_path" not in payload:
        payload["model_list_path"] = str(DEFAULT_MODEL_LIST_PATH)
    if "retry_max_output_tokens" not in payload:
        payload["retry_max_output_tokens"] = DEFAULT_RETRY_MAX_OUTPUT_TOKENS
    if "google_unavailable_retries" not in payload:
        payload["google_unavailable_retries"] = DEFAULT_GOOGLE_UNAVAILABLE_RETRIES
    if "google_unavailable_retry_delay_seconds" not in payload:
        payload["google_unavailable_retry_delay_seconds"] = DEFAULT_GOOGLE_UNAVAILABLE_RETRY_DELAY_SECONDS
    if "anthropic_transient_retries" not in payload:
        payload["anthropic_transient_retries"] = DEFAULT_ANTHROPIC_TRANSIENT_RETRIES
    if "anthropic_transient_retry_delay_seconds" not in payload:
        payload["anthropic_transient_retry_delay_seconds"] = DEFAULT_ANTHROPIC_TRANSIENT_RETRY_DELAY_SECONDS
    if "openrouter_transient_retries" not in payload:
        payload["openrouter_transient_retries"] = DEFAULT_OPENROUTER_TRANSIENT_RETRIES
    if "openrouter_transient_retry_delay_seconds" not in payload:
        payload["openrouter_transient_retry_delay_seconds"] = DEFAULT_OPENROUTER_TRANSIENT_RETRY_DELAY_SECONDS
    if "run_subdir" not in payload:
        payload["run_subdir"] = PERSISTENT_RUN_DIR_NAME
    if "parallel_trials_per_batch" not in payload["search"]:
        payload["search"]["parallel_trials_per_batch"] = DEFAULT_PARALLEL_TRIALS_PER_BATCH
    return payload


def resolve_models(
    config: dict[str, Any],
    requested_models: list[str] | None,
    model_list_path: Path,
    results_dir: Path,
) -> tuple[list[str], list[str], dict[str, Path]]:
    requested = requested_models or load_model_list(model_list_path)
    existing = collect_existing_models(results_dir, config)
    already_completed = [model for model in requested if model in existing]
    pending = [model for model in requested if model not in existing]
    return pending, already_completed, existing


def init_client(api: str) -> Any:
    if api == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to run the OpenRouter benchmark backend")
        return OpenRouterClient(api_key=api_key)

    if api == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required to run the Anthropic benchmark backend")
        return AnthropicClient(api_key=api_key)

    if api == "google_genai":
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required to run the Google GenAI benchmark backend"
            ) from exc

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            return genai.Client(api_key=api_key)
        return genai.Client()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI package is required to run the OpenAI benchmark backend") from exc
    return OpenAI()


def google_finish_reason_name(candidate: Any) -> str | None:
    finish_reason = getattr(candidate, "finish_reason", None)
    if finish_reason is None:
        return None
    name = getattr(finish_reason, "name", None)
    if isinstance(name, str) and name:
        return name
    value = getattr(finish_reason, "value", None)
    if isinstance(value, str) and value:
        return value
    text = str(finish_reason)
    return text if text else None


def google_response_status_and_detail(response: Any) -> tuple[str | None, str | None]:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason is not None:
            block_name = getattr(block_reason, "name", None) or str(block_reason)
            if block_name and block_name != "BLOCK_REASON_UNSPECIFIED":
                return "blocked", str(block_name)

    candidates = list(getattr(response, "candidates", None) or [])
    if not candidates:
        return "completed", None

    finish_reason = google_finish_reason_name(candidates[0])
    if finish_reason == "MAX_TOKENS":
        return "incomplete", "max_output_tokens"
    if finish_reason and finish_reason not in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
        return "completed", finish_reason
    return "completed", None


def is_google_transient_retry_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return (
        ("503" in message and "UNAVAILABLE" in message)
        or ("429" in message and "RESOURCE_EXHAUSTED" in message)
    )


def google_retry_delay_seconds(exc: Exception) -> float | None:
    message = str(exc)
    patterns = [
        re.compile(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
        re.compile(r"'retryDelay':\s*'([0-9]+(?:\.[0-9]+)?)s'", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(message)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def is_google_developer_instruction_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "developer instruction is not enabled" in message


def anthropic_request_json(
    client: AnthropicClient,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{client.base_url}{path}"
    if query:
        encoded_query = urllib.parse.urlencode(
            {key: value for key, value in query.items() if value is not None}
        )
        if encoded_query:
            url = f"{url}?{encoded_query}"
    data: bytes | None = None
    headers = {
        "x-api-key": client.api_key,
        "anthropic-version": client.api_version,
        "accept": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AnthropicAPIError(exc.code, body, exc.headers.get("retry-after")) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from Anthropic: {body[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected Anthropic response shape: {type(parsed).__name__}")
    return parsed


def anthropic_text_from_message(response: dict[str, Any]) -> str:
    content = response.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def anthropic_response_status_and_detail(response: dict[str, Any]) -> tuple[str | None, str | None]:
    stop_reason = response.get("stop_reason")
    if stop_reason == "max_tokens":
        return "incomplete", "max_output_tokens"
    if isinstance(stop_reason, str) and stop_reason not in {"end_turn", ""}:
        return "completed", stop_reason
    return "completed", None


def is_anthropic_transient_retry_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 503, 529}:
        return True
    message = str(exc).upper()
    return any(code in message for code in ("429", "500", "503", "529"))


def anthropic_retry_delay_seconds(exc: Exception) -> float | None:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    message = str(exc)
    patterns = [
        re.compile(r"retry[- ]after[: ]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
        re.compile(r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(message)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def openrouter_request_json(
    client: OpenRouterClient,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{client.base_url}{path}"
    if query:
        encoded_query = urllib.parse.urlencode(
            {key: value for key, value in query.items() if value is not None}
        )
        if encoded_query:
            url = f"{url}?{encoded_query}"
    data: bytes | None = None
    headers = {
        "authorization": f"Bearer {client.api_key}",
        "accept": "application/json",
        "content-type": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise OpenRouterAPIError(exc.code, body, exc.headers.get("retry-after")) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from OpenRouter: {body[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected OpenRouter response shape: {type(parsed).__name__}")
    return parsed


def openrouter_response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def openrouter_response_status_and_detail(response: dict[str, Any]) -> tuple[str | None, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return "completed", None
    finish_reason = choices[0].get("finish_reason")
    if finish_reason == "length":
        return "incomplete", "max_output_tokens"
    if isinstance(finish_reason, str) and finish_reason:
        return "completed", finish_reason
    return "completed", None


def is_openrouter_transient_retry_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504, 524, 529}:
        return True
    message = str(exc).upper()
    return any(code in message for code in ("429", "500", "502", "503", "504", "524", "529"))


def openrouter_retry_delay_seconds(exc: Exception) -> float | None:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    message = str(exc)
    patterns = [
        re.compile(r"retry[- ]after[: ]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
        re.compile(r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(message)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def run_preflight_check(
    client: Any,
    model: str,
    api: str,
    max_output_tokens: int,
    retry_max_output_tokens: int,
    preflight_length: int,
    sequence_format: SequenceFormatSpec = SequenceFormatSpec(),
    user_prompt_factory: Callable[[int], str] | None = None,
    reasoning_effort: str | None = None,
    google_unavailable_retries: int = DEFAULT_GOOGLE_UNAVAILABLE_RETRIES,
    google_unavailable_retry_delay_seconds: float = DEFAULT_GOOGLE_UNAVAILABLE_RETRY_DELAY_SECONDS,
    anthropic_transient_retries: int = DEFAULT_ANTHROPIC_TRANSIENT_RETRIES,
    anthropic_transient_retry_delay_seconds: float = DEFAULT_ANTHROPIC_TRANSIENT_RETRY_DELAY_SECONDS,
    openrouter_transient_retries: int = DEFAULT_OPENROUTER_TRANSIENT_RETRIES,
    openrouter_transient_retry_delay_seconds: float = DEFAULT_OPENROUTER_TRANSIENT_RETRY_DELAY_SECONDS,
) -> tuple[TrialResult, bool]:
    result = run_trial(
        client=client,
        api=api,
        model=model,
        phase="preflight",
        evaluation_index=-1,
        center_length=preflight_length,
        trial_index=0,
        sample_low=preflight_length,
        sample_high=preflight_length,
        expected=preflight_length,
        max_output_tokens=max_output_tokens,
        retry_max_output_tokens=retry_max_output_tokens,
        sequence_format=sequence_format,
        user_prompt_factory=user_prompt_factory,
        reasoning_effort=reasoning_effort,
        google_unavailable_retries=google_unavailable_retries,
        google_unavailable_retry_delay_seconds=google_unavailable_retry_delay_seconds,
        anthropic_transient_retries=anthropic_transient_retries,
        anthropic_transient_retry_delay_seconds=anthropic_transient_retry_delay_seconds,
        openrouter_transient_retries=openrouter_transient_retries,
        openrouter_transient_retry_delay_seconds=openrouter_transient_retry_delay_seconds,
    )
    passed = result.parsed is not None and result.response_status != "error"
    return result, passed


def run_trial(
    client: Any,
    api: str,
    model: str,
    phase: str,
    evaluation_index: int,
    center_length: int,
    trial_index: int,
    sample_low: int,
    sample_high: int,
    expected: int,
    max_output_tokens: int,
    retry_max_output_tokens: int,
    sequence_format: SequenceFormatSpec = SequenceFormatSpec(),
    user_prompt_factory: Callable[[int], str] | None = None,
    reasoning_effort: str | None = None,
    google_unavailable_retries: int = DEFAULT_GOOGLE_UNAVAILABLE_RETRIES,
    google_unavailable_retry_delay_seconds: float = DEFAULT_GOOGLE_UNAVAILABLE_RETRY_DELAY_SECONDS,
    anthropic_transient_retries: int = DEFAULT_ANTHROPIC_TRANSIENT_RETRIES,
    anthropic_transient_retry_delay_seconds: float = DEFAULT_ANTHROPIC_TRANSIENT_RETRY_DELAY_SECONDS,
    openrouter_transient_retries: int = DEFAULT_OPENROUTER_TRANSIENT_RETRIES,
    openrouter_transient_retry_delay_seconds: float = DEFAULT_OPENROUTER_TRANSIENT_RETRY_DELAY_SECONDS,
) -> TrialResult:
    item_label = sequence_item_label(sequence_format.item, sequence_format.item_label)
    custom_prompt = user_prompt_factory(expected) if user_prompt_factory is not None else None
    sequence = "" if custom_prompt is not None else make_sequence(
        expected,
        item=sequence_format.item,
        delimiter=sequence_format.delimiter,
    )
    started = time.perf_counter()
    raw_response = ""
    response_status: str | None = None
    response_detail: str | None = None
    retry_count = 0
    final_max_output_tokens = max_output_tokens
    usage = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "cached_input_tokens": None,
    }

    if api == "chat":
        try:
            response = client.chat.completions.create(
                model=model,
                messages=(
                    build_custom_prompt(custom_prompt)
                    if custom_prompt is not None
                    else build_prompt(sequence, item_label=item_label)
                ),
                max_completion_tokens=max_output_tokens,
            )
            raw_response = response.choices[0].message.content or ""
            response_status = response.choices[0].finish_reason
            usage = extract_usage(response, api)
        except Exception as exc:
            response_status = "error"
            response_detail = f"{type(exc).__name__}: {exc}"
    elif api == "responses":
        token_budget = max_output_tokens
        while True:
            request_kwargs: dict[str, object] = {
                "model": model,
                "instructions": system_instruction(),
                "input": custom_prompt if custom_prompt is not None else user_prompt(sequence, item_label=item_label),
                "max_output_tokens": token_budget,
            }
            reasoning = reasoning_config(reasoning_effort)
            if reasoning is not None:
                request_kwargs["reasoning"] = reasoning
            final_max_output_tokens = token_budget
            try:
                response = client.responses.create(**request_kwargs)
            except Exception as exc:
                response_status = "error"
                response_detail = f"{type(exc).__name__}: {exc}"
                break
            raw_response = getattr(response, "output_text", "") or ""
            response_status = getattr(response, "status", None)
            usage = extract_usage(response, api)
            incomplete_details = getattr(response, "incomplete_details", None)
            response_detail = (
                getattr(incomplete_details, "reason", None)
                if incomplete_details is not None
                else None
            )
            if (
                response_status != "incomplete"
                or response_detail != "max_output_tokens"
                or parse_integer(raw_response) is not None
            ):
                break
            if token_budget >= retry_max_output_tokens:
                response_detail = f"{response_detail}|retry_limit_reached"
                break
            token_budget *= 2
            retry_count += 1
    elif api == "google_genai":
        try:
            from google.genai import types
        except ImportError as exc:
            response_status = "error"
            response_detail = f"{type(exc).__name__}: {exc}"
        else:
            token_budget = max_output_tokens
            use_developer_instruction = True
            while True:
                final_max_output_tokens = token_budget
                unavailable_attempt = 0
                while True:
                    try:
                        if use_developer_instruction:
                            request_kwargs = {
                                "model": model,
                                "contents": custom_prompt if custom_prompt is not None else user_prompt(sequence, item_label=item_label),
                                "config": types.GenerateContentConfig(
                                    system_instruction=system_instruction(),
                                    max_output_tokens=token_budget,
                                    response_mime_type="text/plain",
                                ),
                            }
                        else:
                            request_kwargs = {
                                "model": model,
                                "contents": (
                                    inline_google_custom_prompt(custom_prompt)
                                    if custom_prompt is not None
                                    else inline_google_prompt(sequence, item_label=item_label)
                                ),
                                "config": types.GenerateContentConfig(
                                    max_output_tokens=token_budget,
                                    response_mime_type="text/plain",
                                ),
                            }
                        response = client.models.generate_content(
                            **request_kwargs,
                        )
                        break
                    except Exception as exc:
                        if use_developer_instruction and is_google_developer_instruction_error(exc):
                            use_developer_instruction = False
                            continue
                        if (
                            is_google_transient_retry_error(exc)
                            and unavailable_attempt < google_unavailable_retries
                        ):
                            sleep_seconds = google_unavailable_retry_delay_seconds * (2 ** unavailable_attempt)
                            suggested_delay = google_retry_delay_seconds(exc)
                            if suggested_delay is not None:
                                sleep_seconds = max(sleep_seconds, suggested_delay)
                            time.sleep(min(30.0, sleep_seconds))
                            unavailable_attempt += 1
                            continue
                        response_status = "error"
                        response_detail = f"{type(exc).__name__}: {exc}"
                        break
                if response_status == "error":
                    break
                raw_response = getattr(response, "text", "") or ""
                usage = extract_usage(response, api)
                response_status, response_detail = google_response_status_and_detail(response)
                if (
                    response_status != "incomplete"
                    or response_detail != "max_output_tokens"
                    or parse_integer(raw_response) is not None
                ):
                    break
                if token_budget >= retry_max_output_tokens:
                    response_detail = f"{response_detail}|retry_limit_reached"
                    break
                token_budget *= 2
                retry_count += 1
    elif api == "anthropic":
        token_budget = max_output_tokens
        while True:
            final_max_output_tokens = token_budget
            transient_attempt = 0
            while True:
                try:
                    response = anthropic_request_json(
                        client,
                        "POST",
                        "/messages",
                        payload={
                            "model": model,
                            "system": system_instruction(),
                            "messages": [
                                {
                                    "role": "user",
                                    "content": (
                                        custom_prompt
                                        if custom_prompt is not None
                                        else user_prompt(sequence, item_label=item_label)
                                    ),
                                }
                            ],
                            "max_tokens": token_budget,
                        },
                    )
                    break
                except Exception as exc:
                    if (
                        is_anthropic_transient_retry_error(exc)
                        and transient_attempt < anthropic_transient_retries
                    ):
                        sleep_seconds = anthropic_transient_retry_delay_seconds * (2 ** transient_attempt)
                        suggested_delay = anthropic_retry_delay_seconds(exc)
                        if suggested_delay is not None:
                            sleep_seconds = max(sleep_seconds, suggested_delay)
                        time.sleep(min(10.0, sleep_seconds))
                        transient_attempt += 1
                        continue
                    response_status = "error"
                    response_detail = f"{type(exc).__name__}: {exc}"
                    break
            if response_status == "error":
                break
            raw_response = anthropic_text_from_message(response)
            usage = extract_usage(response, api)
            response_status, response_detail = anthropic_response_status_and_detail(response)
            if (
                response_status != "incomplete"
                or response_detail != "max_output_tokens"
                or parse_integer(raw_response) is not None
            ):
                break
            if token_budget >= retry_max_output_tokens:
                response_detail = f"{response_detail}|retry_limit_reached"
                break
            token_budget *= 2
            retry_count += 1
    else:
        token_budget = max_output_tokens
        while True:
            final_max_output_tokens = token_budget
            transient_attempt = 0
            while True:
                try:
                    response = openrouter_request_json(
                        client,
                        "POST",
                        "/chat/completions",
                        payload={
                            "model": model,
                            "messages": (
                                build_custom_prompt(custom_prompt)
                                if custom_prompt is not None
                                else build_prompt(sequence, item_label=item_label)
                            ),
                            "max_tokens": token_budget,
                            "temperature": 0,
                        },
                    )
                    break
                except Exception as exc:
                    if (
                        is_openrouter_transient_retry_error(exc)
                        and transient_attempt < openrouter_transient_retries
                    ):
                        sleep_seconds = openrouter_transient_retry_delay_seconds * (2 ** transient_attempt)
                        suggested_delay = openrouter_retry_delay_seconds(exc)
                        if suggested_delay is not None:
                            sleep_seconds = max(sleep_seconds, suggested_delay)
                        time.sleep(min(30.0, sleep_seconds))
                        transient_attempt += 1
                        continue
                    response_status = "error"
                    response_detail = f"{type(exc).__name__}: {exc}"
                    break
            if response_status == "error":
                break
            raw_response = openrouter_response_text(response)
            usage = extract_usage(response, api)
            response_status, response_detail = openrouter_response_status_and_detail(response)
            if (
                response_status != "incomplete"
                or response_detail != "max_output_tokens"
                or parse_integer(raw_response) is not None
            ):
                break
            if token_budget >= retry_max_output_tokens:
                response_detail = f"{response_detail}|retry_limit_reached"
                break
            token_budget *= 2
            retry_count += 1

    latency_seconds = time.perf_counter() - started
    parsed = parse_integer(raw_response)
    exact_match = parsed == expected

    if parsed is None:
        absolute_error = None
        squared_error = None
        relative_error = None
    else:
        error = parsed - expected
        absolute_error = abs(error)
        squared_error = float(error * error)
        relative_error = abs(error) / expected

    return TrialResult(
        api=api,
        model=model,
        phase=phase,
        evaluation_index=evaluation_index,
        center_length=center_length,
        trial_index=trial_index,
        sample_low=sample_low,
        sample_high=sample_high,
        expected=expected,
        raw_response=raw_response,
        response_status=response_status,
        response_detail=response_detail,
        parsed=parsed,
        exact_match=exact_match,
        absolute_error=absolute_error,
        squared_error=squared_error,
        relative_error=relative_error,
        latency_seconds=latency_seconds,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
        reasoning_tokens=usage["reasoning_tokens"],
        cached_input_tokens=usage["cached_input_tokens"],
        requested_max_output_tokens=max_output_tokens,
        final_max_output_tokens=final_max_output_tokens,
        retry_count=retry_count,
    )


def summarize_slice(results: list[TrialResult]) -> dict[str, object]:
    total = len(results)
    parsed_results = [result for result in results if result.parsed is not None]
    exact_matches = sum(result.exact_match for result in results)
    absolute_errors = [
        result.absolute_error for result in parsed_results if result.absolute_error is not None
    ]
    squared_errors = [
        result.squared_error for result in parsed_results if result.squared_error is not None
    ]
    relative_errors = [
        result.relative_error for result in parsed_results if result.relative_error is not None
    ]
    signed_errors = [
        float(result.parsed - result.expected)
        for result in parsed_results
        if result.parsed is not None
    ]
    latencies = [result.latency_seconds for result in results]
    input_token_values = [result.input_tokens for result in results if result.input_tokens is not None]
    output_token_values = [result.output_tokens for result in results if result.output_tokens is not None]
    total_token_values = [result.total_tokens for result in results if result.total_tokens is not None]
    reasoning_token_values = [
        result.reasoning_tokens for result in results if result.reasoning_tokens is not None
    ]
    cached_input_token_values = [
        result.cached_input_tokens
        for result in results
        if result.cached_input_tokens is not None
    ]

    return {
        "trials": total,
        "parse_rate": safe_divide(len(parsed_results), total),
        "success_rate": safe_divide(exact_matches, total),
        "mae": mean(absolute_errors),
        "mse": mean(squared_errors),
        "mape": mean(relative_errors),
        "mean_signed_error": mean(signed_errors),
        "avg_latency_seconds": mean(latencies),
        "avg_input_tokens": mean(input_token_values),
        "avg_output_tokens": mean(output_token_values),
        "avg_total_tokens": mean(total_token_values),
        "avg_reasoning_tokens": mean(reasoning_token_values),
        "avg_cached_input_tokens": mean(cached_input_token_values),
        "sum_input_tokens": sum(input_token_values) if input_token_values else 0,
        "sum_output_tokens": sum(output_token_values) if output_token_values else 0,
        "sum_total_tokens": sum(total_token_values) if total_token_values else 0,
        "sum_reasoning_tokens": sum(reasoning_token_values) if reasoning_token_values else 0,
        "sum_cached_input_tokens": (
            sum(cached_input_token_values) if cached_input_token_values else 0
        ),
    }


def passed_trial_records_from_payload(
    summary: dict[str, Any],
    trials: list[TrialResult],
) -> list[TrialResult]:
    passed_keys = {
        (str(item["phase"]), int(item["evaluation_index"]))
        for item in summary.get("evaluations", [])
        if item.get("passed")
    }
    return [
        trial
        for trial in trials
        if (str(trial.phase), int(trial.evaluation_index)) in passed_keys
    ]


def cc_trial_records_from_payload(
    summary: dict[str, Any],
    trials: list[TrialResult],
) -> list[TrialResult]:
    cc = int(summary.get("cc") or 0)
    if cc <= 0:
        return []
    matching_evaluations = [
        item
        for item in summary.get("evaluations", [])
        if item.get("passed")
        and int(item.get("center_length") or 0) == cc
    ]
    if not matching_evaluations:
        return []
    target = max(matching_evaluations, key=lambda item: int(item.get("evaluation_index") or -1))
    target_key = (str(target["phase"]), int(target["evaluation_index"]))
    return [
        trial
        for trial in trials
        if (str(trial.phase), int(trial.evaluation_index)) == target_key
    ]


def average_output_tokens_on_passed_trials(
    summary: dict[str, Any],
    trials: list[TrialResult],
) -> float | None:
    passed_trials = passed_trial_records_from_payload(summary, trials)
    values = [
        float(trial.output_tokens)
        for trial in passed_trials
        if trial.output_tokens is not None and float(trial.output_tokens) > 0
    ]
    return mean(values)


def average_output_tokens_at_cc(
    summary: dict[str, Any],
    trials: list[TrialResult],
) -> float | None:
    cc_trials = cc_trial_records_from_payload(summary, trials)
    values = [
        float(trial.output_tokens)
        for trial in cc_trials
        if trial.output_tokens is not None and float(trial.output_tokens) > 0
    ]
    return mean(values)


def average_total_tokens_at_cc(
    summary: dict[str, Any],
    trials: list[TrialResult],
) -> float | None:
    cc_trials = cc_trial_records_from_payload(summary, trials)
    values = [
        float(trial.total_tokens)
        for trial in cc_trials
        if trial.total_tokens is not None and float(trial.total_tokens) > 0
    ]
    return mean(values)


def trial_dict_is_token_limit_failure(trial: dict[str, Any]) -> bool:
    detail = str(trial.get("response_detail") or "")
    return trial.get("response_status") == "incomplete" and "max_output_tokens" in detail


def model_has_ladder_token_limit_issue(model_payload: dict[str, Any]) -> bool:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for trial in model_payload.get("trials", []):
        key = (str(trial["phase"]), int(trial["center_length"]))
        grouped.setdefault(key, []).append(trial)
    expansion_centers = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1])):
        phase, center_length = key
        if phase != "expansion":
            continue
        failures = sum(1 for trial in grouped[key] if trial_dict_is_token_limit_failure(trial))
        expansion_centers.append((center_length, failures))
    if not expansion_centers:
        return False
    first_center = expansion_centers[0][0]
    for center_length, failures in expansion_centers:
        if failures > 0:
            return center_length > first_center
    return False


def summarize_evaluation(
    center_length: int,
    phase: str,
    evaluation_index: int,
    sample_low: int,
    sample_high: int,
    trials: list[TrialResult],
    tolerance_fraction: float,
) -> dict[str, Any]:
    slice_summary = summarize_slice(trials)
    parse_rate = slice_summary["parse_rate"]
    mape = slice_summary["mape"]
    passed = parse_rate == 1.0 and mape is not None and mape <= tolerance_fraction
    return {
        "center_length": center_length,
        "phase": phase,
        "evaluation_index": evaluation_index,
        "sample_low": sample_low,
        "sample_high": sample_high,
        "passed": passed,
        **slice_summary,
    }


def should_abort_length_evaluation(result: TrialResult) -> bool:
    if result.response_status == "error" and result.retry_count > 0:
        return True
    detail = str(result.response_detail or "")
    return (
        result.response_status == "incomplete"
        and result.retry_count > 0
        and "retry_limit_reached" in detail
    )


def should_promote_output_budget(result: TrialResult, current_budget: int) -> bool:
    if result.final_max_output_tokens <= current_budget:
        return False
    if result.response_status == "error":
        return False
    if "retry_limit_reached" in str(result.response_detail or ""):
        return False
    return bool(result.parsed is not None or str(result.raw_response or "").strip())


def search_is_precise_enough(low: int, high: int, step_fraction: float) -> bool:
    if high <= low:
        return True
    midpoint = max(1, math.ceil((low + high) / 2))
    return (high - low) <= max(1, math.ceil(step_fraction * midpoint))


def evaluate_length(
    client: Any,
    api: str,
    model: str,
    phase: str,
    evaluation_index: int,
    center_length: int,
    rng: random.Random,
    samples_per_length: int,
    jitter_fraction: float,
    tolerance_fraction: float,
    max_output_tokens: int,
    retry_max_output_tokens: int,
    parallel_trials_per_batch: int,
    sequence_format: SequenceFormatSpec = SequenceFormatSpec(),
    user_prompt_factory: Callable[[int], str] | None = None,
    reasoning_effort: str | None = None,
    google_unavailable_retries: int = DEFAULT_GOOGLE_UNAVAILABLE_RETRIES,
    google_unavailable_retry_delay_seconds: float = DEFAULT_GOOGLE_UNAVAILABLE_RETRY_DELAY_SECONDS,
    anthropic_transient_retries: int = DEFAULT_ANTHROPIC_TRANSIENT_RETRIES,
    anthropic_transient_retry_delay_seconds: float = DEFAULT_ANTHROPIC_TRANSIENT_RETRY_DELAY_SECONDS,
    openrouter_transient_retries: int = DEFAULT_OPENROUTER_TRANSIENT_RETRIES,
    openrouter_transient_retry_delay_seconds: float = DEFAULT_OPENROUTER_TRANSIENT_RETRY_DELAY_SECONDS,
) -> tuple[list[TrialResult], dict[str, Any], int]:
    sample_low = max(1, math.floor(center_length * (1 - jitter_fraction)))
    sample_high = max(sample_low, math.ceil(center_length * (1 + jitter_fraction)))
    trial_results: list[TrialResult] = []
    current_max_output_tokens = max_output_tokens
    batch_size = max(1, parallel_trials_per_batch)
    expected_values = [
        rng.randint(sample_low, sample_high)
        for _ in range(samples_per_length)
    ]

    def execute_trial(trial_index: int, expected: int, budget: int) -> TrialResult:
        try:
            return run_trial(
                client=client,
                api=api,
                model=model,
                phase=phase,
                evaluation_index=evaluation_index,
                center_length=center_length,
                trial_index=trial_index,
                sample_low=sample_low,
                sample_high=sample_high,
                expected=expected,
                max_output_tokens=budget,
                retry_max_output_tokens=retry_max_output_tokens,
                sequence_format=sequence_format,
                user_prompt_factory=user_prompt_factory,
                reasoning_effort=reasoning_effort,
                google_unavailable_retries=google_unavailable_retries,
                google_unavailable_retry_delay_seconds=google_unavailable_retry_delay_seconds,
                anthropic_transient_retries=anthropic_transient_retries,
                anthropic_transient_retry_delay_seconds=anthropic_transient_retry_delay_seconds,
                openrouter_transient_retries=openrouter_transient_retries,
                openrouter_transient_retry_delay_seconds=openrouter_transient_retry_delay_seconds,
            )
        except Exception as exc:
            return TrialResult(
                api=api,
                model=model,
                phase=phase,
                evaluation_index=evaluation_index,
                center_length=center_length,
                trial_index=trial_index,
                sample_low=sample_low,
                sample_high=sample_high,
                expected=expected,
                raw_response=f"ERROR: {exc}",
                response_status="error",
                response_detail=type(exc).__name__,
                parsed=None,
                exact_match=False,
                absolute_error=None,
                squared_error=None,
                relative_error=None,
                latency_seconds=0.0,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                reasoning_tokens=None,
                cached_input_tokens=None,
                requested_max_output_tokens=budget,
                final_max_output_tokens=budget,
                retry_count=0,
            )

    if not expected_values:
        summary = summarize_evaluation(
            center_length=center_length,
            phase=phase,
            evaluation_index=evaluation_index,
            sample_low=sample_low,
            sample_high=sample_high,
            trials=trial_results,
            tolerance_fraction=tolerance_fraction,
        )
        return trial_results, summary, current_max_output_tokens

    first_result = execute_trial(0, expected_values[0], current_max_output_tokens)
    trial_results.append(first_result)
    if should_promote_output_budget(first_result, current_max_output_tokens):
        current_max_output_tokens = first_result.final_max_output_tokens
    if not should_abort_length_evaluation(first_result):
        remaining_indices = list(range(1, samples_per_length))
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for start in range(0, len(remaining_indices), batch_size):
                batch_indices = remaining_indices[start : start + batch_size]
                batch_budget = current_max_output_tokens
                futures = [
                    executor.submit(execute_trial, trial_index, expected_values[trial_index], batch_budget)
                    for trial_index in batch_indices
                ]
                batch_results = [future.result() for future in futures]
                batch_results.sort(key=lambda item: item.trial_index)
                trial_results.extend(batch_results)
                abort_batch = False
                for result in batch_results:
                    if should_promote_output_budget(result, current_max_output_tokens):
                        current_max_output_tokens = result.final_max_output_tokens
                    if should_abort_length_evaluation(result):
                        abort_batch = True
                if abort_batch:
                    break

    summary = summarize_evaluation(
        center_length=center_length,
        phase=phase,
        evaluation_index=evaluation_index,
        sample_low=sample_low,
        sample_high=sample_high,
        trials=trial_results,
        tolerance_fraction=tolerance_fraction,
    )
    return trial_results, summary, current_max_output_tokens


def search_counting_capacity(
    client: Any,
    model: str,
    api: str,
    search_config: dict[str, Any],
    max_output_tokens: int,
    retry_max_output_tokens: int,
    seed: int,
    sequence_format: SequenceFormatSpec = SequenceFormatSpec(),
    user_prompt_factory: Callable[[int], str] | None = None,
    reasoning_effort: str | None = None,
    google_unavailable_retries: int = DEFAULT_GOOGLE_UNAVAILABLE_RETRIES,
    google_unavailable_retry_delay_seconds: float = DEFAULT_GOOGLE_UNAVAILABLE_RETRY_DELAY_SECONDS,
    anthropic_transient_retries: int = DEFAULT_ANTHROPIC_TRANSIENT_RETRIES,
    anthropic_transient_retry_delay_seconds: float = DEFAULT_ANTHROPIC_TRANSIENT_RETRY_DELAY_SECONDS,
    openrouter_transient_retries: int = DEFAULT_OPENROUTER_TRANSIENT_RETRIES,
    openrouter_transient_retry_delay_seconds: float = DEFAULT_OPENROUTER_TRANSIENT_RETRY_DELAY_SECONDS,
) -> tuple[list[TrialResult], dict[str, Any]]:
    rng = random.Random(seed)
    initial_length = int(search_config["initial_length"])
    max_length = int(search_config["max_length"])
    samples_per_length = int(search_config["samples_per_length"])
    jitter_fraction = float(search_config["jitter_fraction"])
    tolerance_fraction = float(search_config["tolerance_fraction"])
    step_fraction = float(search_config["step_fraction"])
    parallel_trials_per_batch = int(
        search_config.get("parallel_trials_per_batch", DEFAULT_PARALLEL_TRIALS_PER_BATCH)
    )

    all_trials: list[TrialResult] = []
    evaluations: list[dict[str, Any]] = []
    evaluation_index = 0
    current_max_output_tokens = max_output_tokens

    low_pass: int | None = None
    first_fail: int | None = None
    center = initial_length

    while True:
        phase = "expansion"
        trials, evaluation, current_max_output_tokens = evaluate_length(
            client=client,
            api=api,
            model=model,
            phase=phase,
            evaluation_index=evaluation_index,
            center_length=center,
            rng=rng,
            samples_per_length=samples_per_length,
            jitter_fraction=jitter_fraction,
            tolerance_fraction=tolerance_fraction,
            max_output_tokens=current_max_output_tokens,
            retry_max_output_tokens=retry_max_output_tokens,
            parallel_trials_per_batch=parallel_trials_per_batch,
            sequence_format=sequence_format,
            user_prompt_factory=user_prompt_factory,
            reasoning_effort=reasoning_effort,
            google_unavailable_retries=google_unavailable_retries,
            google_unavailable_retry_delay_seconds=google_unavailable_retry_delay_seconds,
            anthropic_transient_retries=anthropic_transient_retries,
            anthropic_transient_retry_delay_seconds=anthropic_transient_retry_delay_seconds,
            openrouter_transient_retries=openrouter_transient_retries,
            openrouter_transient_retry_delay_seconds=openrouter_transient_retry_delay_seconds,
        )
        all_trials.extend(trials)
        evaluations.append(evaluation)
        evaluation_index += 1

        if evaluation["passed"]:
            low_pass = center
            if center >= max_length:
                break
            next_center = min(max_length, center * 2)
            if next_center == center:
                break
            center = next_center
            continue

        first_fail = center
        break

    if low_pass is None:
        low_pass = 0
        if first_fail is None:
            first_fail = initial_length

    if first_fail is not None and low_pass > 0:
        while not search_is_precise_enough(low_pass, first_fail, step_fraction):
            candidate = max(low_pass + 1, min(first_fail - 1, round((low_pass + first_fail) / 2)))
            trials, evaluation, current_max_output_tokens = evaluate_length(
                client=client,
                api=api,
                model=model,
                phase="refine",
                evaluation_index=evaluation_index,
                center_length=candidate,
                rng=rng,
                samples_per_length=samples_per_length,
                jitter_fraction=jitter_fraction,
                tolerance_fraction=tolerance_fraction,
                max_output_tokens=current_max_output_tokens,
                retry_max_output_tokens=retry_max_output_tokens,
                parallel_trials_per_batch=parallel_trials_per_batch,
                sequence_format=sequence_format,
                user_prompt_factory=user_prompt_factory,
                reasoning_effort=reasoning_effort,
                google_unavailable_retries=google_unavailable_retries,
                google_unavailable_retry_delay_seconds=google_unavailable_retry_delay_seconds,
                anthropic_transient_retries=anthropic_transient_retries,
                anthropic_transient_retry_delay_seconds=anthropic_transient_retry_delay_seconds,
                openrouter_transient_retries=openrouter_transient_retries,
                openrouter_transient_retry_delay_seconds=openrouter_transient_retry_delay_seconds,
            )
            all_trials.extend(trials)
            evaluations.append(evaluation)
            evaluation_index += 1
            if evaluation["passed"]:
                low_pass = candidate
            else:
                first_fail = candidate

    bounded = first_fail is not None
    cc = low_pass
    cc_upper_bound = first_fail
    lower_error = 0 if cc <= 0 else max(1, math.floor(cc * step_fraction))
    upper_error = (
        max(1, cc_upper_bound - cc)
        if cc_upper_bound is not None
        else max(1, math.ceil(cc * step_fraction))
    )

    summary = {
        "model": model,
        "cc": cc,
        "cc_lower_bound": cc,
        "cc_upper_bound": cc_upper_bound,
        "bounded": bounded,
        "search_step_fraction": step_fraction,
        "tolerance_fraction": tolerance_fraction,
        "jitter_fraction": jitter_fraction,
        "samples_per_length": samples_per_length,
        "max_length": max_length,
        "initial_length": initial_length,
        "evaluations": evaluations,
        "evaluation_count": len(evaluations),
        "trial_count": len(all_trials),
        "overall": summarize_slice(all_trials),
        "learned_max_output_tokens": current_max_output_tokens,
        "cc_error_bar": {
            "lower": lower_error,
            "upper": upper_error,
        },
    }
    return all_trials, summary


def print_model_summary(model_summary: dict[str, Any]) -> None:
    upper = model_summary["cc_upper_bound"]
    upper_text = str(upper) if upper is not None else "open"
    print(
        "  cc="
        f"{model_summary['cc']}"
        f"  bracket=[{model_summary['cc_lower_bound']}, {upper_text})"
        f"  trials={model_summary['trial_count']}"
        f"  evals={model_summary['evaluation_count']}"
        f"  mape={format_metric(model_summary['overall']['mape'])}"
        f"  parse={format_metric(model_summary['overall']['parse_rate'])}"
    )


def model_filename(model: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "__", model)
    return f"{safe_name}.json"


def initialize_run(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    requested_models: list[str],
    selected_models: list[str],
    already_completed_models: list[str],
    existing_model_summaries: dict[str, Any],
) -> RunPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_subdir(config)
    run_dir = persistent_run_dir(output_dir, config)
    models_dir = run_dir / "model_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "run_manifest.json"
    cc_json_path = run_dir / "scc_summary_table.json"
    cc_csv_path = run_dir / "scc_summary_table.csv"
    write_run_indexes(
        run_paths=RunPaths(
            run_id=run_id,
            timestamp=timestamp,
            run_dir=run_dir,
            models_dir=models_dir,
            manifest_path=manifest_path,
            cc_json_path=cc_json_path,
            cc_csv_path=cc_csv_path,
        ),
        config_path=config_path,
        config=config,
        requested_models=requested_models,
        selected_models=selected_models,
        already_completed_models=already_completed_models,
        model_summaries=existing_model_summaries,
        status="running",
    )
    return RunPaths(
        run_id=run_id,
        timestamp=timestamp,
        run_dir=run_dir,
        models_dir=models_dir,
        manifest_path=manifest_path,
        cc_json_path=cc_json_path,
        cc_csv_path=cc_csv_path,
    )


def write_model_output(
    run_paths: RunPaths,
    config_path: Path,
    config: dict[str, Any],
    model: str,
    summary: dict[str, Any],
    trials: list[TrialResult],
    model_metadata: dict[str, Any] | None = None,
    parameter_metadata: dict[str, Any] | None = None,
) -> Path:
    model_path = run_paths.models_dir / model_filename(model)
    model_payload = {
        "generated_at_utc": run_paths.timestamp,
        "run_id": run_paths.run_id,
        "config_path": str(config_path),
        "config": config,
        "model": model,
        "model_metadata": model_metadata or {},
        "parameter_metadata": parameter_metadata or {},
        "summary": summary,
        "trials": [asdict(trial) for trial in trials],
    }
    model_path.write_text(
        json.dumps(model_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model_path


def write_run_indexes(
    run_paths: RunPaths,
    config_path: Path,
    config: dict[str, Any],
    requested_models: list[str],
    selected_models: list[str],
    already_completed_models: list[str],
    model_summaries: dict[str, Any],
    status: str,
) -> None:
    model_files = {
        model: str((run_paths.models_dir / model_filename(model)).relative_to(run_paths.run_dir))
        for model in model_summaries
    }

    manifest_payload = {
        "generated_at_utc": run_paths.timestamp,
        "run_id": run_paths.run_id,
        "status": status,
        "config_path": str(config_path),
        "config": config,
        "requested_models": requested_models,
        "selected_models": selected_models,
        "already_completed_models": already_completed_models,
        "models": model_summaries,
        "model_files": model_files,
    }
    run_paths.manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cc_payload = {
        model: {
            "cc": summary["cc"],
            "cc_lower_bound": summary["cc_lower_bound"],
            "cc_upper_bound": summary["cc_upper_bound"],
            "cc_error_bar": summary["cc_error_bar"],
            "total_params_billion": summary.get("parameter_metadata", {}).get("total_params_billion"),
            "active_params_billion": summary.get("parameter_metadata", {}).get("active_params_billion"),
            "parameter_source": summary.get("parameter_metadata", {}).get("parameter_source"),
        }
        for model, summary in model_summaries.items()
    }
    run_paths.cc_json_path.write_text(
        json.dumps(cc_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ladder_models: list[str] = []
    with run_paths.cc_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "cc",
                "cc_lower_bound",
                "cc_upper_bound",
                "error_bar_lower",
                "error_bar_upper",
                "total_params_billion",
                "active_params_billion",
                "avg_output_tokens_at_cc",
                "notes",
            ],
        )
        writer.writeheader()
        for model, summary in model_summaries.items():
            avg_output_tokens_at_cc = None
            model_path = run_paths.models_dir / model_filename(model)
            if model_path.exists():
                try:
                    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
                    trial_objects = [
                        TrialResult(**trial_payload)
                        for trial_payload in model_payload.get("trials", [])
                    ]
                    avg_output_tokens_at_cc = average_output_tokens_on_passed_trials(summary, trial_objects)
                    if model_has_ladder_token_limit_issue(model_payload):
                        ladder_models.append(model)
                except Exception:
                    avg_output_tokens_at_cc = None
            writer.writerow(
                {
                    "model": model,
                    "cc": summary["cc"],
                    "cc_lower_bound": summary["cc_lower_bound"],
                    "cc_upper_bound": summary["cc_upper_bound"],
                    "error_bar_lower": summary["cc_error_bar"]["lower"],
                    "error_bar_upper": summary["cc_error_bar"]["upper"],
                    "total_params_billion": summary.get("parameter_metadata", {}).get("total_params_billion") or "",
                    "active_params_billion": summary.get("parameter_metadata", {}).get("active_params_billion") or "",
                    "avg_output_tokens_at_cc": (
                        f"{avg_output_tokens_at_cc:.4f}"
                        if avg_output_tokens_at_cc is not None
                        else ""
                    ),
                    "notes": "",
                }
            )
        if ladder_models:
            writer.writerow(
                {
                    "model": "__notes__",
                    "cc": "",
                    "cc_lower_bound": "",
                    "cc_upper_bound": "",
                    "error_bar_lower": "",
                    "error_bar_upper": "",
                    "total_params_billion": "",
                    "active_params_billion": "",
                    "avg_output_tokens_at_cc": "",
                    "notes": f"ladder_token_limit_models={','.join(sorted(ladder_models))}",
                }
            )


def main() -> int:
    args = parse_args()
    try:
        config = load_benchmark_config(args.config)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 1

    if args.api:
        config["api"] = args.api
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_output_tokens is not None:
        config["max_output_tokens"] = args.max_output_tokens
    configured_model_list_path = Path(str(config["model_list_path"]))
    if not configured_model_list_path.is_absolute():
        configured_model_list_path = args.config.parent / configured_model_list_path
    model_list_path = args.model_list or configured_model_list_path
    model_catalog_metadata = load_model_catalog_metadata(model_list_path)

    try:
        selected_models, already_completed_models, existing_model_paths = resolve_models(
            config=config,
            requested_models=args.models,
            model_list_path=model_list_path,
            results_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Failed to resolve models: {exc}", file=sys.stderr)
        return 1

    if args.list_models:
        requested_models = args.models or load_model_list(model_list_path)
        for model in requested_models:
            if model in existing_model_paths:
                print(f"{model}\talready_ran\t{existing_model_paths[model]}")
            else:
                print(f"{model}\tpending")
        return 0

    if not selected_models:
        requested_models = args.models or load_model_list(model_list_path)
        print("No pending models to run.")
        print(f"Requested models: {', '.join(requested_models)}")
        if already_completed_models:
            print(f"Already completed: {', '.join(already_completed_models)}")
        return 0

    try:
        client = init_client(str(config["api"]))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    requested_models = args.models or load_model_list(model_list_path)
    api = str(config["api"])
    print("Running SCC benchmark to measure CC")
    print(f"API: {config['api']}")
    print(f"Model list: {model_list_path}")
    print(f"Run subdir: {run_subdir(config)}")
    print(f"Requested: {', '.join(requested_models)}")
    print(f"Models: {', '.join(selected_models)}")
    if already_completed_models:
        print(f"Already completed: {', '.join(already_completed_models)}")
    print(
        "Search config:"
        f" initial={config['search']['initial_length']}"
        f" max={config['search']['max_length']}"
        f" samples={config['search']['samples_per_length']}"
        f" batch={config['search']['parallel_trials_per_batch']}"
        f" jitter={config['search']['jitter_fraction']}"
        f" tolerance={config['search']['tolerance_fraction']}"
        f" step={config['search']['step_fraction']}"
    )
    preflight_length = int(config.get("preflight_length", 8))
    retry_max_output_tokens = int(config.get("retry_max_output_tokens", DEFAULT_RETRY_MAX_OUTPUT_TOKENS))
    google_unavailable_retries = int(
        config.get("google_unavailable_retries", DEFAULT_GOOGLE_UNAVAILABLE_RETRIES)
    )
    google_unavailable_retry_delay_seconds = float(
        config.get(
            "google_unavailable_retry_delay_seconds",
            DEFAULT_GOOGLE_UNAVAILABLE_RETRY_DELAY_SECONDS,
        )
    )
    anthropic_transient_retries = int(
        config.get("anthropic_transient_retries", DEFAULT_ANTHROPIC_TRANSIENT_RETRIES)
    )
    anthropic_transient_retry_delay_seconds = float(
        config.get(
            "anthropic_transient_retry_delay_seconds",
            DEFAULT_ANTHROPIC_TRANSIENT_RETRY_DELAY_SECONDS,
        )
    )
    openrouter_transient_retries = int(
        config.get("openrouter_transient_retries", DEFAULT_OPENROUTER_TRANSIENT_RETRIES)
    )
    openrouter_transient_retry_delay_seconds = float(
        config.get(
            "openrouter_transient_retry_delay_seconds",
            DEFAULT_OPENROUTER_TRANSIENT_RETRY_DELAY_SECONDS,
        )
    )
    print(f"Preflight length: {preflight_length}")
    print(f"Retry max output tokens: {retry_max_output_tokens}")
    if api == "google_genai":
        print(
            "Google transient retry:"
            f" retries={google_unavailable_retries}"
            f" delay={google_unavailable_retry_delay_seconds:.1f}s"
        )
    if api == "anthropic":
        print(
            "Anthropic transient retry:"
            f" retries={anthropic_transient_retries}"
            f" delay={anthropic_transient_retry_delay_seconds:.1f}s"
        )
    if api == "openrouter":
        print(
            "OpenRouter transient retry:"
            f" retries={openrouter_transient_retries}"
            f" delay={openrouter_transient_retry_delay_seconds:.1f}s"
        )

    model_summaries = load_existing_model_summaries(existing_model_paths)
    max_output_tokens = int(config["max_output_tokens"])
    base_seed = int(config["seed"])
    run_paths = initialize_run(
        output_dir=args.output_dir,
        config_path=args.config,
        config=config,
        requested_models=requested_models,
        selected_models=selected_models,
        already_completed_models=already_completed_models,
        existing_model_summaries=model_summaries,
    )

    for model_index, model in enumerate(selected_models):
        print(f"\n[{model}]")
        model_metadata = model_catalog_metadata.get(model, {})
        parameter_metadata = infer_open_source_parameter_metadata(model, model_metadata)
        preflight_result, preflight_passed = run_preflight_check(
            client=client,
            model=model,
            api=api,
            max_output_tokens=max_output_tokens,
            retry_max_output_tokens=retry_max_output_tokens,
            preflight_length=preflight_length,
            google_unavailable_retries=google_unavailable_retries,
            google_unavailable_retry_delay_seconds=google_unavailable_retry_delay_seconds,
            anthropic_transient_retries=anthropic_transient_retries,
            anthropic_transient_retry_delay_seconds=anthropic_transient_retry_delay_seconds,
            openrouter_transient_retries=openrouter_transient_retries,
            openrouter_transient_retry_delay_seconds=openrouter_transient_retry_delay_seconds,
        )
        current_model_trials = [preflight_result]
        if not preflight_passed:
            model_summaries[model] = {
                "model": model,
                "skipped": True,
                "skip_reason": "preflight_failed",
                "preflight_result": asdict(preflight_result),
                "cc": 0,
                "cc_lower_bound": 0,
                "cc_upper_bound": int(config["search"]["initial_length"]),
                "cc_error_bar": {"lower": 0, "upper": int(config["search"]["initial_length"] * config["search"]["step_fraction"])},
                "evaluation_count": 0,
                "trial_count": 1,
                "overall": summarize_slice([preflight_result]),
                "model_metadata": model_metadata,
                "parameter_metadata": parameter_metadata,
            }
            print(
                "  preflight_failed"
                f" status={preflight_result.response_status}"
                f" detail={preflight_result.response_detail}"
                f" raw={preflight_result.raw_response!r}"
            )
            write_model_output(
                run_paths=run_paths,
                config_path=args.config,
                config=config,
                model=model,
                summary=model_summaries[model],
                trials=current_model_trials,
                model_metadata=model_metadata,
                parameter_metadata=parameter_metadata,
            )
            write_run_indexes(
                run_paths=run_paths,
                config_path=args.config,
                config=config,
                requested_models=requested_models,
                selected_models=selected_models,
                already_completed_models=already_completed_models,
                model_summaries=model_summaries,
                status="running",
            )
            continue
        learned_max_output_tokens = max_output_tokens
        if should_promote_output_budget(preflight_result, learned_max_output_tokens):
            learned_max_output_tokens = preflight_result.final_max_output_tokens
        trials, model_summary = search_counting_capacity(
            client=client,
            model=model,
            api=api,
            search_config=config["search"],
            max_output_tokens=learned_max_output_tokens,
            retry_max_output_tokens=retry_max_output_tokens,
            seed=base_seed + model_index,
            google_unavailable_retries=google_unavailable_retries,
            google_unavailable_retry_delay_seconds=google_unavailable_retry_delay_seconds,
            anthropic_transient_retries=anthropic_transient_retries,
            anthropic_transient_retry_delay_seconds=anthropic_transient_retry_delay_seconds,
            openrouter_transient_retries=openrouter_transient_retries,
            openrouter_transient_retry_delay_seconds=openrouter_transient_retry_delay_seconds,
        )
        current_model_trials.extend(trials)
        model_summary["skipped"] = False
        model_summary["preflight_result"] = asdict(preflight_result)
        model_summary["model_metadata"] = model_metadata
        model_summary["parameter_metadata"] = parameter_metadata
        model_summaries[model] = model_summary
        write_model_output(
            run_paths=run_paths,
            config_path=args.config,
            config=config,
            model=model,
            summary=model_summary,
            trials=current_model_trials,
            model_metadata=model_metadata,
            parameter_metadata=parameter_metadata,
        )
        write_run_indexes(
            run_paths=run_paths,
            config_path=args.config,
            config=config,
            requested_models=requested_models,
            selected_models=selected_models,
            already_completed_models=already_completed_models,
            model_summaries=model_summaries,
            status="running",
        )
        print_model_summary(model_summary)

    write_run_indexes(
        run_paths=run_paths,
        config_path=args.config,
        config=config,
        requested_models=requested_models,
        selected_models=selected_models,
        already_completed_models=already_completed_models,
        model_summaries=model_summaries,
        status="completed",
    )

    print(f"\nRun directory: {run_paths.run_dir}")
    print(f"Manifest: {run_paths.manifest_path}")
    print(f"CC JSON: {run_paths.cc_json_path}")
    print(f"CC CSV: {run_paths.cc_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
