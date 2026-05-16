#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import string
import sys
import time
from typing import Any

from run_scc_benchmark import (
    DEFAULT_ANTHROPIC_TRANSIENT_RETRIES,
    DEFAULT_ANTHROPIC_TRANSIENT_RETRY_DELAY_SECONDS,
    DEFAULT_GOOGLE_UNAVAILABLE_RETRIES,
    DEFAULT_GOOGLE_UNAVAILABLE_RETRY_DELAY_SECONDS,
    DEFAULT_OPENROUTER_TRANSIENT_RETRIES,
    DEFAULT_OPENROUTER_TRANSIENT_RETRY_DELAY_SECONDS,
    anthropic_request_json,
    anthropic_response_status_and_detail,
    anthropic_retry_delay_seconds,
    anthropic_text_from_message,
    extract_usage,
    google_response_status_and_detail,
    google_retry_delay_seconds,
    init_client,
    is_anthropic_transient_retry_error,
    is_google_developer_instruction_error,
    is_google_transient_retry_error,
    is_openrouter_transient_retry_error,
    openrouter_request_json,
    openrouter_response_status_and_detail,
    openrouter_response_text,
    openrouter_retry_delay_seconds,
    reasoning_config,
)


SYSTEM_INSTRUCTION = (
    "You are being evaluated. Answer the requested task exactly. "
    "Return only a single JSON object with the requested keys. "
    "Do not include markdown, comments, hidden reasoning, or extra text."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    category: str
    answer_type: str
    prompt: str
    answer: str
    source: str
    metadata: dict[str, Any]


def load_dataset_rows(path: str, split: str, name: str | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face datasets package is required. Install with: "
            "python3 -m pip install datasets"
        ) from exc
    dataset = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
    return [dict(row) for row in dataset]


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_loose(value: Any) -> str:
    text = normalize_space(value).lower()
    return text.strip(string.whitespace + string.punctuation)


def normalize_math_text(value: Any) -> str:
    text = normalize_space(value)
    text = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    return text.strip(".")


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = text.strip()
    if not stripped:
        return None, "empty_response"

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    candidates: list[str] = []
    if fence:
        candidates.append(fence.group(1))
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    for start, char in enumerate(stripped):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for index, char in enumerate(stripped[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start : index + 1])
                    break

    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
        else:
            if isinstance(parsed, dict):
                return parsed, None
            errors.append(f"json_root_{type(parsed).__name__}")
            continue

        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed, None
        errors.append(f"literal_root_{type(parsed).__name__}")
    return None, "; ".join(errors) if errors else "no_json_object"


def extract_letter(value: Any, max_options: int = 10) -> str | None:
    text = normalize_space(value).upper()
    valid = set(LETTERS[:max_options])
    if text in valid:
        return text
    patterns = [
        r"\bANSWER\s*(?:IS|:)?\s*\(?([A-Z])\)?\b",
        r"\bOPTION\s*\(?([A-Z])\)?\b",
        r"^\(?([A-Z])\)?[.)]?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1) in valid:
            return match.group(1)
    matches = [match for match in re.findall(r"\b([A-Z])\b", text) if match in valid]
    return matches[-1] if matches else None


def maybe_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def score_math(prediction: Any, gold: str) -> bool:
    pred_text = str(prediction)
    try:
        from math_verify import parse, verify
    except ImportError:
        return normalize_math_text(pred_text) == normalize_math_text(gold)
    try:
        return bool(verify(parse(gold), parse(pred_text)))
    except Exception:
        return normalize_math_text(pred_text) == normalize_math_text(gold)


def score_answer(answer_type: str, prediction: Any, gold: str, metadata: dict[str, Any]) -> bool:
    if prediction is None:
        return False
    if answer_type == "mcq_letter":
        return extract_letter(prediction, int(metadata.get("option_count", 10))) == str(gold).upper()
    if answer_type == "math":
        return score_math(prediction, gold)
    if answer_type == "python_literal":
        pred_literal = maybe_literal(prediction)
        gold_literal = maybe_literal(gold)
        if not isinstance(pred_literal, str) or not isinstance(gold_literal, str):
            return pred_literal == gold_literal
        return normalize_space(pred_literal) == normalize_space(gold_literal)
    return normalize_loose(prediction) == normalize_loose(gold)


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
    except ImportError:
        return max(1, math.ceil(len(text) / 4))
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


def make_count_sequence(count: int) -> str:
    return ", ".join("a" for _ in range(max(0, count)))


def make_marker_sequence(count: int, marker: str) -> str:
    return ", ".join(marker for _ in range(max(0, count)))


def count_for_target_tokens(target_tokens: int) -> int:
    target_tokens = max(1, int(target_tokens))
    low = 1
    high = 2
    while estimate_tokens(make_count_sequence(high)) < target_tokens:
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if estimate_tokens(make_count_sequence(mid)) < target_tokens:
            low = mid + 1
        else:
            high = mid
    return low


def repeated_to_target_tokens(builder: Any, target_tokens: int) -> str:
    target_tokens = max(1, int(target_tokens))
    units = 1
    text = builder(units)
    while estimate_tokens(text) < target_tokens:
        units *= 2
        text = builder(units)
    low = 1
    high = units
    while low < high:
        mid = (low + high) // 2
        text = builder(mid)
        if estimate_tokens(text) < target_tokens:
            low = mid + 1
        else:
            high = mid
    return builder(low)


def make_irrelevant_code(target_tokens: int, seed: str) -> str:
    names = ["alpha", "beta", "gamma", "delta", "theta", "omega", "sigma", "kappa"]

    def builder(line_count: int) -> str:
        rng = random.Random(f"{seed}:code:{line_count}")
        lines = [
            "def unused_transform(values):",
            "    total = 0",
        ]
        for index in range(line_count):
            name = rng.choice(names)
            value = rng.randint(2, 97)
            lines.append(f"    {name}_{index} = ({value} * {index + 3}) % {value + 11}")
            lines.append(f"    total += {name}_{index}")
        lines.append("    return total")
        return "\n".join(lines)

    return repeated_to_target_tokens(builder, target_tokens)


def format_mmlu_prompt(row: dict[str, Any]) -> str:
    options = list(row["options"])
    lines = [
        "Answer the following multiple-choice question. Return the option letter only in JSON.",
        "",
        f"Question: {row['question']}",
        "",
        "Options:",
    ]
    for index, option in enumerate(options):
        lines.append(f"{LETTERS[index]}. {option}")
    return "\n".join(lines)


def build_mmlu_items(config: dict[str, Any]) -> list[EvalItem]:
    rows = load_dataset_rows(config["path"], config.get("split", "test"))
    excluded = {str(item).lower() for item in config.get("exclude_categories", [])}
    items: list[EvalItem] = []
    for row in rows:
        category = str(row.get("category", "")).lower()
        if category in excluded:
            continue
        options = list(row["options"])
        answer = str(row.get("answer") or "").strip().upper()
        if not answer and row.get("answer_index") is not None:
            answer = LETTERS[int(row["answer_index"])]
        items.append(
            EvalItem(
                item_id=f"mmlu_pro:{row.get('question_id', len(items))}",
                category="mmlu_pro",
                answer_type="mcq_letter",
                prompt=format_mmlu_prompt(row),
                answer=answer,
                source=config["path"],
                metadata={
                    "question_id": row.get("question_id"),
                    "subcategory": row.get("category"),
                    "src": row.get("src"),
                    "option_count": len(options),
                },
            )
        )
    return items


def build_math500_items(config: dict[str, Any]) -> list[EvalItem]:
    rows = load_dataset_rows(config["path"], config.get("split", "test"))
    items: list[EvalItem] = []
    for row in rows:
        prompt = (
            "Solve the following math problem. Return only the final answer in JSON.\n\n"
            f"Problem: {row['problem']}"
        )
        items.append(
            EvalItem(
                item_id=f"math500:{row.get('unique_id', len(items))}",
                category="math500",
                answer_type="math",
                prompt=prompt,
                answer=str(row["answer"]),
                source=config["path"],
                metadata={
                    "unique_id": row.get("unique_id"),
                    "subject": row.get("subject"),
                    "level": row.get("level"),
                },
            )
        )
    return items


def build_cruxeval_items(config: dict[str, Any]) -> list[EvalItem]:
    rows = load_dataset_rows(config["path"], config.get("split", "test"))
    items: list[EvalItem] = []
    for row in rows:
        prompt = (
            "Predict the exact Python return value for this function call. "
            "Return a Python literal as a JSON string.\n\n"
            f"Code:\n{row['code']}\n\n"
            f"Call:\nf({row['input']})"
        )
        items.append(
            EvalItem(
                item_id=f"cruxeval_o:{row.get('id', len(items))}",
                category="cruxeval_o",
                answer_type="python_literal",
                prompt=prompt,
                answer=str(row["output"]),
                source=config["path"],
                metadata={"id": row.get("id")},
            )
        )
    return items


def build_bbh_items(config: dict[str, Any]) -> list[EvalItem]:
    tasks = list(config.get("tasks") or ["date_understanding"])
    items: list[EvalItem] = []
    for task in tasks:
        for index, row in enumerate(load_dataset_rows(config["path"], config.get("split", "test"), task)):
            prompt = (
                "Answer the following reasoning problem. Return only the final answer in JSON.\n\n"
                f"{row.get('input', row.get('question'))}"
            )
            items.append(
                EvalItem(
                    item_id=f"bbh_reasoning:{task}:{index}",
                    category="bbh_reasoning",
                    answer_type="short_text",
                    prompt=prompt,
                    answer=str(row.get("target", "")),
                    source=f"{config['path']}:{task}",
                    metadata={"task": task, "index": index},
                )
            )
    return items


def build_all_items(config: dict[str, Any]) -> dict[str, list[EvalItem]]:
    dataset_config = config["datasets"]
    builders = {
        "mmlu_pro": build_mmlu_items,
        "math500": build_math500_items,
        "cruxeval_o": build_cruxeval_items,
        "bbh_reasoning": build_bbh_items,
    }
    by_category: dict[str, list[EvalItem]] = {}
    for category, builder in builders.items():
        if category not in dataset_config:
            continue
        by_category[category] = builder(dataset_config[category])
    return by_category


def sample_items(
    by_category: dict[str, list[EvalItem]], sample_size: int, seed: int
) -> dict[str, list[EvalItem]]:
    sampled: dict[str, list[EvalItem]] = {}
    for category, items in sorted(by_category.items()):
        if len(items) < sample_size:
            raise ValueError(f"Category {category} has {len(items)} items, need {sample_size}")
        rng = random.Random(f"{seed}:{category}:manifest")
        selected = rng.sample(items, sample_size)
        selected.sort(key=lambda item: item.item_id)
        sampled[category] = selected
    return sampled


def write_manifest(run_dir: Path, sampled: dict[str, list[EvalItem]], config: dict[str, Any]) -> None:
    manifest_dir = run_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for category, items in sorted(sampled.items()):
        rows = [asdict(item) for item in items]
        (manifest_dir / f"{category}.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "seed": config["seed"],
        "sample_size_per_category": config["sample_size_per_category"],
        "categories": {category: [item.item_id for item in items] for category, items in sampled.items()},
    }
    (manifest_dir / "selected_item_ids.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sleep_capped(seconds: float, cap: float = 30.0) -> None:
    time.sleep(min(cap, max(0.0, seconds)))


def generate_response(
    client: Any,
    api: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    retry_max_output_tokens: int,
    config: dict[str, Any],
    reasoning_effort: str | None = None,
) -> tuple[str, str | None, str | None, dict[str, int | None], int, int]:
    token_budget = max_output_tokens
    retry_count = 0
    usage = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "cached_input_tokens": None,
    }

    while True:
        final_max_output_tokens = token_budget
        raw_response = ""
        response_status: str | None = None
        response_detail: str | None = None

        if api == "responses":
            request_kwargs: dict[str, Any] = {
                "model": model,
                "instructions": SYSTEM_INSTRUCTION,
                "input": prompt,
                "max_output_tokens": token_budget,
            }
            reasoning = reasoning_config(reasoning_effort)
            if reasoning is not None:
                request_kwargs["reasoning"] = reasoning
            response = client.responses.create(**request_kwargs)
            raw_response = getattr(response, "output_text", "") or ""
            response_status = getattr(response, "status", None)
            incomplete_details = getattr(response, "incomplete_details", None)
            response_detail = (
                getattr(incomplete_details, "reason", None)
                if incomplete_details is not None
                else None
            )
            usage = extract_usage(response, api)

        elif api == "chat":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=token_budget,
            )
            raw_response = response.choices[0].message.content or ""
            response_status = response.choices[0].finish_reason
            response_detail = None
            usage = extract_usage(response, api)

        elif api == "google_genai":
            try:
                from google.genai import types
            except ImportError as exc:
                raise RuntimeError("google-genai is required for api=google_genai") from exc
            use_developer_instruction = True
            transient_attempt = 0
            while True:
                try:
                    if use_developer_instruction:
                        response = client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_INSTRUCTION,
                                max_output_tokens=token_budget,
                                response_mime_type="application/json",
                            ),
                        )
                    else:
                        response = client.models.generate_content(
                            model=model,
                            contents=f"{SYSTEM_INSTRUCTION}\n\n{prompt}",
                            config=types.GenerateContentConfig(
                                max_output_tokens=token_budget,
                                response_mime_type="application/json",
                            ),
                        )
                    break
                except Exception as exc:
                    if use_developer_instruction and is_google_developer_instruction_error(exc):
                        use_developer_instruction = False
                        continue
                    if (
                        is_google_transient_retry_error(exc)
                        and transient_attempt < int(config.get("google_unavailable_retries", DEFAULT_GOOGLE_UNAVAILABLE_RETRIES))
                    ):
                        sleep_seconds = float(config.get("google_unavailable_retry_delay_seconds", DEFAULT_GOOGLE_UNAVAILABLE_RETRY_DELAY_SECONDS)) * (2 ** transient_attempt)
                        suggested = google_retry_delay_seconds(exc)
                        sleep_capped(max(sleep_seconds, suggested or 0.0))
                        transient_attempt += 1
                        continue
                    raise
            raw_response = getattr(response, "text", "") or ""
            response_status, response_detail = google_response_status_and_detail(response)
            usage = extract_usage(response, api)

        elif api == "anthropic":
            transient_attempt = 0
            while True:
                try:
                    response = anthropic_request_json(
                        client,
                        "POST",
                        "/messages",
                        payload={
                            "model": model,
                            "system": SYSTEM_INSTRUCTION,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": token_budget,
                        },
                    )
                    break
                except Exception as exc:
                    if (
                        is_anthropic_transient_retry_error(exc)
                        and transient_attempt < int(config.get("anthropic_transient_retries", DEFAULT_ANTHROPIC_TRANSIENT_RETRIES))
                    ):
                        sleep_seconds = float(config.get("anthropic_transient_retry_delay_seconds", DEFAULT_ANTHROPIC_TRANSIENT_RETRY_DELAY_SECONDS)) * (2 ** transient_attempt)
                        suggested = anthropic_retry_delay_seconds(exc)
                        sleep_capped(max(sleep_seconds, suggested or 0.0), cap=10.0)
                        transient_attempt += 1
                        continue
                    raise
            raw_response = anthropic_text_from_message(response)
            response_status, response_detail = anthropic_response_status_and_detail(response)
            usage = extract_usage(response, api)

        elif api == "openrouter":
            transient_attempt = 0
            while True:
                try:
                    response = openrouter_request_json(
                        client,
                        "POST",
                        "/chat/completions",
                        payload={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": SYSTEM_INSTRUCTION},
                                {"role": "user", "content": prompt},
                            ],
                            "max_tokens": token_budget,
                            "temperature": 0,
                        },
                    )
                    break
                except Exception as exc:
                    if (
                        is_openrouter_transient_retry_error(exc)
                        and transient_attempt < int(config.get("openrouter_transient_retries", DEFAULT_OPENROUTER_TRANSIENT_RETRIES))
                    ):
                        sleep_seconds = float(config.get("openrouter_transient_retry_delay_seconds", DEFAULT_OPENROUTER_TRANSIENT_RETRY_DELAY_SECONDS)) * (2 ** transient_attempt)
                        suggested = openrouter_retry_delay_seconds(exc)
                        sleep_capped(max(sleep_seconds, suggested or 0.0))
                        transient_attempt += 1
                        continue
                    raise
            raw_response = openrouter_response_text(response)
            response_status, response_detail = openrouter_response_status_and_detail(response)
            usage = extract_usage(response, api)
        else:
            raise ValueError(f"Unsupported api: {api}")

        if (
            response_status == "incomplete"
            and response_detail == "max_output_tokens"
            and token_budget < retry_max_output_tokens
        ):
            token_budget = min(retry_max_output_tokens, token_budget * 2)
            retry_count += 1
            continue
        return raw_response, response_status, response_detail, usage, final_max_output_tokens, retry_count


def selected_cases(config: dict[str, Any], model_filter: set[str] | None) -> list[dict[str, Any]]:
    cases = []
    for case in config["cases"]:
        model = str(case.get("model") or "")
        if not model:
            raise ValueError(f"Case missing model: {case!r}")
        if model_filter and model not in model_filter:
            continue
        if not str(case.get("api") or ""):
            raise ValueError(f"Case missing api: {case!r}")
        cases.append(dict(case))
    return cases
