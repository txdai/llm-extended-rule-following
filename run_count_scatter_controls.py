#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

from run_scc_benchmark import init_client
from count_scatter_control_common import (
    EvalItem,
    build_all_items,
    count_for_target_tokens,
    estimate_tokens,
    extract_json_object,
    generate_response,
    make_irrelevant_code,
    make_marker_sequence,
    sample_items,
    score_answer,
    selected_cases,
    write_manifest,
)


DEFAULT_CONFIG_PATH = Path("configs/benchmarks/count_scatter_control.json")


@dataclass
class CountScatterRecord:
    run_id: str
    model: str
    api: str
    condition: str
    category: str
    source_category: str
    item_id: str
    target_count: int
    parsed_count: int | None
    count_error: int | None
    count_abs_error: int | None
    count_correct: bool
    secondary_target_count: int | None
    parsed_secondary_count: int | None
    secondary_correct: bool | None
    task_correct: bool | None
    matched_prompt_tokens: int
    distractor_estimated_tokens: int
    prompt_estimated_tokens: int
    raw_response: str
    response_status: str | None
    response_detail: str | None
    parsed_json: dict[str, Any] | None
    parse_error: str | None
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
        description="Matched dual-task count-scatter control sweep."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--models", nargs="+", help="Optional model filter.")
    parser.add_argument("--count-start", type=int, help="Override inclusive start count.")
    parser.add_argument("--count-end", type=int, help="Override inclusive end count.")
    parser.add_argument("--samples-per-count", type=int, help="Override samples per count per category.")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ["run_subdir", "seed", "sample_size_per_category", "count_start", "count_end", "cases", "datasets"]:
        if key not in payload:
            raise ValueError(f"Config {path} is missing {key!r}")
    return payload


def real_task_count_prompt(item: EvalItem, target_count: int) -> str:
    return (
        "You must answer the task and count the marker sequence. Return JSON with exactly this schema: "
        "{\"task_answer\": <answer to Task>, \"count\": <number of a markers>}.\n\n"
        f"Task:\n{item.prompt}\n\n"
        "Marker sequence:\n"
        f"{make_marker_sequence(target_count, 'a')}"
    )


def plain_count_prompt(target_count: int) -> str:
    return (
        "Count the marker sequence. Return JSON with exactly this schema: "
        "{\"count\": <number of a markers>}.\n\n"
        "Marker sequence:\n"
        f"{make_marker_sequence(target_count, 'a')}"
    )


def control_count_prompt(
    control_type: str,
    item: EvalItem,
    target_count: int,
    seed: str,
) -> tuple[str, int | None, int]:
    matched_tokens = estimate_tokens(item.prompt)
    primary = make_marker_sequence(target_count, "a")
    secondary_count: int | None = None

    if control_type == "irrelevant_code":
        distractor = make_irrelevant_code(matched_tokens, seed)
        schema = "{\"count\": <number of a markers>}"
        instruction = "Count only the a markers in the marker sequence. Ignore the unrelated code."
        label = "Unrelated code"
    elif control_type == "dual_counting":
        secondary_count = count_for_target_tokens(matched_tokens)
        distractor = make_marker_sequence(secondary_count, "b")
        schema = "{\"count\": <number of a markers>, \"secondary_count\": <number of b markers>}"
        instruction = "Count both marker sequences exactly."
        label = "Secondary marker sequence"
    else:
        raise ValueError(f"Unsupported control type: {control_type}")

    prompt = (
        f"{instruction} Return JSON with exactly this schema: {schema}.\n\n"
        f"Primary marker sequence:\n{primary}\n\n"
        f"{label} matched to the sampled benchmark question length:\n{distractor}"
    )
    return prompt, secondary_count, estimate_tokens(distractor)


def build_specs(sampled: dict[str, list[EvalItem]], config: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(config["seed"])
    count_start = int(config["count_start"])
    count_end = int(config["count_end"])
    if count_end < count_start:
        raise ValueError("count_end must be >= count_start")
    samples_per_count = int(config.get("samples_per_count_per_category", 6))
    control_samples_per_type = int(config.get("control_samples_per_count_per_type", samples_per_count))
    pool_controls = bool(config.get("pool_controls_across_categories", True))
    controls = list(config.get("control_types") or ["irrelevant_code", "dual_counting"])

    specs: list[dict[str, Any]] = []
    for target_count in range(count_start, count_end + 1):
        for sample_index in range(control_samples_per_type):
            specs.append(
                {
                    "condition": "plain_count",
                    "category": "plain_count",
                    "source_category": "",
                    "item": None,
                    "target_count": target_count,
                    "secondary_target_count": None,
                    "matched_prompt_tokens": 0,
                    "distractor_estimated_tokens": 0,
                    "prompt": plain_count_prompt(target_count),
                    "sample_index": sample_index,
                }
            )

        selected_by_category: dict[str, list[EvalItem]] = {}
        pooled_real_items: list[EvalItem] = []
        for category, items in sorted(sampled.items()):
            rng = random.Random(f"{seed}:scatter:{category}:{target_count}")
            selected = rng.sample(items, min(samples_per_count, len(items)))
            selected_by_category[category] = selected
            pooled_real_items.extend(selected)
            for offset, item in enumerate(selected):
                real_prompt = real_task_count_prompt(item, target_count)
                specs.append(
                    {
                        "condition": "real_task",
                        "category": category,
                        "source_category": category,
                        "item": item,
                        "target_count": target_count,
                        "secondary_target_count": None,
                        "matched_prompt_tokens": estimate_tokens(item.prompt),
                        "distractor_estimated_tokens": estimate_tokens(item.prompt),
                        "prompt": real_prompt,
                    }
                )

        if pool_controls:
            control_pool = pooled_real_items
            for control_type in controls:
                rng = random.Random(f"{seed}:pooled-control:{target_count}:{control_type}")
                if len(control_pool) >= control_samples_per_type:
                    selected_controls = rng.sample(control_pool, control_samples_per_type)
                else:
                    selected_controls = [rng.choice(control_pool) for _ in range(control_samples_per_type)]
                for offset, item in enumerate(selected_controls):
                    prompt, secondary_count, distractor_tokens = control_count_prompt(
                        control_type=control_type,
                        item=item,
                        target_count=target_count,
                        seed=f"{seed}:{target_count}:pooled:{control_type}:{item.item_id}:{offset}",
                    )
                    specs.append(
                        {
                            "condition": control_type,
                            "category": "pooled_control",
                            "source_category": item.category,
                            "item": item,
                            "target_count": target_count,
                            "secondary_target_count": secondary_count,
                            "matched_prompt_tokens": estimate_tokens(item.prompt),
                            "distractor_estimated_tokens": distractor_tokens,
                            "prompt": prompt,
                        }
                    )
            continue

        for category, selected in sorted(selected_by_category.items()):
            for offset, item in enumerate(selected):
                for control_type in controls:
                    prompt, secondary_count, distractor_tokens = control_count_prompt(
                        control_type=control_type,
                        item=item,
                        target_count=target_count,
                        seed=f"{seed}:{target_count}:{category}:{control_type}:{item.item_id}:{offset}",
                    )
                    specs.append(
                        {
                            "condition": control_type,
                            "category": category,
                            "source_category": category,
                            "item": item,
                            "target_count": target_count,
                            "secondary_target_count": secondary_count,
                            "matched_prompt_tokens": estimate_tokens(item.prompt),
                            "distractor_estimated_tokens": distractor_tokens,
                            "prompt": prompt,
                        }
                    )
    return specs


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_one(
    run_id: str,
    client: Any,
    case: dict[str, Any],
    spec: dict[str, Any],
    config: dict[str, Any],
) -> CountScatterRecord:
    api = str(case["api"])
    model = str(case["model"])
    prompt = str(spec["prompt"])
    started = time.perf_counter()
    try:
        raw, status, detail, usage, final_budget, retries = generate_response(
            client=client,
            api=api,
            model=model,
            prompt=prompt,
            max_output_tokens=int(config.get("max_output_tokens", 2048)),
            retry_max_output_tokens=int(config.get("retry_max_output_tokens", 8192)),
            config=config,
            reasoning_effort=case.get("reasoning_effort"),
        )
    except Exception as exc:
        raw = f"ERROR: {type(exc).__name__}: {exc}"
        status = "error"
        detail = type(exc).__name__
        usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "cached_input_tokens": None,
        }
        final_budget = int(config.get("max_output_tokens", 2048))
        retries = 0
    latency = time.perf_counter() - started
    parsed, parse_error = extract_json_object(raw)
    target_count = int(spec["target_count"])
    parsed_count = parse_int(parsed.get("count")) if parsed else None
    count_error = parsed_count - target_count if parsed_count is not None else None
    count_abs_error = abs(count_error) if count_error is not None else None
    count_correct = parsed_count == target_count

    secondary_target = spec.get("secondary_target_count")
    parsed_secondary = parse_int(parsed.get("secondary_count")) if parsed else None
    secondary_correct = None
    if secondary_target is not None:
        secondary_correct = parsed_secondary == int(secondary_target)

    item: EvalItem | None = spec["item"]
    task_correct = None
    if spec["condition"] == "real_task" and parsed:
        if item is None:
            raise RuntimeError("real_task spec missing item")
        task_correct = score_answer(
            item.answer_type,
            parsed.get("task_answer"),
            item.answer,
            item.metadata,
        )

    return CountScatterRecord(
        run_id=run_id,
        model=model,
        api=api,
        condition=spec["condition"],
        category=spec["category"],
        source_category=spec.get("source_category", spec["category"]),
        item_id=item.item_id if item is not None else f"plain_count:{target_count}:{spec.get('sample_index', 0)}",
        target_count=target_count,
        parsed_count=parsed_count,
        count_error=count_error,
        count_abs_error=count_abs_error,
        count_correct=count_correct,
        secondary_target_count=secondary_target,
        parsed_secondary_count=parsed_secondary,
        secondary_correct=secondary_correct,
        task_correct=task_correct,
        matched_prompt_tokens=int(spec["matched_prompt_tokens"]),
        distractor_estimated_tokens=int(spec["distractor_estimated_tokens"]),
        prompt_estimated_tokens=estimate_tokens(prompt),
        raw_response=raw,
        response_status=status,
        response_detail=detail,
        parsed_json=parsed,
        parse_error=parse_error,
        latency_seconds=latency,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
        reasoning_tokens=usage["reasoning_tokens"],
        cached_input_tokens=usage["cached_input_tokens"],
        requested_max_output_tokens=int(config.get("max_output_tokens", 2048)),
        final_max_output_tokens=final_budget,
        retry_count=retries,
    )


def case_id(case: dict[str, Any]) -> str:
    return f"{case['api']}__{case['model']}".replace("/", "__").replace(":", "_")


def write_csv(path: Path, records: list[CountScatterRecord]) -> None:
    fieldnames = [key for key in CountScatterRecord.__dataclass_fields__ if key not in {"raw_response", "parsed_json"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row.pop("raw_response", None)
            row.pop("parsed_json", None)
            writer.writerow(row)


def append_jsonl(path: Path, records: list[CountScatterRecord]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def write_summary(path: Path, records: list[CountScatterRecord]) -> None:
    grouped: dict[tuple[str, str, str], list[CountScatterRecord]] = {}
    for record in records:
        grouped.setdefault((record.model, record.condition, record.category), []).append(record)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "condition",
                "category",
                "n",
                "count_accuracy",
                "mean_abs_count_error",
                "secondary_accuracy",
                "task_accuracy",
                "parse_rate",
                "avg_matched_prompt_tokens",
                "avg_prompt_estimated_tokens",
                "avg_input_tokens",
            ],
        )
        writer.writeheader()
        for (model, condition, category), rows in sorted(grouped.items()):
            count_accuracy = sum(r.count_correct for r in rows) / len(rows)
            abs_errors = [r.count_abs_error for r in rows if r.count_abs_error is not None]
            secondary = [r.secondary_correct for r in rows if r.secondary_correct is not None]
            task = [r.task_correct for r in rows if r.task_correct is not None]
            input_tokens = [r.input_tokens for r in rows if r.input_tokens is not None]
            writer.writerow(
                {
                    "model": model,
                    "condition": condition,
                    "category": category,
                    "n": len(rows),
                    "count_accuracy": f"{count_accuracy:.6f}",
                    "mean_abs_count_error": f"{(sum(abs_errors) / len(abs_errors)):.6f}" if abs_errors else "",
                    "secondary_accuracy": f"{(sum(secondary) / len(secondary)):.6f}" if secondary else "",
                    "task_accuracy": f"{(sum(task) / len(task)):.6f}" if task else "",
                    "parse_rate": f"{(sum(1 for r in rows if r.parsed_json is not None) / len(rows)):.6f}",
                    "avg_matched_prompt_tokens": f"{(sum(r.matched_prompt_tokens for r in rows) / len(rows)):.6f}",
                    "avg_prompt_estimated_tokens": f"{(sum(r.prompt_estimated_tokens for r in rows) / len(rows)):.6f}",
                    "avg_input_tokens": f"{(sum(input_tokens) / len(input_tokens)):.6f}" if input_tokens else "",
                }
            )


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        if args.count_start is not None:
            config["count_start"] = args.count_start
        if args.count_end is not None:
            config["count_end"] = args.count_end
        if args.samples_per_count is not None:
            config["samples_per_count_per_category"] = args.samples_per_count
        sampled = sample_items(
            build_all_items(config),
            sample_size=int(config["sample_size_per_category"]),
            seed=int(config["seed"]),
        )
        cases = selected_cases(config, set(args.models or []) or None)
        specs = build_specs(sampled, config)
    except Exception as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

    run_id = utc_timestamp()
    run_dir = args.output_dir / str(config["run_subdir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(run_dir, sampled, config)
    planned = [
        {
            "condition": spec["condition"],
            "category": spec["category"],
            "source_category": spec.get("source_category", spec["category"]),
            "item_id": (
                spec["item"].item_id
                if spec["item"] is not None
                else f"plain_count:{spec['target_count']}:{spec.get('sample_index', 0)}"
            ),
            "target_count": spec["target_count"],
            "secondary_target_count": spec["secondary_target_count"],
            "matched_prompt_tokens": spec["matched_prompt_tokens"],
            "distractor_estimated_tokens": spec["distractor_estimated_tokens"],
            "prompt_estimated_tokens": estimate_tokens(spec["prompt"]),
        }
        for spec in specs
    ]
    (run_dir / "planned_trials.json").write_text(json.dumps(planned, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Run directory: {run_dir}")
    print(f"Count range: {config['count_start']}..{config['count_end']}")
    print(f"Specs per model: {len(specs)}")
    print(f"Cases: {', '.join(case_id(case) for case in cases)}")
    if args.prepare_only:
        print("Prepared manifests and planned trials without API calls.")
        return 0

    clients: dict[str, Any] = {}
    try:
        for api in sorted({str(case["api"]) for case in cases}):
            clients[api] = init_client(api)
    except Exception as exc:
        print(f"Client initialization failed: {exc}", file=sys.stderr)
        return 1

    all_records: list[CountScatterRecord] = []
    jsonl_path = run_dir / "trials.jsonl"
    max_workers = max(1, int(config.get("parallel_requests", 4)))
    for case in cases:
        case_records: list[CountScatterRecord] = []
        print(f"\n[{case_id(case)}] starting {len(specs)} trials")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_one, run_id, clients[str(case["api"])], case, spec, config)
                for spec in specs
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                all_records.append(record)
                case_records.append(record)
                append_jsonl(jsonl_path, [record])
                if index % 100 == 0 or index == len(specs):
                    print(f"  completed {index}/{len(specs)}")
        (run_dir / f"{case_id(case)}.json").write_text(
            json.dumps([asdict(record) for record in case_records], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_csv(run_dir / "trials.csv", all_records)
        write_summary(run_dir / "summary.csv", all_records)

    print(f"\nComplete: {run_dir}")
    print(f"Trials: {run_dir / 'trials.csv'}")
    print(f"Summary: {run_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
