#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from run_scc_benchmark import (
    DEFAULT_RETRY_MAX_OUTPUT_TOKENS,
    RunPaths,
    SequenceFormatSpec,
    TrialResult,
    average_output_tokens_at_cc,
    average_total_tokens_at_cc,
    collect_existing_models,
    format_metric,
    infer_open_source_parameter_metadata,
    init_client,
    initialize_run,
    load_benchmark_config,
    load_existing_model_summaries,
    model_filename,
    print_model_summary,
    run_preflight_check,
    search_counting_capacity,
    should_promote_output_budget,
    write_model_output,
    write_run_indexes,
)


DEFAULT_MODIFIED_CONFIG_PATH = Path("configs/benchmarks/stimulus_ablation.json")
DEFAULT_MODIFIED_RUN_SUBDIR = "scc_stimulus_ablation_runs"
DEFAULT_SUMMARY_CSV_NAME = "stimulus_ablation_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modified-token ablation benchmark using the SCC CC search logic."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MODIFIED_CONFIG_PATH,
        help="Path to the modified benchmark config JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the benchmark result folders.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional base-model filter. If omitted, all configured base models are used.",
    )
    parser.add_argument(
        "--stimuli",
        nargs="+",
        help="Optional stimulus-id filter. If omitted, all configured stimuli are used.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the selected ablation cases and exit.",
    )
    return parser.parse_args()


def load_modified_config(path: Path) -> dict[str, Any]:
    config = load_benchmark_config(path)
    if "cases" not in config or not isinstance(config["cases"], list):
        raise ValueError(f"Config file {path} must contain a 'cases' list")
    if "stimuli" not in config or not isinstance(config["stimuli"], list):
        raise ValueError(f"Config file {path} must contain a 'stimuli' list")
    if "run_subdir" not in config or not str(config["run_subdir"]).strip():
        config["run_subdir"] = DEFAULT_MODIFIED_RUN_SUBDIR
    return config


def case_identifier(base_model: str, stimulus_id: str, variant_id: str | None = None) -> str:
    if variant_id:
        return f"{base_model}__{variant_id}__{stimulus_id}"
    return f"{base_model}__{stimulus_id}"


def sequence_spec_from_stimulus(stimulus: dict[str, Any]) -> SequenceFormatSpec:
    return SequenceFormatSpec(
        item=str(stimulus["item"]),
        delimiter=str(stimulus["delimiter"]),
        item_label=str(stimulus.get("item_label") or stimulus["item"]),
    )


def build_ablation_cases(
    config: dict[str, Any],
    model_filter: set[str] | None,
    stimulus_filter: set[str] | None,
) -> list[dict[str, Any]]:
    stimuli_by_id: dict[str, dict[str, Any]] = {}
    for raw_stimulus in config["stimuli"]:
        if not isinstance(raw_stimulus, dict):
            raise ValueError(f"Stimulus entries must be objects, got: {raw_stimulus!r}")
        stimulus_id = str(raw_stimulus.get("id") or "").strip()
        if not stimulus_id:
            raise ValueError(f"Stimulus entry is missing an id: {raw_stimulus!r}")
        stimuli_by_id[stimulus_id] = raw_stimulus

    cases: list[dict[str, Any]] = []
    for raw_case in config["cases"]:
        if not isinstance(raw_case, dict):
            raise ValueError(f"Case entries must be objects, got: {raw_case!r}")
        api = str(raw_case.get("api") or "").strip()
        model = str(raw_case.get("model") or "").strip()
        variant_id = str(raw_case.get("variant_id") or "").strip() or None
        reasoning_effort = str(raw_case.get("reasoning_effort") or "").strip() or None
        if not api or not model:
            raise ValueError(f"Case entry must contain 'api' and 'model': {raw_case!r}")
        if model_filter and model not in model_filter:
            continue
        for stimulus_id, stimulus in stimuli_by_id.items():
            if stimulus_filter and stimulus_id not in stimulus_filter:
                continue
            cases.append(
                {
                    "id": case_identifier(model, stimulus_id, variant_id),
                    "api": api,
                    "base_model": model,
                    "variant_id": variant_id,
                    "reasoning_effort": reasoning_effort,
                    "stimulus_id": stimulus_id,
                    "stimulus": stimulus,
                }
            )
    return cases


def write_modified_summary_csv(run_paths: RunPaths) -> Path:
    output_path = run_paths.run_dir / DEFAULT_SUMMARY_CSV_NAME
    rows: list[dict[str, Any]] = []
    for model_path in sorted(run_paths.models_dir.glob("*.json")):
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        summary = dict(payload.get("summary", {}))
        metadata = dict(payload.get("model_metadata", {}))
        trials = [TrialResult(**trial_payload) for trial_payload in payload.get("trials", [])]
        rows.append(
            {
                "case_id": str(payload.get("model") or ""),
                "model": str(metadata.get("base_model") or ""),
                "api": str(metadata.get("api") or ""),
                "stimulus_id": str(metadata.get("stimulus_id") or ""),
                "variant_id": str(metadata.get("variant_id") or ""),
                "reasoning_effort": str(metadata.get("reasoning_effort") or ""),
                "item": str(metadata.get("item") or ""),
                "delimiter": str(metadata.get("delimiter") or ""),
                "delimiter_label": str(metadata.get("delimiter_label") or ""),
                "cc": summary.get("cc", ""),
                "cc_lower_bound": summary.get("cc_lower_bound", ""),
                "cc_upper_bound": summary.get("cc_upper_bound", ""),
                "avg_output_tokens_at_cc": (
                    f"{average_output_tokens_at_cc(summary, trials):.4f}"
                    if average_output_tokens_at_cc(summary, trials) is not None
                    else ""
                ),
                "avg_total_tokens_at_cc": (
                    f"{average_total_tokens_at_cc(summary, trials):.4f}"
                    if average_total_tokens_at_cc(summary, trials) is not None
                    else ""
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["model"],
            row["variant_id"],
            row["reasoning_effort"],
            row["stimulus_id"],
            row["api"],
        )
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "model",
                "api",
                "stimulus_id",
                "variant_id",
                "reasoning_effort",
                "item",
                "delimiter",
                "delimiter_label",
                "cc",
                "cc_lower_bound",
                "cc_upper_bound",
                "avg_output_tokens_at_cc",
                "avg_total_tokens_at_cc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> int:
    args = parse_args()
    try:
        config = load_modified_config(args.config)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 1

    model_filter = set(args.models or [])
    stimulus_filter = set(args.stimuli or [])
    try:
        requested_cases = build_ablation_cases(config, model_filter or None, stimulus_filter or None)
    except Exception as exc:
        print(f"Failed to build ablation cases: {exc}", file=sys.stderr)
        return 1

    existing_model_paths = collect_existing_models(args.output_dir, config)
    requested_case_ids = [case["id"] for case in requested_cases]
    already_completed = [case_id for case_id in requested_case_ids if case_id in existing_model_paths]
    pending_cases = [case for case in requested_cases if case["id"] not in existing_model_paths]

    if args.list_cases:
        for case in requested_cases:
            status = "already_ran" if case["id"] in existing_model_paths else "pending"
            extra = f"\t{existing_model_paths[case['id']]}" if case["id"] in existing_model_paths else ""
            print(
                f"{case['id']}\t{status}"
                f"\tapi={case['api']}"
                f"\tmodel={case['base_model']}"
                f"\tvariant={case.get('variant_id') or ''}"
                f"\treasoning={case.get('reasoning_effort') or ''}"
                f"\tstimulus={case['stimulus_id']}"
                f"{extra}"
            )
        return 0

    if not pending_cases:
        print("No pending ablation cases to run.")
        if requested_case_ids:
            print(f"Requested cases: {', '.join(requested_case_ids)}")
        if already_completed:
            print(f"Already completed: {', '.join(already_completed)}")
        return 0

    clients: dict[str, Any] = {}
    try:
        for api in sorted({case["api"] for case in pending_cases}):
            clients[api] = init_client(api)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Running SCC stimulus-ablation benchmark to measure CC")
    print(f"Config: {args.config}")
    print(f"Run subdir: {config['run_subdir']}")
    print(f"Cases: {', '.join(case['id'] for case in pending_cases)}")
    if already_completed:
        print(f"Already completed: {', '.join(already_completed)}")

    model_summaries = load_existing_model_summaries(existing_model_paths)
    run_paths = initialize_run(
        output_dir=args.output_dir,
        config_path=args.config,
        config=config,
        requested_models=requested_case_ids,
        selected_models=[case["id"] for case in pending_cases],
        already_completed_models=already_completed,
        existing_model_summaries=model_summaries,
    )

    max_output_tokens = int(config["max_output_tokens"])
    retry_max_output_tokens = int(config.get("retry_max_output_tokens", DEFAULT_RETRY_MAX_OUTPUT_TOKENS))
    preflight_length = int(config.get("preflight_length", 8))
    google_unavailable_retries = int(config.get("google_unavailable_retries", 20))
    google_unavailable_retry_delay_seconds = float(config.get("google_unavailable_retry_delay_seconds", 2.0))
    anthropic_transient_retries = int(config.get("anthropic_transient_retries", 8))
    anthropic_transient_retry_delay_seconds = float(config.get("anthropic_transient_retry_delay_seconds", 2.0))
    openrouter_transient_retries = int(config.get("openrouter_transient_retries", 8))
    openrouter_transient_retry_delay_seconds = float(config.get("openrouter_transient_retry_delay_seconds", 2.0))
    base_seed = int(config["seed"])

    for case_index, case in enumerate(pending_cases):
        case_id = case["id"]
        api = case["api"]
        base_model = case["base_model"]
        stimulus = case["stimulus"]
        sequence_format = sequence_spec_from_stimulus(stimulus)
        print(
            f"\n[{case_id}]"
            f" api={api}"
            f" model={base_model}"
            f" reasoning={case.get('reasoning_effort') or '-'}"
            f" item={sequence_format.item!r}"
            f" delimiter={sequence_format.delimiter!r}"
        )
        model_metadata = {
            "api": api,
            "base_model": base_model,
            "variant_id": case.get("variant_id") or "",
            "reasoning_effort": case.get("reasoning_effort") or "",
            "stimulus_id": case["stimulus_id"],
            "item": sequence_format.item,
            "item_label": sequence_format.item_label,
            "delimiter": sequence_format.delimiter,
            "delimiter_label": str(stimulus.get("delimiter_label") or ""),
        }
        parameter_metadata = infer_open_source_parameter_metadata(base_model, {})
        preflight_result, preflight_passed = run_preflight_check(
            client=clients[api],
            model=base_model,
            api=api,
            max_output_tokens=max_output_tokens,
            retry_max_output_tokens=retry_max_output_tokens,
            preflight_length=preflight_length,
            sequence_format=sequence_format,
            reasoning_effort=case.get("reasoning_effort"),
            google_unavailable_retries=google_unavailable_retries,
            google_unavailable_retry_delay_seconds=google_unavailable_retry_delay_seconds,
            anthropic_transient_retries=anthropic_transient_retries,
            anthropic_transient_retry_delay_seconds=anthropic_transient_retry_delay_seconds,
            openrouter_transient_retries=openrouter_transient_retries,
            openrouter_transient_retry_delay_seconds=openrouter_transient_retry_delay_seconds,
        )
        current_trials = [preflight_result]
        if not preflight_passed:
            model_summaries[case_id] = {
                "model": case_id,
                "base_model": base_model,
                "variant_id": case.get("variant_id") or "",
                "reasoning_effort": case.get("reasoning_effort") or "",
                "stimulus_id": case["stimulus_id"],
                "stimulus": stimulus,
                "skipped": True,
                "skip_reason": "preflight_failed",
                "preflight_result": asdict(preflight_result),
                "cc": 0,
                "cc_lower_bound": 0,
                "cc_upper_bound": int(config["search"]["initial_length"]),
                "cc_error_bar": {
                    "lower": 0,
                    "upper": int(config["search"]["initial_length"] * config["search"]["step_fraction"]),
                },
                "evaluation_count": 0,
                "trial_count": 1,
                "overall": {
                    "mape": preflight_result.relative_error,
                    "parse_rate": 1.0 if preflight_result.parsed is not None else 0.0,
                },
            }
            write_model_output(
                run_paths=run_paths,
                config_path=args.config,
                config=config,
                model=case_id,
                summary=model_summaries[case_id],
                trials=current_trials,
                model_metadata=model_metadata,
                parameter_metadata=parameter_metadata,
            )
            write_run_indexes(
                run_paths=run_paths,
                config_path=args.config,
                config=config,
                requested_models=requested_case_ids,
                selected_models=[case["id"] for case in pending_cases],
                already_completed_models=already_completed,
                model_summaries=model_summaries,
                status="running",
            )
            write_modified_summary_csv(run_paths)
            print(
                "  preflight_failed"
                f" status={preflight_result.response_status}"
                f" detail={preflight_result.response_detail}"
                f" raw={preflight_result.raw_response!r}"
            )
            continue

        learned_max_output_tokens = max_output_tokens
        if should_promote_output_budget(preflight_result, learned_max_output_tokens):
            learned_max_output_tokens = preflight_result.final_max_output_tokens
        trials, model_summary = search_counting_capacity(
            client=clients[api],
            model=base_model,
            api=api,
            search_config=config["search"],
            max_output_tokens=learned_max_output_tokens,
            retry_max_output_tokens=retry_max_output_tokens,
            seed=base_seed + case_index,
            sequence_format=sequence_format,
            reasoning_effort=case.get("reasoning_effort"),
            google_unavailable_retries=google_unavailable_retries,
            google_unavailable_retry_delay_seconds=google_unavailable_retry_delay_seconds,
            anthropic_transient_retries=anthropic_transient_retries,
            anthropic_transient_retry_delay_seconds=anthropic_transient_retry_delay_seconds,
            openrouter_transient_retries=openrouter_transient_retries,
            openrouter_transient_retry_delay_seconds=openrouter_transient_retry_delay_seconds,
        )
        current_trials.extend(trials)
        model_summary["model"] = case_id
        model_summary["base_model"] = base_model
        model_summary["variant_id"] = case.get("variant_id") or ""
        model_summary["reasoning_effort"] = case.get("reasoning_effort") or ""
        model_summary["stimulus_id"] = case["stimulus_id"]
        model_summary["stimulus"] = stimulus
        model_summary["skipped"] = False
        model_summary["preflight_result"] = asdict(preflight_result)
        model_summaries[case_id] = model_summary
        write_model_output(
            run_paths=run_paths,
            config_path=args.config,
            config=config,
            model=case_id,
            summary=model_summary,
            trials=current_trials,
            model_metadata=model_metadata,
            parameter_metadata=parameter_metadata,
        )
        write_run_indexes(
            run_paths=run_paths,
            config_path=args.config,
            config=config,
            requested_models=requested_case_ids,
            selected_models=[case["id"] for case in pending_cases],
            already_completed_models=already_completed,
            model_summaries=model_summaries,
            status="running",
        )
        write_modified_summary_csv(run_paths)
        print_model_summary(model_summary)
        output_avg = average_output_tokens_at_cc(model_summary, current_trials)
        total_avg = average_total_tokens_at_cc(model_summary, current_trials)
        print(
            "  cc_metrics"
            f" avg_output={format_metric(output_avg)}"
            f" avg_total={format_metric(total_avg)}"
        )

    write_run_indexes(
        run_paths=run_paths,
        config_path=args.config,
        config=config,
        requested_models=requested_case_ids,
        selected_models=[case["id"] for case in pending_cases],
        already_completed_models=already_completed,
        model_summaries=model_summaries,
        status="completed",
    )
    summary_csv = write_modified_summary_csv(run_paths)
    print(f"\nModified summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
