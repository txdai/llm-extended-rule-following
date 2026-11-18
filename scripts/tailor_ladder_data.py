#!/usr/bin/env python3
"""Build the analysis-ready ladder corpus without changing the raw results.

The derivative merges the original cohort scans, lower and upper extensions,
and the Gemini 3.7 pilot scans into one JSONL file per logical task/model scan.
Every retained request must have the fixed 8,192-token output budget.  A scan
is truncated immediately after its tenth consecutive budget-exhausted scan
response, and that tenth row is marked as the terminal record.

No provider calls are made by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "results" / "ladder-tailored-8192"
ARCHIVE = REPO / "archive" / "ladder-grid-raw-complete-pre-tailoring-2026-09-01.tar.gz"
MAX_OUTPUT_TOKENS = 8192
BUDGET_EXHAUSTION_STREAK = 10
EXCLUDED_MODELS = {"claude-fable-5": "provider_content_filter"}
SOURCE_PRIORITY = {"lower": 0, "pilot": 1, "original": 2, "upper": 3}

PILOT_RUNS = {
    "r2-colortotal": ["results/ladder-rung2/color-total"],
    "r3-grow": ["results/ladder-rung3/grow-run"],
    "r3-merge": ["results/ladder-rung3/recolor-merge"],
    "r3-erase": ["results/ladder-rung3/erase-singletons"],
    "r4-merge": ["results/ladder-rung4/recolor-merge"],
    "r4-erase": ["results/ladder-rung4/erase-singletons"],
}


def archive_sha256() -> str:
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_budget_exhausted(row: dict[str, Any]) -> bool:
    return (
        row.get("response_status") == "incomplete"
        and str(row.get("response_detail") or "").startswith("max_output_tokens")
    )


def read_trial_file(
    path: Path,
    *,
    scan: str,
    model: str,
    source_run: str,
    rows_by_scan: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    with path.open(encoding="utf-8") as source:
        for source_line, line in enumerate(source, start=1):
            row = json.loads(line)
            requested = row.get("requested_max_output_tokens")
            final = row.get("final_max_output_tokens")
            if requested != MAX_OUTPUT_TOKENS or final != MAX_OUTPUT_TOKENS:
                raise ValueError(
                    f"{path}:{source_line} has budget requested={requested}, final={final}"
                )
            row["tailoring"] = {
                "source_run": source_run,
                "source_path": str(path.relative_to(REPO)),
                "source_line": source_line,
                "budget_exhausted": is_budget_exhausted(row),
                "termination_record": False,
            }
            rows_by_scan[(scan, model)].append(row)


def load_rows() -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]], int, Counter[str]
]:
    rows_by_scan: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded_records = 0
    source_counts: Counter[str] = Counter()

    for source_run, root_name in (
        ("original", "results/ladder-full"),
        ("lower", "results/ladder-extended-lower"),
        ("upper", "results/ladder-extended-upper-4x"),
    ):
        root = REPO / root_name
        for path in sorted(root.glob("*/*/trials/*.jsonl")):
            scan = path.parents[2].name
            model = path.parents[1].name
            if model in EXCLUDED_MODELS:
                excluded_records += sum(1 for _ in path.open(encoding="utf-8"))
                continue
            before = len(rows_by_scan[(scan, model)])
            read_trial_file(
                path,
                scan=scan,
                model=model,
                source_run=source_run,
                rows_by_scan=rows_by_scan,
            )
            source_counts[source_run] += len(rows_by_scan[(scan, model)]) - before

    pilot_model = "gemini-3p7-flash"
    for scan, run_names in PILOT_RUNS.items():
        for run_name in run_names:
            for path in sorted((REPO / run_name / "trials").glob("*.jsonl")):
                before = len(rows_by_scan[(scan, pilot_model)])
                read_trial_file(
                    path,
                    scan=scan,
                    model=pilot_model,
                    source_run="pilot",
                    rows_by_scan=rows_by_scan,
                )
                source_counts["pilot"] += (
                    len(rows_by_scan[(scan, pilot_model)]) - before
                )

    return rows_by_scan, excluded_records, source_counts


def upper_targets() -> set[tuple[str, str]]:
    root = REPO / "results" / "ladder-extended-upper-4x"
    return {
        (directory.parent.name, directory.name)
        for directory in root.glob("*/*")
        if directory.is_dir()
    }


def upper_remaining(scan: str, model: str, last_load: int) -> int:
    config = (
        REPO
        / "configs"
        / "experiments"
        / f"ladderextend_upper_4x_{scan.replace('-', '')}_{model}.yaml"
    )
    search = yaml.safe_load(config.read_text(encoding="utf-8"))["search"]
    return max(0, (int(search["stop"]) - last_load) // int(search["step"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing tailored data: {output}")
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"raw-data archive is required first: {ARCHIVE}")

    rows_by_scan, excluded_records, source_counts = load_rows()
    targets = upper_targets()
    scans_dir = output / "scans"
    scans_dir.mkdir(parents=True)

    summaries: list[dict[str, Any]] = []
    termination_counts: Counter[str] = Counter()
    kept_records = 0
    dropped_records = 0
    needed: list[dict[str, Any]] = []

    for (scan, model), all_rows in sorted(rows_by_scan.items()):
        preflight = [row for row in all_rows if row.get("phase") == "preflight"]
        scan_rows = [row for row in all_rows if row.get("phase") != "preflight"]
        preflight.sort(
            key=lambda row: (
                SOURCE_PRIORITY[row["tailoring"]["source_run"]],
                row["tailoring"]["source_line"],
            )
        )
        scan_rows.sort(
            key=lambda row: (
                int(row["expected"]),
                SOURCE_PRIORITY[row["tailoring"]["source_run"]],
                row["tailoring"]["source_line"],
            )
        )

        streak = 0
        terminal_index: int | None = None
        for index, row in enumerate(scan_rows):
            if row["tailoring"]["budget_exhausted"]:
                streak += 1
            else:
                streak = 0
            if streak == BUDGET_EXHAUSTION_STREAK:
                terminal_index = index
                row["tailoring"]["termination_record"] = True
                break

        if terminal_index is None:
            retained_scan_rows = scan_rows
            terminal_load: int | str = ""
            termination_reason = ""
        else:
            retained_scan_rows = scan_rows[: terminal_index + 1]
            terminal_load = int(scan_rows[terminal_index]["expected"])
            termination_reason = "10_consecutive_budget_exhaustions"
            termination_counts[termination_reason] += 1

        retained = preflight + retained_scan_rows
        dropped = len(scan_rows) - len(retained_scan_rows)
        kept_records += len(retained)
        dropped_records += dropped

        destination = scans_dir / scan / f"{model}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as sink:
            for row in retained:
                sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        upper_directory = (
            REPO / "results" / "ladder-extended-upper-4x" / scan / model
        )
        upper_targeted = (scan, model) in targets
        upper_registry = (upper_directory / "registry.csv").exists()
        protocol_complete = (
            not upper_targeted or upper_registry or terminal_index is not None
        )
        needs_additional_scan = not protocol_complete
        additional_probe_upper_bound = 0
        last_upper_load: int | str = ""
        if needs_additional_scan:
            upper_rows = [
                row
                for row in retained_scan_rows
                if row["tailoring"]["source_run"] == "upper"
            ]
            if not upper_rows:
                raise RuntimeError(f"upper-targeted partial scan has no rows: {scan}/{model}")
            last_upper_load = max(int(row["expected"]) for row in upper_rows)
            additional_probe_upper_bound = upper_remaining(
                scan, model, int(last_upper_load)
            )
            needed.append(
                {
                    "scan": scan,
                    "model": model,
                    "last_upper_load": last_upper_load,
                    "additional_probe_upper_bound": additional_probe_upper_bound,
                    "ending_budget_exhaustion_streak": streak,
                }
            )

        summaries.append(
            {
                "scan": scan,
                "model": model,
                "source_records": len(all_rows),
                "retained_records": len(retained),
                "dropped_after_termination": dropped,
                "termination_reason": termination_reason,
                "termination_load": terminal_load,
                "upper_targeted": upper_targeted,
                "upper_registry_complete": upper_registry,
                "protocol_complete": protocol_complete,
                "needs_additional_scan": needs_additional_scan,
                "last_upper_load": last_upper_load,
                "additional_probe_upper_bound": additional_probe_upper_bound,
            }
        )

    summary_path = output / "scan_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    manifest = {
        "dataset": "ladder-tailored-8192",
        "raw_archive": str(ARCHIVE.relative_to(REPO)),
        "raw_archive_sha256": archive_sha256(),
        "rules": {
            "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
            "budget_exhaustion_definition": (
                "response_status=incomplete and response_detail starts with "
                "max_output_tokens"
            ),
            "termination": (
                f"truncate after {BUDGET_EXHAUSTION_STREAK} consecutive "
                "budget-exhausted scan responses"
            ),
            "preflight_affects_streak": False,
            "protocol_exclusions": EXCLUDED_MODELS,
        },
        "counts": {
            "models": len({model for _, model in rows_by_scan}),
            "logical_scans": len(rows_by_scan),
            "source_records": sum(len(rows) for rows in rows_by_scan.values()),
            "source_records_by_run": dict(sorted(source_counts.items())),
            "excluded_records": excluded_records,
            "retained_records": kept_records,
            "dropped_after_budget_termination": dropped_records,
            "budget_terminated_scans": sum(termination_counts.values()),
            "upper_targeted_scans": len(targets),
            "completed_scans": len(rows_by_scan) - len(needed),
            "additional_scans_needed": len(needed),
            "additional_probe_upper_bound": sum(
                int(row["additional_probe_upper_bound"]) for row in needed
            ),
        },
        "additional_scans_needed": needed,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Tailored grid-ladder data\n\n"
        "This directory is a derivative of the lossless raw archive named in "
        "`manifest.json`. It excludes Claude Fable 5 under the documented "
        "content-filter protocol exclusion, merges the Gemini 3.7 pilot with "
        "the original/lower/upper cohort records, enforces an 8,192-token "
        "per-request output cap, and truncates each logical scan at its tenth "
        "consecutive budget-exhausted scan response. The tenth row is retained "
        "with `tailoring.termination_record=true`; later rows remain available "
        "only in the archive and raw `results/` roots.\n\n"
        "`scan_summary.csv` records completion and the worst-case remaining "
        "probe count. Creating this derivative makes no provider calls.\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    for row in needed:
        print(
            f"NEEDS {row['scan']}/{row['model']}: "
            f"last={row['last_upper_load']}, "
            f"at most {row['additional_probe_upper_bound']} probes, "
            f"budget streak={row['ending_budget_exhaustion_streak']}"
        )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
