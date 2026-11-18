#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


from counting.paths import REPO_ROOT as ROOT
MODEL_DIRS = (ROOT / "results" / "counting-data" / "models",)
TRIALS_DIR = ROOT / "results" / "counting-data" / "trials"
OUTPUT_REPORT = ROOT / "results" / "non_single_number_responses_report.md"
OUTPUT_CASES = ROOT / "results" / "non_single_number_responses_cases.csv"

SINGLE_INTEGER_RE = re.compile(r"^\s*[+-]?\d+\s*$")
NUMBER_RE = re.compile(r"[+-]?\d+")
REASONING_MARKERS = (
    "let me count",
    "i will count",
    "i'll count",
    "count by",
    "count the",
    "step-by-step",
    "re-count",
    "recount",
    "systematically",
    "there are",
    "the sequence contains",
    "the number of",
    "provided sequence",
    "carefully count",
    "row 1",
    "groups of",
    "group of",
)


@dataclass(frozen=True)
class ResponseCase:
    run_group: str
    model: str
    model_file: str
    expected: int | None
    center_length: int | None
    phase: str
    trial_index: int | None
    evaluation_index: int | None
    parsed: int | None
    exact_match: bool | None
    response_status: str | None
    category: str
    raw_response: str


def is_single_integer_response(raw_response: str | None) -> bool:
    return bool(SINGLE_INTEGER_RE.fullmatch(raw_response or ""))


def normalized_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_motif_tokens(text: str) -> int:
    return len(re.findall(r"\ba\b", text.lower()))


def categorize_response(raw_response: str | None) -> str:
    raw = raw_response or ""
    stripped = raw.strip()
    lowered = stripped.lower()

    if not stripped:
        return "Blank / whitespace"

    if "<think>" in lowered or "</think>" in lowered:
        return "Reasoning-tag or hidden-CoT markup"

    if "```" in raw or re.search(r"`[^`]+`", raw):
        return "Code / markdown formatted output"

    if re.fullmatch(r"[+-]?\d+\W+", stripped) or re.fullmatch(r"\W+[+-]?\d+", stripped):
        return "Number with stray punctuation / formatting"

    has_reasoning_marker = any(marker in lowered for marker in REASONING_MARKERS)
    number_count = len(NUMBER_RE.findall(stripped))
    if has_reasoning_marker or (
        number_count >= 3 and ("row" in lowered or "group" in lowered or "=" in stripped or ":" in stripped)
    ):
        return "Step-by-step counting or explanation"

    if (
        (lowered.count("a,") >= 10 or count_motif_tokens(stripped) >= 20)
        and not re.search(r"\b(count|there are|sequence contains|number of|row|group|step)\b", lowered)
    ):
        return "Prompt echo / copied sequence"

    if not NUMBER_RE.search(stripped):
        return "Text with no parseable number"

    if re.search(r"[A-Za-z\u0080-\uffff]", stripped):
        return "Answer with extra prose / instruction leakage"

    return "Other mixed numeric format"



def audited_trial_logs() -> list[Path]:
    """Trial logs behind the released tables.

    Configurations remeasured at the 32k cap replace their catalogue scan in
    the tables (registry.csv records the run of origin); the audit follows the
    same substitution so its totals match the dataset.
    """
    import csv

    replaced = set()
    registry = TRIALS_DIR.parent / "registry.csv"
    if registry.exists():
        with registry.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["run"].startswith("budget-32k"):
                    replaced.add(row["run_key"].replace("/", "__").replace("#", "__") + ".jsonl")
    logs = [path for path in sorted(TRIALS_DIR.glob("*.jsonl")) if path.name not in replaced]
    if replaced:
        logs += sorted((ROOT / "results" / "budget-32k").glob("*/trials/*.jsonl"))
    return logs


def response_cases() -> tuple[list[ResponseCase], int]:
    """Every scan probe, from the per-trial logs of the current dataset."""
    cases: list[ResponseCase] = []
    total_trials = 0
    for path in audited_trial_logs():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                trial = json.loads(line)
            except json.JSONDecodeError:
                continue
            if trial.get("phase") != "scan":
                continue
            total_trials += 1
            raw = trial.get("raw_response")
            if is_single_integer_response(raw):
                continue
            cases.append(ResponseCase(
                run_group="counting-data",
                model=str(trial.get("model") or path.stem.replace("__", "/", 1)),
                model_file=path.stem,
                expected=trial.get("expected"),
                center_length=trial.get("center_length"),
                phase=str(trial.get("phase") or "scan"),
                trial_index=trial.get("trial_index"),
                evaluation_index=trial.get("evaluation_index"),
                parsed=trial.get("parsed"),
                exact_match=trial.get("exact_match"),
                response_status=trial.get("response_status"),
                category=categorize_response(raw),
                raw_response=str(raw) if raw is not None else "",
            ))
    return cases, total_trials


def write_cases_csv(cases: list[ResponseCase]) -> None:
    OUTPUT_CASES.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CASES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_group",
                "model",
                "model_file",
                "expected",
                "center_length",
                "phase",
                "trial_index",
                "evaluation_index",
                "parsed",
                "exact_match",
                "response_status",
                "category",
                "raw_response",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "run_group": case.run_group,
                    "model": case.model,
                    "model_file": case.model_file,
                    "expected": case.expected,
                    "center_length": case.center_length,
                    "phase": case.phase,
                    "trial_index": case.trial_index,
                    "evaluation_index": case.evaluation_index,
                    "parsed": case.parsed,
                    "exact_match": case.exact_match,
                    "response_status": case.response_status,
                    "category": case.category,
                    "raw_response": case.raw_response,
                }
            )


def category_examples(cases: list[ResponseCase], category: str, limit: int = 5) -> list[ResponseCase]:
    category_cases = [case for case in cases if case.category == category]
    category_cases.sort(key=lambda case: (case.model.lower(), case.expected or -1, case.phase, case.trial_index or -1))

    selected: list[ResponseCase] = []
    used_models: set[str] = set()
    for case in category_cases:
        if case.model in used_models:
            continue
        selected.append(case)
        used_models.add(case.model)
        if len(selected) >= limit:
            return selected
    for case in category_cases:
        if case in selected:
            continue
        selected.append(case)
        if len(selected) >= limit:
            break
    return selected


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(cases: list[ResponseCase], total_trials: int) -> None:
    category_counts = Counter(case.category for case in cases)
    category_models = {category: len({case.model for case in cases if case.category == category}) for category in category_counts}
    category_parsed = {
        category: sum(1 for case in cases if case.category == category and case.parsed is not None)
        for category in category_counts
    }
    run_counts = Counter(case.run_group for case in cases)
    model_counts = Counter(case.model for case in cases)
    parsed_count = sum(1 for case in cases if case.parsed is not None)

    lines: list[str] = []
    lines.append("# Non-single-number response audit")
    lines.append("")
    lines.append(
        "This report flags every trial whose raw response was **not** a single integer under the strict regex "
        "`^\\s*[+-]?\\d+\\s*$`."
    )
    lines.append(
        "Cases are categorized from the raw response text, even when the benchmark parser still extracted an integer "
        "from the response."
    )
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        f"- Total trials scanned: **{total_trials}**"
    )
    lines.append(
        f"- Non-single-number responses: **{len(cases)}** ({len(cases) / total_trials:.1%} of all trials)"
    )
    lines.append(
        f"- Cases where an integer was still parsed: **{parsed_count}** ({parsed_count / len(cases):.1%} of flagged cases)"
    )
    lines.append(
        f"- Affected models: **{len(model_counts)}**"
    )
    lines.append(
        f"- Closed run cases: **{run_counts.get('counting-stable-search', 0)}**; open-source run cases: **{run_counts.get('counting-stable-search-open-source', 0)}**"
    )
    lines.append("")
    lines.append("## Category breakdown")
    lines.append("")
    category_rows = []
    for category, count in category_counts.most_common():
        category_rows.append(
            [
                category,
                str(count),
                f"{count / len(cases):.1%}",
                str(category_models[category]),
                str(category_parsed[category]),
            ]
        )
    lines.append(
        markdown_table(
            ["Category", "Cases", "Share", "Affected models", "Integer still parsed"],
            category_rows,
        )
    )
    lines.append("")
    lines.append("## Most affected models")
    lines.append("")
    model_rows = []
    for model, count in model_counts.most_common(15):
        parsed_here = sum(1 for case in cases if case.model == model and case.parsed is not None)
        categories = Counter(case.category for case in cases if case.model == model)
        top_category = categories.most_common(1)[0][0]
        model_rows.append([model, str(count), str(parsed_here), top_category])
    lines.append(markdown_table(["Model", "Flagged cases", "Integer still parsed", "Dominant category"], model_rows))
    lines.append("")
    lines.append("## Representative examples by category")
    lines.append("")

    for category, count in category_counts.most_common():
        lines.append(f"### {category} ({count} cases)")
        lines.append("")
        lines.append(
            markdown_table(
                ["Model", "Expected length", "Phase", "Parsed integer", "Status", "Raw response snippet"],
                [
                    [
                        case.model,
                        str(case.expected),
                        case.phase,
                        "" if case.parsed is None else str(case.parsed),
                        str(case.response_status),
                        normalized_whitespace(case.raw_response)[:140].replace("|", "\\|"),
                    ]
                    for case in category_examples(cases, category)
                ],
            )
        )
        lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append(f"- Full case table: `{OUTPUT_CASES.relative_to(ROOT)}`")
    lines.append("")

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cases, total_trials = response_cases()
    write_cases_csv(cases)
    write_report(cases, total_trials)
    print(f"Report: {OUTPUT_REPORT}")
    print(f"Cases CSV: {OUTPUT_CASES}")
    print(f"Flagged {len(cases)} / {total_trials} trials as non-single-number responses.")
    for category, count in Counter(case.category for case in cases).most_common():
        print(f"  {category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
