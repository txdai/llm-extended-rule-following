"""The dataset two tables wide, built to be appended to.

A run writes, alongside its per-model JSON, two flat tables:

    registry.csv    one row per measurement -- a model in one configuration
    responses.csv   one row per probe -- the count asked for, the count returned

`registry.csv` is the unit of analysis. A model that can run with reasoning on
and off is two rows, not one, distinguished by `variant`; the row carries the
model's static metadata (release date, parameters, price, context window)
alongside the measured outcome, so a plot needs no join against the catalogue.

`responses.csv` is the raw curve. One row per length probed, recording the input
count and the output count and nothing derived, so any later definition of
capacity -- first mistake, last correct, longest exact prefix, a tolerance band --
can be recomputed from it without re-querying anything.

Both are keyed on `run_key` (`model` or `model#variant`) plus `run`, so later
runs stack: a 1..50 characterisation and a full scan of the same model coexist as
separate rows and can be compared or concatenated. Rebuild or merge with
scripts/build_dataset.py.

This is the only output format. Runs produced by the earlier adaptive ladder are
read into the same two tables, tagged `method=ladder`, so the archive and new
scans are analysed the same way. The shapes differ in ways the columns preserve
rather than hide: a ladder probes each tier with sixteen jittered targets and
records a bracket, so its rows carry `unstable_upper_bound` and repeat lengths,
while a scan probes each length once and records `first_mistake`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

REGISTRY_FIELDS = [
    # identity
    "run_key", "run", "model", "variant", "lab",
    # configuration this measurement was taken under
    "reasoning_effort", "reasoning_supported", "reasoning_mandatory",
    "temperature_locked", "max_output_tokens", "scan_start", "scan_stop", "scan_step",
    # outcome
    "method", "cc", "cc_bound_by", "first_mistake", "unstable_upper_bound",
    "first_error", "last_correct", "n_budget_exhausted",
    "n_probed", "n_correct", "n_mistake", "n_error",
    "highest_length", "stopped_early", "correct_above_first_mistake",
    # static model metadata, carried so analysis needs no join
    "canonical_slug", "hugging_face_id", "display_name", "release_date", "knowledge_cutoff", "total_params_b", "active_params_b",
    "architecture_class", "context_length",
    "prompt_usd_per_mtok", "completion_usd_per_mtok",
]

RESPONSE_FIELDS = [
    "run_key", "run", "model", "variant", "phase",
    "input_count", "output_count", "outcome", "exact",
    "signed_error", "response_status", "response_detail",
    "output_tokens", "reasoning_tokens", "input_tokens", "latency_seconds",
]


def _outcome(trial: dict[str, Any]) -> str:
    """correct / mistake / error -- the scan's three-way distinction."""
    if trial.get("parsed") is None:
        return "error"
    return "correct" if trial.get("exact_match") else "mistake"


# Anthropic documents that from Claude Opus 4.7 onward the temperature, top_p
# and top_k parameters are accepted and silently ignored (OpenRouter's Claude
# 4.7 migration guide); gemini-3.7-flash behaves the same way.  A request with
# temperature 0 therefore succeeds but is
# still a provider-side sample, so these models carry the locked flag alongside
# the ones that reject the parameter.
TEMPERATURE_IGNORED = frozenset({
    "anthropic/claude-opus-4.8",
    "anthropic/claude-opus-5",
    "google/gemini-3.7-flash",
})


def registry_row(
    payload: dict[str, Any], run: str, catalogue: dict[str, dict[str, str]]
) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    meta = payload.get("model_metadata") or {}
    key = payload.get("run_key") or summary.get("run_key") or payload["model"]
    # Early records stored the run key in the `model` field, so strip any
    # variant suffix rather than trusting it to be a plain id.
    model = payload["model"].split("#", 1)[0]
    variant = summary.get("variant") or meta.get("variant") or (
        key.split("#", 1)[1] if "#" in key else ""
    )
    static = dict(catalogue.get(model, {}))
    # Runs made against the direct provider APIs recorded parameter counts the
    # OpenRouter catalogue has no row for -- the Gemma and Llama families among
    # them. Fall back to what the run itself measured rather than lose it.
    archived = summary.get("parameter_metadata") or {}
    if not static.get("total_params_b") and archived.get("total_params_billion"):
        static["total_params_b"] = archived["total_params_billion"]
        static.setdefault("active_params_b", archived.get("active_params_billion") or "")

    # Same for the label fields. Benchmark tables name models inconsistently, so
    # the aliases matter for joining; a model retired from the catalogue since
    # its run would otherwise lose every label but its id.
    run_meta = payload.get("model_metadata") or {}
    for column, source in (("canonical_slug", "canonical_slug"),
                           ("hugging_face_id", "hugging_face_id"),
                           ("name", "name")):
        if not static.get(column) and run_meta.get(source):
            static[column] = run_meta[source]
    search = (payload.get("config") or {}).get("search") or {}

    curve = summary.get("curve") or []
    first_mistake = summary.get("first_mistake")
    above = (
        sum(1 for c in curve if first_mistake and c["length"] > first_mistake
            and c["outcome"] == "correct")
        if first_mistake else 0
    )

    # A scan records its own tallies. A ladder run predates them, so derive the
    # same three counts from its trials rather than leaving the columns blank.
    method = summary.get("strategy") or ("scan" if curve else "ladder")
    counts = _counts(summary, payload)
    bound_by, n_exhausted = _bound_by(summary, payload)

    return {
        "run_key": key,
        "run": run,
        "model": model,
        "variant": variant,
        "lab": model.split("/")[0] if "/" in model else "",
        "reasoning_effort": summary.get("reasoning_effort") or "",
        "reasoning_supported": static.get("reasoning_supported", ""),
        "reasoning_mandatory": static.get("reasoning_mandatory", ""),
        "temperature_locked": (
            True if model in TEMPERATURE_IGNORED
            else summary.get("temperature_locked", meta.get("temperature_locked", ""))
        ),
        "max_output_tokens": summary.get("max_output_tokens", ""),
        "scan_start": search.get("start", ""),
        "scan_stop": search.get("stop", ""),
        "scan_step": search.get("step", ""),
        "method": method,
        "cc": summary.get("stable_count"),
        "cc_bound_by": bound_by,
        "first_mistake": first_mistake if first_mistake is not None else "",
        "unstable_upper_bound": (
            summary.get("unstable_upper_bound")
            if summary.get("unstable_upper_bound") is not None else ""
        ),
        "first_error": summary.get("first_error") if summary.get("first_error") is not None else "",
        "last_correct": summary.get("last_correct") if summary.get("last_correct") is not None else "",
        "n_budget_exhausted": n_exhausted,
        "n_probed": counts["n_probed"],
        "n_correct": counts["n_correct"],
        "n_mistake": counts["n_mistake"],
        "n_error": counts["n_error"],
        "highest_length": counts["highest_length"],
        "stopped_early": summary.get("stopped_early") or "",
        "correct_above_first_mistake": above,
        "canonical_slug": static.get("canonical_slug", ""),
        "hugging_face_id": static.get("hugging_face_id", ""),
        "display_name": static.get("name", ""),
        "release_date": static.get("release_date", ""),
        "knowledge_cutoff": static.get("knowledge_cutoff", ""),
        "total_params_b": static.get("total_params_b", ""),
        "active_params_b": static.get("active_params_b", ""),
        "architecture_class": static.get("architecture_class", ""),
        "context_length": static.get("context_length", ""),
        "prompt_usd_per_mtok": static.get("prompt_usd_per_mtok", ""),
        "completion_usd_per_mtok": static.get("completion_usd_per_mtok", ""),
    }


def _bound_by(summary: dict[str, Any], payload: dict[str, Any]) -> tuple[str, int]:
    """What actually ended the scan, and how many replies blew the output cap.

    A capacity bounded by a wrong answer measures counting. One bounded by the
    output budget measures verbosity against the cap: several models here answer
    correctly right up to the point where their reply no longer fits, never
    returning a wrong number at all. Reading those two as the same quantity
    would put an artefact of the token budget into the results.

        mistake    a wrong answer ended it -- the real measurement
        budget     ran out of output tokens, having never answered wrong
        error      stopped on errors of some other kind
        range_end  reached the end of the scan range without stopping
    """
    exhausted = sum(
        1 for trial in payload.get("trials", [])
        if "fixed_budget_exhausted" in (trial.get("response_detail") or "")
    )
    if summary.get("first_mistake") is not None:
        return "mistake", exhausted
    if summary.get("stopped_early") == "consecutive_errors":
        return ("budget" if exhausted else "error"), exhausted
    if summary.get("stopped_early"):
        return summary["stopped_early"], exhausted
    return "range_end", exhausted


def _counts(summary: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Probe tallies, from the summary where recorded and from trials otherwise."""
    if summary.get("lengths_probed") is not None:
        return {
            "n_probed": summary.get("lengths_probed", ""),
            "n_correct": summary.get("correct_count", ""),
            "n_mistake": summary.get("mistake_count", ""),
            "n_error": summary.get("error_count", ""),
            "highest_length": summary.get("highest_length_probed", ""),
        }
    trials = [t for t in payload.get("trials", []) if t.get("phase") != "preflight"]
    outcomes = [_outcome(t) for t in trials]
    lengths = [t.get("expected") for t in trials if t.get("expected") is not None]
    return {
        "n_probed": len(trials),
        "n_correct": outcomes.count("correct"),
        "n_mistake": outcomes.count("mistake"),
        "n_error": outcomes.count("error"),
        "highest_length": max(lengths) if lengths else "",
    }


def response_rows(payload: dict[str, Any], run: str) -> list[dict[str, Any]]:
    """One row per probe: the count asked for against the count returned."""
    summary = payload.get("summary") or {}
    meta = payload.get("model_metadata") or {}
    key = payload.get("run_key") or summary.get("run_key") or payload["model"]
    model = payload["model"].split("#", 1)[0]
    variant = summary.get("variant") or meta.get("variant") or (
        key.split("#", 1)[1] if "#" in key else ""
    )

    rows = []
    for trial in payload.get("trials", []):
        # The preflight probe is a connectivity check, not part of the curve.
        # Every other phase is kept: a ladder's expansion and refinement probes
        # are the same measurement -- a count asked for, a count returned -- and
        # dropping them would throw away the archive.
        if trial.get("phase") == "preflight":
            continue
        parsed = trial.get("parsed")
        expected = trial.get("expected")
        rows.append({
            "run_key": key,
            "run": run,
            "model": model,
            "variant": variant,
            "phase": trial.get("phase") or "",
            "input_count": expected,
            "output_count": parsed if parsed is not None else "",
            "outcome": _outcome(trial),
            "exact": bool(trial.get("exact_match")),
            "signed_error": (parsed - expected) if parsed is not None else "",
            "response_status": trial.get("response_status") or "",
            "response_detail": (trial.get("response_detail") or "")[:120],
            "output_tokens": trial.get("output_tokens") if trial.get("output_tokens") is not None else "",
            "reasoning_tokens": trial.get("reasoning_tokens") if trial.get("reasoning_tokens") is not None else "",
            "input_tokens": trial.get("input_tokens") if trial.get("input_tokens") is not None else "",
            "latency_seconds": round(trial.get("latency_seconds") or 0.0, 3),
        })
    rows.sort(key=lambda r: (r["input_count"], r["phase"]))
    return rows


def load_catalogue(path: Path) -> dict[str, dict[str, str]]:
    """Static per-model metadata, keyed by model id. Missing file is not fatal."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def write_tables(
    out_dir: Path,
    registry: Iterable[dict[str, Any]],
    responses: Iterable[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write both tables, sorted so a diff between runs is readable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_path = out_dir / "registry.csv"
    resp_path = out_dir / "responses.csv"

    reg = sorted(registry, key=lambda r: (r["run"], r["run_key"]))
    with reg_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reg)

    resp = sorted(responses, key=lambda r: (r["run"], r["run_key"], r["input_count"], r["phase"]))
    with resp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(resp)

    return reg_path, resp_path


def build_from_run(
    run_dir: Path, catalogue: dict[str, dict[str, str]], run: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read one run directory into registry rows and response rows."""
    run = run or run_dir.name
    registry, responses = [], []
    models_dir = run_dir / "models"
    if not models_dir.exists():
        return registry, responses
    for path in sorted(models_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "stable_count" not in (payload.get("summary") or {}):
            continue
        registry.append(registry_row(payload, run, catalogue))
        responses.extend(response_rows(payload, run))
    return registry, responses


def load_registry(run_dir: Path) -> list[dict[str, Any]]:
    """Read one run's registry.csv, with numbers parsed.

    `stable_count` is provided as an alias for `cc` so analysis written against
    the older flat file keeps working against the new table.
    """
    path = run_dir / "registry.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- build it with scripts/build_dataset.py {run_dir}"
        )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("cc", "first_mistake", "unstable_upper_bound", "first_error",
                        "last_correct", "n_budget_exhausted",
                        "n_probed", "n_correct", "n_mistake", "n_error",
                        "highest_length", "correct_above_first_mistake", "context_length"):
                if row.get(key) not in (None, ""):
                    try:
                        row[key] = int(float(row[key]))
                    except ValueError:
                        pass
            for key in ("total_params_b", "active_params_b",
                        "prompt_usd_per_mtok", "completion_usd_per_mtok"):
                if row.get(key) not in (None, ""):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            row["stable_count"] = row.get("cc")
            rows.append(row)
    return rows
