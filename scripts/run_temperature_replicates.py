#!/usr/bin/env python3
"""Replicate the counting scan through the transition region, five times.

    OPENROUTER_API_KEY=... python scripts/run_temperature_replicates.py [--dry-run]

Nine models, most of which do not expose temperature control, are rescanned
five times each under the catalogue protocol (same prompt, same fixed 8,192
token budget, same estimator).  Each replicate starts a short distance below
the catalogue CC and runs until the criterion-native stop rule resolves the
transition, so the record covers the passage from mostly-right to
mostly-wrong without paying for the long stable prefix.  The stimulus is
identical across replicates: the only thing that varies is the provider's own
sampling, which is the effect being measured.

Output: results/temperature-replicates/rep-<k>/<slug>/ (one scan each).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "configs" / "experiments"
OUT = REPO / "results" / "temperature-replicates"
REPLICATES = 5

# (model id, variant, reasoning effort, temperature control, catalogue CC)
# `locked` mirrors the catalogue flag; `ignored` marks models that accept the
# parameter and silently ignore it (Anthropic from Opus 4.7 onward, and
# gemini-3.7-flash); `pinned` is the one model that honours temperature 0.
MODELS = [
    ("openai/gpt-5",              "",        "medium", "locked",  101),
    ("openai/gpt-5.2",            "think",   "medium", "locked",  152),
    ("openai/gpt-5.4",            "nothink", "none",   "locked",  208),
    ("openai/o3",                 "think",   "medium", "locked",  103),
    ("openai/gpt-5.6-sol",        "think",   "medium", "locked",   96),
    ("anthropic/claude-opus-5",   "nothink", "none",   "ignored", 179),
    ("anthropic/claude-sonnet-5", "nothink", "none",   "locked",   92),
    ("anthropic/claude-fable-5",  "",        "medium", "locked",  317),
    ("google/gemini-3.7-flash",   "",        "medium", "ignored", 352),
    ("anthropic/claude-sonnet-4.6", "nothink", "none",  "pinned",  320),
]
START_FRACTION = {"anthropic/claude-fable-5": 0.85}
DEFAULT_START_FRACTION = 0.80
STOP_FRACTION = 1.5


def slug_of(model: str) -> str:
    return model.split("/")[-1].replace(".", "p")


def write_config(model, variant, effort, control, cc, rep) -> tuple[str, Path]:
    entry: dict = {"id": model}
    if variant:
        entry["variant"] = variant
    if effort != "none" or variant == "nothink":
        entry["reasoning_effort"] = effort
    if control != "pinned":
        entry["temperature_locked"] = True
    start = round(START_FRACTION.get(model, DEFAULT_START_FRACTION) * cc)
    conf = {
        "extends": ["base.yaml", "providers/openrouter.yaml",
                    "tasks/homogeneous_count.yaml", "strategies/scan.yaml"],
        "models": [entry],
        "max_output_tokens": 8192,
        "retry_max_output_tokens": 8192,
        "escalate_output_budget": False,
        # Four consecutive loads per request batch: the stop rule is checked
        # after each batch, so resolution overshoots by at most three loads
        # and no scored response changes (see RunnerSettings).
        "parallel_scan_probes": 4,
        "search": {
            "start": start,
            "step": 1,
            "stop": round(STOP_FRACTION * cc),
            "stop_after_consecutive_errors": 10,
            "stop_at_first_mistake_multiple": None,
            "stop_when_resolved": True,
        },
    }
    slug = slug_of(model)
    path = CONF / f"temprep_{slug}_rep{rep}.yaml"
    path.write_text(yaml.safe_dump(conf, sort_keys=False))
    return f"temperature-replicates/rep-{rep}/{slug}", path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--models", nargs="*", help="only these model ids")
    args = ap.parse_args()

    # Slowest models first (reasoning tokens per probe times probes per scan),
    # all replicates of a model together, so the long poles start immediately.
    order = ["openai/gpt-5", "anthropic/claude-fable-5", "google/gemini-3.7-flash",
             "openai/o3", "openai/gpt-5.6-sol", "openai/gpt-5.2",
             "anthropic/claude-opus-5", "openai/gpt-5.4", "anthropic/claude-sonnet-5",
             "anthropic/claude-sonnet-4.6"]
    by_id = {m[0]: m for m in MODELS}
    if args.models:
        order = [m for m in order if m in args.models]
    jobs = []
    for model_id in order:
        model, variant, effort, control, cc = by_id[model_id]
        for rep in range(1, args.replicates + 1):
            jobs.append(write_config(model, variant, effort, control, cc, rep))
    print(f"{len(jobs)} scans queued", file=sys.stderr)
    if args.dry_run:
        for name, path in jobs[:len(MODELS)]:
            s = yaml.safe_load(path.read_text())["search"]
            print(f"  {name}  start={s['start']} stop={s['stop']}")
        return 0

    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(job):
        name, path = job
        log = log_dir / (name.replace("/", "_") + ".log")
        with log.open("w") as sink:
            rc = subprocess.call(["counting", "run", str(path), "--run-name", name],
                                 stdout=sink, stderr=sink)
        print(f"{name}: exit {rc}", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, jobs))
    print(f"done: {codes.count(0)}/{len(codes)} clean", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
