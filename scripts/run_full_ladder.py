#!/usr/bin/env python3
"""Full execution ladder (rungs 2-4, 8 scans) across the ARC post-release
cohort plus the interference trio, leap-frogged and parallelized.

    OPENROUTER_API_KEY=... python scripts/run_full_ladder.py [--dry-run]
    python scripts/run_full_ladder.py --only-trio   # calibration scans only

Scan brackets come from the gemini-3.7-flash pilot ratios (boundary as a
multiple of CC); step = pred/80 keeps each scan near ~80 probes. The trio's
interference-calibration scans (sonnet-5 and gpt-5.6-terra on r2-colortotal,
r3-merge, r4-merge) are queued first so the interference phase can launch
while the rest of the ladder is still running; the remainder runs
longest-first to minimize wall-clock under the worker pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "results" / "counting-data"
CONF = REPO / "configs" / "experiments"
STATE = REPO / "results" / "ladder-full" / "launch_state.json"

COHORT = [
    "anthropic/claude-fable-5", "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.7", "anthropic/claude-opus-4.8",
    "anthropic/claude-opus-5", "anthropic/claude-sonnet-4.6",
    "deepseek/deepseek-v3.2", "deepseek/deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-pro-0813", "google/gemini-3.1-pro-preview",
    "google/gemini-3.5-flash", "google/gemini-3.5-flash-lite",
    "google/gemini-3.6-flash", "minimax/minimax-m2.5",
    "moonshotai/kimi-k2.5", "moonshotai/kimi-k3",
    "openai/gpt-5.2", "openai/gpt-5.4", "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano", "openai/gpt-5.5", "openai/gpt-5.6-luna",
    "openai/gpt-5.6-sol", "openai/gpt-5.6-terra",
    "x-ai/grok-4.5", "x-ai/grok-4.6", "z-ai/glm-5", "z-ai/glm-5.2",
]
TRIO_EXTRA = ["anthropic/claude-sonnet-5"]  # terra is already in the cohort
# gemini-3.7-flash: full ladder measured in the pilot; never queued here.

# scan -> (task dict, boundary/CC ratio from the flash pilot, min start)
SCANS = {
    "r2-colortotal": ({"name": "run_structure", "probe": "color_total"}, 0.93, 1),
    "r3-grow": ({"name": "rule_execution", "rule": "grow_run"}, 0.80, 4),
    "r3-merge": ({"name": "rule_execution", "rule": "recolor_merge"}, 0.72, 4),
    "r3-erase": ({"name": "rule_execution", "rule": "erase_singletons"}, 0.68, 4),
    "r4-merge": ({"name": "grid_rule_execution", "rule": "recolor_merge"}, 0.97, 4),
    "r4-erase": ({"name": "grid_rule_execution", "rule": "erase_singletons"}, 0.95, 4),
}
TRIO_CALIBRATION = [
    (m, s) for m in ("anthropic/claude-sonnet-5", "openai/gpt-5.6-terra")
    for s in ("r2-colortotal", "r3-merge", "r4-merge")
]


def model_specs() -> dict[str, dict]:
    scc = defaultdict(list)
    for r in csv.DictReader((DATA / "scc.csv").open()):
        try:
            scc[r["model"]].append((float(r["scc"]), r.get("variant") or "", r["run_key"]))
        except ValueError:
            continue
    registry = {r["run_key"]: r for r in csv.DictReader((DATA / "registry.csv").open())}
    specs = {}
    for m in COHORT + TRIO_EXTRA:
        cc, variant, run_key = max(scc[m])
        reg = registry.get(run_key, {})
        entry: dict = {"id": m}
        if variant:
            entry["variant"] = variant
            entry["reasoning_effort"] = "medium" if variant == "think" else "none"
        elif reg.get("reasoning_effort") in ("low", "medium", "high"):
            entry["reasoning_effort"] = reg["reasoning_effort"]
        specs[m] = {"cc": cc, "entry": entry}
    return specs


def write_config(model: str, scan: str, spec: dict) -> tuple[str, Path, float]:
    task, ratio, min_start = SCANS[scan]
    pred = max(min_start + 4, round(ratio * spec["cc"]))
    conf = {
        "extends": ["base.yaml", "providers/openrouter.yaml", "strategies/scan.yaml"],
        "models": [spec["entry"]],
        "task": task,
        "max_output_tokens": 8192,
        "retry_max_output_tokens": 8192,
        "escalate_output_budget": False,
        "search": {
            "start": max(min_start, round(0.35 * pred)),
            "step": max(1, round(pred / 80)),
            "stop": max(round(2.0 * pred), pred + 40),
            "stop_after_consecutive_errors": 10,
            "stop_at_first_mistake_multiple": None,
            "stop_when_resolved": True,
        },
    }
    slug = model.split("/")[-1].replace(".", "p")
    path = CONF / f"ladderfull_{scan.replace('-', '')}_{slug}.yaml"
    path.write_text(yaml.safe_dump(conf, sort_keys=False))
    return f"ladder-full/{scan}/{slug}", path, pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-trio", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    specs = model_specs()
    jobs: list[tuple[str, Path, float, bool]] = []
    for model, scan in TRIO_CALIBRATION:
        name, path, pred = write_config(model, scan, specs[model])
        jobs.append((name, path, pred, True))
    if not args.only_trio:
        rest = []
        for model in COHORT + TRIO_EXTRA:
            for scan in SCANS:
                if (model, scan) in TRIO_CALIBRATION:
                    continue
                name, path, pred = write_config(model, scan, specs[model])
                rest.append((name, path, pred, False))
        rest.sort(key=lambda j: -j[2])  # longest-first
        jobs += rest

    print(f"{len(jobs)} scans queued "
          f"({sum(1 for j in jobs if j[3])} trio-calibration first)", file=sys.stderr)
    if args.dry_run:
        for name, _, pred, prio in jobs[:12]:
            print(f"  {'*' if prio else ' '} {name}  pred~{pred:.0f}")
        return 0

    (REPO / "results" / "ladder-full").mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps([j[0] for j in jobs], indent=1))
    log_dir = REPO / "results" / "ladder-full" / "logs"
    log_dir.mkdir(exist_ok=True)

    def run(job):
        name, path, _, _ = job
        log = log_dir / (name.replace("/", "_") + ".log")
        with log.open("w") as sink:
            rc = subprocess.call(
                ["counting", "run", str(path), "--run-name", name,
                 "--concurrency", str(args.concurrency)],
                stdout=sink, stderr=sink)
        print(f"{name}: exit {rc}", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, jobs))
    print(f"done: {codes.count(0)}/{len(codes)} clean", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
