#!/usr/bin/env python3
"""Rebuttal Experiment 2: 16-model x 11-motif sweep for mixed-effects variance decomposition."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
CONF = REPO / "configs" / "experiments" / "rebuttal_exp2"
LOG_DIR = REPO / "results" / "rebuttal-motif16" / "logs"

MOTIFS = [
    ("baseline-a-comma",    "a",  ", ",  "comma"),
    ("letter-b-comma",      "b",  ", ",  "comma"),
    ("letter-x-comma",      "x",  ", ",  "comma"),
    ("double-aa-comma",     "aa", ", ",  "comma"),
    ("greek-alpha-comma",   "α",  ", ",  "comma"),
    ("greek-beta-comma",    "β",  ", ",  "comma"),
    ("chinese-han-comma",   "汉", ", ",  "comma"),
    ("chinese-zhong-comma", "中", ", ",  "comma"),
    ("letter-a-space",      "a",  " ",   "space"),
    ("letter-a-pipe",       "a",  " | ", "pipe"),
    ("letter-a-section",    "a",  " § ", "section"),
]

# 16 models across 7 families (3 Fig. 2F frontier configs + 13 immediate-answer configs)
MODELS = [
    {"id": "anthropic/claude-sonnet-4.6",       "variant": "nothink", "reasoning_effort": "none"},
    {"id": "anthropic/claude-haiku-4.5",        "variant": "nothink", "reasoning_effort": "none"},
    {"id": "google/gemini-2.5-flash",           "variant": "nothink", "reasoning_effort": "none"},
    {"id": "google/gemma-4-26b-a4b-it",         "variant": "nothink", "reasoning_effort": "none"},
    {"id": "openai/gpt-4.1"},
    {"id": "openai/gpt-4.1-mini"},
    {"id": "openai/gpt-5.4",                    "variant": "nothink", "reasoning_effort": "none",   "temperature_locked": True},
    {"id": "meta-llama/llama-3.3-70b-instruct"},
    {"id": "meta-llama/llama-4-maverick"},
    {"id": "qwen/qwen3-235b-a22b-2507"},
    {"id": "qwen/qwen3.6-35b-a3b",              "variant": "nothink", "reasoning_effort": "none"},
    {"id": "deepseek/deepseek-v3.2",            "variant": "nothink", "reasoning_effort": "none"},
    {"id": "mistralai/mistral-medium-3.1"},
]


def main() -> int:
    import sys as _sys
    if not os.environ.get("OPENROUTER_API_KEY") and not _sys.stdin.isatty():
        os.environ["OPENROUTER_API_KEY"] = _sys.stdin.readline().strip()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif-parallel", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    CONF.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for motif_id, item, delimiter, label in MOTIFS:
        cfg = {
            "extends": ["base.yaml", "providers/openrouter.yaml", "tasks/homogeneous_count.yaml", "strategies/scan.yaml"],
            "seed": 20260824,
            "models": MODELS,
            "task": {"name": "homogeneous_count", "item": item, "delimiter": delimiter, "delimiter_label": label},
            "max_output_tokens": 256,
            "retry_max_output_tokens": 256,
            "escalate_output_budget": False,
            "parallel_scan_probes": 4,
            "search": {
                "start": 1,
                "step": 1,
                "stop": 1200,
                "stop_after_consecutive_errors": 10,
                "stop_at_first_mistake_multiple": None,
                "stop_when_resolved": True,
            },
        }
        cfg_path = CONF / f"motif16_{motif_id}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        jobs.append((motif_id, cfg_path))

    print(f"Exp2 Motif16: {len(jobs)} motifs x {len(MODELS)} models = {len(jobs)*len(MODELS)} scans", flush=True)
    if args.dry_run:
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")

    def run_motif(job):
        motif_id, cfg_path = job
        run_name = f"rebuttal-motif16/{motif_id}"
        log_file = LOG_DIR / f"{motif_id}.log"
        with log_file.open("w") as sink:
            rc = subprocess.call(
                [PY, "-m", "counting", "run", str(cfg_path), "--run-name", run_name, "--concurrency", str(args.concurrency)],
                cwd=str(REPO), env=env, stdout=sink, stderr=sink,
            )
        print(f"[Exp2] {motif_id}: exit {rc}", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.motif_parallel) as pool:
        rcs = list(pool.map(run_motif, jobs))
    print(f"Exp2 Motif16 complete: {rcs.count(0)}/{len(rcs)} motifs clean", flush=True)
    return 0 if all(r == 0 for r in rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
