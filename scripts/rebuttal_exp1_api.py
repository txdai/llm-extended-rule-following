#!/usr/bin/env python3
"""Rebuttal Experiment 1 (API): Output-format & mod-10 controls across frontier models."""

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
CONF = REPO / "configs" / "experiments" / "rebuttal_exp1"
LOG_DIR = REPO / "results" / "rebuttal-exp1" / "logs"

FORMATS = ["int_bare", "english_words", "zero_padded_spaced", "mod_10"]

# (model_id, variant, reasoning_effort, temp_locked, max_out_tokens, stop_len, step)
MODELS = [
    ("anthropic/claude-sonnet-4.6", "nothink", "none",   False, 128,  460, 1),
    ("anthropic/claude-sonnet-5",   "nothink", "none",   True,  128,  160, 1),
    ("openai/gpt-5.4",              "nothink", "none",   True,  128,  300, 1),
    ("google/gemini-2.5-flash",     "nothink", "none",   False, 128,  260, 1),
    ("google/gemini-3.7-flash",     "",        "medium", True,  4096, 480, 2),
]


def slug_of(model: str, variant: str) -> str:
    base = model.split("/")[-1].replace(".", "p")
    return f"{base}_{variant}" if variant else base


def main() -> int:
    import sys as _sys
    if not os.environ.get("OPENROUTER_API_KEY") and not _sys.stdin.isatty():
        os.environ["OPENROUTER_API_KEY"] = _sys.stdin.readline().strip()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    CONF.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for model_id, variant, effort, locked, max_tok, stop_len, step in MODELS:
        entry: dict = {"id": model_id}
        if variant:
            entry["variant"] = variant
        if effort:
            entry["reasoning_effort"] = effort
        if locked:
            entry["temperature_locked"] = True
        slug = slug_of(model_id, variant)
        for fmt in FORMATS:
            cfg = {
                "extends": ["base.yaml", "providers/openrouter.yaml", "strategies/scan.yaml"],
                "seed": 20261111,
                "models": [entry],
                "task": {"name": "formatted_count", "output_format": fmt, "item": "a", "delimiter": ", "},
                "max_output_tokens": max_tok,
                "retry_max_output_tokens": max_tok,
                "escalate_output_budget": False,
                "parallel_scan_probes": 6 if max_tok <= 256 else 4,
                "search": {
                    "start": 1,
                    "step": step,
                    "stop": stop_len,
                    "stop_after_consecutive_errors": 15,
                    "stop_at_first_mistake_multiple": None,
                    # Scan past CC so we capture post-boundary offset & attractor distributions too!
                    "stop_when_resolved": False,
                },
            }
            cfg_path = CONF / f"{slug}_{fmt}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            run_name = f"rebuttal-exp1/{fmt}/{slug}"
            jobs.append((run_name, cfg_path))

    print(f"Exp1 API: {len(jobs)} scans queued", flush=True)
    if args.dry_run:
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")

    def run_one(job):
        run_name, cfg_path = job
        log_file = LOG_DIR / (run_name.replace("/", "__") + ".log")
        with log_file.open("w") as sink:
            rc = subprocess.call(
                [PY, "-m", "counting", "run", str(cfg_path), "--run-name", run_name],
                cwd=str(REPO), env=env, stdout=sink, stderr=sink,
            )
        print(f"[Exp1] {run_name}: exit {rc}", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rcs = list(pool.map(run_one, jobs))
    print(f"Exp1 API complete: {rcs.count(0)}/{len(rcs)} clean", flush=True)
    return 0 if all(r == 0 for r in rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
