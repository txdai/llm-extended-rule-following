#!/usr/bin/env python3
"""Rebuttal Experiment 3A (API): R=5 Dual-Task Interference Replicates + Non-Executive Semantic Control."""

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
CONF = REPO / "configs" / "experiments" / "rebuttal_exp3"
LOG_DIR = REPO / "results" / "rebuttal-interference" / "logs"

# Calibrated from Table 14 / Appendix H solo capacities
SPECS = {
    "gemini-3p7-flash": {
        "cc": 352,
        "entry": {"id": "google/gemini-3.7-flash", "reasoning_effort": "medium", "temperature_locked": True},
        "grid_u": 85,
        "row3_u": 103,
    },
    "claude-sonnet-5": {
        "cc": 89,
        "entry": {"id": "anthropic/claude-sonnet-5", "variant": "think", "reasoning_effort": "medium", "temperature_locked": True},
        "grid_u": 21,
        "row3_u": 15,
    },
}


def main() -> int:
    import sys as _sys
    if not os.environ.get("OPENROUTER_API_KEY") and not _sys.stdin.isatty():
        os.environ["OPENROUTER_API_KEY"] = _sys.stdin.readline().strip()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    CONF.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for slug, spec in SPECS.items():
        cc = spec["cc"]
        grid_u = spec["grid_u"]
        row3_u = spec["row3_u"]
        conds = [
            ("solo",     0,                      "solo"),
            ("inert",    round(0.90 * grid_u),   "inert-90"),
            ("semantic", round(0.90 * grid_u),   "semantic-90"),
            ("pointer",  round(0.90 * grid_u),   "pointer-90"),
            ("row3",     round(0.60 * row3_u),   "row3-60"),
            ("grid",     round(0.60 * grid_u),   "grid-60"),
        ]
        for rep in range(1, args.replicates + 1):
            seed = 20261100 + rep
            for kind, size, tag in conds:
                if kind == "solo":
                    task_cfg = {"name": "homogeneous_count", "item": "a", "delimiter": ", "}
                else:
                    task_cfg = {"name": "count_with_load", "load_kind": kind, "load_size": max(3, size)}
                cfg = {
                    "extends": ["base.yaml", "providers/openrouter.yaml", "strategies/scan.yaml"],
                    "seed": seed,
                    "models": [spec["entry"]],
                    "task": task_cfg,
                    "max_output_tokens": 4096,
                    "retry_max_output_tokens": 4096,
                    "escalate_output_budget": False,
                    "parallel_scan_probes": 4,
                    "search": {
                        "start": max(5, round(0.20 * cc)),
                        "step": max(1, round(cc / 100)),
                        "stop": round(1.5 * cc) + 60,
                        "stop_after_consecutive_errors": 10,
                        "stop_at_first_mistake_multiple": None,
                        "stop_when_resolved": True,
                    },
                }
                cfg_path = CONF / f"{slug}_{tag}_rep{rep}.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                run_name = f"rebuttal-interference/rep-{rep}/{slug}/{tag}"
                jobs.append((run_name, cfg_path))

    print(f"Exp3 API: {len(jobs)} scans queued", flush=True)
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
        print(f"[Exp3] {run_name}: exit {rc}", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rcs = list(pool.map(run_one, jobs))
    print(f"Exp3 API complete: {rcs.count(0)}/{len(rcs)} clean", flush=True)
    return 0 if all(r == 0 for r in rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
