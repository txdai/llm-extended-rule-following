#!/usr/bin/env python3
"""Rescan the budget-bounded reasoning configurations at a 32k output cap.

    OPENROUTER_API_KEY=... python scripts/run_budget_rerun.py [--dry-run]

Under the catalogue protocol (fixed 8,192-token cap) sixteen of the 77
verified hybrid pairs have a reasoning member whose scan ended on budget
exhaustion rather than on wrong answers, so its capacity is a lower bound.
This driver rescans exactly those configurations with the cap raised to
32,768 tokens and everything else unchanged (start at 1, step 1, the same
criterion-native stop rule), so each bound can be replaced by a point
estimate or confirmed as a bound at the larger budget.

Output: results/budget-32k/<slug>/ (one scan each).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from counting.analysis.reasoning_pairs import hybrid_pairs  # noqa: E402

CONF = REPO / "configs" / "experiments"
OUT = REPO / "results" / "budget-32k"
CAP = 32768


def bounded_verified_think_configs() -> list[tuple[str, int]]:
    scc = {(r["model"], r["variant"]): r
           for r in csv.DictReader((REPO / "results/counting-data/scc.csv").open())}
    verified = {p.model for p in hybrid_pairs() if p.classification == "verified"}
    out = []
    for (model, variant), r in scc.items():
        if variant == "think" and model in verified and r["scc_bound_by"] != "mistake":
            out.append((model, int(r["scc"])))
    return sorted(out)


def slug_of(model: str) -> str:
    return model.split("/")[-1].replace(".", "p")


def write_config(model: str) -> tuple[str, Path]:
    conf = {
        "extends": ["base.yaml", "providers/openrouter.yaml",
                    "tasks/homogeneous_count.yaml", "strategies/scan.yaml"],
        "models": [{"id": model, "variant": "think", "reasoning_effort": "medium"}],
        "max_output_tokens": CAP,
        "retry_max_output_tokens": CAP,
        "escalate_output_budget": False,
        "parallel_scan_probes": 4,
        "search": {"start": 1, "step": 1, "stop": 3000,
                   "stop_after_consecutive_errors": 10,
                   "stop_at_first_mistake_multiple": None,
                   "stop_when_resolved": True},
    }
    path = CONF / f"budget32k_{slug_of(model)}.yaml"
    path.write_text(yaml.safe_dump(conf, sort_keys=False))
    return f"budget-32k/{slug_of(model)}", path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    configs = bounded_verified_think_configs()
    jobs = [write_config(m) for m, _ in configs]
    print(f"{len(jobs)} scans queued", file=sys.stderr)
    for (m, lb), (name, _) in zip(configs, jobs):
        print(f"  {name}  catalogue bound >= {lb}", file=sys.stderr)
    if args.dry_run:
        return 0
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(job):
        name, path = job
        with (log_dir / (name.replace("/", "_") + ".log")).open("w") as sink:
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
