#!/usr/bin/env python3
"""Extend the grid ladder beyond both sides of its pilot-derived brackets.

The original scans live under ``results/ladder-full``.  A scan resumed in
place can only continue above its largest completed load, so lower-bracket
probes need independent checkpoints.  This launcher therefore writes:

* ``ladder-extended-lower`` for scans classified below-start or mixed; and
* ``ladder-extended-upper-<N>x`` for ceiling-stable or mixed scans.

The consolidated fitter merges these checkpoints with the original record.
Upper scans stop as soon as the transition criterion resolves; the multiple
is only a safety ceiling.  Re-running the command resumes without paying for
completed probes again.

    python scripts/run_ladder_extensions.py --phase all --upper-multiple 4
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
FITS = REPO / "results" / "ladder-full" / "fits.csv"
CONF = REPO / "configs" / "experiments"

LOWER_CLASSES = {"below-start", "unresolved-mixed"}
UPPER_CLASSES = {"ceiling-stable", "unresolved-mixed"}
MIN_START = {
    "r2-colortotal": 1,
    "r3-grow": 4,
    "r3-merge": 4,
    "r3-erase": 4,
    "r4-merge": 4,
    "r4-erase": 4,
}


def original_config(scan: str, slug: str) -> Path:
    return CONF / f"ladderfull_{scan.replace('-', '')}_{slug}.yaml"


def multiple_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p") + "x"


def write_extension_config(
    row: dict[str, str], phase: str, upper_multiple: float
) -> tuple[str, Path] | None:
    scan, slug = row["scan"], row["model"]
    source = original_config(scan, slug)
    if not source.exists():
        # The gemini-3.7-flash pilot is stored outside ladder-full and is not
        # currently censored, but keep the diagnostic explicit if that changes.
        print(f"skip {scan}/{slug}: no original config {source.name}", file=sys.stderr)
        return None

    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    search = config["search"]
    step = int(search["step"])
    config["preflight_length"] = 0
    search["stop_at_first_mistake_multiple"] = None
    search["stop_when_resolved"] = True

    if phase == "lower":
        start = MIN_START[scan]
        stop = int(search["start"]) - step
        if stop < start:
            return None
        root = "ladder-extended-lower"
    else:
        start = int(float(row["max_load"])) + step
        predicted = int(float(row["predicted"]))
        stop = max(start, round(upper_multiple * predicted))
        root = f"ladder-extended-upper-{multiple_tag(upper_multiple)}"
        seed_logs = []
        for seed_root in ("ladder-full", "ladder-extended-lower"):
            seed_logs.extend(
                sorted(
                    str(path.resolve())
                    for path in (
                        REPO / "results" / seed_root / scan / slug / "trials"
                    ).glob("*.jsonl")
                )
            )
        config["scan_seed_trial_logs"] = seed_logs

    search["start"] = start
    search["stop"] = stop
    out = CONF / f"ladderextend_{phase}_{multiple_tag(upper_multiple)}_{scan.replace('-', '')}_{slug}.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return f"{root}/{scan}/{slug}", out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "lower", "upper"), default="all")
    parser.add_argument("--upper-multiple", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--parallel-scan-probes",
        type=int,
        default=1,
        help="consecutive loads per request batch (criterion checked after each batch)",
    )
    parser.add_argument(
        "--resume-incomplete",
        action="store_true",
        help="resume every existing extension directory lacking registry.csv",
    )
    parser.add_argument(
        "--skip-existing-run-dirs",
        action="store_true",
        help="queue only jobs whose result directory has not been created",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.upper_multiple <= 2:
        parser.error("--upper-multiple must exceed the original 2x ceiling")

    phases = ("lower", "upper") if args.phase == "all" else (args.phase,)
    jobs: list[tuple[str, Path, str]] = []
    if args.resume_incomplete:
        roots = {
            "lower": REPO / "results" / "ladder-extended-lower",
            "upper": REPO / "results" / f"ladder-extended-upper-{multiple_tag(args.upper_multiple)}",
        }
        for phase in phases:
            for run_dir in sorted(roots[phase].glob("*/*")):
                if (run_dir / "registry.csv").exists():
                    continue
                scan, slug = run_dir.parent.name, run_dir.name
                config = CONF / (
                    f"ladderextend_{phase}_{multiple_tag(args.upper_multiple)}_"
                    f"{scan.replace('-', '')}_{slug}.yaml"
                )
                if config.exists():
                    resolved = yaml.safe_load(config.read_text(encoding="utf-8"))
                    resolved["parallel_scan_probes"] = args.parallel_scan_probes
                    config.write_text(
                        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
                    )
                    jobs.append((str(run_dir.relative_to(REPO / "results")), config, phase))
    else:
        rows = list(csv.DictReader(FITS.open(encoding="utf-8")))
        rows = [r for r in rows if r["analysis_included"].lower() == "true"]
        for phase in phases:
            wanted = LOWER_CLASSES if phase == "lower" else UPPER_CLASSES
            for row in rows:
                if row["class"] not in wanted:
                    continue
                job = write_extension_config(row, phase, args.upper_multiple)
                if job is not None:
                    jobs.append((*job, phase))

    if args.skip_existing_run_dirs:
        jobs = [job for job in jobs if not (REPO / "results" / job[0]).exists()]

    jobs.sort(key=lambda item: (item[2], item[0]))
    counts = {phase: sum(j[2] == phase for j in jobs) for phase in phases}
    print(f"queued {len(jobs)} extensions: {counts}", file=sys.stderr)
    if args.dry_run:
        for name, config, _ in jobs:
            resolved = yaml.safe_load(config.read_text(encoding="utf-8"))
            search = resolved["search"]
            print(f"{name}: {search['start']}..{search['stop']} step {search['step']}")
        return 0

    state_dir = REPO / "results" / "ladder-extensions"
    state_dir.mkdir(parents=True, exist_ok=True)
    state = state_dir / f"launch-{args.phase}-{multiple_tag(args.upper_multiple)}.json"
    state.write_text(json.dumps([j[0] for j in jobs], indent=1) + "\n", encoding="utf-8")
    log_dir = state_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    def run(job: tuple[str, Path, str]) -> int:
        name, config, _ = job
        log = log_dir / (name.replace("/", "_") + ".log")
        with log.open("a", encoding="utf-8") as sink:
            rc = subprocess.call(
                [
                    "counting",
                    "run",
                    str(config),
                    "--run-name",
                    name,
                    "--concurrency",
                    str(args.concurrency),
                ],
                cwd=REPO,
                stdout=sink,
                stderr=sink,
            )
        print(f"{name}: exit {rc}", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, jobs))
    clean = codes.count(0)
    print(f"done: {clean}/{len(codes)} clean", flush=True)
    return 0 if clean == len(codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
