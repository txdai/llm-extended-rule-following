#!/usr/bin/env python3
"""Phase 2: interference scans on the trio, loads calibrated from phase 1.

    OPENROUTER_API_KEY=... python scripts/launch_interference.py [--dry-run]

For each trio model (claude-sonnet-5, gpt-5.6-terra from ladder-full;
gemini-3.7-flash from the pilot runs), fit the three solo records
(r2-colortotal, r3-merge, r4-merge), convert each fitted boundary into the
task's own unit count (runs / special runs / regions, measured from the
builders at the boundary load), then scan counting under eight conditions:

    grid @ 30/60/90/120% of the model's r4-merge object capacity
    row2 @ 60% of r2-colortotal unit capacity
    row3 @ 60% of r3-merge unit capacity
    inert and pointer controls @ 90% of the grid capacity

Falls back to CC-ratio predictions for any solo record that is censored.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "configs" / "experiments"

TRIO = {
    "anthropic/claude-sonnet-5": {
        "cc": 89, "entry": {"id": "anthropic/claude-sonnet-5", "variant": "think",
                            "reasoning_effort": "medium"},
        "solo_runs": {k: REPO / f"results/ladder-full/{k}/claude-sonnet-5"
                      for k in ("r2-colortotal", "r3-merge", "r4-merge")},
    },
    "openai/gpt-5.6-terra": {
        "cc": 117, "entry": {"id": "openai/gpt-5.6-terra", "variant": "think",
                             "reasoning_effort": "medium"},
        "solo_runs": {k: REPO / f"results/ladder-full/{k}/gpt-5p6-terra"
                      for k in ("r2-colortotal", "r3-merge", "r4-merge")},
    },
    "google/gemini-3.7-flash": {
        "cc": 352, "entry": {"id": "google/gemini-3.7-flash",
                             "reasoning_effort": "medium"},
        "solo_runs": {"r2-colortotal": REPO / "results/ladder-rung2/color-total",
                      "r3-merge": REPO / "results/ladder-rung3/recolor-merge",
                      "r4-merge": REPO / "results/ladder-rung4/recolor-merge"},
    },
}
FALLBACK_UNIT_RATIO = {"r2-colortotal": 161 / 352, "r3-merge": 103 / 352,
                       "r4-merge": 85 / 352}  # units per CC, flash pilot


def fit_boundary(run_dir: Path) -> int | None:
    from counting.analysis.capacity import fit_transition
    trials = list(run_dir.glob("trials/*.jsonl"))
    if not trials:
        return None
    rows = sorted((json.loads(l) for f in trials for l in open(f)),
                  key=lambda r: r["expected"])
    t = fit_transition([r["expected"] for r in rows],
                       [r.get("parsed") == r.get("expected") for r in rows])
    return int(t.scc) if t.scc and not t.censored else None


def units_at(scan: str, boundary: int) -> int:
    from counting.tasks import build_task
    def measure(seed: int) -> int:
        rng = random.Random(seed)
        if scan == "r2-colortotal":
            stim = build_task("run_structure", probe="color_total").build(boundary, rng)
            cells = stim.user.split("\n\n")[1].split(" ")
            return sum(1 for i, c in enumerate(cells)
                       if c == "7" and (i == 0 or cells[i - 1] != "7"))
        if scan == "r3-merge":
            stim = build_task("rule_execution", rule="recolor_merge").build(boundary, rng)
            cells = stim.user.split("\n\n")[1].split(" ")
            return sum(1 for i, c in enumerate(cells)
                       if c in "73" and (i == 0 or cells[i - 1] != c))
        stim = build_task("grid_rule_execution", rule="recolor_merge").build(boundary, rng)
        grid = [r.split(" ") for r in stim.user.split("\n\n")[1].split("\n")]
        h, w = len(grid), len(grid[0])
        seen, count = set(), 0
        for r in range(h):
            for c in range(w):
                if (r, c) in seen or grid[r][c] not in "73":
                    continue
                stack = [(r, c)]; seen.add((r, c)); d = grid[r][c]; count += 1
                while stack:
                    rr, cc = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen
                                and grid[nr][nc] == d):
                            seen.add((nr, nc)); stack.append((nr, nc))
        return count
    return round(statistics.mean(measure(s) for s in range(30)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    jobs = []
    for model, spec in TRIO.items():
        caps = {}
        for scan, run_dir in spec["solo_runs"].items():
            boundary = fit_boundary(run_dir)
            if boundary is None:
                caps[scan] = round(FALLBACK_UNIT_RATIO[scan] * spec["cc"])
                print(f"{model} {scan}: unresolved solo, fallback units {caps[scan]}",
                      file=sys.stderr)
            else:
                caps[scan] = units_at(scan, boundary)
                print(f"{model} {scan}: boundary {boundary} -> {caps[scan]} units",
                      file=sys.stderr)
        grid_u = caps["r4-merge"]
        conds = [("grid", round(0.30 * grid_u), "grid-30"),
                 ("grid", round(0.60 * grid_u), "grid-60"),
                 ("grid", round(0.90 * grid_u), "grid-90"),
                 ("grid", round(1.20 * grid_u), "grid-120"),
                 ("row2", round(0.60 * caps["r2-colortotal"]), "row2-60"),
                 ("row3", round(0.60 * caps["r3-merge"]), "row3-60"),
                 ("inert", round(0.90 * grid_u), "inert-90"),
                 ("pointer", round(0.90 * grid_u), "pointer-90")]
        cc = spec["cc"]
        slug = model.split("/")[-1].replace(".", "p")
        for kind, size, tag in conds:
            conf = {
                "extends": ["base.yaml", "providers/openrouter.yaml",
                            "strategies/scan.yaml"],
                "models": [spec["entry"]],
                "task": {"name": "count_with_load", "load_kind": kind,
                         "load_size": max(3, size)},
                "max_output_tokens": 8192,
                "retry_max_output_tokens": 8192,
                "escalate_output_budget": False,
                "search": {"start": max(5, round(0.20 * cc)),
                           "step": max(1, round(cc / 100)),
                           "stop": round(1.5 * cc) + 60,
                           "stop_after_consecutive_errors": 10,
                           "stop_at_first_mistake_multiple": None,
                           "stop_when_resolved": True},
            }
            path = CONF / f"interfere_{slug}_{tag}.yaml"
            path.write_text(yaml.safe_dump(conf, sort_keys=False))
            jobs.append((f"interference/{slug}/{tag}", path))

    print(f"{len(jobs)} interference scans", file=sys.stderr)
    if args.dry_run:
        return 0
    log_dir = REPO / "results" / "interference" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(job):
        name, path = job
        with (log_dir / (name.replace("/", "_") + ".log")).open("w") as sink:
            rc = subprocess.call(["counting", "run", str(path), "--run-name", name,
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
