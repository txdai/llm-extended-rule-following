#!/usr/bin/env python3
"""Solo capacities of the state-machine ladder: each level alone, per model.

    python scripts/run_state_solo.py --dry-run
    python scripts/run_state_solo.py

Four measurements per model, one per machine level. The scan variable is the
machine load. Level 1 (table lookup) is expected to survive to extreme loads,
so it steps 50 at a time to 1000 rather than densely; the derived-state levels
scan densely from 1 under the same budget and stop criteria as every other
measurement in the paper.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    {"id": "anthropic/claude-sonnet-5", "variant": "think",
     "reasoning_effort": "medium", "temperature_locked": True},
    {"id": "openai/gpt-5.6-terra", "variant": "think",
     "reasoning_effort": "medium", "temperature_locked": True},
    {"id": "google/gemini-3.7-flash", "reasoning_effort": "medium"},
]

LEVELS = {
    # level: (start, step, ceiling)
    1: (50, 50, 1000),
    2: (1, 1, 1500),   # ceiling capped by the word pool (chains need m+1 words)
    3: (1, 1, 1500),
    4: (1, 1, 1500),
}


def config_for(level: int) -> dict:
    start, step, stop = LEVELS[level]
    return {
        "extends": ["base.yaml", "providers/openrouter.yaml", "strategies/scan.yaml"],
        "models": MODELS,
        "task": {"name": "state_solo", "machine_level": level},
        "max_output_tokens": 8192,
        "retry_max_output_tokens": 8192,
        "escalate_output_budget": False,
        "search": {
            "start": start, "step": step, "stop": stop,
            "stop_after_consecutive_errors": 10,
            "stop_at_first_mistake_multiple": None,
            "stop_when_resolved": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", type=int, help="only these levels")
    args = parser.parse_args()

    levels = [l for l in LEVELS if not args.only or l in args.only]

    def run_one(level: int) -> None:
        cfg = config_for(level)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                         dir=REPO_ROOT / "configs" / "experiments") as handle:
            yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
            path = Path(handle.name)
        cmd = ["counting", "run", str(path),
               "--run-name", f"state-solo/L{level}",
               "--concurrency", str(args.concurrency)]
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"=== start L{level}", file=sys.stderr)
        try:
            subprocess.run(cmd, check=False,
                           stdout=None if args.dry_run else subprocess.DEVNULL)
        finally:
            path.unlink(missing_ok=True)
        print(f"=== done  L{level}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=len(levels)) as pool:
        list(pool.map(run_one, levels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
