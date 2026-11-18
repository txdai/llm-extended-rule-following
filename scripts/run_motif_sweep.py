#!/usr/bin/env python3
"""Run the motif-perturbation sweep under the SCC scan protocol.

    python scripts/run_motif_sweep.py --dry-run
    python scripts/run_motif_sweep.py --concurrency 4

One run per motif, into results/motif-sweep/<motif-id>/. Splitting by directory
rather than by run key keeps every motif's record a complete scan that the same
loader, datastore and fitting code can read without special cases.

The motif set is the one used for the syntax-sensitivity result: the item is
varied (letter, doubled letter, Greek, Han) and the delimiter is varied (comma,
space, pipe, section sign), each against the same baseline. The models are the
three carried in the main text figure.
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

MOTIFS = [
    ("baseline-a-comma",    "a",  ", ",  "comma"),
    ("letter-b-comma",      "b",  ", ",  "comma"),
    ("letter-x-comma",      "x",  ", ",  "comma"),
    ("double-aa-comma",     "aa", ", ",  "comma"),
    ("greek-alpha-comma",   "α", ", ", "comma"),
    ("greek-beta-comma",    "β", ", ", "comma"),
    ("chinese-han-comma",   "汉", ", ", "comma"),
    ("chinese-zhong-comma", "中", ", ", "comma"),
    ("letter-a-space",      "a",  " ",   "space"),
    ("letter-a-pipe",       "a",  " | ", "pipe"),
    ("letter-a-section",    "a",  " § ", "section"),
]

MODELS = [
    {"id": "anthropic/claude-sonnet-5", "variant": "think",
     "reasoning_effort": "medium", "temperature_locked": True},
    {"id": "openai/gpt-5.6-terra", "variant": "think",
     "reasoning_effort": "medium", "temperature_locked": True},
    {"id": "google/gemini-3.7-flash", "reasoning_effort": "medium"},
]


def config_for(item: str, delimiter: str, delimiter_label: str) -> dict:
    return {
        "extends": ["base.yaml", "providers/openrouter.yaml",
                    "tasks/homogeneous_count.yaml", "strategies/scan.yaml"],
        "models": MODELS,
        "task": {"name": "homogeneous_count", "item": item,
                 "delimiter": delimiter, "delimiter_label": delimiter_label},
        "max_output_tokens": 8192,
        "retry_max_output_tokens": 8192,
        "escalate_output_budget": False,
        "search": {
            "start": 1, "step": 1, "stop": 3000,
            "stop_after_consecutive_errors": 10,
            # Same criterion-native rule as the main sweep: scan until the
            # fitted transition is settled, not to a multiple of a landmark.
            "stop_at_first_mistake_multiple": None,
            "stop_when_resolved": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--concurrency", type=int, default=3,
                        help="models in parallel within one motif")
    parser.add_argument("--motif-parallel", type=int, default=4,
                        help="motifs running at once; motifs are independent runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", help="only these motif ids")
    args = parser.parse_args()

    motifs = [m for m in MOTIFS if not args.only or m[0] in args.only]
    print(f"{len(motifs)} motifs x {len(MODELS)} models", file=sys.stderr)

    def run_one(motif):
        motif_id, item, delimiter, label = motif
        cfg = config_for(item, delimiter, label)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                         dir=REPO_ROOT / "configs" / "experiments") as handle:
            yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
            path = Path(handle.name)
        cmd = ["counting", "run", str(path),
               "--run-name", f"motif-sweep/{motif_id}",
               "--concurrency", str(args.concurrency)]
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"=== start {motif_id}  item={item!r} delimiter={delimiter!r}", file=sys.stderr)
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL)
        finally:
            path.unlink(missing_ok=True)
        print(f"=== done  {motif_id}", file=sys.stderr)

    # Motifs are independent runs writing to separate directories, so they can
    # overlap; within one motif the three models already run in parallel.
    with ThreadPoolExecutor(max_workers=max(1, args.motif_parallel)) as pool:
        list(pool.map(run_one, motifs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
