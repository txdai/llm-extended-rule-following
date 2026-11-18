#!/usr/bin/env python3
"""Refresh the external benchmark scores used in the correlation analysis.

    python scripts/fetch_benchmarks.py
    python scripts/fetch_benchmarks.py --dry-run

Pulls Epoch AI's combined benchmark table and splits it into the per-benchmark
CSVs under results/benchmark_data/, preserving the column layout the analysis
already reads so no downstream code has to change.

What this refreshes: the benchmarks Epoch scores itself and publishes in
https://epoch.ai/data/benchmarks.csv -- GPQA Diamond, SWE-Bench Verified,
OTIS Mock AIME, MATH Level 5, FrontierMath, SimpleQA Verified, and the puzzle
sets. Two of the paper's three headline benchmarks are in that set.

It also refreshes the *_external.csv files and the Epoch Capabilities Index:
since mid-2026 Epoch publishes the full dashboard, external benchmarks
included, as a bulk bundle at https://epoch.ai/data/benchmark_data.zip. The
bundle's CSVs are copied over results/benchmark_data/ wholesale (our derived
counting_benchmark_* outputs are never in the bundle and are left alone).

Epoch AI's data is CC-BY; see results/benchmark_data/README.md for the citation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "benchmark_data"
PROVENANCE = OUT_DIR / "provenance.json"

BENCHMARKS_URL = "https://epoch.ai/data/benchmarks.csv"
EXTERNAL_BUNDLE_URL = "https://epoch.ai/data/benchmark_data.zip"

# Epoch task name -> the filename the analysis already expects. Anything not
# listed here is written under a slugified name, so a newly added Epoch task
# shows up rather than being silently dropped.
TASK_FILES = {
    "GPQA diamond": "gpqa_diamond.csv",
    "SWE-Bench verified": "swe_bench_verified.csv",
    "OTIS Mock AIME 2024-2025": "otis_mock_aime_2024_2025.csv",
    "MATH level 5": "math_level_5.csv",
    "SimpleQA Verified": "simpleqa_verified.csv",
    "Chess Puzzles": "chess_puzzles.csv",
    "FrontierMath-2025-02-28-Private": "frontiermath.csv",
    "FrontierMath-Tier-4-2025-07-01-Private": "frontiermath_tier_4.csv",
}

# Source column in benchmarks.csv -> destination column in the per-task CSV.
COLUMN_MAP = [
    ("model", "Model version"),
    ("mean_score", "mean_score"),
    ("best_score", "Best score (across scorers)"),
    ("Version release date", "Release date"),
    ("Organization", "Organization"),
    ("Country (of organization)", "Country"),
    ("Training compute (FLOP)", "Training compute (FLOP)"),
    ("Training compute notes", "Training compute notes"),
    ("stderr", "stderr"),
    ("log viewer", "Log viewer"),
    ("logs", "Logs"),
    ("started_at", "Started at"),
    ("id_runs", "id"),
]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".csv"


def fetch_csv(url: str) -> list[dict[str, str]]:
    request = urllib.request.Request(url, headers={"accept": "text/csv"})
    with urllib.request.urlopen(request, timeout=180) as response:
        text = response.read().decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def fetch_external_bundle(dry_run: bool) -> dict[str, int]:
    """Download Epoch's full benchmark bundle and copy its CSVs into OUT_DIR."""
    import io as _io
    import zipfile
    with urllib.request.urlopen(EXTERNAL_BUNDLE_URL, timeout=120) as resp:
        payload = resp.read()
    copied: dict[str, int] = {}
    with zipfile.ZipFile(_io.BytesIO(payload)) as bundle:
        for name in bundle.namelist():
            base = name.rsplit("/", 1)[-1]
            if "/" in name.rstrip("/") and not name.endswith(base):
                continue
            if not base.endswith(".csv") or name.count("/") > 0:
                continue
            data = bundle.read(name)
            copied[base] = data.count(b"\n")
            if not dry_run:
                (OUT_DIR / base).write_bytes(data)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="report changes, write nothing")
    args = parser.parse_args()

    print(f"fetching {BENCHMARKS_URL}", file=sys.stderr)
    rows = fetch_csv(BENCHMARKS_URL)
    print(f"  {len(rows)} scored runs", file=sys.stderr)

    by_task: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written: dict[str, Any] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task, task_rows in sorted(by_task.items()):
        filename = TASK_FILES.get(task, slugify(task))
        path = OUT_DIR / filename

        # Newest first, matching how the existing files are ordered.
        task_rows.sort(key=lambda r: (r.get("Version release date") or ""), reverse=True)
        out_rows = [
            {dst: (row.get(src) or "") for src, dst in COLUMN_MAP} for row in task_rows
        ]

        # Merge rather than overwrite. Epoch's combined table is not a strict
        # superset of the per-benchmark dashboard exports already on disk -- a
        # handful of older runs appear only in the export -- so replacing the
        # file would silently drop them. Union on the run id, new row wins.
        previous = None
        kept_only_old = 0
        if path.exists():
            with path.open(encoding="utf-8", newline="") as handle:
                existing = list(csv.DictReader(handle))
            previous = len(existing)
            new_ids = {r["id"] for r in out_rows if r["id"]}
            carried = [
                {dst: (row.get(dst) or "") for _, dst in COLUMN_MAP}
                for row in existing
                if row.get("id") and row["id"] not in new_ids
            ]
            kept_only_old = len(carried)
            out_rows.extend(carried)
            out_rows.sort(key=lambda r: (r.get("Release date") or ""), reverse=True)

        latest = max((r["Release date"] for r in out_rows if r["Release date"]), default="")
        if previous is None:
            delta = "  (new)"
        else:
            delta = f"  (was {previous}, +{len(out_rows) - previous}"
            delta += f", {kept_only_old} carried)" if kept_only_old else ")"
        mark = " " if TASK_FILES.get(task) else "+"
        print(f"  {mark} {filename:<44} {len(out_rows):>4} rows{delta}   latest {latest}",
              file=sys.stderr)

        written[filename] = {
            "task": task,
            "rows": len(out_rows),
            "previous_rows": previous,
            "carried_from_previous": kept_only_old,
            "latest_release_date": latest,
        }

        if not args.dry_run:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[dst for _, dst in COLUMN_MAP])
                writer.writeheader()
                writer.writerows(out_rows)

    bundle_files = fetch_external_bundle(args.dry_run)
    for name, rows in sorted(bundle_files.items()):
        if name not in written:
            print(f"  ~ {name:<44} {rows:>4} rows (bundle)", file=sys.stderr)

    untouched = sorted(
        p.name for p in OUT_DIR.glob("*.csv")
        if p.name not in written and p.name not in bundle_files
        and not p.name.startswith("counting_benchmark")
    )

    if not args.dry_run:
        PROVENANCE.write_text(
            json.dumps(
                {
                    "refreshed_at_utc": stamp,
                    "source": [BENCHMARKS_URL, EXTERNAL_BUNDLE_URL],
                    "license": "CC-BY 4.0, Epoch AI (see README.md)",
                    "refreshed_files": written,
                    "bundle_files": sorted(bundle_files),
                    "not_refreshed": {"files": untouched},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"\n  refreshed {len(written)} files + {len(bundle_files)} from the bundle",
          file=sys.stderr)
    print(f"  left alone {len(untouched)} files "
          f"(see {PROVENANCE.relative_to(REPO_ROOT)})", file=sys.stderr)
    if args.dry_run:
        print("\n  dry run: nothing written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
