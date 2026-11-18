#!/usr/bin/env python3
"""Summarise a scan run.

    python scripts/report_scan.py results/openrouter-probe-50
    python scripts/report_scan.py results/openrouter-probe-50 --csv out.csv

Reads the per-model summaries a run wrote and reports the counting-capacity
distribution, the failure modes, and the models that never answered at all.

The distinction the report keeps throughout is the one the scan makes:

    correct   answered exactly
    mistake   answered, wrong number   -- the measurement
    error     no parseable integer     -- ran out of budget, looped, or empty

A model whose failures are sporadic -- some lengths past the first mistake still
correct -- is flagged, because capacity read as "first mistake minus one"
understates such a model, and the shape is itself a finding.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load(run_dir: Path) -> list[dict[str, Any]]:
    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise SystemExit(f"no models/ under {run_dir}")
    out = []
    for path in sorted(models_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = payload.get("summary") or {}
        if "stable_count" not in summary:
            continue
        # A model can appear twice, once with reasoning on and once off. Label
        # the rows by run key so the two are never confused for each other.
        summary["label"] = summary.get("run_key") or summary["model"]
        out.append(summary)
    return out


def bucket(cc: int) -> str:
    if cc == 0:
        return "0 (fails immediately)"
    if cc < 10:
        return "1-9"
    if cc < 25:
        return "10-24"
    if cc < 50:
        return "25-49"
    return "50 (ceiling, no mistake)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--csv", type=Path, help="also write a per-model CSV")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = load(args.run_dir)
    if not rows:
        raise SystemExit(f"no completed models in {args.run_dir}")

    ccs = sorted(r["stable_count"] for r in rows)
    print(f"=== {args.run_dir.name}: {len(rows)} models ===\n")
    print(f"  counting capacity   median {statistics.median(ccs):.0f}   "
          f"mean {statistics.mean(ccs):.1f}   min {ccs[0]}   max {ccs[-1]}")

    counts: dict[str, int] = {}
    for r in rows:
        counts[bucket(r["stable_count"])] = counts.get(bucket(r["stable_count"]), 0) + 1
    print()
    for key in ("0 (fails immediately)", "1-9", "10-24", "25-49", "50 (ceiling, no mistake)"):
        n = counts.get(key, 0)
        bar = "#" * int(46 * n / len(rows))
        print(f"  CC {key:<26} {n:>4}  {bar}")

    perfect = [r for r in rows if r.get("first_mistake") is None and r.get("error_count", 0) == 0]
    all_error = [r for r in rows if r.get("correct_count", 0) == 0]
    any_error = [r for r in rows if r.get("error_count", 0) > 0]
    print(f"\n  perfect through the ceiling      {len(perfect):>4}")
    print(f"  never answered correctly         {len(all_error):>4}")
    print(f"  produced at least one error      {len(any_error):>4}")

    # Sporadic failure: correct answers exist above the first mistake.
    sporadic = []
    for r in rows:
        fm = r.get("first_mistake")
        if fm is None:
            continue
        later_ok = [p for p in (r.get("curve") or [])
                    if p["length"] > fm and p["outcome"] == "correct"]
        if later_ok:
            sporadic.append((len(later_ok), r))
    print(f"  sporadic (correct past the first mistake)  {len(sporadic):>4}")

    print(f"\n  strongest {args.top}:")
    print(f"    {'measurement':<46} {'CC':>4} {'ok':>4} {'wrong':>6} {'err':>4}  stopped")
    for r in sorted(rows, key=lambda r: (-r["stable_count"], r["label"]))[: args.top]:
        print(f"    {r['label']:<46} {r['stable_count']:>4} {r.get('correct_count',0):>4} "
              f"{r.get('mistake_count',0):>6} {r.get('error_count',0):>4}  "
              f"{r.get('stopped_early') or 'range end'}")

    zero = [r for r in rows if r["stable_count"] == 0]
    if zero:
        print(f"\n  {len(zero)} models with CC=0 (first mistake or error at length 1):")
        for r in sorted(zero, key=lambda r: r["label"])[:15]:
            fm, fe = r.get("first_mistake"), r.get("first_error")
            why = f"mistake at {fm}" if fm == 1 else (f"error at {fe}" if fe == 1 else "see curve")
            print(f"    {r['label']:<46} {why}")
        if len(zero) > 15:
            print(f"    ... and {len(zero) - 15} more")

    if sporadic:
        print(f"\n  most sporadic (correct answers above the first mistake):")
        for n, r in sorted(sporadic, reverse=True, key=lambda x: x[0])[:8]:
            print(f"    {r['label']:<46} CC={r['stable_count']:<4} "
                  f"{n} correct above length {r['first_mistake']}")

    if args.csv:
        fields = ["label", "model", "variant", "temperature_locked", "reasoning_effort",
              "stable_count", "first_mistake", "first_error", "correct_count",
                  "mistake_count", "error_count", "lengths_probed", "highest_length_probed",
                  "stopped_early"]
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: -r["stable_count"]))
        print(f"\n  wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
