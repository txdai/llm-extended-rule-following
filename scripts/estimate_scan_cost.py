#!/usr/bin/env python3
"""Estimate what a linear scan will cost, and what the levers are worth.

    python scripts/estimate_scan_cost.py                 # per-model measured CC
    python scripts/estimate_scan_cost.py --cc 500        # flat assumption instead
    python scripts/estimate_scan_cost.py --multiple 1.5

Prices come from results/model_catalogue/openrouter_models.csv. Token rates come
from the archived runs under results/ where a model was measured before, and
from the measured median otherwise -- not from a guess, because the guess is
wrong by an order of magnitude for reasoning models.

Capacity is taken per model from the archived runs, not assumed flat. That
matters more than any other input: the measured distribution is heavily skewed --
30 of 126 models score 0, the median is 44, and only 3 exceed 1000 -- so a flat
CC=1000 overstates the request count by more than an order of magnitude.

The archived numbers come from the ladder, which passes a tier at nMAE <= 5%
over 16 jittered samples. The scan is stricter: it wants an exact answer at every
single length, so its first mistake lands at or below the ladder's figure. Using
the ladder number is therefore conservative -- the real scan should cost less.

Two further facts drive everything:

  * Input is quadratic in the stop multiple. A scan to 2*CC issues 2*CC requests
    whose prompt grows linearly, so total input tokens go as (2*CC)^2. Halving
    the multiple quarters the input bill.

  * Output cost is bimodal, not proportional to length. The archive shows most
    models answering with a bare integer -- the median request emits 5 output
    tokens, and 79 of 123 models have a median under 100. A minority of thinking
    models instead emit thousands per request regardless of length. Modelling
    output as "tokens per counted item" is wrong for both groups, so this script
    uses each model's measured median request instead, falling back to the class
    median only for models never measured.

Prompt caching cannot touch output, which is why it saves far less overall than
the input-side arithmetic suggests.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOGUE = REPO_ROOT / "results" / "model_catalogue" / "openrouter_models.csv"
ARCHIVE_RUNS = [
    "counting-stable-search",
    "counting-stable-search-open-source",
    "counting-openrouter-formal",
]

INPUT_TOK_PER_ITEM = 2.38   # measured median across the archive
PROMPT_HEADER_TOK = 30
# Measured medians of *output tokens per request*, over the archived runs.
# Output is near-constant per request for most models rather than growing with
# length: the median model answers in 5 tokens whatever it was asked to count.
DEFAULT_OUT_REASONING = 1202.0
DEFAULT_OUT_PLAIN = 4.0


def measured_output_rates() -> dict[str, float]:
    """Median output tokens per request, per model, from archived runs."""
    rates: dict[str, float] = {}
    for run in ARCHIVE_RUNS:
        models_dir = REPO_ROOT / "results" / run / "models"
        if not models_dir.exists():
            continue
        for path in sorted(models_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            outputs = [
                t["output_tokens"]
                for t in payload.get("trials", [])
                if t.get("output_tokens") is not None
            ]
            if outputs:
                rates[payload["model"]] = statistics.median(outputs)
    return rates


def measured_capacities() -> dict[str, int]:
    """Measured counting capacity per model, from whatever runs are on disk."""
    caps: dict[str, int] = {}
    _ = None
    for run in ARCHIVE_RUNS:
        models_dir = REPO_ROOT / "results" / run / "models"
        if not models_dir.exists():
            continue
        for path in sorted(models_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            value = (payload.get("summary") or {}).get("stable_count")
            if isinstance(value, int):
                caps[payload["model"]] = value
    return caps


def match_rate(model_id: str, rates: dict[str, float]) -> float | None:
    """Archived runs used bare ids; the roster is namespaced."""
    if model_id in rates:
        return rates[model_id]
    tail = model_id.split("/", 1)[-1]
    if tail in rates:
        return rates[tail]
    for name, rate in rates.items():
        if name.split("/", 1)[-1] == tail:
            return rate
    return None


def scan_cost(
    cc: int,
    multiple: float,
    prompt_usd: float,
    completion_usd: float,
    cache_read_usd: float | None,
    out_per_request: float,
    output_cap: int,
    use_cache: bool,
    batch_discount: float,
) -> dict[str, float]:
    """Cost of scanning length 1 upward, stopping at `multiple` x cc."""
    highest = int(multiple * cc)
    lengths = range(1, highest + 1)
    fresh_in = sum(INPUT_TOK_PER_ITEM * n + PROMPT_HEADER_TOK for n in lengths)
    cached_in = sum(INPUT_TOK_PER_ITEM * (n - 1) + PROMPT_HEADER_TOK for n in lengths)
    out = highest * min(float(output_cap), out_per_request)

    p = prompt_usd * (1 - batch_discount)
    c = completion_usd * (1 - batch_discount)
    if use_cache and cache_read_usd is not None:
        cr = cache_read_usd * (1 - batch_discount)
        input_cost = cached_in / 1e6 * cr + (fresh_in - cached_in) / 1e6 * p
    else:
        input_cost = fresh_in / 1e6 * p

    return {
        "requests": highest,
        "input_usd": input_cost,
        "output_usd": out / 1e6 * c,
        "total_usd": input_cost + out / 1e6 * c,
        # A model whose typical reply already sits near the cap will start
        # returning nothing as soon as it needs a little more room.
        "near_cap": out_per_request > 0.5 * output_cap,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--cc",
        type=int,
        help="flat counting capacity for every model; default uses measured values",
    )
    parser.add_argument("--multiple", type=float, default=2.0, help="stop at this x first mistake")
    parser.add_argument("--output-cap", type=int, default=8192)
    parser.add_argument("--no-cache", action="store_true", help="assume no prefix cache hits")
    parser.add_argument("--exclude-pro", action="store_true", help="drop -pro / o1-pro tier models")
    parser.add_argument("--top", type=int, default=12, help="how many models to itemise")
    args = parser.parse_args()

    if not CATALOGUE.exists():
        print(f"no catalogue at {CATALOGUE}; run scripts/fetch_openrouter_models.py first",
              file=sys.stderr)
        return 1

    rows = list(csv.DictReader(CATALOGUE.open(encoding="utf-8")))
    if args.exclude_pro:
        rows = [r for r in rows if not r["id"].endswith("-pro") and "o1-pro" not in r["id"]]

    rates = measured_output_rates()
    caps = measured_capacities()
    print(f"archive: {len(rates)} models with token usage, {len(caps)} with a capacity",
          file=sys.stderr)

    # Capacity trends upward with release year (2024 median 32, 2026 median 52,
    # and the p75 rises from 44 to 128). Unmeasured models are newer than
    # measured ones by about seven months, so a single global median would bias
    # the estimate low. Fall back per release year instead.
    by_year: dict[str, list[int]] = {}
    for row in rows:
        value = match_rate(row["id"], {k: float(v) for k, v in caps.items()})
        if value is not None and row["release_date"]:
            by_year.setdefault(row["release_date"][:4], []).append(int(value))
    year_median = {y: int(statistics.median(v)) for y, v in by_year.items()}
    global_median = int(statistics.median(caps.values())) if caps else 100

    def fallback_for(row: dict[str, Any]) -> int:
        year = (row.get("release_date") or "")[:4]
        if year in year_median:
            return year_median[year]
        # No measured peer from that year: use the latest year that has one.
        return year_median[max(year_median)] if year_median else global_median

    results = []
    matched = 0
    cc_matched = 0
    for r in rows:
        prompt = float(r["prompt_usd_per_mtok"] or 0)
        completion = float(r["completion_usd_per_mtok"] or 0)
        cache_raw = r["cache_read_usd_per_mtok"]
        cache = float(cache_raw) if cache_raw not in (None, "") else None
        reasoning = r["reasoning_supported"] == "True"

        rate = match_rate(r["id"], rates)
        if rate is not None:
            matched += 1
        else:
            rate = DEFAULT_OUT_REASONING if reasoning else DEFAULT_OUT_PLAIN

        if args.cc is not None:
            cc = args.cc
        else:
            measured = match_rate(r["id"], {k: float(v) for k, v in caps.items()})
            cc = int(measured) if measured is not None else fallback_for(r)
            cc_matched += measured is not None

        cost = scan_cost(
            cc, args.multiple, prompt, completion, cache, rate,
            args.output_cap, use_cache=not args.no_cache, batch_discount=0.0,
        )
        cost["id"] = r["id"]
        cost["rate"] = rate
        cost["cc"] = cc
        cost["reasoning"] = reasoning
        results.append(cost)

    print(f"  {matched}/{len(rows)} matched an archived token measurement", file=sys.stderr)
    if args.cc is None:
        summary_by_year = ", ".join(f"{y}:{v}" for y, v in sorted(year_median.items()))
        print(f"  {cc_matched}/{len(rows)} matched an archived capacity; rest use the "
              f"median for their release year ({summary_by_year})\n", file=sys.stderr)
    else:
        print(f"  capacity forced to {args.cc} for every model\n", file=sys.stderr)

    total = sum(c["total_usd"] for c in results)
    total_in = sum(c["input_usd"] for c in results)
    total_out = sum(c["output_usd"] for c in results)
    requests = sum(c["requests"] for c in results)

    flags = []
    if args.no_cache:
        flags.append("no cache")
    if args.exclude_pro:
        flags.append("no -pro tier")
    label = f"  [{', '.join(flags)}]" if flags else ""

    cc_label = f"CC={args.cc} (flat)" if args.cc is not None else "CC per model (measured)"
    print(f"=== {len(results)} models, {cc_label}, stop at {args.multiple}x, "
          f"cap {args.output_cap}{label} ===")
    print(f"  requests   {requests:>12,}")
    print(f"  input      ${total_in:>11,.0f}   ({100*total_in/total:.0f}%)")
    print(f"  output     ${total_out:>11,.0f}   ({100*total_out/total:.0f}%)")
    print(f"  TOTAL      ${total:>11,.0f}")

    results.sort(key=lambda c: c["total_usd"], reverse=True)
    print(f"\n  most expensive {args.top}:")
    print(f"    {'model':<42} {'CC':>6} {'total':>9} {'in':>8} {'out':>9} {'out tok/req':>12}")
    for c in results[: args.top]:
        flag = " *" if c["near_cap"] else ""
        print(f"    {c['id']:<42} {c['cc']:>6} ${c['total_usd']:>8,.2f} "
              f"${c['input_usd']:>7,.2f} ${c['output_usd']:>8,.2f} {c['rate']:>11,.0f}{flag}")

    near = [c for c in results if c["near_cap"]]
    if near:
        print(f"\n  * {len(near)} models typically emit more than half the {args.output_cap}-token")
        print("    cap per reply. They will start returning nothing as soon as they need a")
        print("    little more room, so their measured capacity reflects verbosity against")
        print("    the cap as much as tracking ability. Worth separating in the analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
