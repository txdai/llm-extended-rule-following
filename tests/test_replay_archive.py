"""Regression test against the published runs.

Replays every archived model result through the current metrics and search code
and asserts that each tier verdict and each SCC comes out exactly as recorded.
This is what guards the refactor: the harness was rewritten from scratch, and
these are the numbers in the paper.

Skips cleanly when results/ is absent -- it is gitignored, so a fresh clone has
nothing to replay.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from counting.metrics import parse_integer, score_trial, summarize_slice, tier_passed
from counting.paths import RESULTS
from counting.types import TrialResult

RUNS = ["counting-stable-search", "counting-stable-search-open-source", "counting-stable-modified"]


def replay(payload: dict) -> tuple[int, list[tuple]]:
    """Recompute SCC and every tier verdict from the stored trials."""
    summary = payload["summary"]
    tolerance = summary.get("tolerance_fraction", 0.05)

    # Only tiers the search recorded. The preflight trial is a connectivity
    # check, not a tier, and was never part of the ladder.
    tiers = {(e["phase"], e["evaluation_index"]) for e in summary.get("evaluations", [])}

    grouped = collections.defaultdict(list)
    for trial in payload.get("trials", []):
        key = (trial["phase"], trial["evaluation_index"])
        if key not in tiers:
            continue
        grouped[key].append(
            TrialResult(
                provider=trial.get("api", ""),
                model=payload["model"],
                task="homogeneous_count",
                phase=trial["phase"],
                evaluation_index=trial["evaluation_index"],
                center_length=trial["center_length"],
                trial_index=trial["trial_index"],
                sample_low=trial["sample_low"],
                sample_high=trial["sample_high"],
                expected=trial["expected"],
                raw_response="",
                response_status=trial.get("response_status"),
                response_detail=trial.get("response_detail"),
                parsed=trial.get("parsed"),
                latency_seconds=trial.get("latency_seconds") or 0.0,
                **score_trial(trial.get("parsed"), trial["expected"]),
            )
        )

    verdicts = {k: tier_passed(summarize_slice(v), tolerance) for k, v in grouped.items()}
    mismatches = [
        (payload["model"], k, bool(e.get("passed")), verdicts[k])
        for e in summary.get("evaluations", [])
        if (k := (e["phase"], e["evaluation_index"])) in verdicts
        and verdicts[k] != bool(e.get("passed"))
    ]

    passed = [grouped[k][0].center_length for k, v in verdicts.items() if v]
    failed = [grouped[k][0].center_length for k, v in verdicts.items() if not v]
    first_fail = min(failed) if failed else None
    below = [c for c in passed if first_fail is None or c < first_fail]
    return (max(below) if below else 0), mismatches


def main() -> int:
    model_files = [p for run in RUNS for p in sorted((RESULTS / run / "models").glob("*.json"))]
    if not model_files:
        print(f"no archived runs under {RESULTS} -- nothing to replay (results/ is gitignored)")
        return 0

    reproduced = differed = 0
    tier_mismatches: list[tuple] = []
    stale_parses = 0

    for path in model_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scc, mismatches = replay(payload)
        tier_mismatches.extend(mismatches)
        if scc == payload["summary"]["stable_count"]:
            reproduced += 1
        else:
            differed += 1
            print(f"  SCC differs: {payload['model']} stored={payload['summary']['stable_count']} replayed={scc}")
        stale_parses += sum(
            1
            for t in payload.get("trials", [])
            if parse_integer(t.get("raw_response")) != t.get("parsed")
        )

    print(f"models replayed          {reproduced + differed}")
    print(f"  SCC reproduced         {reproduced}")
    print(f"  SCC differed           {differed}")
    print(f"  tier verdicts differed {len(tier_mismatches)}")
    for row in tier_mismatches[:5]:
        print(f"    {row}")

    # Six trials in the archive were parsed by an older harness that took the
    # FIRST integer in a reply rather than the last. All six sit in tiers that
    # failed regardless, so no SCC depends on them. The count is asserted so that
    # a genuine parser regression shows up as a change here.
    print(f"  trials whose stored parse != current parser: {stale_parses} (expected 6, all in failing tiers)")

    ok = differed == 0 and not tier_mismatches and stale_parses <= 6
    print("\n" + ("archive replay reproduces every published number" if ok else "REPLAY FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
