#!/usr/bin/env python3
"""Aggregate per-trace classifications into the mode table of the
reasoning-trace appendix.

Inputs:
  results/trace-classification/trace_labels.jsonl -- one label record per
    classified trace (strategy, failure_mode, trace_answer_relation), produced
    by the batch classification pass over the harvested traces.
  results/harvested-traces-enriched/*.jsonl -- enriched trace records
    (scripts/enrich_harvested_traces.py), used for the validation rates and
    the token-expenditure medians.

Prints every number quoted in the appendix and asserts internal consistency.
"""
from __future__ import annotations

import glob
import json
import re
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FAMILY_RULES = [
    # (family, pattern over "failure strategy relation"); first match wins
    ("delivery-failure",
     r"stranded|reasoning-only|reasoning-channel|no-answer-emitted|formatting"
     r"|extraction-artifact|output-bleed|bleed|refus|filtered"),
    ("runaway-enumeration", r"runaway|overrun|past-end|past-n|momentum"),
    ("verification-livelock",
     r"death-loop|livelock|budget|exhaust|timeout|no-commit|nontermination"
     r"|non-termination|oscillat|restart|stall|loop-no|deliberation-loop"
     r"|repetition|recount-loop|doubt|overrun-no-answer|discarded|arbitrat"
     r"|conflict|lowest-picked|overwritten"),
    ("round-number-capture", r"round|snap|attractor|capitulat"),
    ("boundary-slip",
     r"remainder|tail|boundary-error|off-by|partial|drift|stops-short"
     r"|overshoot|dropped-one|miscount|slip|inflat|stops-early|early-stop"
     r"|overcount|offset-error|self-correct-to-wrong|self-revision"),
    ("fabricated-operand",
     r"fabricat|phantom|invented|launder|unmeasured|assumed-layout"
     r"|grid-tiling|premise|circular-length"),
    ("counts-own-copy",
     r"regenerat|transcri|own-copy|copy-drift|self-generated-list|terminus"),
    ("anchor-then-ratify",
     r"anchor|ratif|precede|prior|first-impression|opening-estimate"
     r"|estimate-first|answer-first|asserted-then|is-opening|latent-estimate"
     r"|early-estimate|recall|benchmark-prior|triangulation|theater"
     r"|confirms-error|concordant|agreement"),
    ("opaque-or-vacuous-trace",
     r"vacuous|narrat|opaque|summary|no-arithmetic|without-visible|stub"
     r"|decoupl|elided|truncated|no-enumeration|without-execution|assert"
     r"|exceeds-trace-progress"),
    ("anchor-then-ratify", r"abandon|guess|estimate|surrender|mismatch"),
]

GROUPS = {
    "Answer decoupled from trace": ["anchor-then-ratify", "opaque-or-vacuous-trace"],
    "Enumeration of own generation": ["counts-own-copy", "runaway-enumeration",
                                      "round-number-capture"],
    "Slips in genuine counts": ["boundary-slip", "fabricated-operand"],
    "Commitment and delivery": ["verification-livelock", "delivery-failure"],
}

DERIVABLE = re.compile(
    r"answer-from-trace|answer-from-self|answer-from-early|answer-in-reason")


def family(t: dict) -> str:
    text = " ".join([(t.get("failure_mode") or ""), (t.get("strategy") or ""),
                     (t.get("trace_answer_relation") or "")]).lower()
    # slips only count when the words appear in the failure/strategy fields
    fs = " ".join([(t.get("failure_mode") or ""), (t.get("strategy") or "")]).lower()
    for fam, pat in FAMILY_RULES:
        hay = fs if fam in ("runaway-enumeration", "verification-livelock",
                            "round-number-capture", "boundary-slip") else text
        if re.search(pat, hay):
            return fam
    return "other"


def main() -> int:
    labels = [json.loads(line) for line in
              (REPO / "results/trace-classification/trace_labels.jsonl").open()]
    wrong = [t for t in labels if not t["correct"]]
    right = [t for t in labels if t["correct"]]
    fams = Counter(family(t) for t in wrong)
    print(f"classified {len(labels)} ({len(wrong)} incorrect, {len(right)} "
          "correct near/above capacity)")
    total = 0
    for group, members in GROUPS.items():
        for m in members:
            print(f"  {group:32s} {m:28s} {fams.get(m, 0):4d}")
            total += fams.get(m, 0)
    other = fams.get("other", 0)
    print(f"  unassigned: {other}")
    assert total + other == len(wrong)

    nd = sum(1 for t in wrong
             if not (DERIVABLE.match((t.get("trace_answer_relation") or "").lower())
                     and "fabricat" not in (t.get("trace_answer_relation") or "")))
    print(f"incorrect traces whose answer is not derivable from the trace: "
          f"{nd}/{len(wrong)} ({nd/len(wrong):.0%})")
    above_ok = [t for t in right if t["relative_to_cc"] == "above"]
    no_work = sum(1 for t in above_ok if re.search(
        r"vacuous|narrat|opaque|summary|truncated|anchor|assert|precede|prior"
        r"|estimate|readout|impression",
        " ".join([(t.get("strategy") or ""),
                  (t.get("trace_answer_relation") or "")]).lower()))
    print(f"correct above capacity with no executed enumeration: "
          f"{no_work}/{len(above_ok)}")

    rows = [json.loads(line) for f in
            glob.glob(str(REPO / "results/harvested-traces-enriched/*.jsonl"))
            for line in open(f)]
    if rows:
        for zone in ("below", "near", "above"):
            z = [r for r in rows if r["relative_to_cc"] == zone]
            ok = sum(r["correct"] for r in z)
            print(f"validation {zone}: {ok}/{len(z)} exact ({ok/len(z):.0%})")
        above = [r for r in rows if r["relative_to_cc"] == "above"]
        med = lambda xs: int(statistics.median(xs)) if xs else None
        print("above-capacity median reasoning tokens: "
              f"correct {med([r['tokens_reasoning'] for r in above if r['correct']])}, "
              f"wrong integer {med([r['tokens_reasoning'] for r in above if not r['correct'] and r['answer_returned'] is not None])}, "
              f"no integer {med([r['tokens_reasoning'] for r in above if r['answer_returned'] is None])}")
        none_ = [r for r in above if r["answer_returned"] is None]
        at_cap = sum(1 for r in none_ if r["finish"] == "length")
        print(f"no-integer records at token cap: {at_cap}/{len(none_)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
