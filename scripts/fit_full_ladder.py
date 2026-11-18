#!/usr/bin/env python3
"""Fit every ladder scan and write the consolidated table the figures read.

    python scripts/fit_full_ladder.py

Output: results/ladder-full/fits.csv with one row per (scan, model):
boundary, CI, regime accuracies, outcome class, prediction from the flash
ratios, and explicit censoring bounds.  ``bound_low``/``bound_high`` are the
observed constraints on the latent boundary: a missing low value denotes a
left-censored scan and a missing high value denotes a right-censored scan.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

from counting.analysis.capacity import fit_transition, sufficiency

REPO = Path(__file__).resolve().parents[1]
RATIO = {"r2-colortotal": 0.93, "r3-grow": 0.80, "r3-merge": 0.72,
         "r3-erase": 0.68, "r4-merge": 0.97, "r4-erase": 0.95}
PROTOCOL_EXCLUSIONS = {
    # The serving layer returned content_filter on every ladder task,
    # including a load-8 preflight.  Keep the raw records in the release but
    # do not interpret provider-suppressed responses as execution capacity.
    "claude-fable-5": "provider_content_filter",
}
FLASH = {  # the pilot runs, folded in under the same slug
    ("r2-colortotal", "gemini-3p7-flash"): ["results/ladder-rung2/color-total"],
    ("r3-grow", "gemini-3p7-flash"): ["results/ladder-rung3/grow-run"],
    ("r3-merge", "gemini-3p7-flash"): ["results/ladder-rung3/recolor-merge"],
    ("r3-erase", "gemini-3p7-flash"): ["results/ladder-rung3/erase-singletons"],
    ("r4-merge", "gemini-3p7-flash"): ["results/ladder-rung4/recolor-merge"],
    ("r4-erase", "gemini-3p7-flash"): ["results/ladder-rung4/erase-singletons"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-only",
        action="store_true",
        help="ignore extension directories (for bracket-sensitivity comparisons)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/ladder-full/fits.csv",
    )
    args = parser.parse_args()
    scc = defaultdict(list)
    for r in csv.DictReader((REPO / "results/counting-data/scc.csv").open()):
        try:
            scc[r["model"].split("/")[-1].replace(".", "p")].append(float(r["scc"]))
        except ValueError:
            continue
    cc_of = {slug: max(v) for slug, v in scc.items()}

    dirs: dict[tuple[str, str], list[str]] = {
        key: list(paths) for key, paths in FLASH.items()
    }
    result_globs = ["results/ladder-full/*/*/registry.csv"]
    if not args.original_only:
        result_globs.append("results/ladder-extended-*/*/*/registry.csv")
    for pattern in result_globs:
        for f in sorted(glob.glob(str(REPO / pattern))):
            scan, slug = f.split("/")[-3], f.split("/")[-2]
            directory = "/".join(f.split("/")[:-1])
            dirs.setdefault((scan, slug), []).append(directory)

    rows_out = []
    for (scan, slug), ds in sorted(dirs.items()):
        raw_rows = []
        for d in ds:
            for tf in glob.glob(str(REPO / d / "trials" / "*.jsonl")):
                raw_rows += [json.loads(l) for l in open(tf)]
        # A provider/transport error is a missing probe, not evidence that the
        # model failed the execution task. Content-filtered and budget-limited
        # replies have completed/incomplete response statuses and remain scored
        # under the fixed protocol; only genuine request failures are omitted.
        transport_errors = sum(r.get("response_status") == "error" for r in raw_rows)
        rows = [r for r in raw_rows if r.get("response_status") != "error"]
        if not rows:
            continue
        rows.sort(key=lambda r: r["expected"])
        L = [r["expected"] for r in rows]
        ok = [r.get("parsed") == r.get("expected") for r in rows]
        error = [r.get("parsed") is None for r in rows]
        blanks = sum(1 for r in rows if r.get("parsed") is None)
        t = fit_transition(L, ok, error)
        start = sorted(set(L))[1] if len(set(L)) > 1 else L[0]
        cc = cc_of.get(slug, 0)
        pred = RATIO[scan] * cc if cc else None
        rate = sum(ok) / len(ok)
        if blanks >= 0.7 * len(rows):
            cls = "blank-dead"
        elif t.censored or t.scc is None:
            cls = ("ceiling-stable" if rate >= 0.75 else
                   "below-start" if rate <= 0.3 else "unresolved-mixed")
        elif t.scc <= start:
            cls = "below-start"
        else:
            cls = "resolved"
        if cls == "resolved":
            censoring, bound_low, bound_high = "exact", t.scc, t.scc
        elif cls == "below-start":
            # The model is already predominantly wrong at the smallest task
            # load.  Its transition, if positive, is no larger than that load.
            censoring, bound_low, bound_high = "left", "", min(L)
        elif cls == "ceiling-stable":
            # No collapse is supported through the largest probed load.
            censoring, bound_low, bound_high = "right", max(L), ""
        elif cls == "unresolved-mixed":
            # A transition was not identifiable, but any transition the scan
            # could have located lies within the explicitly tested interval.
            censoring, bound_low, bound_high = "interval", min(L), max(L)
        else:
            censoring, bound_low, bound_high = "none", "", ""
        above = [c for l, c in zip(L, ok) if l > (t.scc or 0)]
        rows_out.append({
            "scan": scan, "model": slug,
            "analysis_included": slug not in PROTOCOL_EXCLUSIONS,
            "exclusion_reason": PROTOCOL_EXCLUSIONS.get(slug, ""),
            "class": cls,
            "boundary": t.scc if cls == "resolved" else "",
            "ci_low": t.ci_low if cls == "resolved" else "",
            "ci_high": t.ci_high if cls == "resolved" else "",
            "censoring": censoring,
            "bound_low": bound_low,
            "bound_high": bound_high,
            "p_stable": round(t.p_stable, 4) if t.p_stable is not None else "",
            "p_after": round(t.p_after, 4) if t.p_after is not None else "",
            "bound_by": t.bound_by,
            "sufficiency": sufficiency(t, len(above), sum(above)) if t.scc else "",
            "n_probes": len(rows), "n_transport_errors": transport_errors,
            "min_load": min(L), "max_load": max(L),
            "cc": cc, "predicted": round(pred) if pred else "",
            "rel": round(t.scc / pred, 4) if (cls == "resolved" and pred) else "",
        })
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as sink:
        w = csv.DictWriter(sink, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out} ({len(rows_out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
