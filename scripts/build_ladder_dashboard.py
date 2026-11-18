#!/usr/bin/env python3
"""Build the ladder-scan dashboard from the tailored corpus.

    python scripts/build_ladder_dashboard.py
    python scripts/build_ladder_dashboard.py --out /tmp/ladder.html

Reads the tailored per-scan JSONL and the consolidated fit and writes one
self-contained HTML page: every ladder probe, plotted as the true count
against the count the model answered.

The page exists because the outcome class of a scan (`resolved`,
`below-start`, `ceiling-stable`, `unresolved-mixed`) is a summary of a curve
that nobody has looked at.  Plotting asked against answered separates the
scans whose boundary sits at the task floor from the scans that are wrong
everywhere and from the scans whose accuracy is not monotone in load at all.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "results" / "ladder-tailored-8192"
DEFAULT_FITS = REPO / "results" / "ladder-full" / "fits.csv"
ORIGINAL_FITS = REPO / "results" / "ladder-full" / "fits-original.csv"
TEMPLATE = Path(__file__).with_name("ladder_dashboard_template.html")
DEFAULT_OUT = REPO / "results" / "ladder-tailored-8192" / "dashboard.html"

# The smallest load each generator can construct; a scan classified below-start
# at this load is censored by the task, not by the choice of scan bracket.
FLOOR = {"r2-colortotal": 1, "r3-grow": 4, "r3-merge": 4,
         "r3-erase": 4, "r4-merge": 4, "r4-erase": 4}
SOURCE_CODE = {"original": 0, "lower": 1, "upper": 2, "pilot": 3}
TASK_LABEL = {"r2-colortotal": "rung 2 · scattered cells",
              "r3-grow": "rung 3 · grow (row)",
              "r3-merge": "rung 3 · recolor (row)",
              "r3-erase": "rung 3 · erase (row)",
              "r4-merge": "rung 4 · recolor (grid)",
              "r4-erase": "rung 4 · erase (grid)"}


def _num(v):
    try:
        return float(v) if v not in ("", None) else None
    except (TypeError, ValueError):
        return None


def read_fits(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    return {(r["scan"], r["model"]): r for r in csv.DictReader(path.open())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    fits = read_fits(args.fits)
    original = read_fits(ORIGINAL_FITS)
    summary = {}
    summary_path = args.corpus / "scan_summary.csv"
    if summary_path.exists():
        summary = {(r["scan"], r["model"]): r for r in csv.DictReader(summary_path.open())}

    scans = []
    for f in sorted(args.corpus.glob("scans/*/*.jsonl")):
        scan, slug = f.parent.name, f.stem
        pts = []
        for line in f.open():
            r = json.loads(line)
            t = r.get("tailoring") or {}
            flags = (1 if t.get("budget_exhausted") else 0)
            flags |= (2 if r.get("phase") == "preflight" else 0)
            flags |= (4 if r.get("response_status") == "error" else 0)
            pts.append([r["expected"], r.get("parsed"),
                        SOURCE_CODE.get(t.get("source_run"), 0), flags])
        pts.sort(key=lambda p: (p[0], p[2]))
        fit = fits.get((scan, slug), {})
        orig = original.get((scan, slug), {})
        sm = summary.get((scan, slug), {})
        scans.append({
            "t": scan, "m": slug, "n": len(pts),
            "cls": fit.get("class", ""), "cls0": orig.get("class", ""),
            "cens": fit.get("censoring", ""),
            "inc": fit.get("analysis_included", "True") == "True",
            "b": _num(fit.get("boundary")), "cl": _num(fit.get("ci_low")),
            "ch": _num(fit.get("ci_high")),
            "ps": _num(fit.get("p_stable")), "pa": _num(fit.get("p_after")),
            "cc": _num(fit.get("cc")), "pr": _num(fit.get("predicted")),
            "fl": FLOOR.get(scan), "term": sm.get("termination_reason", ""),
            "need": sm.get("needs_additional_scan") == "True",
            "p": pts,
        })

    payload = {
        "meta": {
            "corpus": str(args.corpus.relative_to(REPO)),
            "fits": str(args.fits.relative_to(REPO)),
            "scans": len(scans),
            "models": len({s["m"] for s in scans}),
            "probes": sum(s["n"] for s in scans),
            "tasks": TASK_LABEL,
        },
        "scans": scans,
    }
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"{args.out}  {len(scans)} scans  {payload['meta']['probes']} probes  "
          f"{args.out.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
