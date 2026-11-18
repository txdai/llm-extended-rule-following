#!/usr/bin/env python3
"""Build the counting-curves dashboard from the dataset.

    python scripts/build_dashboard.py
    python scripts/build_dashboard.py results/counting-data --out dash.html

Reads `registry.csv` and `responses.csv` and writes one self-contained HTML
page: every probe, plotted as asked-count against answered-count.

The page exists because `cc = first_mistake - 1` is a weak statistic -- almost
every measurement keeps answering correctly above its first mistake -- so the
curve, not the scalar, is the thing worth looking at. The definitions view
recomputes capacity under other rules so the choice can be made by eye.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "results" / "counting-data"


def load(run_dir: Path):
    registry = list(csv.DictReader((run_dir / "registry.csv").open(encoding="utf-8")))
    curves: dict[str, dict[int, int]] = {}
    for row in csv.DictReader((run_dir / "responses.csv").open(encoding="utf-8")):
        if row.get("phase") != "scan":
            continue
        try:
            length = int(row["input_count"])
        except (TypeError, ValueError):
            continue
        # -1 marks a probe that returned no parseable integer: the budget ran
        # out, the model looped, or the provider failed. It is not an answer of
        # zero and must never be plotted as one.
        if row["outcome"] == "error" or not row["output_count"]:
            answer = -1
        else:
            try:
                answer = int(row["output_count"])
            except ValueError:
                answer = -1
        curves.setdefault(row["run_key"], {})[length] = answer
    return registry, curves


def num(value, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def build_rows(registry, curves):
    rows = []
    for entry in registry:
        curve = curves.get(entry["run_key"])
        if not curve:
            continue
        start = min(curve)
        end = max(curve)
        # A dense array indexed from `start`; null is a length never probed,
        # which resume gaps can leave behind and which must stay distinct from
        # an error.
        answers = [curve.get(length) for length in range(start, end + 1)]
        rows.append({
            "k": entry["run_key"],
            "m": entry["model"],
            "lab": entry.get("lab") or "",
            "v": entry.get("variant") or "",
            "cc": num(entry.get("cc"), int),
            "bb": entry.get("cc_bound_by") or "",
            "fm": num(entry.get("first_mistake"), int),
            "eff": entry.get("reasoning_effort") or "",
            "tl": entry.get("temperature_locked") in ("True", "true", "1"),
            "rd": entry.get("release_date") or "",
            "tp": num(entry.get("total_params_b")),
            "ap": num(entry.get("active_params_b")),
            "px": num(entry.get("prompt_usd_per_mtok")),
            "s": start,
            "o": answers,
        })
    rows.sort(key=lambda r: -(r["cc"] or 0))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    registry, curves = load(args.run_dir)
    rows = build_rows(registry, curves)
    if not rows:
        raise SystemExit(f"no curves found under {args.run_dir}")

    probes = sum(len([a for a in r["o"] if a is not None]) for r in rows)
    above = sum(
        1 for r in rows
        if r["fm"] and any(
            a == r["s"] + i for i, a in enumerate(r["o"]) if a is not None and r["s"] + i > r["fm"]
        )
    )
    meta = {
        "measurements": len(rows),
        "models": len({r["m"] for r in rows}),
        "probes": probes,
        "above": above,
        "labs": sorted({r["lab"] for r in rows if r["lab"]}),
        "run": args.run_dir.name,
    }

    template = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    payload = json.dumps({"meta": meta, "rows": rows}, separators=(",", ":"))
    html = template.replace("/*__DATA__*/null", payload)

    out = args.out or (args.run_dir / "dashboard.html")
    out.write_text(html, encoding="utf-8")
    size = out.stat().st_size / 1024
    print(f"wrote {out}  ({size:.0f} KB)")
    print(f"  {meta['measurements']} measurements, {meta['models']} models, {probes:,} probes")
    print(f"  {above} answer correctly above their first mistake")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
