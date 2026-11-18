#!/usr/bin/env python3
"""Roster of the measurements whose sweep does not settle the capacity.

    python scripts/build_extension_roster.py

Reads `scc.csv`, selects the measurements marked `censored` or
`tail_unresolved`, and writes a roster containing exactly those entries, drawn
verbatim from the main roster so decoding and reasoning settings are unchanged.

`budget_bound` measurements are deliberately excluded: their collapse is the
output cap, so probing further cannot resolve them -- only a larger budget can,
which is a separate cost decision.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counting.storage import run_key  # noqa: E402

EXTEND = ("censored", "tail_unresolved")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scc", type=Path,
                        default=REPO_ROOT / "results" / "counting-data" / "scc.csv")
    parser.add_argument("--roster", type=Path,
                        default=REPO_ROOT / "configs" / "models" / "openrouter_frontier.yaml")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "configs" / "models" / "extension_unresolved.yaml")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.scc.open(encoding="utf-8")))
    wanted = {r["run_key"]: r for r in rows if r["sufficiency"] in EXTEND}

    roster = yaml.safe_load(args.roster.read_text(encoding="utf-8"))["models"]
    by_key = {run_key(e): e for e in roster}

    picked, missing = [], []
    for key, row in sorted(wanted.items(), key=lambda kv: -int(kv[1]["scc"])):
        entry = by_key.get(key)
        if entry is None:
            missing.append(key)
            continue
        picked.append((entry, row))

    lines = [
        "# Measurements whose sweep does not settle the capacity -- GENERATED.",
        "#   python scripts/build_extension_roster.py",
        "#",
        f"# {len(picked)} of {len(rows)} measurements, selected by the sufficiency column of",
        "# scc.csv: the fitted transition is either undeclared (the model was still",
        "# mostly exact when probing stopped) or declared on a tail too thin to",
        "# establish it. Both are artifacts of stopping at 2x the first mistake, a",
        "# landmark the fitted criterion does not use.",
        "#",
        "# Run with configs/experiments/openrouter_extension.yaml, which drops the",
        "# multiple rule and stops when the criterion resolves instead. Scans resume",
        "# from their existing checkpoints, so no probe is paid for twice.",
        "",
        "models:",
        "",
    ]
    for entry, row in picked:
        flat = ", ".join(
            f'{k}: "{v}"' if k == "id" else f"{k}: {v}"
            for k, v in entry.items()
        )
        note = f'{row["sufficiency"]}, scc>={row["scc"]}, p_stable={row["p_stable"]}'
        lines.append(f"  - {{{flat}}}   # {note}")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.out}  ({len(picked)} measurements)")
    for label in EXTEND:
        n = sum(1 for _, r in picked if r["sufficiency"] == label)
        print(f"  {label:<16} {n:>3}")
    if missing:
        print(f"\n  not in the roster, skipped ({len(missing)}):")
        for key in missing:
            print(f"    {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
