#!/usr/bin/env python3
"""Fit the transition capacity for every measurement.

    python scripts/compute_scc.py
    python scripts/compute_scc.py results/counting-data

Reads `responses.csv`, fits the two-regime transition per measurement
(see counting/analysis/capacity.py), writes `scc.csv` next to it, and prints
the comparison against first-mistake capacity -- including the subsampling
check: refit every curve on every 2nd and 5th probe and watch which
definition holds still.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counting.analysis.capacity import (  # noqa: E402
    fit_step01, fit_transition, sufficiency,
)


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, nargs="?",
                        default=REPO_ROOT / "results" / "counting-data")
    args = parser.parse_args()

    curves: dict[tuple, list] = defaultdict(list)
    with (args.run_dir / "responses.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != "scan":
                continue
            curves[(row["run"], row["run_key"])].append(
                (int(row["input_count"]), row["outcome"] == "correct",
                 row["outcome"] == "error"))

    registry = {(r["run"], r["run_key"]): r
                for r in csv.DictReader((args.run_dir / "registry.csv").open(encoding="utf-8"))}

    out_rows, table = [], []
    for key, probes in sorted(curves.items()):
        probes.sort()
        lengths = [p[0] for p in probes]
        ok = [p[1] for p in probes]
        err = [p[2] for p in probes]
        t = fit_transition(lengths, ok, err)
        # Is the record deep enough for the fit to mean what it claims?
        #   censored        no collapse observed; capacity is a lower bound
        #   budget_bound    the collapse observed is the token cap, not counting
        #   tail_unresolved transition declared, but the probes above it are too
        #                   few to establish the broken regime: the 95% Wilson
        #                   interval on p_after still reaches 1/2
        n2 = sum(1 for l in lengths if l > t.scc)
        c2 = sum(1 for l, o in zip(lengths, ok) if l > t.scc and o)
        suff = sufficiency(t, n2, c2)
        reg = registry.get(key, {})
        cc = int(reg["cc"]) if reg.get("cc") else None
        row = {
            "run": key[0], "run_key": key[1],
            "model": reg.get("model", key[1]), "variant": reg.get("variant", ""),
            "scc": t.scc, "scc_ci_low": t.ci_low, "scc_ci_high": t.ci_high,
            "p_stable": round(t.p_stable, 4),
            "p_after": round(t.p_after, 4) if t.p_after is not None else "",
            "censored": t.censored, "delta_ll": round(t.delta_ll, 2),
            "scc_bound_by": t.bound_by, "scc_step01": fit_step01(lengths, ok),
            "cc_first_mistake": cc if cc is not None else "",
            "n_probes": t.n_probes, "n_above": n2, "sufficiency": suff,
        }
        out_rows.append(row)
        table.append((key[1], cc, t, lengths, ok, err))

    out = args.run_dir / "scc.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {out}  ({len(out_rows)} rows)\n")

    # -- comparison ---------------------------------------------------------
    paired = [(cc, t) for _, cc, t, *_ in table if cc is not None]
    sccs = [t.scc for _, t in paired]
    ccs = [cc for cc, _ in paired]
    med = lambda v: statistics.median(v)
    print(f"  capacity        first-mistake  median {med(ccs):>5.0f}   mean {statistics.mean(ccs):>6.1f}")
    print(f"                  transition     median {med(sccs):>5.0f}   mean {statistics.mean(sccs):>6.1f}")
    print(f"  spearman(first-mistake, transition)  {spearman(ccs, sccs):.3f}")
    up = sum(1 for cc, t in paired if t.scc > cc)
    print(f"  transition above first-mistake       {up}/{len(paired)}")
    print(f"  censored (no transition in range)    {sum(1 for _, t in paired if t.censored)}")
    dll = sorted(t.delta_ll for _, t in paired if not t.censored)
    print(f"  delta-LL of declared transitions     median {med(dll):.0f}   "
          f"p10 {dll[len(dll)//10]:.1f}   (BIC bar: ln n, 4.6-6.9 here)")
    agree = [abs(t.scc - r["scc_step01"]) for t, r in zip((t for _, t in paired), out_rows)]
    print(f"  |transition - zero-one twin|         median {med(agree):.0f}   "
          f"within 5: {sum(1 for a in agree if a <= 5)}/{len(agree)}")

    from collections import Counter
    counts = Counter(r["sufficiency"] for r in out_rows)
    print("\n  sweep sufficiency under the fitted criterion:")
    for k in ("ok", "censored", "budget_bound", "tail_unresolved"):
        print(f"    {k:<16} {counts.get(k, 0):>4}")
    for k, label in (("censored", "no collapse observed in range"),
                     ("budget_bound", "collapse observed is the 8192 cap"),
                     ("tail_unresolved", "too few probes above the transition")):
        rows = [r for r in out_rows if r["sufficiency"] == k]
        if not rows:
            continue
        print(f"\n  {k} -- {label}:")
        for r in sorted(rows, key=lambda r: -r["scc"]):
            print(f"    {r['run_key']:<48} scc>={r['scc']:<5} p_stable={r['p_stable']:<7} "
                  f"probed_to={r['scc'] + r['n_above'] if r['n_above'] else r['scc']}")

    print("\n  largest moves, first-mistake -> transition:")
    movers = sorted(table, key=lambda x: -(x[2].scc - (x[1] or 0)))[:8]
    for key, cc, t, *_ in movers:
        ci = f"[{t.ci_low},{t.ci_high}]"
        print(f"    {key:<46} {cc:>4} -> {t.scc:<4} {ci:<12} "
              f"p_stable={t.p_stable:.3f} p_after={t.p_after if t.p_after is None else round(t.p_after,3)}")

    # -- the subsampling check ---------------------------------------------
    print("\n  refit on every k-th probe (does the estimate move?):")
    print(f"    {'k':>3} {'median shift, transition':>26} {'median shift, first-mistake':>29}")
    for step in (2, 5):
        d_scc, d_fm = [], []
        for _, cc, t, lengths, ok, err in table:
            if len(lengths) < 3 * step:
                continue
            sub = fit_transition(lengths[::step], ok[::step], err[::step])
            d_scc.append(sub.scc - t.scc)
            fm_full = next((l for l, o in zip(lengths, ok) if not o), None)
            fm_sub = next((l for l, o in zip(lengths[::step], ok[::step]) if not o), None)
            if fm_full and fm_sub:
                d_fm.append(fm_sub - fm_full)
        print(f"    {step:>3} {med(d_scc):>+21.0f} ({statistics.mean(d_scc):>+5.1f}) "
              f"{med(d_fm):>+22.0f} ({statistics.mean(d_fm):>+5.1f})   median (mean)")

    ps = sorted(t.p_stable for _, t in paired)
    print(f"\n  p_stable  median {med(ps):.3f}   p10 {ps[len(ps)//10]:.3f}   "
          f"below 0.9: {sum(1 for v in ps if v < 0.9)}/{len(ps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
