#!/usr/bin/env python3
"""Rank stability of the execution ladder across tasks and against CC.

    python scripts/analyze_ladder_ranks.py

Reads results/ladder-full/fits.csv (analysis cohort only) and writes
results/ladder-full/rank_stability.json with pairwise Spearman correlations
between tasks, each task's correlation with CC, and Kendall's W across the
six tasks.  Censored scans enter either not at all ("resolved") or at their
observed bound ("limit-substituted": left-censored at the task minimum,
right-censored at the expanded ceiling, mixed at the geometric midpoint of
the tested interval).
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FITS = REPO / "results" / "ladder-full" / "fits.csv"
OUT = REPO / "results" / "ladder-full" / "rank_stability.json"


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def kendall_w(table, tasks, models):
    k, n = len(tasks), len(models)
    R = [ranks([table[t][m] for m in models]) for t in tasks]
    sums = [sum(R[j][i] for j in range(k)) for i in range(n)]
    s = sum((x - k * (n + 1) / 2) ** 2 for x in sums)
    return 12 * s / (k * k * (n ** 3 - n))


def main() -> int:
    rows = [r for r in csv.DictReader(FITS.open()) if r["analysis_included"] != "False"]
    tasks = sorted({r["scan"] for r in rows})
    exact, subst, cc = defaultdict(dict), defaultdict(dict), {}
    for r in rows:
        cc[r["model"]] = float(r["cc"]) if r["cc"] else None
        c = r["censoring"]
        if c == "exact":
            exact[r["scan"]][r["model"]] = subst[r["scan"]][r["model"]] = float(r["boundary"])
        elif c == "left":
            subst[r["scan"]][r["model"]] = float(r["bound_high"])
        elif c == "right":
            subst[r["scan"]][r["model"]] = float(r["bound_low"])
        elif c == "interval":
            subst[r["scan"]][r["model"]] = math.sqrt(float(r["bound_low"]) * float(r["bound_high"]))

    def pairwise(table):
        out = []
        for a, b in itertools.combinations(tasks, 2):
            common = [m for m in table[a] if m in table[b]]
            if len(common) >= 6:
                out.append({"task_a": a, "task_b": b, "n": len(common),
                            "rho": spearman([table[a][m] for m in common],
                                            [table[b][m] for m in common])})
        rhos = [o["rho"] for o in out]
        return {"pairs": out, "mean": statistics.mean(rhos), "median": statistics.median(rhos),
                "min": min(rhos), "max": max(rhos)}

    def versus_cc(table):
        out = {}
        for t in tasks:
            ms = [m for m in table[t] if cc.get(m)]
            out[t] = {"n": len(ms), "rho": spearman([table[t][m] for m in ms], [cc[m] for m in ms])}
        return out

    all_subst = [m for m in cc if cc[m] and all(m in subst[t] for t in tasks)]
    all_exact = [m for m in cc if cc[m] and all(m in exact[t] for t in tasks)]
    result = {
        "tasks": tasks,
        "resolved": {"pairwise": pairwise(exact), "versus_cc": versus_cc(exact),
                     "kendall_w": {"n_models": len(all_exact), "w": kendall_w(exact, tasks, all_exact)}},
        "limit_substituted": {"pairwise": pairwise(subst), "versus_cc": versus_cc(subst),
                              "kendall_w": {"n_models": len(all_subst), "w": kendall_w(subst, tasks, all_subst)}},
    }
    OUT.write_text(json.dumps(result, indent=1))
    for key in ("resolved", "limit_substituted"):
        r = result[key]
        print(f"{key}: pairwise rho mean {r['pairwise']['mean']:.2f} (min {r['pairwise']['min']:.2f}, "
              f"max {r['pairwise']['max']:.2f}); Kendall W {r['kendall_w']['w']:.2f} over "
              f"{r['kendall_w']['n_models']} models; task-vs-CC rho "
              + ", ".join(f"{t}={v['rho']:.2f}" for t, v in r["versus_cc"].items()))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
