"""Coverage check for the interval on tau, fixed-length and under the stop rule.

    python scripts/check_interval_calibration.py [--reps N]

Simulates two-regime records, fits them with the paper's estimator, and reports
how often the chi-squared profile set (bar 1.92) and the simulation-calibrated
set contain the true changepoint. The sequential design mirrors the harness:
probe lengths ascend from 1, and the scan stops once a transition is declared
and the accuracy above it is significantly below one half (Wilson upper < 0.5).
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from counting.analysis.capacity import fit_transition, sufficiency, PROFILE_BAR, transition_profile

def interval(lengths, correct, bar):
    cuts, _, _ = transition_profile(lengths, correct)
    best = max(ll for _, ll in cuts)
    inside = [L for L, ll in cuts if best - ll <= bar]
    return min(inside), max(inside)

def one(nscan, p1, p2, tau, rng, sequential):
    L, c = [], []
    n = 0
    while True:
        n += 1
        L.append(n); c.append(rng.random() < (p1 if n <= tau else p2))
        if sequential:
            if n < 12: continue
            t = fit_transition(L, c, calibrate=False)
            if t.censored: 
                if n >= 6 * tau: break
                continue
            above = c[t.n_probes - (len(L) - L.index(t.scc) - 1):] if False else c[L.index(t.scc) + 1:]
            if sufficiency(t, len(above), sum(above)) == "ok": break
        elif n >= nscan: break
    t = fit_transition(L, c)              # calibrated
    if t.censored: return None
    lo, hi = interval(L, c, PROFILE_BAR)
    return dict(chi=lo <= tau <= hi, cal=t.ci_low <= tau <= t.ci_high,
                w_chi=hi - lo, w_cal=t.ci_high - t.ci_low, bar=t.bar, n=len(L))

def run(nscan, p1, p2, tau, reps, sequential, seed=11):
    rng = random.Random(seed); rows = [r for r in (one(nscan, p1, p2, tau, rng, sequential) for _ in range(reps)) if r]
    med = lambda k: sorted(r[k] for r in rows)[len(rows) // 2]
    return dict(design="sequential" if sequential else f"fixed n={nscan}", p1=p1, p2=p2, tau=tau, declared=len(rows),
                coverage_chi=sum(r["chi"] for r in rows) / len(rows), coverage_cal=sum(r["cal"] for r in rows) / len(rows),
                median_width_chi=med("w_chi"), median_width_cal=med("w_cal"), median_bar=round(med("bar"), 2), median_n=med("n"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=1000); ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = [run(100, 0.96, 0.12, 50, a.reps, False), run(300, 0.96, 0.12, 50, a.reps, False),
           run(0, 0.96, 0.12, 50, a.reps, True), run(0, 0.90, 0.20, 150, a.reps, True), run(0, 0.96, 0.12, 300, a.reps // 2, True)]
    txt = json.dumps(out, indent=1); print(txt)
    if a.out: a.out.write_text(txt)
