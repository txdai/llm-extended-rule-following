"""Robustness of the strongest CC correlates: METR, Balrog, WeirdML.

    python scripts/analyze_benchmark_robustness.py

For each evaluation: Spearman rho with CC, the partial correlation controlling
for the Epoch Capabilities Index (rank residuals), the reverse partial, the
range of rho leaving out one vendor at a time, rho within the largest vendor,
and an incremental-prediction test: leave-one-vendor-out and leave-one-model-out
prediction of the score from the index alone versus index plus log CC.
Reads results/benchmark_data/counting_benchmark_merged.csv; writes
results/benchmark_data/robustness.json.
"""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "results/benchmark_data/counting_benchmark_merged.csv"
OUT = ROOT / "results/benchmark_data/robustness.json"
TARGETS = {"metr_time_horizons_external.csv": ("METR time horizon", True),
           "balrog_external.csv": ("Balrog", False),
           "weirdml_external.csv": ("WeirdML", False)}
ECI_FILE = "epoch_capabilities_index.csv"

def rank(x):
    x = np.asarray(x, float); r = np.empty(len(x)); order = np.argsort(x); r[order] = np.arange(len(x))
    # average ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r

def spearman(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])

def partial(a, b, z):
    ra, rb, rz = rank(a), rank(b), rank(z)
    X = np.column_stack([np.ones(len(rz)), rz])
    resid = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(resid(ra), resid(rb))[0, 1])

def load():
    rows = list(csv.DictReader(open(MERGED)))
    eci = {}
    for r in rows:
        if r["benchmark_file"] == ECI_FILE and r["stable_count"] not in ("", "0"):
            eci[r["counting_model"]] = float(r["benchmark_score"])
    per = defaultdict(list)
    for r in rows:
        if r["benchmark_file"] in TARGETS and r["stable_count"] not in ("", "0"):
            per[r["benchmark_file"]].append(dict(model=r["counting_model"], vendor=r["provider"],
                                                 score=float(r["benchmark_score"]), logcc=math.log10(float(r["stable_count"])),
                                                 eci=eci.get(r["counting_model"])))
    return per

def oos(rows, use_cc, groups):
    """Pooled out-of-sample R^2 and RMSE over held-out groups."""
    y = np.array([r["y"] for r in rows]); pred = np.zeros(len(rows))
    for g in set(groups):
        tr = [i for i, gg in enumerate(groups) if gg != g]; te = [i for i, gg in enumerate(groups) if gg == g]
        X = np.array([[1, r["eci"]] + ([r["logcc"]] if use_cc else []) for r in rows])
        beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; pred[te] = X[te] @ beta
    sse = float(((y - pred) ** 2).sum()); sst = float(((y - y.mean()) ** 2).sum())
    return dict(r2=1 - sse / sst, rmse=math.sqrt(sse / len(y)))

def analyse(name, rows, log_y):
    out = dict(name=name, n=len(rows), rho=spearman([r["logcc"] for r in rows], [r["score"] for r in rows]))
    vendors = defaultdict(int)
    for r in rows: vendors[r["vendor"]] += 1
    lovo = {}
    for v in vendors:
        keep = [r for r in rows if r["vendor"] != v]
        if len(keep) >= 5: lovo[v] = spearman([r["logcc"] for r in keep], [r["score"] for r in keep])
    out["lovo"] = lovo; out["lovo_range"] = [min(lovo.values()), max(lovo.values())] if lovo else None
    big = max(vendors, key=vendors.get); within = [r for r in rows if r["vendor"] == big]
    out["largest_vendor"] = dict(vendor=big, n=len(within), rho=spearman([r["logcc"] for r in within], [r["score"] for r in within]) if len(within) >= 5 else None)
    with_eci = [r for r in rows if r["eci"] is not None]
    out["n_with_eci"] = len(with_eci)
    if len(with_eci) >= 8:
        a = [r["logcc"] for r in with_eci]; b = [r["score"] for r in with_eci]; z = [r["eci"] for r in with_eci]
        out["rho_on_eci_subset"] = spearman(a, b); out["rho_eci_score"] = spearman(z, b); out["rho_eci_cc"] = spearman(z, a)
        out["partial_cc_score_given_eci"] = partial(a, b, z); out["partial_eci_score_given_cc"] = partial(z, b, a)
        for r in with_eci: r["y"] = math.log10(r["score"]) if log_y else r["score"]
        groups_v = [r["vendor"] for r in with_eci]; groups_m = list(range(len(with_eci)))
        out["incremental"] = dict(
            lovo=dict(eci_only=oos(with_eci, False, groups_v), eci_plus_cc=oos(with_eci, True, groups_v)),
            loo=dict(eci_only=oos(with_eci, False, groups_m), eci_plus_cc=oos(with_eci, True, groups_m)),
            target="log10 score" if log_y else "score")
    return out

if __name__ == "__main__":
    per = load(); res = [analyse(TARGETS[f][0], per[f], TARGETS[f][1]) for f in TARGETS]
    OUT.write_text(json.dumps(res, indent=1))
    for r in res:
        print(f"\n{r['name']}: n={r['n']} rho={r['rho']:.2f}  LOVO {r['lovo_range']}  largest vendor {r['largest_vendor']}")
        if "partial_cc_score_given_eci" in r:
            print(f"  with ECI n={r['n_with_eci']}: rho={r['rho_on_eci_subset']:.2f}  partial(CC,score|ECI)={r['partial_cc_score_given_eci']:.2f}  reverse={r['partial_eci_score_given_cc']:.2f}  rho(ECI,score)={r['rho_eci_score']:.2f}")
            inc = r["incremental"]
            for k in ("lovo", "loo"):
                e, c = inc[k]["eci_only"], inc[k]["eci_plus_cc"]
                print(f"  {k}: R2 {e['r2']:.2f} -> {c['r2']:.2f}   RMSE {e['rmse']:.3f} -> {c['rmse']:.3f}  ({inc['target']})")
