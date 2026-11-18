"""Analysis for the state-tracking scan (docs/plans/state-tracking-scan.md).

Reads results/tracking/*.summary.json, pairs each run with the catalogue's CC of
the same model and configuration (scc.csv), the Epoch Capabilities Index and
vendor from the merged benchmark table, and tests the three prespecified
predictions:

  P1  rho(TC immediate, CC immediate) across models; partial given ECI; vendor-
      held-out R^2 of log TC from ECI with and without log CC.
  P2  reasoning-on minus reasoning-off change in log TC against the same change
      in log CC on hybrids; rho(TC reasoning, CC reasoning) against P1's rho.
  P3  event-access and supplied-state accuracy at and beyond the boundary against
      the tracking accuracy above it.

Writes paper/generated/tracking_table.tex, paper/generated/tracking_stats.json,
and paper/figures/tracking-vs-cc.pdf.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_benchmark_robustness as A  # noqa: E402

TRACK = ROOT / "results" / "tracking"
SCC = ROOT / "results" / "counting-data" / "scc.csv"
MERGED = ROOT / "results" / "benchmark_data" / "counting_benchmark_merged.csv"
GEN = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"


def load():
    runs = {}
    for f in TRACK.glob("*.summary.json"):
        s = json.loads(f.read_text())
        if "failed" in s or s.get("tc") is None:
            continue
        rows = [json.loads(l) for l in f.with_suffix("").with_suffix(".jsonl").read_text().splitlines() if l.strip()]
        rt = [r.get("reasoning_tokens") or 0 for r in rows if r["kind"] == "scan" and r["status"] != "error"]
        s["reasoning_tokens_median"] = float(np.median(rt)) if rt else 0.0
        s["reasoning_fraction"] = float(np.mean([x > 0 for x in rt])) if rt else 0.0
        if s["sufficiency"] == "ok" and (s["tc"] <= 0 or (s.get("p_stable") or 0) == 0):
            s["sufficiency"] = "never_stable"   # exact nowhere: no stable regime
        runs[s["key"]] = s
    cc = {}
    for r in csv.DictReader(SCC.open()):
        try:
            v = float(r["scc"])
        except ValueError:
            continue
        key = r["model"] + (f"#{r['variant']}" if r["variant"] else "")
        cc[key] = dict(cc=v, bound=r.get("scc_bound_by", ""), censored=r.get("censored", ""))
    eci, vendor = {}, {}
    for r in csv.DictReader(MERGED.open()):
        vendor[r["counting_model"]] = r["provider"]
        if r["benchmark_file"] == "epoch_capabilities_index.csv" and r["stable_count"] not in ("", "0"):
            eci[r["counting_model"]] = float(r["benchmark_score"])
    params = {}
    for r in csv.DictReader((ROOT / "results" / "counting-data" / "registry.csv").open()):
        try:
            v = float(r.get("total_params_b") or "")
        except ValueError:
            continue
        if v > 0:
            params[r["model"]] = v
    return runs, cc, eci, vendor, params


def mode_of(key: str, effort=None, s: dict | None = None) -> str:
    """Verified mode: 'reasoning' when the run's own replies carried reasoning tokens in more
    than half of the scan probes, otherwise 'immediate'. A #nothink run that still reasoned,
    or a #think run that did not, is 'unverified' and excluded from the mode comparisons."""
    if s is None:
        return "reasoning" if key.endswith("#think") else "immediate"
    reasoned = s.get("reasoning_fraction", 0.0) > 0.5
    if key.endswith("#nothink") and reasoned:
        return "unverified"
    if key.endswith("#think") and not reasoned:
        return "unverified"
    return "reasoning" if reasoned else "immediate"


def per_model(runs, cc):
    """Per model: immediate and reasoning TC and CC where the configuration exists."""
    out = defaultdict(dict)
    for key, s in runs.items():
        model = key.split("#")[0]
        m = mode_of(key, s.get("reasoning_effort"), s)
        c = cc.get(key)
        if m == "unverified":
            out[model].setdefault("unverified", []).append(key)
            continue
        tc = 0 if s["sufficiency"] == "never_stable" else s["tc"]   # no stable regime: capacity below one swap
        out[model][m] = dict(tc=max(1, tc), tc_ci=s["ci"], tc_suff=s["sufficiency"], tc_censored=s["censored"], tc_raw=tc,
                             cc=c["cc"] if c else None, cc_bound=c["bound"] if c else None,
                             side=s["side"], side_lengths=s["side_lengths"], scan=s["scan"], key=key)
    return out


def spearman_boot(x, y, reps=2000, seed=7):
    rho = A.spearman(x, y)
    rng = random.Random(seed)
    vals = []
    n = len(x)
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        xs, ys = [x[i] for i in idx], [y[i] for i in idx]
        if len(set(xs)) > 2 and len(set(ys)) > 2:
            vals.append(A.spearman(xs, ys))
    vals.sort()
    return rho, (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]) if vals else (None, None)


def main() -> int:
    runs, cc, eci, vendor, params = load()
    pm = per_model(runs, cc)
    stats = {}
    # immediate-answer configuration: nothink, or the single configuration when reasoning cannot be switched
    rows_imm = []
    for model, d in pm.items():
        r = d.get("immediate")
        if r and r["cc"]:
            rows_imm.append(dict(model=model, vendor=vendor.get(model, model.split("/")[0]), tc=r["tc"], cc=r["cc"],
                                 eci=eci.get(model), logcc=math.log10(r["cc"]), y=math.log10(r["tc"]), mode="immediate"))
    # reasoning-on configurations against their own CC
    rows_reas = []
    for model, d in pm.items():
        r = d.get("reasoning")
        if r and r["cc"]:
            rows_reas.append(dict(model=model, tc=r["tc"], cc=r["cc"], logcc=math.log10(r["cc"]), y=math.log10(r["tc"])))
    for r in rows_imm:
        r["params"] = params.get(r["model"])
    x = [r["logcc"] for r in rows_imm]; y = [r["y"] for r in rows_imm]
    rho, ci = spearman_boot(x, y)
    stats["P1"] = dict(n=len(rows_imm), rho=rho, rho_ci=ci)
    # scaling: log TC on log CC, OLS slope with a bootstrap interval, and the ratio TC/CC across models
    X = np.c_[np.ones(len(x)), x]; beta = np.linalg.lstsq(X, np.array(y), rcond=None)[0]
    rng = random.Random(11); slopes = []
    for _ in range(2000):
        idx = [rng.randrange(len(x)) for _ in x]
        Xb = np.c_[np.ones(len(idx)), [x[i] for i in idx]]; yb = np.array([y[i] for i in idx])
        if len(set(Xb[:, 1])) > 2:
            slopes.append(float(np.linalg.lstsq(Xb, yb, rcond=None)[0][1]))
    slopes.sort()
    ratios = [r["tc"] / r["cc"] for r in rows_imm]
    stats["P1"].update(slope_log=float(beta[1]), slope_ci=(slopes[int(0.025 * len(slopes))], slopes[int(0.975 * len(slopes))]) if slopes else None,
                       intercept_log=float(beta[0]), ratio_median=float(np.median(ratios)),
                       ratio_iqr=(float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))),
                       ratio_min_max=(float(min(ratios)), float(max(ratios))))
    rp = [r for r in rows_imm if r["params"]]
    if len(rp) >= 8:
        stats["P1"]["n_params"] = len(rp)
        stats["P1"]["partial_given_log_params"] = A.partial([r["logcc"] for r in rp], [r["y"] for r in rp], [math.log10(r["params"]) for r in rp])
        stats["P1"]["rho_params_tc"] = A.spearman([math.log10(r["params"]) for r in rp], [r["y"] for r in rp])
    re_ = [r for r in rows_imm if r["eci"] is not None]
    if len(re_) >= 8:
        g = [r["vendor"] for r in re_]
        stats["P1"].update(n_eci=len(re_),
                           partial_given_eci=A.partial([r["logcc"] for r in re_], [r["y"] for r in re_], [r["eci"] for r in re_]),
                           rho_eci_tc=A.spearman([r["eci"] for r in re_], [r["y"] for r in re_]),
                           lovo_r2_eci=A.oos(re_, False, g)["r2"], lovo_r2_eci_cc=A.oos(re_, True, g)["r2"])
    if len(rows_reas) >= 6:
        rr_rho, rr_ci = spearman_boot([r["logcc"] for r in rows_reas], [r["y"] for r in rows_reas])
        stats["P1"]["reasoning_on"] = dict(n=len(rows_reas), rho=rr_rho, rho_ci=rr_ci)
    # sensitivity to censoring: drop floor (no stable regime) runs; then keep only resolved runs
    def subset_stats(label, keep):
        sub = [r for r in rows_imm if keep(pm[r["model"]]["immediate"])]
        if len(sub) >= 6:
            rho_s, ci_s = spearman_boot([r["logcc"] for r in sub], [r["y"] for r in sub])
            stats["P1"][label] = dict(n=len(sub), rho=rho_s, rho_ci=ci_s)
    subset_stats("no_floor", lambda r: r["tc_suff"] != "never_stable")
    subset_stats("resolved_only", lambda r: r["tc_suff"] == "ok")
    stats["by_cause"] = dict(zip(*np.unique([s["sufficiency"] for s in runs.values()], return_counts=True)))
    stats["by_cause"] = {k: int(v) for k, v in stats["by_cause"].items()}
    stats["unverified"] = sorted(k for d in pm.values() for k in d.get("unverified", []))
    stats["n_models"] = len({k.split("#")[0] for k in runs})
    # P2: hybrids with both modes
    hyb = [(m, d) for m, d in pm.items() if "immediate" in d and "reasoning" in d and d["immediate"]["cc"] and d["reasoning"]["cc"]]
    if hyb:
        dtc = [math.log10(d["reasoning"]["tc"] / d["immediate"]["tc"]) for _, d in hyb]
        dcc = [math.log10(d["reasoning"]["cc"] / d["immediate"]["cc"]) for _, d in hyb]
        rr = [(math.log10(d["reasoning"]["cc"]), math.log10(d["reasoning"]["tc"])) for _, d in hyb]
        ii = [(math.log10(d["immediate"]["cc"]), math.log10(d["immediate"]["tc"])) for _, d in hyb]
        rng = random.Random(3); dgain, drho = [], []
        for _ in range(2000):
            idx = [rng.randrange(len(hyb)) for _ in hyb]
            dgain.append(float(np.median([dtc[i] for i in idx])))
            if len({ii[i][0] for i in idx}) > 2 and len({rr[i][0] for i in idx}) > 2:
                drho.append(A.spearman([rr[i][0] for i in idx], [rr[i][1] for i in idx]) - A.spearman([ii[i][0] for i in idx], [ii[i][1] for i in idx]))
        dgain.sort(); drho.sort()
        stats["P2"] = dict(n=len(hyb), median_log10_gain_tc=float(np.median(dtc)), median_log10_gain_cc=float(np.median(dcc)),
                           gain_tc_ci=(dgain[int(0.025 * len(dgain))], dgain[int(0.975 * len(dgain))]),
                           gains_tc=dict(zip([m for m, _ in hyb], dtc)), gains_cc=dict(zip([m for m, _ in hyb], dcc)),
                           rho_reasoning=A.spearman([a for a, _ in rr], [b for _, b in rr]),
                           rho_immediate_same_models=A.spearman([a for a, _ in ii], [b for _, b in ii]),
                           delta_rho_ci=(drho[int(0.025 * len(drho))], drho[int(0.975 * len(drho))]) if drho else None,
                           n_bounded_on=sum(1 for _, d in hyb if d["reasoning"]["tc_suff"] != "ok"),
                           hybrids_resolved_both=[m for m, d in hyb if d["reasoning"]["tc_suff"] == "ok" and d["immediate"]["tc_suff"] == "ok"])
    # P3: side questions at 1.0x and 1.5x against tracking accuracy above the boundary
    ev, sup, above = [], [], []
    for key, s in runs.items():
        tc = max(1, s["tc"])
        for L in s["side_lengths"]:
            if L >= tc:
                for kind, acc in (("event", ev), ("supplied", sup)):
                    v = s["side"].get(f"{kind}@{L}")
                    if v and v["n"]:
                        acc.append(v["correct"] / v["n"])
        ab = [c for (L, c, _) in s["scan"] if L > tc]
        if ab:
            above.append(sum(ab) / len(ab))
    stats["P3"] = dict(runs=len(runs), event_at_or_beyond=float(np.mean(ev)) if ev else None,
                       supplied_at_or_beyond=float(np.mean(sup)) if sup else None,
                       tracking_above_boundary=float(np.mean(above)) if above else None)
    stats["sufficiency"] = dict(zip(*np.unique([s["sufficiency"] for s in runs.values()], return_counts=True))) if runs else {}
    stats["sufficiency"] = {k: int(v) for k, v in stats["sufficiency"].items()}
    GEN.mkdir(exist_ok=True)
    (GEN / "tracking_stats.json").write_text(json.dumps(stats, indent=1, default=float))

    # table
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             "\\caption{\\textbf{State-tracking capacity against counting capacity.} Tracking capacity TC (four objects, swaps until the location of a moved object is no longer the majority answer) and CC for the same model and the same configuration; the event-access and supplied-state rates at the fitted boundary (five probes each). Reasoning is the verified setting: on when the run's replies carried reasoning tokens in more than half of the scan probes. Marks: \\(b\\), bounded by the output budget, entered at the bound; \\(u\\), transition declared but its tail not established; \\(c\\), still exact at the top of the probed range; 0, no stable regime at any length, entered at one in the correlations and dropped in the sensitivity analysis. The subset is fixed by rule (every model with a METR horizon and an index score, the six probe-suite models, and the five lowest-capacity served models) and is skewed toward recent models, as the METR leaderboard is.}",
             "\\label{tab:tracking}", "\\begin{tabular}{llrrrrr}", "\\toprule",
             "Model & Reasoning & CC & TC & TC interval & Event at TC & Supplied at TC \\\\", "\\midrule"]
    def fmt_side(s, kind):
        L = max(1, s["tc"]); v = s["side"].get(f"{kind}@{L}")
        return f"{v['correct']}/{v['n']}" if v else "--"
    ordered = sorted(runs.values(), key=lambda s: (-(cc.get(s["key"], {}).get("cc") or 0), s["key"]))
    for s in ordered:
        c = cc.get(s["key"])
        name = s["model"].split("/", 1)[-1].replace("_", "\\_")
        mode = {"reasoning": "on", "immediate": "off", "unverified": "unverified"}[mode_of(s["key"], s.get("reasoning_effort"), s)]
        mark = {"ok": "", "budget_bound": "$^{b}$", "tail_unresolved": "$^{u}$", "censored": "$^{c}$", "never_stable": "$^{0}$"}[s["sufficiency"]]
        dag = mark
        shown = "0" if s["sufficiency"] == "never_stable" else str(max(1, s["tc"]))
        ci = "no stable regime" if s["sufficiency"] == "never_stable" else f"[{s['ci'][0]}, {s['ci'][1]}]"
        lines.append(f"\\texttt{{{name}}} & {mode} & {c['cc']:.0f} & {shown}{dag} & {ci} & {fmt_side(s, 'event')} & {fmt_side(s, 'supplied')} \\\\" if c else
                     f"\\texttt{{{name}}} & {mode} & -- & {shown}{dag} & {ci} & {fmt_side(s, 'event')} & {fmt_side(s, 'supplied')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (GEN / "tracking_table.tex").write_text("\n".join(lines))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial"; plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for m, d in pm.items():
        for mode, marker, fill in (("immediate", "o", True), ("reasoning", "s", False)):
            r = d.get(mode)
            if not r or not r["cc"]:
                continue
            ax.plot(r["cc"], r["tc"], marker, ms=5.5, color="#0f8f80" if fill else "#c25555",
                    mfc=("#0f8f80" if fill else "white"), mec=("#0f8f80" if fill else "#c25555"), mew=1.3, alpha=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Counting capacity (items)", fontsize=10.5); ax.set_ylabel("Tracking capacity (swaps)", fontsize=10.5)
    ax.tick_params(labelsize=9); ax.grid(alpha=0.2, linewidth=0.9)
    ax.plot([], [], "o", color="#0f8f80", label="reasoning off"); ax.plot([], [], "s", mfc="white", mec="#c25555", label="reasoning on")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    p1 = stats["P1"]
    ax.set_title(f"State Tracking vs CC (ρ = {p1['rho']:.2f}, n = {p1['n']})", fontsize=11)
    fig.tight_layout(); FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "tracking-vs-cc.pdf"); fig.savefig(FIG / "tracking-vs-cc.png", dpi=200)
    print(json.dumps(stats, indent=1, default=float))
    print(f"wrote {GEN / 'tracking_table.tex'} ({len(runs)} runs) and {FIG / 'tracking-vs-cc.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
