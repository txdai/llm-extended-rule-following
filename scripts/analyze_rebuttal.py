#!/usr/bin/env python3
"""Numbers behind the rebuttal experiments, recomputed from the raw records.

    PYTHONPATH=src python scripts/analyze_rebuttal.py

Reads results/rebuttal-* (plus results/motif-sweep, results/counting-data and
results/benchmark_data) and prints, in the order the paper uses them:

  1. encodings across sixteen models   (Appendix A.6, Table 2)
  2. output formats                    (Appendix A.7, Table 3)
  3. concurrent loads, five scans each (Appendix H, Table 17; Fig. 3C)
  4. Claude Opus 4 past its boundary   (reply to the 5%-tolerance comparison)
  5. METR partial correlation against the whole benchmark screen (Appendix J)

Needs numpy and scipy. Every fit is the catalogue's two-regime fit on
scan-phase probes, without interval calibration.
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

from counting.analysis.capacity import fit_transition

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MOTIFS = ["baseline-a-comma", "letter-b-comma", "letter-x-comma", "double-aa-comma",
          "greek-alpha-comma", "greek-beta-comma", "chinese-han-comma",
          "chinese-zhong-comma", "letter-a-space", "letter-a-pipe", "letter-a-section"]
FORMATS = ["int_bare", "english_words", "zero_padded_spaced", "mod_10"]


def scan_trials(path) -> tuple[dict, list[dict]]:
    payload = json.loads(Path(path).read_text())
    trials = sorted((t for t in payload.get("trials", []) if t.get("phase") == "scan"),
                    key=lambda t: t["center_length"])
    return payload, trials


def fit(trials):
    return fit_transition([t["center_length"] for t in trials],
                          [bool(t["exact_match"]) for t in trials],
                          [t["parsed"] is None for t in trials], calibrate=False)


# -- 1. encodings ------------------------------------------------------------

def variance_shares(Y: np.ndarray) -> np.ndarray:
    """Two-way random effects, one observation per cell: model, encoding, residual."""
    I, J = Y.shape
    g = Y.mean()
    ss_m = J * ((Y.mean(1) - g) ** 2).sum()
    ss_f = I * ((Y.mean(0) - g) ** 2).sum()
    ss_e = ((Y - g) ** 2).sum() - ss_m - ss_f
    ms_m, ms_f, ms_e = ss_m / (I - 1), ss_f / (J - 1), ss_e / ((I - 1) * (J - 1))
    vm, vf = max(0.0, (ms_m - ms_e) / J), max(0.0, (ms_f - ms_e) / I)
    return np.array([vm, vf, ms_e]) / (vm + vf + ms_e)


def kendall_w(Y: np.ndarray) -> tuple[float, float, float]:
    I, J = Y.shape
    R = np.array([stats.rankdata(Y[:, j]) for j in range(J)]).sum(0)
    W = 12 * ((R - R.mean()) ** 2).sum() / (J ** 2 * (I ** 3 - I))
    rhos = [stats.spearmanr(Y[:, a], Y[:, b]).statistic
            for a in range(J) for b in range(a + 1, J)]
    return float(W), float(np.mean(rhos)), float(np.min(rhos))


def encodings() -> None:
    print("== 1. encodings across sixteen models")
    cells, slopes, repeats = {}, {}, collections.defaultdict(list)
    # The three models of Fig. 2F keep their original scans; motif16 adds the rest.
    for source in ("motif-sweep", "rebuttal-motif16"):
        for path in sorted(glob.glob(str(RES / source / "*" / "models" / "*.json"))):
            payload, trials = scan_trials(path)
            key = (payload["run_key"], Path(path).parts[-3])
            cc = fit(trials).scc
            repeats[key].append(cc)
            if key in cells:
                continue
            cells[key] = cc
            pts = [(t["center_length"], t["input_tokens"]) for t in trials
                   if t.get("input_tokens") is not None]
            if len(pts) >= 5:
                slopes[key] = float(np.polyfit(*np.array(pts, float).T, 1)[0])
    local, local_slopes = {}, {}
    for path in sorted(glob.glob(str(RES / "rebuttal-motif-local" / "*" / "*" / "summary.json"))):
        d = json.loads(Path(path).read_text())
        key = ("local/" + d["model"], d["motif"])
        local[key] = d["transition"]["scc"]
        recs = [json.loads(line) for line in (Path(path).parent / "records.jsonl").open()]
        pts = [(r["length"], r["input_tokens"]) for r in recs if r.get("input_tokens") is not None]
        if len(pts) >= 5:
            local_slopes[key] = float(np.polyfit(*np.array(pts, float).T, 1)[0])

    def report(name, table, token_slopes, seed=1):
        models = sorted({m for m, _ in table})
        Y = np.array([[table[(m, f)] for f in MOTIFS] for m in models], float)
        L = np.log(np.maximum(Y, 1))
        rng = np.random.default_rng(seed)
        boot = np.array([variance_shares(L[rng.integers(0, len(models), len(models))])
                         for _ in range(4000)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        sh, raw = variance_shares(L), variance_shares(Y)
        W, rho, rho_min = kendall_w(Y)
        med = sorted(int(np.median(r)) for r in Y)
        print(f"  {name}: {len(models)} models, {Y.size} scans, median CC {med[0]} to {med[-1]}")
        for label, k in (("model", 0), ("encoding", 1), ("residual", 2)):
            print(f"    {label:9s} {100*sh[k]:5.1f}% [{100*lo[k]:.1f}, {100*hi[k]:.1f}]"
                  f"   raw CC {100*raw[k]:.1f}%")
        print(f"    Kendall W {W:.3f}; Spearman between encodings mean {rho:.3f}, min {rho_min:.3f}")
        ratio = Y[:, MOTIFS.index("letter-a-section")] / Y[:, 0]
        print(f"    section sign / comma: <= 1/3 in {(ratio <= 1/3 + 1e-9).sum()} models, "
              f"within 5% in {(np.abs(ratio - 1) <= 0.05).sum()}")
        xs, ys = [], []
        for m in models:
            t = np.array([token_slopes.get((m, f), np.nan) for f in MOTIFS])
            y = np.log(np.maximum([table[(m, f)] for f in MOTIFS], 1))
            ok = ~np.isnan(t)
            xs += list(t[ok] - t[ok].mean()); ys += list(y[ok] - y[ok].mean())
        r = np.corrcoef(xs, ys)[0, 1]
        print(f"    tokens per item vs log CC within model: r = {r:.2f} ({100*r*r:.0f}%), n = {len(xs)}")

    report("API panel", cells, slopes)
    report("open-weight, local", local, local_slopes)
    report("all", {**cells, **local}, {**slopes, **local_slopes})
    print("  cells scanned twice (first, repeat):",
          {f"{m.split('/')[-1]}|{f}": v for (m, f), v in repeats.items() if len(v) > 1})


# -- 2. output formats -------------------------------------------------------

def strictly_formatted(fmt: str, reply: str | None) -> bool:
    reply = (reply or "").strip()
    if fmt == "int_bare":
        return re.fullmatch(r"\d+", reply) is not None
    if fmt == "mod_10":
        return re.fullmatch(r"\d", reply) is not None
    if fmt == "zero_padded_spaced":
        return re.fullmatch(r"\d \d \d \d", reply) is not None
    return re.fullmatch(r"[a-z\- ]+", reply) is not None and len(reply) < 60


def output_formats() -> None:
    print("\n== 2. output formats")
    table = {}
    for slug in ("gemini-3p7-flash", "gpt-5p4_nothink", "gemini-2p5-flash_nothink",
                 "claude-sonnet-5_nothink", "claude-sonnet-4p6_nothink"):
        for fmt in FORMATS:
            path = glob.glob(str(RES / "rebuttal-exp1" / fmt / slug / "models" / "*.json"))[0]
            _, trials = scan_trials(path)
            f = fit(trials)
            ok = sum(strictly_formatted(fmt, t.get("raw_response")) for t in trials)
            cut = sum(t.get("response_status") != "completed" for t in trials)
            reasoning = np.median([t.get("reasoning_tokens") or 0 for t in trials])
            table[(slug, fmt)] = f.scc
            print(f"  {slug:26s} {fmt:19s} CC {f.scc:4d}{'+' if f.censored else ' '} "
                  f"format followed {ok:3d}/{len(trials):3d}  cut off {cut:3d}  "
                  f"median reasoning tokens {reasoning:.0f}")
    for path in sorted(glob.glob(str(RES / "rebuttal-exp1-local" / "*" / "*" / "summary.json"))):
        d = json.loads(Path(path).read_text())
        recs = [json.loads(line) for line in (Path(path).parent / "records.jsonl").open()]
        ok = sum(strictly_formatted(d["format"], r.get("reply")) for r in recs)
        table[(d["model"], d["format"])] = d["transition"]["scc"]
        print(f"  {d['model']:26s} {d['format']:19s} CC {d['transition']['scc']:4d}  "
              f"format followed {ok:3d}/{len(recs):3d}")
    kept = ["gemini-3p7-flash", "gpt-5p4_nothink", "gemini-2p5-flash_nothink",
            "gemma-4-26b-a4b-it", "qwen3.6-35b-a3b", "llama-3.3-70b-instruct", "gpt-oss-20b"]
    bare = np.array([table[(m, "int_bare")] for m in kept], float)
    for fmt in FORMATS[1:]:
        r = np.array([table[(m, fmt)] for m in kept]) / bare
        print(f"  {fmt:19s} / integer, seven configurations: {np.round(r, 2).tolist()}")
    words = [table[(m, "english_words")] for m in kept]
    print(f"  words: median ratio {np.median(np.array(words) / bare):.2f}, "
          f"rank correlation {stats.spearmanr(bare, words).statistic:.2f}")

    print("  gemini-3.7-flash, errors past the boundary:")
    for fmt in FORMATS[:3]:
        path = glob.glob(str(RES / "rebuttal-exp1" / fmt / "gemini-3p7-flash" / "models" / "*.json"))[0]
        _, trials = scan_trials(path)
        cc = fit(trials).scc
        post = [t for t in trials if t["center_length"] > cc]
        errs = [t["parsed"] - t["center_length"] for t in post
                if not t["exact_match"] and t["parsed"] is not None]
        c = collections.Counter(errs)
        last = sum(t["parsed"] is not None and t["parsed"] % 10 == t["center_length"] % 10 for t in post)
        print(f"    {fmt:19s} CC {cc}: {len(errs)} errors, -100: {c[-100]}, -99: {c[-99]}, "
              f"-50: {c[-50]}; last digit right in {last}/{len(post)}, exact in "
              f"{sum(bool(t['exact_match']) for t in post)}")
    path = glob.glob(str(RES / "rebuttal-exp1" / "mod_10" / "gemini-3p7-flash" / "models" / "*.json"))[0]
    _, trials = scan_trials(path)
    print(f"    last digit alone: exact in {sum(bool(t['exact_match']) for t in trials)}/{len(trials)}"
          f" probes up to {trials[-1]['center_length']}")
    for fmt in ("int_bare", "english_words"):
        recs = [json.loads(line) for line in
                (RES / "rebuttal-exp1-local" / "gemma-4-26b-a4b-it" / fmt / "records.jsonl").open()]
        errs = [r["parsed"] - r["length"] for r in recs if not r["correct"] and r["parsed"] is not None]
        print(f"    gemma-4-26b {fmt}: {len(errs)} errors, ten short in {errs.count(-10)}")


# -- 3. concurrent loads -----------------------------------------------------

def concurrent_loads() -> None:
    print("\n== 3. concurrent loads, five scans per cell")
    conds = ["solo", "inert-90", "semantic-90", "pointer-90", "row3-60", "grid-60"]
    for slug in ("gemini-3p7-flash", "claude-sonnet-5"):
        rows = {}
        for cond in conds:
            ccs, accs = [], []
            for rep in range(1, 6):
                path = glob.glob(str(RES / "rebuttal-interference" / f"rep-{rep}" / slug / cond / "models" / "*.json"))[0]
                _, trials = scan_trials(path)
                ccs.append(fit(trials).scc)
                graded = []
                for t in trials:
                    answer = (t.get("metadata") or {}).get("subtask_answer")
                    if answer is None:
                        continue
                    numbers = re.findall(r"-?\d+", t.get("raw_response") or "")
                    graded.append(len(numbers) >= 2 and int(numbers[0]) == answer)
                if graded:
                    accs.append(float(np.mean(graded)))
            rows[cond] = (np.array(ccs, float), accs)
        solo, control = rows["solo"][0], rows["inert-90"][0]
        print(f"  {slug}")
        for cond in conds:
            ccs, accs = rows[cond]
            line = (f"    {cond:12s} {' '.join(str(int(c)) for c in ccs):22s} "
                    f"{ccs.mean():6.1f} +- {ccs.std(ddof=1):4.1f}  ratio {ccs.mean()/solo.mean():.2f}")
            if accs:
                line += f"  load accuracy {np.mean(accs):.2f} +- {np.std(accs, ddof=1):.2f}"
            if cond != "solo":
                line += f"  p vs solo {stats.ttest_ind(ccs, solo, equal_var=False).pvalue:.3f}"
            if cond not in ("solo", "inert-90"):
                line += f"  p vs control {stats.ttest_ind(ccs, control, equal_var=False).pvalue:.3f}"
            print(line)
    print("  open-weight models (ratio to the single solo scan; load accuracy)")
    for model in ("gpt-oss-20b", "gemma-4-26b-a4b-it", "qwen3.6-35b-a3b"):
        solo = None
        out = []
        for cond in conds:
            ccs, accs = [], []
            for path in sorted(glob.glob(str(RES / "rebuttal-interference-local" / model / cond / "seed-*" / "summary.json"))):
                d = json.loads(Path(path).read_text())
                ccs.append(d["transition"]["scc"])
                if d.get("subtask_accuracy") is not None:
                    accs.append(d["subtask_accuracy"])
            if cond == "solo":
                solo = float(np.mean(ccs))
                continue
            out.append(f"{cond} {np.mean(ccs)/solo:.2f} ({np.mean(accs):.2f})")
        print(f"    {model:20s} solo {solo:.0f}; " + "; ".join(out))


# -- 4. Claude Opus 4 --------------------------------------------------------

def opus4() -> None:
    print("\n== 4. claude-opus-4, reasoning off, past the boundary")
    path = RES / "counting-data" / "trials" / "anthropic__claude-opus-4__nothink.jsonl"
    trials = sorted((t for t in map(json.loads, path.open()) if t.get("phase") == "scan"),
                    key=lambda t: t["center_length"])
    cc = fit(trials).scc
    post = [t for t in trials if t["center_length"] > cc]
    rel = [abs(t["parsed"] - t["center_length"]) / t["center_length"] for t in post]
    errs = [t["parsed"] - t["center_length"] for t in post if not t["exact_match"]]
    print(f"  CC {cc}; lengths {post[0]['center_length']} to {post[-1]['center_length']}: exact "
          f"{np.mean([bool(t['exact_match']) for t in post]):.3f}, one short in "
          f"{errs.count(-1)}/{len(errs)} errors, mean relative error {100*np.mean(rel):.3f}%, "
          f"largest {100*max(rel):.1f}%")


# -- 5. METR against the whole screen ----------------------------------------

def rank_partial(a, b, z) -> float:
    ra, rb, rz = (stats.rankdata(v) for v in (a, b, z))
    X = np.column_stack([np.ones(len(rz)), rz])
    resid = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(resid(ra), resid(rb))[0, 1])


def metr_screen() -> None:
    print("\n== 5. partial correlation with CC given the Epoch index, every benchmark")
    rows = list(csv.DictReader((RES / "benchmark_data" / "counting_benchmark_merged.csv").open()))
    usable = [r for r in rows if r["stable_count"] not in ("", "0") and r["benchmark_score"] != ""]
    eci = {r["counting_model"]: float(r["benchmark_score"]) for r in usable
           if r["benchmark_file"] == "epoch_capabilities_index.csv"}
    per = collections.defaultdict(dict)
    for r in usable:
        per[r["benchmark_file"]][r["counting_model"]] = (
            math.log10(float(r["stable_count"])), float(r["benchmark_score"]))
    out = []
    for name, d in per.items():
        shared = [m for m in d if m in eci]
        if name == "epoch_capabilities_index.csv" or len(d) < 10 or len(shared) < 8:
            continue
        pr = rank_partial([d[m][0] for m in shared], [d[m][1] for m in shared], [eci[m] for m in shared])
        if not np.isfinite(pr):
            continue
        t = pr * math.sqrt((len(shared) - 3) / (1 - pr * pr))
        out.append((pr, 2 * stats.t.sf(abs(t), len(shared) - 3), len(shared), name))
    out.sort(reverse=True)
    print(f"  {len(per) - 1} benchmarks in the merged file, partial computed for {len(out)}")
    for pr, p, n, name in out[:4]:
        print(f"    {name:40s} partial {pr:+.2f}  n {n:3d}  p {p:.1e}  x{len(per) - 1} = {min(1, p * (len(per) - 1)):.4f}")


if __name__ == "__main__":
    encodings()
    output_formats()
    concurrent_loads()
    opus4()
    metr_screen()
