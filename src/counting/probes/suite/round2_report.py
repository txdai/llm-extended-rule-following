"""Aggregate the round-2 controls (E1-E4) into cross-model tables.

Reads results/suite/<model>/round2/ (E4 `accumulate.jsonl`, E1 `output_*.jsonl`,
E2 `logit_interp.jsonl`), computes E3 offline from the multi-prompt and
mixed-grid caches, and writes results/suite/round2_tables.{json,md}.

    python -m counting.probes.suite.round2_report --suite-dir results/suite
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .report import first_int, first_token_match, layer_sort, load_json, load_rows
from .subspace import LAMBDA_GRID, choose_lambda_grouped, count_folds, grouped_cv, ridge_multi

MODEL_ORDER = ["gpt-oss-20b", "gpt-oss-120b", "gemma-4-26b-a4b-it",
               "muse-glimmer-30b", "qwen3.6-35b-a3b", "llama-3.3-70b-instruct"]


# -- E4 ----------------------------------------------------------------------

def slope_with_ci(deltas: np.ndarray, shifts: np.ndarray, n_boot: int = 300, seed: int = 0) -> dict:
    ok = np.isfinite(shifts)
    d, s = deltas[ok].astype(float), shifts[ok].astype(float)
    if len(d) < 3 or np.var(d) == 0:
        return {"slope": None, "ci": None, "n": int(ok.sum()), "broken": int((~ok).sum())}
    rng = np.random.default_rng(seed)
    fit = lambda dd, ss: float(np.polyfit(dd, ss, 1)[0])
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        if np.var(d[idx]) > 0:
            boots.append(fit(d[idx], s[idx]))
    return {"slope": round(fit(d, s), 3), "ci": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            "n": int(ok.sum()), "broken": int((~ok).sum()), "moved": int((s != 0).sum()),
            "mean_shift_by_delta": {str(int(k)): round(float(s[d == k].mean()), 2) for k in sorted(set(d.tolist()))}}


def summarize_accumulate(model_dir: Path) -> dict | None:
    rows = load_rows(model_dir / "round2" / "accumulate.jsonl")
    fit = load_json(model_dir / "round2" / "fit.json")
    if not rows or not fit:
        return None
    baseline = {}
    for r in rows:
        if r["group"] == "baseline":
            baseline[(r["band"], r["length"])] = r.get("parsed")
    out: dict = {"fit": fit, "baselines": {f"{b}:{l}": v for (b, l), v in baseline.items()}, "shift": {}, "rescale": {}}
    groups: dict = defaultdict(list)
    for r in rows:
        if r["group"] != "shift":
            continue
        b = baseline.get((r["band"], r["length"]))
        if b is None:
            continue
        shift = (r["parsed"] - b) if r.get("parsed") is not None else np.nan
        groups[(r["site"], r["band"], r["family"], r["variant"])].append((r["delta"], shift, r["j"], r["length"]))
    for (site, band, family, variant), pts in groups.items():
        deltas = np.array([p[0] for p in pts]); shifts = np.array([p[1] for p in pts], dtype=float)
        key = f"{site}|{band}|{family}|{variant}"
        out["shift"][key] = slope_with_ci(deltas, shifts)
        # by j (edit-start fraction), by prompt length, and by sign of delta
        by_j: dict = {}
        for j in sorted({p[2] for p in pts if p[2] is not None}):
            sel = [p for p in pts if p[2] == j]
            by_j[str(j)] = slope_with_ci(np.array([p[0] for p in sel]), np.array([p[1] for p in sel], dtype=float), n_boot=100)
        out["shift"][key]["by_j"] = by_j
        by_len: dict = {}
        for length in sorted({p[3] for p in pts}):
            sel = [p for p in pts if p[3] == length]
            by_len[str(length)] = slope_with_ci(np.array([p[0] for p in sel]), np.array([p[1] for p in sel], dtype=float), n_boot=100)
        out["shift"][key]["by_length"] = {k: v["slope"] for k, v in by_len.items()}
        pos = [p for p in pts if p[0] > 0]
        neg = [p for p in pts if p[0] < 0]
        ok = lambda sel: np.isfinite([p[1] for p in sel])
        out["shift"][key]["slope_positive_delta"] = round(float(np.sum([p[0] * p[1] for p in pos if np.isfinite(p[1])]) / max(np.sum([p[0] ** 2 for p in pos if np.isfinite(p[1])]), 1e-9)), 3) if pos else None
        out["shift"][key]["slope_negative_delta"] = round(float(np.sum([p[0] * p[1] for p in neg if np.isfinite(p[1])]) / max(np.sum([p[0] ** 2 for p in neg if np.isfinite(p[1])]), 1e-9)), 3) if neg else None
    for r in rows:
        if r["group"] == "rescale":
            b = baseline.get(("beyond", r["length"]))
            out["rescale"].setdefault(f"{r['site']}|L{r['length']}|target{r['target']}", []).append(
                {"baseline": b, "parsed": r.get("parsed"), "target": r["target"], "length": r["length"]})
    return out


# -- E1 ----------------------------------------------------------------------

def summarize_output(model_dir: Path, task: str) -> dict | None:
    rows = load_rows(model_dir / "round2" / f"output_{task}.jsonl")
    meta = load_json(model_dir / "round2" / f"output_{task}_meta.json")
    if not rows or not meta:
        return None
    base = meta["base"]
    out: dict = {"meta": {k: meta[k] for k in ("n_stable", "first_token_exact_rate", "linear_cv_r2", "example_prompt")}}
    loc = [r for r in rows if r["group"] == "localize" and r["layer"] is not None]
    if loc:
        by_layer: dict = defaultdict(list)
        for r in loc:
            by_layer[r["layer"]].append(first_token_match(r["decoded"], r["donor"]))
        curve = {int(l): round(sum(v) / len(v), 3) for l, v in sorted(by_layer.items())}
        out["localize"] = {"curve": curve, "n_layers": len(curve),
                           "consolidation_layer": next((l for l, v in curve.items() if v >= 0.8), None),
                           "half_transfer_layer": next((l for l, v in curve.items() if v >= 0.5), None)}
    tr = [r for r in rows if r["group"] in ("transplant", "layers")]
    if tr:
        out["transplant"] = {}
        for site in sorted({r["site"] for r in tr}, key=layer_sort):
            rs = [r for r in tr if r["site"] == site and str(r["donor"])[0] != str(base)[0]]
            def rate(fam, comp):
                sel = [r for r in rs if r["family"] == fam and r["complement"] == comp]
                return round(sum(first_token_match(r["decoded"], r["donor"]) for r in sel) / len(sel), 3) if sel else None
            out["transplant"][site] = {"n_informative": len({r["donor"] for r in rs}), "full": rate("full", False),
                                       "U": rate("U", False), "U_comp": rate("U", True),
                                       "group": "layers" if any(r["group"] == "layers" for r in rs) else "transplant"}
    ab = [r for r in rows if r["group"] == "ablate" and r["family"] != "none"]
    if ab:
        out["ablate"] = {}
        for site in sorted({r["site"] for r in ab}, key=layer_sort):
            fams: dict = {}
            for fam in sorted({r["family"] for r in ab if r["site"] == site}):
                sel = [r for r in ab if r["site"] == site and r["family"] == fam and r.get("own")]
                fams[fam] = f"{sum(1 for r in sel if r['decoded'].strip()[:1] == r['own'].strip()[:1])}/{len(sel)}"
            out["ablate"][site] = fams
    it = [r for r in rows if r["group"] == "interp" and str(r["donor"])[0] != str(base)[0]]
    if it:
        by_alpha = {}
        for alpha in sorted({r["alpha"] for r in it}):
            sel = [r for r in it if r["alpha"] == alpha]
            by_alpha[str(alpha)] = round(sum(first_token_match(r["decoded"], r["donor"]) for r in sel) / len(sel), 3)
        out["interp"] = {"site": it[0]["site"], "donor_rate_by_alpha": by_alpha,
                         "first_alpha_50": next((a for a, v in by_alpha.items() if v >= 0.5), None),
                         "first_alpha_90": next((a for a, v in by_alpha.items() if v >= 0.9), None)}
    return out


def counting_reference(model_entry: dict) -> dict | None:
    """The counting task's numbers from the round-1 subspace block."""
    sub = model_entry.get("subspace")
    if not sub:
        return None
    out: dict = {"localize": {"consolidation_layer": sub["localize"]["consolidation_layer"],
                              "half_transfer_layer": sub["localize"]["half_transfer_layer"],
                              "n_layers": sub["localize"]["n_layers"], "curve": {int(k): v for k, v in sub["localize"]["curve"].items()}},
                 "transplant": {}, "ablate": {}}
    for site, tr in sub["transplant"].items():
        fam = tr["families"]
        u = next((v for k, v in fam.items() if k.startswith("U|") and k.endswith("|proj")), None)
        uc = next((v for k, v in fam.items() if k.startswith("U|") and k.endswith("|comp")), None)
        out["transplant"][site] = {"full": fam["full"]["rate"], "U": u["rate"] if u else None,
                                   "U_comp": uc["rate"] if uc else None, "n_informative": fam["full"]["n_informative"]}
    for site, ab in sub.get("ablate", {}).items():
        fams = {}
        for key, st in ab.items():
            f, k = key.split("|")
            if f in ("U", "L", "F") or (f in ("R", "A") and k in ("8", "16")):
                name = f if f in ("U", "L", "F") else f"{f}{k}"
                fams.setdefault(name, [0, 0])
                fams[name][0] += st["first_token_survive"]; fams[name][1] += st["n"]
        out["ablate"][site] = {k: f"{v[0]}/{v[1]}" for k, v in fams.items()}
    if sub.get("interp"):
        site = sorted(sub["interp"], key=layer_sort)[-1]
        tbl = sub["interp"][site]
        rates = {a: (v["donor"] / v["n"] if v["n"] else 0) for a, v in tbl.items()}
        out["interp"] = {"site": site, "first_alpha_50": next((a for a, v in rates.items() if v >= 0.5), None),
                         "first_alpha_90": next((a for a, v in rates.items() if v >= 0.9), None)}
    return out


# -- E2 ----------------------------------------------------------------------

def summarize_logits(model_dir: Path) -> dict | None:
    rows = load_rows(model_dir / "round2" / "logit_interp.jsonl")
    meta = load_json(model_dir / "round2" / "logit_interp_meta.json")
    if not rows or not meta:
        return None
    base = meta["base"]
    pairs: dict = defaultdict(list)
    for r in rows:
        pairs[r["donor"]].append(r)
    informative = {d: rs for d, rs in pairs.items() if str(d)[0] != str(base)[0]}
    with_inter = {d: rs for d, rs in informative.items() if rs[0]["n_intermediate_tokens"] > 0}
    leader_top = sum(1 for rs in with_inter.values() if any(r["top_is_intermediate"] for r in rs))
    leader_logit = sum(1 for rs in with_inter.values()
                       if any(r["max_intermediate_logit"] is not None and 0 < r["alpha"] < 1
                              and r["max_intermediate_logit"] > max(r["base_logit"], r["donor_logit"]) for r in rs))
    # Which intermediate leads: a neighbour of an endpoint (within 2) or an integer in between?
    def leader_value(r):
        s = r.get("top_string")
        try:
            return int(str(s).strip())
        except (TypeError, ValueError):
            return None
    far_leader = 0
    neighbour_leader = 0
    for d, rs in with_inter.items():
        vals = [leader_value(r) for r in rs if r["top_is_intermediate"]]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        if any(min(abs(v - base), abs(v - d)) > 2 for v in vals):
            far_leader += 1
        else:
            neighbour_leader += 1
    mass_half = [r["intermediate_mass"] for rs in with_inter.values() for r in rs if abs(r["alpha"] - 0.5) < 1e-6]
    # Split of the intermediate mass at alpha 0.5 by distance from the endpoints
    # (needs the per-token logits stored by the current logit_interp).
    near_half, far_half = [], []
    strings = meta.get("answer_token_strings")
    if strings and rows and rows[0].get("answer_logits"):
        values = [int(s.strip()) if s.strip().isdigit() else None for s in strings]
        for d, rs in with_inter.items():
            for r in rs:
                if abs(r["alpha"] - 0.5) > 1e-6:
                    continue
                lg = np.array(r["answer_logits"], dtype=float)
                p = np.exp(lg - lg.max()); p /= p.sum()
                lo, hi = sorted((base, d))
                near_half.append(float(sum(pi for pi, v in zip(p, values) if v is not None and lo < v < hi and min(abs(v - base), abs(v - d)) <= 2)))
                far_half.append(float(sum(pi for pi, v in zip(p, values) if v is not None and lo < v < hi and min(abs(v - base), abs(v - d)) > 2)))
    mass_ends = [r["intermediate_mass"] for rs in with_inter.values() for r in rs if r["alpha"] in (0.0, 1.0)]
    mass_max = [max(r["intermediate_mass"] for r in rs) for rs in with_inter.values()]
    switch = []
    for rs in informative.values():
        srt = sorted(rs, key=lambda r: r["alpha"])
        a = next((r["alpha"] for r in srt if r["top_is_donor"]), None)
        if a is not None:
            switch.append(a)
    return {"site": meta["site"], "digit_split": meta["digit_split"], "n_pairs": len(informative),
            "n_pairs_with_intermediate_tokens": len(with_inter),
            "pairs_intermediate_leads_argmax": leader_top,
            "pairs_leader_is_endpoint_neighbour": neighbour_leader,
            "pairs_leader_far_from_both_endpoints": far_leader,
            "pairs_intermediate_logit_exceeds_both_endpoints": leader_logit,
            "median_intermediate_mass_at_half": round(float(np.median(mass_half)), 4) if mass_half else None,
            "median_intermediate_mass_at_half_near_endpoints": round(float(np.median(near_half)), 4) if near_half else None,
            "median_intermediate_mass_at_half_far": round(float(np.median(far_half)), 4) if far_half else None,
            "max_intermediate_mass_at_half_far": round(float(np.max(far_half)), 4) if far_half else None,
            "median_intermediate_mass_at_endpoints": round(float(np.median(mass_ends)), 4) if mass_ends else None,
            "max_intermediate_mass_median_over_pairs": round(float(np.median(mass_max)), 4) if mass_max else None,
            "median_switch_alpha": round(float(np.median(switch)), 3) if switch else None}


# -- E3 ----------------------------------------------------------------------

def exact_rates(pred: np.ndarray, truth: np.ndarray) -> dict:
    rounded = np.round(pred)
    return {"exact": round(float(np.mean(rounded == truth)), 3), "within1": round(float(np.mean(np.abs(rounded - truth) <= 1)), 3),
            "n": int(len(truth))}


def summarize_exact(model_dir: Path, scan_records: list[dict] | None) -> dict | None:
    multi = model_dir / "collect-multi" / "activations.npz"
    meta = load_json(model_dir / "collect-multi" / "meta.json")
    if not multi.exists() or not meta:
        return None
    c = np.load(multi)
    counts, exact = c["counts"].astype(int), c["greedy_exact"].astype(bool)
    scc = int(meta["scc"])
    sites = [k.split("|")[0] for k in c.files if k.endswith("|decode_point")]
    in_regime = counts <= scc
    beyond = (counts > scc) & (counts <= int(round(1.25 * scc)))
    model_answers = {r["length"]: r.get("parsed") for r in (scan_records or [])}
    mixed_path = model_dir / "mixed" / "states.npz"
    mixed_json = load_json(model_dir / "mixed" / "mixed.json")
    out: dict = {"scc": scc, "n_in_regime": int(in_regime.sum()), "n_beyond": int(beyond.sum()), "sites": {},
                 "model": {
                     "in_regime_exact": round(float(exact[in_regime].mean()), 3),
                     "beyond_exact": round(float(exact[beyond].mean()), 3) if beyond.any() else None,
                     "scan_in_regime_within1": round(float(np.mean([abs(model_answers[n] - n) <= 1 for n in range(1, scc + 1)
                                                                    if model_answers.get(n) is not None])), 3) if model_answers else None,
                     "scan_beyond_within1": round(float(np.mean([abs(model_answers[n] - n) <= 1 for n in range(scc + 1, int(round(1.25 * scc)) + 1)
                                                                 if model_answers.get(n) is not None])), 3) if model_answers else None,
                     "mixed_exact": mixed_json["behavioral_accuracy"] if mixed_json else None,
                     "mixed_within1": round(float(np.mean([abs((t.get("parsed") or -99) - t["k"]) <= 1 for t in mixed_json["trials"]])), 3) if mixed_json else None,
                 }}
    for site in sites:
        x = c[f"{site}|decode_point"].astype(np.float32)
        x_in, y_in = x[in_regime], counts[in_regime].astype(float)
        folds = count_folds(counts[in_regime], seed=0)
        lam, r2, pred_in = choose_lambda_grouped(x_in, y_in, folds, grid=LAMBDA_GRID)
        entry = {"in_regime": exact_rates(pred_in, counts[in_regime]), "cv_r2": round(float(r2), 4)}
        w, xc, yc = ridge_multi(x_in, y_in, lam)
        if beyond.any():
            pred_b = (x[beyond].astype(np.float64) - xc) @ w[:, 0] + yc[0]
            entry["beyond"] = exact_rates(pred_b, counts[beyond])
        if mixed_path.exists() and mixed_json and site in np.load(mixed_path).files:
            m = np.load(mixed_path)
            xm, km = m[site].astype(np.float32), m["k"].astype(int)
            pred_h = (xm.astype(np.float64) - xc) @ w[:, 0] + yc[0]
            entry["mixed_homogeneous_readout"] = exact_rates(pred_h, km)
            fm = count_folds(km, seed=0)
            _, r2m, pred_m = choose_lambda_grouped(xm, km.astype(float), fm, grid=LAMBDA_GRID)
            entry["mixed_in_grid_cv"] = exact_rates(pred_m, km)
            entry["mixed_in_grid_cv"]["cv_r2"] = round(float(r2m), 3)
        out["sites"][site] = entry
    return out


# -- driver ------------------------------------------------------------------

def to_markdown(models: dict[str, dict]) -> str:
    lines = ["# Round-2 mechanistic controls: cross-model tables", ""]

    lines += ["## E4. Multi-position shift of the running count: slope of answer shift vs delta", "",
              "| model | site | band | S local (CI) | S local: +delta / -delta (through origin) | S local by length L | S mean (CI) | random local | random mean | decode-point local | decode-point mean | moved S local | broken |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in models.items():
        acc = m.get("accumulate")
        if not acc:
            continue
        sites = sorted({k.split("|")[0] for k in acc["shift"]}, key=layer_sort)
        for site in sites:
            for band in ("stable", "beyond"):
                def cell(fam, var):
                    st = acc["shift"].get(f"{site}|{band}|{fam}|{var}")
                    if not st or st["slope"] is None:
                        return "--"
                    return f"{st['slope']} [{st['ci'][0]}, {st['ci'][1]}]"
                s_loc = acc["shift"].get(f"{site}|{band}|S|local", {})
                if not s_loc:
                    continue
                by_len = " ".join(f"{k}:{v}" for k, v in s_loc.get("by_length", {}).items())
                lines.append(f"| {name} | {site} | {band} | {cell('S','local')} | {s_loc.get('slope_positive_delta')} / {s_loc.get('slope_negative_delta')} | {by_len} "
                             f"| {cell('S','mean')} | {cell('R','local')} | {cell('R','mean')} "
                             f"| {cell('S_decode_point','local')} | {cell('S_decode_point','mean')} | {s_loc.get('moved')}/{s_loc.get('n')} | {s_loc.get('broken')} |")

    lines += ["", "## E4. Running-count readout at item positions and the subspace S_k", "",
              "| model | site | k | LOO R2 | LOO RMSE (items) | S_k explained | mean-step norm / state norm |", "|---|---|---|---|---|---|---|"]
    for name, m in models.items():
        acc = m.get("accumulate")
        if not acc:
            continue
        for site, f in sorted(acc["fit"]["sites"].items(), key=lambda kv: layer_sort(kv[0])):
            lines.append(f"| {name} | {site} | {f['k']} | {f['loo_r2']} | {f['loo_rmse_items']} | {f['cum_explained_k']} | {f['mean_step_norm_over_state_norm']} |")

    lines += ["", "## E4. Rescaling the accumulation beyond CC (answer under compression of the running count)", "",
              "| model | site | L | target | unedited answer | edited answer |", "|---|---|---|---|---|---|"]
    for name, m in models.items():
        acc = m.get("accumulate")
        if not acc:
            continue
        for key, rs in sorted(acc["rescale"].items(), key=lambda kv: (layer_sort(kv[0].split("|")[0]), kv[0])):
            site, L, target = key.split("|")
            lines.append(f"| {name} | {site} | {L[1:]} | {target[6:]} | {rs[0]['baseline']} | {', '.join(str(r['parsed']) for r in rs)} |")

    lines += ["", "## E1. Consolidation layer by task (full transplant at every layer, >= 80% transfer)", "",
              "| model | layers | counting (>=50% / >=80%) | copy | retrieval | copy first-token exact | retrieval first-token exact |", "|---|---|---|---|---|---|---|"]
    for name, m in models.items():
        ct, cp, rt = m.get("counting"), m.get("copy"), m.get("retrieval")
        if not (cp or rt):
            continue
        def loc(t):
            if not t or not t.get("localize"):
                return "--"
            return f"{t['localize']['half_transfer_layer']} / {t['localize']['consolidation_layer']}"
        lines.append(f"| {name} | {ct['localize']['n_layers'] if ct else '--'} | {loc(ct)} | {loc(cp)} | {loc(rt)} "
                     f"| {cp['meta']['first_token_exact_rate'] if cp else '--'} | {rt['meta']['first_token_exact_rate'] if rt else '--'} |")

    lines += ["", "## E1. U transplant, complement and ablation by task (first-token transfer, informative donors)", "",
              "| model | site | task | full | U | U comp. | U ablation survives | L | F | R8 | R16 | A8 | A16 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in models.items():
        for task in ("counting", "copy", "retrieval"):
            t = m.get(task)
            if not t or not t.get("transplant"):
                continue
            for site in sorted(t["transplant"], key=layer_sort):
                tr = t["transplant"][site]
                ab = t.get("ablate", {}).get(site, {})
                lines.append(f"| {name} | {site} | {task} | {tr['full']} | {tr['U']} | {tr['U_comp']} | {ab.get('U', '--')} "
                             f"| {ab.get('L', '--')} | {ab.get('F', '--')} | {ab.get('R8', '--')} | {ab.get('R16', '--')} | {ab.get('A8', '--')} | {ab.get('A16', '--')} |")

    lines += ["", "## E1. Interpolation switch by task (first alpha with >= 50% / >= 90% donor)", "",
              "| model | counting | copy | retrieval |", "|---|---|---|---|"]
    for name, m in models.items():
        def sw(t):
            it = (t or {}).get("interp")
            return f"{it['first_alpha_50']} / {it['first_alpha_90']}" if it else "--"
        if m.get("copy") or m.get("retrieval"):
            lines.append(f"| {name} | {sw(m.get('counting'))} | {sw(m.get('copy'))} | {sw(m.get('retrieval'))} |")

    lines += ["", "## E2. Logit-level interpolation at the last site", "",
              "| model | pairs (with intermediate tokens) | intermediate leads argmax | of which the leader is an endpoint neighbour (within 2) / farther | intermediate logit exceeds both endpoints | median intermediate mass at alpha 0.5 | of which within 2 of an endpoint / farther (median; max farther) | same at the endpoints | median switch alpha |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, m in models.items():
        lg = m.get("logits")
        if lg:
            split = (f"{lg.get('median_intermediate_mass_at_half_near_endpoints')} / {lg.get('median_intermediate_mass_at_half_far')} "
                     f"(max {lg.get('max_intermediate_mass_at_half_far')})") if lg.get("median_intermediate_mass_at_half_far") is not None else "--"
            lines.append(f"| {name} | {lg['n_pairs']} ({lg['n_pairs_with_intermediate_tokens']}) | {lg['pairs_intermediate_leads_argmax']} "
                         f"| {lg.get('pairs_leader_is_endpoint_neighbour')} / {lg.get('pairs_leader_far_from_both_endpoints')} "
                         f"| {lg['pairs_intermediate_logit_exceeds_both_endpoints']} | {lg['median_intermediate_mass_at_half']} | {split} "
                         f"| {lg['median_intermediate_mass_at_endpoints']} | {lg['median_switch_alpha']} |")

    lines += ["", "## E3. Exact-match accuracy of the rounded ridge readout (count-held-out folds) against the model", "",
              "| model | site | in-regime probe exact / +-1 | model exact / +-1 (scan) | beyond-CC probe exact / +-1 | model exact / +-1 | mixed grid: in-grid CV exact / +-1 | homogeneous readout exact / +-1 | model exact / +-1 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, m in models.items():
        ex = m.get("exact")
        if not ex:
            continue
        md = ex["model"]
        for site in sorted(ex["sites"], key=layer_sort):
            s = ex["sites"][site]
            def pair(d):
                return f"{d['exact']} / {d['within1']}" if d else "--"
            lines.append(f"| {name} | {site} | {pair(s['in_regime'])} | {md['in_regime_exact']} / {md['scan_in_regime_within1']} "
                         f"| {pair(s.get('beyond'))} | {md['beyond_exact']} / {md['scan_beyond_within1']} "
                         f"| {pair(s.get('mixed_in_grid_cv'))} | {pair(s.get('mixed_homogeneous_readout'))} | {md['mixed_exact']} / {md['mixed_within1']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite-dir", type=Path, default=Path("results/suite"))
    args = parser.parse_args()
    tables = {m["model"]: m for m in json.loads((args.suite_dir / "tables.json").read_text())}
    models: dict[str, dict] = {}
    for name in MODEL_ORDER:
        d = args.suite_dir / name
        if not d.is_dir():
            continue
        scan_dirs = sorted(d.glob("scan-*"))
        scan_records = None
        immediate = {"gpt-oss-20b": "final", "gpt-oss-120b": "final", "gemma-4-26b-a4b-it": "nothink",
                     "muse-glimmer-30b": "final", "qwen3.6-35b-a3b": "nothink", "llama-3.3-70b-instruct": "final"}
        sd = d / f"scan-{immediate.get(name, 'final')}"
        if sd.exists():
            scan_records = load_rows(sd / "records.jsonl")
        entry = {
            "accumulate": summarize_accumulate(d),
            "copy": summarize_output(d, "copy"),
            "retrieval": summarize_output(d, "retrieval"),
            "counting_layers": summarize_output(d, "counting"),
            "counting": counting_reference(tables.get(name, {})),
            "logits": summarize_logits(d),
            "exact": summarize_exact(d, scan_records),
        }
        if any(v for k, v in entry.items() if k != "counting"):
            models[name] = entry
    (args.suite_dir / "round2_tables.json").write_text(json.dumps(models, indent=1))
    (args.suite_dir / "round2_tables.md").write_text(to_markdown(models))
    print(f"aggregated {len(models)} models -> {args.suite_dir / 'round2_tables.md'}")


if __name__ == "__main__":
    main()
