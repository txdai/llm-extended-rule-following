"""Aggregate every model's suite outputs into cross-model tables.

Walks results/suite/<model>/ for scan-*/summary.json, geometry.json,
interventions/, and mixed/mixed.json, and writes tables.json plus a readable
tables.md next to them. The narrative report in docs/ cites these tables.

    python -m counting.probes.suite.report --suite-dir results/suite
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def first_token_match(decoded: str, target: int) -> bool:
    """Does the decoded answer start with the target's leading digit?

    The transfer metric that is fair to single-digit tokenizers (Gemma, Llama
    split numbers into digits): an output-patch at one position can only force
    the token it directly feeds, so full-integer match understates transfer.
    """
    text = decoded.strip()
    return bool(text) and text[0] == str(target)[0]


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def summarize_interventions(model_dir: Path) -> dict | None:
    rows_path = model_dir / "interventions" / "interventions.jsonl"
    if not rows_path.exists():
        return None
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    meta = load_json(model_dir / "interventions" / "meta.json") or {}
    by_site: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_site[row["site"]][row["experiment"]].append(row)

    sites = {}
    for site, groups in by_site.items():
        baseline = groups.get("baseline", [{}])[0].get("parsed")
        entry: dict = {"baseline_answer": baseline}
        for name in ("clamp", "step_translate"):
            rs = groups.get(name, [])
            if rs:
                entry[name] = {
                    "n": len(rs),
                    "moved": sum(1 for r in rs if r["parsed"] != baseline),
                    "exact": sum(1 for r in rs if r["parsed"] == r["target"]),
                }
        for name in ("chimera_full", "chimera_linear_only", "chimera_residual_only"):
            rs = groups.get(name, [])
            if rs:
                entry[name] = {
                    "n": len(rs),
                    "parsed_hit": sum(1 for r in rs if r["parsed"] == r["target"]),
                    "first_token_hit": sum(
                        1 for r in rs if first_token_match(r["decoded"], r["target"])
                    ),
                    "stayed_base": sum(1 for r in rs if r["parsed"] == baseline),
                }
        rs = groups.get("random_control", [])
        if rs:
            entry["random_control"] = {
                "n": len(rs),
                "moved": sum(1 for r in rs if r["parsed"] != baseline),
            }
        sites[site] = entry
    return {"base_count": meta.get("base_count"), "sites": sites}


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def transfer_stats(rows: list[dict], base: int) -> dict:
    """First-token transfer of the donor answer, with and without the donors
    that share the base's leading digit (which can never fail the test), and
    the graded displacement (parsed - base) / (donor - base)."""
    informative = [r for r in rows if str(r["donor"])[0] != str(base)[0]]
    disp = [
        (r["parsed"] - base) / (r["donor"] - base)
        for r in rows if r.get("parsed") is not None and r["donor"] != base
    ]
    return {
        "n": len(rows),
        "first_token_hit": sum(1 for r in rows if first_token_match(r["decoded"], r["donor"])),
        "n_informative": len(informative),
        "informative_hit": sum(1 for r in informative if first_token_match(r["decoded"], r["donor"])),
        "rate": round(
            sum(1 for r in informative if first_token_match(r["decoded"], r["donor"])) / len(informative), 3
        ) if informative else None,
        "mean_displacement": round(sum(disp) / len(disp), 3) if disp else None,
        "intermediate": sum(
            1 for r in rows if r.get("parsed") is not None
            and min(base, r["donor"]) < r["parsed"] < max(base, r["donor"])
        ),
    }


def effective_dimension(curve: dict[int, float | None], full_rate: float | None, level: float) -> int | None:
    """Smallest k whose transfer rate reaches `level` x the full-state rate."""
    if not full_rate:
        return None
    for k in sorted(curve):
        rate = curve[k]
        if rate is not None and rate >= level * full_rate:
            return k
    return None


def summarize_subspace(model_dir: Path) -> dict | None:
    sub = model_dir / "subspace"
    readouts = load_json(sub / "readouts.json")
    if readouts is None:
        return None
    meta = load_json(sub / "meta.json") or {}
    base = meta.get("base_count")
    unembed_meta = load_json(sub / "unembed.json") or {}
    # With digit-splitting tokenizers a decode-point edit controls the first
    # digit only and the rest is re-generated from the base, so a parsed
    # "intermediate" integer (donor 12 into base 38 giving 18) is an artifact.
    digit_split = bool(unembed_meta.get("digit_split"))
    out: dict = {"n_stable": readouts["n_stable"], "n_counts": readouts["n_counts"],
                 "digit_split": digit_split, "sites": {}}

    for site, s in readouts["sites"].items():
        entry: dict = {
            "linear_cv_r2": s["linear"]["cv_r2"],
            "cos_paper_u": s["linear"].get("cos_multi_ridge_vs_paper_u"),
            "last_digit": {k: s["digits"]["last"].get(k) for k in
                           ("accuracy", "perm_mean", "perm_max", "neighbour_confusion", "chance")},
            "last_digit_resid": {k: s["digits"].get("last_resid", {}).get(k) for k in
                                 ("accuracy", "perm_mean", "perm_max", "neighbour_confusion", "chance")},
            "tens_digit": {k: s["digits"]["tens"].get(k) for k in ("accuracy", "perm_mean", "n_classes")},
            "readable_periods": s["periodic"].get("readable_periods", []),
            "periodic": {p: {k: v.get(k) for k in ("cycles", "r2_cos", "r2_sin", "null_cos", "null_sin", "readable")}
                         for p, v in s["periodic"].items() if isinstance(v, dict)},
            "trajectory_rank": s["trajectory"]["rank"],
            "cum_evr": s["trajectory"]["cum_evr"],
        }
        if "decoder" in s:
            d = s["decoder"]
            entry["decoder"] = {
                "rank_U": d["rank_U"],
                "max_cos_L_U": max(d["cos_L_U"]) if d["cos_L_U"] else None,
                "max_cos_F_U": max(d["cos_F_U"]) if d["cos_F_U"] else None,
                "max_cos_G_U": max(d["cos_G_U"]) if d["cos_G_U"] else None,
                "max_cos_T8_U": max(d["cos_T8_U"]) if d["cos_T8_U"] else None,
                "logit_lens": d.get("logit_lens_stable"),
            }
        if "chord" in s:
            entry["chord"] = s["chord"]
        out["sites"][site] = entry

    # Localization: first layer at which the full transplant carries the answer.
    loc = load_rows(sub / "localize.jsonl")
    if loc:
        base_loc = loc[0]["base"]
        by_layer: dict[int, list] = defaultdict(list)
        for r in loc:
            if r["layer"] is not None:
                by_layer[r["layer"]].append(first_token_match(r["decoded"], r["donor"]))
        curve = {int(l): round(sum(v) / len(v), 3) for l, v in sorted(by_layer.items())}
        consolidation = next((l for l, v in curve.items() if v >= 0.8), None)
        half = next((l for l, v in curve.items() if v >= 0.5), None)
        out["localize"] = {"n_layers": len(curve), "curve": curve, "consolidation_layer": consolidation,
                           "half_transfer_layer": half, "base": base_loc}

    # Transplant curves and effective dimensions.
    tr = load_rows(sub / "transplant.jsonl")
    if tr and base is None:
        base = tr[0].get("base")  # meta.json is only written when the run ends
    if tr and base is not None:
        out["transplant"] = {}
        for site in sorted({r["site"] for r in tr}, key=layer_sort):
            rows = [r for r in tr if r["site"] == site]
            full = transfer_stats([r for r in rows if r["family"] == "full"], base)
            fam: dict = {"full": full}
            groups: dict[tuple, list] = defaultdict(list)
            for r in rows:
                if r["family"] != "full":
                    groups[(r["family"], r["k"], r["complement"])].append(r)
            for (family, k, comp), rs in groups.items():
                key = f"{family}|{k}|{'comp' if comp else 'proj'}"
                fam[key] = transfer_stats(rs, base)
            curves: dict = {}
            for family in ("T", "R", "A", "U_k"):
                ks = sorted({r["k"] for r in rows if r["family"] == family})
                curve = {}
                for k in ks:
                    st = fam.get(f"{family}|{k}|proj")
                    curve[k] = st["rate"] if st else None
                if curve:
                    curves[family] = {
                        "curve": curve,
                        "d50": effective_dimension(curve, full["rate"], 0.5),
                        "d90": effective_dimension(curve, full["rate"], 0.9),
                    }
            out["transplant"][site] = {"families": fam, "curves": curves}

    # Interpolation.
    it = load_rows(sub / "interp.jsonl")
    if it and base is None:
        base = it[0].get("base")
    if it and base is not None:
        out["interp"] = {}
        for site in sorted({r["site"] for r in it}, key=layer_sort):
            # Incidental donors (same leading digit as the base) cannot show a
            # switch under the first-token metric and are excluded here.
            rows = [r for r in it if r["site"] == site and str(r["donor"])[0] != str(base)[0]]
            by_alpha: dict = {}
            for alpha in sorted({r["alpha"] for r in rows}):
                rs = [r for r in rows if r["alpha"] == alpha]
                by_alpha[str(alpha)] = {
                    "donor": sum(1 for r in rs if first_token_match(r["decoded"], r["donor"])),
                    "base": sum(1 for r in rs if r.get("parsed") == base),
                    "intermediate": None if digit_split else sum(
                        1 for r in rs if r.get("parsed") is not None
                        and min(base, r["donor"]) < r["parsed"] < max(base, r["donor"])
                    ),
                    "other": sum(
                        1 for r in rs if r.get("parsed") is not None and r["parsed"] != base
                        and not first_token_match(r["decoded"], r["donor"])
                        and not (min(base, r["donor"]) < r["parsed"] < max(base, r["donor"]))
                    ),
                    "n": len(rs),
                }
            out["interp"][site] = by_alpha

    # Ablation survival.
    ab = load_rows(sub / "ablate.jsonl")
    if ab:
        out["ablate"] = {}
        for site in sorted({r["site"] for r in ab}, key=layer_sort):
            rows = [r for r in ab if r["site"] == site]
            groups = defaultdict(list)
            for r in rows:
                groups[(r["family"], r["k"])].append(r)
            fam = {}
            for (family, k), rs in groups.items():
                key = f"{family}|{k}"
                own_ok = [r for r in rs if r.get("own")]
                fam[key] = {
                    "n": len(own_ok),
                    "first_token_survive": sum(
                        1 for r in own_ok if r["decoded"].strip()[:1] == r["own"].strip()[:1]),
                    "exact_survive": sum(1 for r in own_ok if r["parsed"] == first_int(r["own"])),
                }
            out["ablate"][site] = fam

    # Periodic clamps.
    pe = load_rows(sub / "periodic.jsonl")
    if pe:
        out["periodic"] = {}
        for site in sorted({r["site"] for r in pe}, key=layer_sort):
            rows = [r for r in pe if r["site"] == site]
            per_site = {}
            for period in sorted({r["period"] for r in rows}):
                for family in sorted({r["family"] for r in rows if r["period"] == period}):
                    rs = [r for r in rows if r["period"] == period and r["family"] == family
                          and first_int(r["own"]) is not None]
                    parsed = [r for r in rs if r.get("parsed") is not None]
                    agree = sum(1 for r in parsed if (r["parsed"] - first_int(r["own"])) % period == r["d"])
                    # An unparseable output after the edit is a destroyed answer: moved, not dropped.
                    moved = sum(1 for r in rs if r.get("parsed") != first_int(r["own"]))
                    matrix: dict = defaultdict(lambda: defaultdict(int))
                    for r in parsed:
                        matrix[r["d"]][int((r["parsed"] - first_int(r["own"])) % period)] += 1
                    readable_periods = readouts["sites"].get(site, {}).get("periodic", {}).get("readable_periods", [])
                    ratios = [r["edit_norm_ratio"] for r in rs if r.get("edit_norm_ratio") is not None]
                    per_site[f"P{period}|{family}"] = {
                        "n": len(rs), "moved": moved, "unparseable": len(rs) - len(parsed), "agree": agree,
                        "pair_readable": (period in readable_periods) if family == "F" else None,
                        "mean_edit_norm_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
                        "matrix": {str(d): dict(v) for d, v in matrix.items()},
                    }
            out["periodic"][site] = per_site
    return out


def first_int(text: str | None) -> int | None:
    from ...metrics import parse_integer

    return parse_integer(text) if text else None


def collect_model(model_dir: Path) -> dict:
    out: dict = {"model": model_dir.name, "scans": {}}
    for scan_dir in sorted(model_dir.glob("scan-*")):
        summary = load_json(scan_dir / "summary.json")
        if summary:
            t = summary["transition"]
            out["scans"][scan_dir.name.removeprefix("scan-")] = {
                "scc": t["scc"], "ci": [t["ci_low"], t["ci_high"]],
                "p_stable": t["p_stable"], "n_probes": summary["n_probes"],
                "sufficiency": summary["sufficiency"],
            }
    geometry = load_json(model_dir / "geometry.json")
    if geometry:
        out["margin_collapse"] = geometry["margin_collapse"]
        out["geometry"] = {
            site: {
                "cv_r2": m["cv_r2"], "cv_rmse_items": m["cv_rmse_items"],
                "rho": m["rho"],
                "beyond_slopes": {k: v["local_slope"] for k, v in m["beyond_regime"].items()},
            }
            for site, m in geometry["sites"].items()
            if site.endswith("decode_point")
        }
    interventions = summarize_interventions(model_dir)
    if interventions:
        out["interventions"] = interventions
    mixed = load_json(model_dir / "mixed" / "mixed.json")
    if mixed:
        out["mixed"] = {
            "behavioral_accuracy": mixed["behavioral_accuracy"],
            "grid_size": mixed["grid_size"],
            "sites": mixed["sites"],
        }
    subspace = summarize_subspace(model_dir)
    if subspace:
        out["subspace"] = subspace
    return out


def subspace_markdown(models: list[dict]) -> list[str]:
    lines = ["", "## Subspace controls: readouts (multi-prompt cache, count-held-out folds)", "",
             "| model | site | linear R2 | last digit raw acc (neighbour) | last digit resid acc (perm max) | P10 R2 resid cos/sin | periods readable | T rank | max cos(G,U) | max cos(L,U) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        sub = m.get("subspace")
        if not sub:
            continue
        for site in sorted(sub["sites"], key=layer_sort):
            s = sub["sites"][site]
            ld, lr = s["last_digit"], s.get("last_digit_resid", {})
            p10 = s["periodic"].get("10", {})
            dec = s.get("decoder") or {}
            lines.append(
                f"| {m['model']} | {site} | {s['linear_cv_r2']:.3f} "
                f"| {ld.get('accuracy')} ({ld.get('neighbour_confusion')}) "
                f"| {lr.get('accuracy')} ({lr.get('perm_max')}) "
                f"| {p10.get('r2_cos_resid', '--')}/{p10.get('r2_sin_resid', '--')} "
                f"| {s['readable_periods']} | {s['trajectory_rank']} "
                f"| {dec.get('max_cos_G_U', '--')} | {dec.get('max_cos_L_U', '--')} |"
            )

    lines += ["", "## Subspace controls: consolidation layer (full transplant at every layer)", "",
              "| model | layers | first layer >= 50% | first layer >= 80% |", "|---|---|---|---|"]
    for m in models:
        loc = (m.get("subspace") or {}).get("localize")
        if loc:
            lines.append(f"| {m['model']} | {loc['n_layers']} | {loc['half_transfer_layer']} | {loc['consolidation_layer']} |")

    lines += ["", "## Subspace controls: transplant (first-token transfer, incidental donors excluded)", "",
              "| model | site | full | L | F | G | U | T d50/d90 | R d50/d90 | A d50/d90 | U_k d50/d90 | T curve |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        tr = (m.get("subspace") or {}).get("transplant")
        if not tr:
            continue
        for site in sorted(tr, key=layer_sort):
            fam, curves = tr[site]["families"], tr[site]["curves"]

            def rate(key: str) -> str:
                st = next((v for k, v in fam.items() if k.startswith(key + "|") and k.endswith("|proj")), None)
                return f"{st['rate']}" if st and st["rate"] is not None else "--"

            def dd(family: str) -> str:
                c = curves.get(family)
                return f"{c['d50']}/{c['d90']}" if c else "--"

            t_curve = " ".join(f"{k}:{v}" for k, v in curves.get("T", {}).get("curve", {}).items())
            lines.append(
                f"| {m['model']} | {site} | {fam['full']['rate']} | {rate('L')} | {rate('F')} | {rate('G')} "
                f"| {rate('U')} | {dd('T')} | {dd('R')} | {dd('A')} | {dd('U_k')} | {t_curve} |"
            )

    lines += ["", "## Subspace controls: ablation to the mean (first-token survival of the model's own answer)", "",
              "| model | site | L | F | G | U | T8 | R8 | T16 | R16 | A16 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        ab = (m.get("subspace") or {}).get("ablate")
        if not ab:
            continue
        for site in sorted(ab, key=layer_sort):
            fam = ab[site]

            def surv(prefix: str, k: int | None = None) -> str:
                keys = [key for key in fam if key.split("|")[0] == prefix and (k is None or key.endswith(f"|{k}"))]
                if not keys:
                    return "--"
                n = sum(fam[key]["n"] for key in keys)
                s = sum(fam[key]["first_token_survive"] for key in keys)
                return f"{s}/{n}"

            lines.append(
                f"| {m['model']} | {site} | {surv('L')} | {surv('F')} | {surv('G')} | {surv('U')} "
                f"| {surv('T', 8)} | {surv('R', 8)} | {surv('T', 16)} | {surv('R', 16)} | {surv('A', 16)} |"
            )

    lines += ["", "## Subspace controls: periodic clamps (answer moved / shift agrees with phase)", "",
              "| model | site | P10 fitted pair | P10 random pair | P5 | P2 |", "|---|---|---|---|---|---|"]
    for m in models:
        pe = (m.get("subspace") or {}).get("periodic")
        if not pe:
            continue
        for site in sorted(pe, key=layer_sort):
            p = pe[site]

            def cell(key: str) -> str:
                v = p.get(key)
                if not v:
                    return "--"
                broken = f" ({v['unparseable']} unparseable)" if v.get("unparseable") else ""
                flag = " [pair not readable]" if v.get("pair_readable") is False else ""
                return f"{v['moved']}/{v['agree']} of {v['n']}{broken}{flag}"

            lines.append(f"| {m['model']} | {site} | {cell('P10|F')} | {cell('P10|R2')} | {cell('P5|F')} | {cell('P2|F')} |")

    lines += ["", "## Subspace controls: interpolation along the donor chord (informative donors only)", "",
              "| model | site | n | first alpha with >= 50% donor | first alpha with >= 90% donor | intermediate answers (all alphas) | other answers |",
              "|---|---|---|---|---|---|---|"]
    for m in models:
        it = (m.get("subspace") or {}).get("interp")
        if not it:
            continue
        for site in sorted(it, key=layer_sort):
            table = it[site]
            n = next(iter(table.values()))["n"] if table else 0

            def first_alpha(level: float) -> str:
                for alpha, v in table.items():
                    if v["n"] and v["donor"] / v["n"] >= level:
                        return alpha
                return "--"

            inter = "n/a (digit tokenizer)" if any(v["intermediate"] is None for v in table.values()) else sum(v["intermediate"] for v in table.values())
            other = sum(v["other"] for v in table.values())
            lines.append(f"| {m['model']} | {site} | {n} | {first_alpha(0.5)} | {first_alpha(0.9)} | {inter} | {other} |")
    return lines


def layer_sort(site: str) -> int:
    try:
        return int(site.rsplit("layer", 1)[1].split("|")[0])
    except (IndexError, ValueError):
        return 0


def to_markdown(models: list[dict]) -> str:
    lines = ["# Probe-suite cross-model tables", ""]

    lines += ["## Behavioral scans (local greedy)", "",
              "| model | mode | SCC | 95% CI | p_stable | probes |",
              "|---|---|---|---|---|---|"]
    for m in models:
        for mode, s in m["scans"].items():
            lines.append(
                f"| {m['model']} | {mode} | {s['scc']} | [{s['ci'][0]}, {s['ci'][1]}] "
                f"| {s['p_stable']:.2f} | {s['n_probes']} |"
            )

    lines += ["", "## Linear readability of the count (decode-point states, held-out ridge)", "",
              "| model | site | CV R2 | CV RMSE (items) | AR rho | slope 1.0-1.2x | 1.2-1.6x |",
              "|---|---|---|---|---|---|---|"]
    for m in models:
        for site in sorted(m.get("geometry", {}), key=layer_sort):
            g = m["geometry"][site]
            b = g["beyond_slopes"]
            lines.append(
                f"| {m['model']} | {site.split('|')[0]} | {g['cv_r2']:.3f} | {g['cv_rmse_items']:.2f} "
                f"| {g['rho']:.3f} | {b.get('1.0-1.2x', '--')} | {b.get('1.2-1.6x', '--')} |"
            )

    lines += ["", "## Margin collapse vs behavior", "",
              "| model | first negative margin | first greedy divergence | behavioral SCC |",
              "|---|---|---|---|"]
    for m in models:
        mc = m.get("margin_collapse")
        if mc:
            lines.append(
                f"| {m['model']} | {mc['first_negative_min_gap']} "
                f"| {mc['first_greedy_divergence']} | {mc['behavioral_scc']} |"
            )

    lines += ["", "## Interventions at the decode point", "",
              "| model | site | clamp moved/exact | step moved | full: parsed / 1st-tok "
              "| linear-only: parsed / 1st-tok | residual-only: parsed / 1st-tok | random moved |",
              "|---|---|---|---|---|---|---|---|"]
    for m in models:
        iv = m.get("interventions")
        if not iv:
            continue
        for site in sorted(iv["sites"], key=layer_sort):
            e = iv["sites"][site]

            def frac(d, *keys):
                if not d:
                    return "--"
                return "/".join(str(d[k]) for k in keys) + f" of {d['n']}"

            lines.append(
                f"| {m['model']} | {site} | {frac(e.get('clamp'), 'moved', 'exact')} "
                f"| {frac(e.get('step_translate'), 'moved')} "
                f"| {frac(e.get('chimera_full'), 'parsed_hit', 'first_token_hit')} "
                f"| {frac(e.get('chimera_linear_only'), 'parsed_hit', 'first_token_hit')} "
                f"| {frac(e.get('chimera_residual_only'), 'parsed_hit', 'first_token_hit')} "
                f"| {frac(e.get('random_control'), 'moved')} |"
            )

    lines += ["", "## Mixed-sequence deconfound (count k among distractors, length L)", "",
              "| model | behavioral acc | site | CV R2 count | CV R2 length | cos(d_count, d_len) |",
              "|---|---|---|---|---|---|"]
    for m in models:
        mx = m.get("mixed")
        if not mx:
            continue
        for site in sorted(mx["sites"], key=layer_sort):
            s = mx["sites"][site]
            lines.append(
                f"| {m['model']} | {mx['behavioral_accuracy']:.2%} | {site} "
                f"| {s['cv_r2_count']:.3f} | {s['cv_r2_length']:.3f} "
                f"| {s['cosine_count_length']:.3f} |"
            )
    lines += subspace_markdown(models)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite-dir", type=Path, default=Path("results/suite"))
    args = parser.parse_args()

    models = [
        collect_model(d) for d in sorted(args.suite_dir.iterdir())
        if d.is_dir() and any(d.glob("scan-*"))
    ]
    (args.suite_dir / "tables.json").write_text(json.dumps(models, indent=1))
    (args.suite_dir / "tables.md").write_text(to_markdown(models))
    print(f"aggregated {len(models)} models -> {args.suite_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
