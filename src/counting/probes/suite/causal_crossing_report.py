"""Aggregate Stage 1 without selecting on held-out outcomes.

    python -m counting.probes.suite.causal_crossing_report
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .causal_crossing import SCC, Route, cases

ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "results/suite"


def bootstrap(values, counts):
    """95% percentile interval, resampling counts, retaining their positions."""
    values, counts = np.asarray(values, dtype=float), np.asarray(counts)
    unique = np.unique(counts)
    sums = np.array([values[counts == c].sum() for c in unique])
    sizes = np.array([(counts == c).sum() for c in unique])
    indices = np.random.default_rng(7).integers(len(unique), size=(10000, len(unique)))
    samples = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return dict(rate=float(values.mean()), successes=int(values.sum()), total=len(values),
                count_clusters=len(unique), ci95=np.quantile(samples, [.025, .975]).tolist())


def summarize(model):
    directory = SUITE / model / "causal-crossing"
    complete = json.loads((directory / "complete.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    frozen = json.loads((directory / "frozen.json").read_text())
    rows = [json.loads(l) for l in (directory / "rows.jsonl").read_text().splitlines()]
    keys = {(r["phase"], r["case"], r["route"], r["control"]) for r in rows}
    assert len(keys) == len(rows) == complete["rows"], "duplicate/incomplete records"
    baseline = {r["case"]: r for r in rows if r["phase"] == "heldout_baseline"}
    expected = {c["id"] for c in cases(model) if c["split"] == "heldout"}
    assert set(baseline) == expected
    b = list(baseline.values())
    result = dict(model=model, rows=len(rows), elapsed_seconds=complete["elapsed_seconds"],
                  calibration_counts=sorted({c["n"] for c in cases(model) if c["split"] == "calibration"}),
                  heldout_counts=sorted({c["n"] for c in cases(model) if c["split"] == "heldout"}),
                  baseline={}, calibration=[], interventions=[])
    for name, field, expected_answer in [("target_count", "target_count", lambda r: r["n"]),
                                         ("neutral_count", "donor_count", lambda r: r["n"] - 1),
                                         ("target_fact", "target_fact", lambda r: 1),
                                         ("neutral_fact", "donor_fact", lambda r: 2)]:
        result["baseline"][name] = bootstrap([r[field]["parsed"] == expected_answer(r) for r in b],
                                              [r["n"] for r in b])
    result["baseline"]["n_minus_one"] = bootstrap([r["target_count"]["parsed"] == r["n"] - 1 for r in b],
                                                    [r["n"] for r in b])
    for candidate in manifest["routes"]:
        rr = [r for r in rows if r["phase"] == "screen" and r["route"] == candidate["name"]]
        assert len(rr) == 18, "incomplete calibration candidate"
        result["calibration"].append(dict(**candidate,
            n_minus_one=bootstrap([r["count"]["parsed"] == r["n"] - 1 for r in rr], [r["n"] for r in rr])))
    for route_dict in frozen["selected"]:
        route = Route(route_dict["family"], tuple(route_dict["layers"]))
        for control in ("edit", "sham", "random_source", "random_layer"):
            rr = [r for r in rows if r["phase"] == "heldout" and r["route"] == route.name and r["control"] == control]
            assert {r["case"] for r in rr} == expected
            ns = [r["n"] for r in rr]
            omission = [r["count"]["parsed"] == r["n"] - 1 for r in rr]
            fact = [r["fact"]["parsed"] == 1 for r in rr]
            base_correct = [baseline[r["case"]]["target_count"]["parsed"] == r["n"] for r in rr]
            record = dict(family=route.family, layers=list(route.layers), control=control,
                          n_minus_one=bootstrap(omission, ns), factual=bootstrap(fact, ns),
                          donor_answer_match=bootstrap([r["count"]["parsed"] is not None and
                              r["count"]["parsed"] == baseline[r["case"]]["donor_count"]["parsed"] for r in rr], ns),
                          joint=bootstrap(np.array(omission) & fact, ns),
                          unchanged_count=bootstrap([r["count"]["parsed"] == baseline[r["case"]]["target_count"]["parsed"] for r in rr], ns),
                          factual_losses=sum(baseline[r["case"]]["target_fact"]["parsed"] == 1 and not f for r, f in zip(rr, fact)),
                          factual_gains=sum(baseline[r["case"]]["target_fact"]["parsed"] != 1 and f for r, f in zip(rr, fact)),
                          unparsed_count=sum(r["count"]["parsed"] is None for r in rr),
                          error_histogram=dict(Counter(str(r["count"]["parsed"] - r["n"]) if r["count"]["parsed"] is not None else "unparsed" for r in rr)),
                          degenerate_layer_controls=sum(r.get("layer_control_degenerate", False) for r in rr))
            record["unpatched_attention_to_target_source"] = {
                str(layer): float(np.mean([np.mean(next(s["attention_heads"] for s in r["ordinary_route"]
                                                         if s["layer"] == layer)) for r in b]))
                for layer in route.layers}
            eligible = [r for r, ok in zip(rr, base_correct) if ok]
            record["baseline_correct_subset"] = (bootstrap([r["count"]["parsed"] == r["n"] - 1 for r in eligible],
                                                           [r["n"] for r in eligible]) if eligible else None)
            record["max_source_residual_change"] = (max(r["max_source_residual_change"] for r in rr)
                                                     if route.family == "message" else None)
            record["max_protected_prefix_roundoff_relative_l2"] = (max(
                d["relative_l2"] for r in rr for d in r["protected_prefix_numerical_drift"].values())
                if route.family == "message" else None)
            record["passes_gate"] = bool(control == "edit" and record["n_minus_one"]["rate"] >= .8
                and record["factual"]["rate"] >= result["baseline"]["target_fact"]["rate"]
                and record["joint"]["rate"] >= .8
                and result["baseline"]["target_fact"]["rate"] >= .8
                and result["baseline"]["neutral_fact"]["rate"] >= .8)
            if control == "sham":
                record["sham_count_disagreements"] = sum(r["count"]["decoded"] != baseline[r["case"]]["target_count"]["decoded"] for r in rr)
                record["sham_fact_disagreements"] = sum(r["fact"]["decoded"] != baseline[r["case"]]["target_fact"]["decoded"] for r in rr)
                assert record["sham_count_disagreements"] == record["sham_fact_disagreements"] == 0, "sham regression failed"
            result["interventions"].append(record)
    result["passes_gate"] = any(r["passes_gate"] for r in result["interventions"])
    if (directory / "ordinary_readout.json").exists():
        readouts = json.loads((directory / "ordinary_readout.json").read_text())
        weights = json.loads((directory / "historical_readout_weights.json").read_text())
        result["ordinary_readout"] = []
        for site in weights["sites"]:
            rr = [r for r in readouts if r["site"] == site and r["split"] == "heldout"]
            assert len(rr) == 24
            layer = int(site.rsplit("layer", 1)[1])
            attention = [np.mean(next(s["attention_heads"] for s in r["ordinary_route"] if s["layer"] == layer)) for r in b]
            result["ordinary_readout"].append(dict(site=site,
                mean_attention_to_source=float(np.mean(attention)),
                median_target_minus_ordinal=float(np.median([r["target_readout"] - r["j"] for r in rr])),
                median_neutral_minus_target=float(np.median([r["neutral_minus_target"] for r in rr])),
                historical_loo_rmse=weights["sites"][site]["historical_loo_rmse"]))
        decoys = [r for r in json.loads((directory / "factual_decoy.json").read_text()) if r["split"] == "heldout"]
        assert len(decoys) == 24
        result["baseline"]["neutral_other_fact"] = bootstrap([r["parsed"] == r["expected"] for r in decoys], [r["n"] for r in decoys])
    return result


def compact_rate(metric):
    percent = f"{100 * metric['rate']:.1f}".removesuffix(".0")
    return f"{metric['successes']}/{metric['total']} ({percent}%)"


def route_label(layers):
    if len(layers) > 1 and layers == list(range(layers[0], layers[-1] + 1)):
        return f"{layers[0]}–{layers[-1]}"
    return ", ".join(map(str, layers))


def write_report(tables):
    lines = ["# Causal crossing: natural event-message replacements", "",
             "2026-09-15 · `mechanistic-suite` · Local Torch/MPS bf16, greedy decoding.", "",
             "## Result and decision", ""]
    if not any(t["passes_gate"] for t in tables.values()):
        lines += ["**Stage 1 did not identify a selective missed-update intervention meeting the specified gate in either model.** "
                  "The frozen interventions were tested on 24 held-out count/position pairs per model. "
                  "The required result was the exact integer N−1 on at least 80% of cases, with factual access preserved. "
                  "Following the supplied plan's kill criterion, no non-counting transfer or repair experiment was run.", "",
                  "This is a bounded negative result for the tested residual and all-head source-message replacements. "
                  "It does not show that a shared event-integration mechanism is absent. "
                  "The counting-to-execution mechanistic claim remains unestablished; the existing behavioral associations are unchanged.", ""]
    else:
        lines += ["**A Stage 1 gate passed. Transfer and repair require separate evaluation before any shared-mechanism conclusion.**", ""]
    for model, table in tables.items():
        best = max(table["calibration"], key=lambda r: r["n_minus_one"]["rate"])
        parts = [f"{r['family']} at {'layer' if len(r['layers']) == 1 else 'layers'} {route_label(r['layers'])}: {compact_rate(r['n_minus_one'])} exact N−1; "
                 f"factual {compact_rate(r['factual'])}" for r in table["interventions"] if r["control"] == "edit"]
        lines += [f"- **{model}:** best calibration count score {compact_rate(best['n_minus_one'])}; held-out " + "; ".join(parts) + "."]
    if not any(t["passes_gate"] for t in tables.values()):
        lines += ["", "**What this adds.** Residual replacements changed counts while often damaging factual access. "
                  "The gpt-oss message patch spared the factual check but rarely produced the requested one-item omission. "
                  "Gemma's stronger message effect came with substantial factual losses even though the item's own "
                  "residual was preserved exactly. Preserving that local state is therefore insufficient to guarantee "
                  "accessibility under these message edits. The experiment does not identify a reliable, discrete "
                  "event-update operation to carry into another task."]
    lines += ["", "[Stage 1 calibration and held-out controls (PDF)](../paper/figures/causal-crossing-stage1.pdf)", "",
              "**Figure.** Left: calibration scores for residual suffix replacements and individual-layer message replacements. "
              "The additional multi-layer message candidates are included in the machine-readable tables. "
              "Center/right: the two frozen interventions and their controls on held-out cases. "
              "Dashed line: 80% count criterion; dotted line: unedited baseline. "
              "Intervals resample the eight held-out counts, retaining their three positions (10,000 draws, seed 7). "
              "Bootstrap intervals can be degenerate at 0%/100%; these are not population bounds or equivalence tests.", "",
              "## Design and selection", "",
              "The [supplied plan](plans/causal-crossing.md) and [execution protocol](plans/causal-crossing-stage1-protocol.md) "
              "record the design and the numerical-precision amendment made during smoke validation, before calibration. "
              "No counting scan was rerun. Counts were sampled within the existing SCC envelopes (36 and 75), "
              "not filtered by the new model responses. Six counts × three positions form calibration; "
              "eight disjoint counts × three different relative positions form the held-out set. "
              "The SCC envelope is probabilistic: unedited exact accuracy must be measured, not assumed.", "",
              "| Model | Calibration counts | Held-out counts | Candidates |", "|---|---|---|---:|"]
    for model, table in tables.items():
        lines.append(f"| {model} | {', '.join(map(str, table['calibration_counts']))} | {', '.join(map(str, table['heldout_counts']))} | {len(table['calibration'])} |")
    lines += ["", "Each donor changes exactly one item token from `a` to `x`; length, delimiters and positions are identical. "
              "Recipient input IDs never change. The history ends at the final item. Its cache is copied before branching "
              "to the original count question's chat suffix or to a factual continuation asking for 1 (`a`) or 2 (`x`) at item j. "
              "Neither question suffix nor any generated token is patched. Layer indices are zero-based. "
              "This allows downstream reconstruction, as the design requires. "
              "Replies are scored as complete integers, not by leading digit. Decoding is greedy with an eight-token cap; "
              "the gpt-oss final-channel prefill and template date are explicit in the manifest.", "",
              "- **Residual:** donor post-layer state at j, at every layer from L onward; every start layer is screened.",
              "- **Message:** replace only the weighted attention contribution a(q,j)v(j), before output projection, "
              "for later history positions q>j, across all heads. Screen every individual layer, suffixes starting "
              "at layer 0 and the four suite sites, and three-layer windows at those sites. Other source contributions "
              "are not renormalized. These are natural weighted-message replacements, not key/value substitutions.",
              "- **Selection:** screen count accuracy first, then factual responses for the top three candidates per family. "
              "Prefer candidates meeting the calibration gate, then joint N−1/factual success, then count success, "
              "then fewer/earlier layers. Freeze one candidate per family before any held-out response is computed. "
              "Keep the best candidate for a negative audit even when calibration fails.", "",
              "## Unedited baselines and donor adequacy", "",
              "| Model | Target count = N | Neutral count = N−1 | Target fact at j | Neutral fact at j | Neutral fact elsewhere |",
              "|---|---:|---:|---:|---:|---:|"]
    for model, table in tables.items():
        b = table["baseline"]
        lines.append(f"| {model} | " + " | ".join(compact_rate(b[k]) for k in ("target_count", "neutral_count", "target_fact", "neutral_fact", "neutral_other_fact")) + " |")
    lines += ["", "The neutral donor's integer answer is not assumed to be N−1. "
              "Poor donor counting limits what a failed transplant can establish: replacing one symbol may disturb "
              "more than one part of the model's counting computation. Donor-answer copying and the desired omitted-item "
              "counterfactual are scored separately below. "
              "The extra factual query asks about another `a` position while `x` remains at j; it checks whether merely "
              "detecting an `x` could explain factual performance. It was added as a descriptive validation check and "
              "does not select the intervention. These simple factual questions do not establish general event recall.", "",
              "## Frozen interventions and controls", "",
              "| Model | Family / layers | Condition | Exact N−1 | Factual correct | Joint | Factual losses / gains |",
              "|---|---|---|---:|---:|---:|---:|"]
    for model, table in tables.items():
        for r in table["interventions"]:
            control = r["control"].replace("_", " ") + (" (identical layers)" if r["degenerate_layer_controls"] else "")
            lines.append(f"| {model} | {r['family']} / {route_label(r['layers'])} | {control} | "
                         f"{compact_rate(r['n_minus_one'])} | {compact_rate(r['factual'])} | {compact_rate(r['joint'])} | "
                         f"{r['factual_losses']} / {r['factual_gains']} |")
    lines += ["", "Random-source controls use another item position in the same natural donor. "
              "Later donor positions can already carry consequences of j, so this control is not an assumption of zero signal. "
              "Random-layer controls use the same number of layers, selected once with seed 7. A full-depth route necessarily "
              "has a degenerate random-layer control; this supplies no layer-specificity evidence. "
              "Shams use recipient messages/states while retaining the same donor computation and branching procedure. "
              "All held-out sham count and factual replies must exactly reproduce their unedited replies.", "",
              "### Conditional count scores and error structure", "",
              "| Model | Family | Donor-answer match | Exact N−1 on baseline-correct cases | Errors relative to true N (error: cases) |",
              "|---|---|---:|---:|---|"]
    for model, table in tables.items():
        for r in table["interventions"]:
            if r["control"] == "edit":
                conditional = compact_rate(r["baseline_correct_subset"]) if r["baseline_correct_subset"] else "No eligible cases"
                hist = ", ".join(f"{k}: {v}" for k, v in sorted(r["error_histogram"].items()))
                lines.append(f"| {model} | {r['family']} | {compact_rate(r['donor_answer_match'])} | {conditional} | {hist} |")
    lines += ["", "## Ordinary route and running-count readout checks", "",
              "| Model | Site | Mean attention to j | Median readout minus j, target | Median neutral−target readout | Historical LOO RMSE |",
              "|---|---|---:|---:|---:|---:|"]
    for model, table in tables.items():
        for r in table["ordinary_readout"]:
            lines.append(f"| {model} | {r['site']} | {r['mean_attention_to_source']:.4f} | "
                         f"{r['median_target_minus_ordinal']:.2f} | {r['median_neutral_minus_target']:.2f} | {r['historical_loo_rmse']:.2f} |")
    lines += ["", "Attention is averaged over heads and later history queries in ordinary unpatched runs. "
              "Readouts use the existing E4 stable item-position cache and its previously selected ridge penalties; "
              "the small fitted weights are released. This is a descriptive projection check, not new held-out probe "
              "validation: the historical training counts may overlap the present test counts, and the all-`a` training "
              "trajectory confounds running count with ordinal position. Neither a nonzero attention weight nor a "
              "probe shift proves that this route carries a discrete unit increment. Neutral replacements do not "
              "generally shift this historical readout by −1; Gemma's early-site shifts are positive and large. "
              "These mixed-symbol projections require separate calibration before being interpreted as running counts.", "",
              "At the frozen message routes specifically:", ""]
    for model, table in tables.items():
        r = next(r for r in table["interventions"] if r["family"] == "message" and r["control"] == "edit")
        mean_attention = np.mean(list(r["unpatched_attention_to_target_source"].values()))
        lines.append(f"- {model}, {'layer' if len(r['layers']) == 1 else 'layers'} {route_label(r['layers'])}: mean ordinary attention to j = {mean_attention:.4f}.")
    lines += ["",
              "## Implementation validation and scope", "",
              "The source-message sum was checked against an independent explicit sum over all source contributions. "
              "Same-input bf16 replacement is exact, and q≤j is untouched by that attention edit. "
              "On the native gpt-oss MoE implementation, changing later tokens nevertheless changed subsequent "
              "earlier-token rounding: the initial unguarded smoke showed up to 1.4% relative source-state drift. "
              "The frozen implementation therefore restores the recipient's own unpatched post-layer states at q≤j, "
              "enforcing the exact-arithmetic causal-prefix invariant. It inserts no donor state at these positions. "
              "Pre-restoration numerical drift is retained per layer, and the protected source states must be "
              "bit-identical after restoration. This guard is part of the intervention definition.", "",
              "Both models passed cache-branch versus fresh-pass answer checks on a calibration smoke case; "
              "Gemma's eager and default SDPA answers also agreed there (gpt-oss's native backend is already eager). "
              "All 42 target/donor token pairs per model were checked for exactly one changed token and donor "
              "tokenizer round-trip equality. These checks do not establish equality across all possible kernels or prompts.", "",
              "The search is finite: all-head routes can miss selective head subsets, multi-source interactions, "
              "different neutral symbols or context-dependent mechanisms. Only eight distinct counts per model are "
              "held out. A threshold failure is not proof of no causal participation, and a factual rate match is not "
              "a powered equivalence test. A null count response is also compatible with downstream reconstruction "
              "from the unchanged history after the patch. No result here establishes the bottleneck of a natural beyond-capacity failure.", "",
              "## Paper implications", ""]
    if not any(t["passes_gate"] for t in tables.values()):
        lines += ["> Natural residual and source-message replacements at item positions did not yield a reliable "
                  "selective one-event omission with preserved factual access under the tested protocol. "
                  "Consequently, the present results do not establish that counting and non-counting execution "
                  "share the same limiting computation.", "",
                  "Keep the existing counting intervention results within their tested conditions. "
                  "Do not add a shared-mechanism or bottleneck-rescue claim to the abstract. "
                  "Do not classify this as failed non-counting transfer: transfer was not run because Stage 1 failed. "
                  "The non-commuting task generator is prepared and unit-tested, but has no model results.", ""]
        lines += ["For a future attempt, establish a reliable *natural* one-item counterfactual on counting calibration "
                  "before circuit search: select a fixed neutral rendering, verify its unpatched N−1 answers and "
                  "positional factual access, and validate running-count readouts on mixed target/neutral trajectories. "
                  "Then test more selective head routes with fresh held-out counts. Do not choose donors by their test answers.", ""]
    lines += ["## Reproduction and artifacts", "",
              "Run from the repository root with local checkpoints and the recorded Torch/Transformers versions:", "",
              "```sh", ".venv/bin/python -m unittest discover -s tests -p test_causal_crossing.py -v"]
    for model in tables:
        lines += [f".venv/bin/python -u -m counting.probes.suite.causal_crossing --model {model}",
                  f".venv/bin/python -u -m counting.probes.suite.causal_crossing_readouts --model {model}"]
    lines += [".venv/bin/python -m counting.probes.suite.causal_crossing_report", "```", "",
              "The intervention runner resumes saved rows. To repeat model inference, use a fresh `--out` directory; "
              "retain the released output directory when regenerating the report from the recorded results.", "",
              "- `results/suite/<model>/causal-crossing/`: manifest with exact inputs/IDs, seed, versions and source hash; "
              "raw `rows.jsonl`; `frozen.json`; completion record; token validation; historical readout weights and "
              "ordinary readouts; factual decoy responses.",
              "- `results/suite/<model>/causal-crossing-smoke/`: smoke replies and fresh/default-backend comparisons. "
              "The gpt-oss directory also retains the unguarded numerical-drift comparison.",
              "- [Machine-readable tables](../results/suite/causal_crossing_tables.json) and "
              "[PDF plate](../paper/figures/causal-crossing-stage1.pdf).", "",
              f"Primary runs produced {sum(t['rows'] for t in tables.values()):,} records. "
              "Baseline records contain four replies; intervention records contain one or two. "
              "Run times and selected candidates are recorded per model; the models overlapped in execution, "
              "so their elapsed times must not be summed as wall time. No paid model API was used.", ""]
    (ROOT / "docs/causal-crossing-report.md").write_text("\n".join(lines))


def make_plate(tables, output_stem=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from counting.analysis.figures.suite_round2 import letter
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 10})
    fig, axes = plt.subplots(len(tables), 3, figsize=(15, 4 * len(tables)), squeeze=False)
    colors = {"residual": "#b56338", "message": "#267e9d"}
    for i, (model, table) in enumerate(tables.items()):
        ax = axes[i, 0]
        for family in ("residual", "message"):
            rows = [r for r in table["calibration"] if r["family"] == family
                    and (family == "residual" or len(r["layers"]) == 1)]
            ax.plot([r["layers"][0] for r in rows], [r["n_minus_one"]["rate"] for r in rows],
                    marker=".", color=colors[family], label="Residual suffix" if family == "residual" else "Message, one layer")
        ax.axhline(.8, color="0.5", ls="--", lw=1)
        ax.set(xlabel="Layer (zero-based)", ylabel="Exact N−1 rate", title=f"{model}\nCalibration: 18 cases")
        ax.legend(fontsize=8, frameon=False)
        for j, metric, title in [(1, "n_minus_one", "Held-out exact N−1"), (2, "factual", "Held-out factual accuracy")]:
            ax = axes[i, j]
            controls = ["edit", "sham", "random_source", "random_layer"]
            for fi, family in enumerate(("residual", "message")):
                rr = [next(r for r in table["interventions"] if r["family"] == family and r["control"] == c) for c in controls]
                vals = np.array([r[metric]["rate"] for r in rr])
                ci = np.array([r[metric]["ci95"] for r in rr])
                ax.bar(np.arange(4) + (fi - .5) * .36, vals, width=.35,
                       color=colors[family], label=family.capitalize(),
                       yerr=np.maximum(0, np.array([vals - ci[:, 0], ci[:, 1] - vals])), capsize=2)
            base = table["baseline"]["n_minus_one" if metric == "n_minus_one" else "target_fact"]["rate"]
            ax.axhline(base, color="black", lw=1, ls=":", label="Unedited")
            if metric == "n_minus_one":
                ax.axhline(.8, color="0.5", ls="--", lw=1)
            selected = {r["family"]: route_label(r["layers"]) for r in table["interventions"] if r["control"] == "edit"}
            subtitle = (f"Residual {selected['residual']}; message {selected['message']}" if j == 1
                        else "Question about item j")
            ax.set(xticks=np.arange(4), xticklabels=["Patch", "Sham", "Random\nsource", "Random\nlayer"], title=title + "\n" + subtitle,
                   ylabel="Fraction of 24 cases")
            if j == 1:
                ax.legend(fontsize=8, frameon=False, loc="upper right")
            if any(r["degenerate_layer_controls"] for r in table["interventions"]):
                ax.set_xticklabels(["Patch", "Sham", "Random\nsource", "Random\nlayer*"])
                ax.set_xlabel("* Full-depth controls equal the patch", fontsize=8)
        for j, ax in enumerate(axes[i]):
            ax.set_ylim(-.02, 1.1)
            ax.spines[["top", "right"]].set_visible(False)
            letter(ax, chr(65 + i * 3 + j))
    fig.tight_layout(pad=2)
    path = Path(output_stem) if output_stem else ROOT / "paper/figures/causal-crossing-stage1"
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    tables = {model: summarize(model) for model in SCC}
    assert all("ordinary_readout" in t for t in tables.values()), "run the descriptive readout/factual audit first"
    path = SUITE / "causal_crossing_tables.json"
    path.write_text(json.dumps(tables, indent=2, allow_nan=False) + "\n")
    make_plate(tables)
    write_report(tables)
    print(json.dumps({m: dict(passes_gate=t["passes_gate"], baseline=t["baseline"],
                       interventions=t["interventions"]) for m, t in tables.items()}, indent=2))


if __name__ == "__main__":
    main()
