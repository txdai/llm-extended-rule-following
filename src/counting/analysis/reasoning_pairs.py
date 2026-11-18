#!/usr/bin/env python3
"""Reasoning on against reasoning off, for every hybrid model measured.

A hybrid model is one measurement with the reasoning knob on (#think) and one
with it off (#nothink); the dataset carries 91 such pairs. Both configurations
are the same weights, so the pair isolates what test-time reasoning buys --
unlike a comparison across different models, which confounds the knob with
everything else that changed between them.

Not every pair is a real contrast. Some providers ignore `effort: "none"` and
reason anyway, and a few report no reasoning tokens in either configuration.
Pairs are classified from the measured reasoning tokens, the figure draws only
the verified contrasts, and the appendix table lists every pair with its
classification so the exclusions are auditable.

Outputs:
    paper/figures/post-hoc-reasoning.pdf     delta-CC against token multiplier
    paper/figures/token-consumption.pdf      tokens per probe at the boundary
    paper/appendix/reasoning_pairs_table.tex the full pair table

    python -m counting.analysis.reasoning_pairs
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

from counting.analysis.figures.attractors import BASELINE_DARK, set_spine_width
from counting.analysis.figures.search_panels import PROVIDER_COLORS, infer_provider_family
from counting.paths import REPO_ROOT as ROOT

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

RUN = ROOT / "results" / "counting-data"
OUTPUT_DIR = ROOT / "paper" / "figures"
OUTPUT_PDF = OUTPUT_DIR / "post-hoc-reasoning.pdf"
TOKENS_PDF = OUTPUT_DIR / "token-consumption.pdf"
TABLE_TEX = ROOT / "paper" / "appendix" / "reasoning_pairs_table.tex"

# Kept importable for landscape_triptych; the new figure labels no points.
LABEL_OFFSETS: dict[str, tuple[int, int]] = {}
PLOT_LABELS: dict[str, str] = {}

TOKENS_PER_ITEM = 2.49


@dataclass(frozen=True)
class Pair:
    model: str
    provider: str
    cc_think: int
    cc_nothink: int
    tokens_think: float          # mean completion tokens per probe, shared stable range
    tokens_nothink: float
    rtok_think: float            # mean reasoning tokens per probe
    rtok_nothink: float
    classification: str          # verified | knob ignored | no reasoning | unreported
    rescanned: bool = False      # reasoning member measured at the 32k cap (results/budget-32k)

    @property
    def delta_cc(self) -> int:
        return self.cc_think - self.cc_nothink

    @property
    def token_multiplier(self) -> float:
        return self.tokens_think / self.tokens_nothink if self.tokens_nothink else float("nan")


def _probe_tokens() -> dict[str, list[tuple[int, float, float]]]:
    """Per run key: (length, completion tokens, reasoning tokens) per probe."""
    per_key: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with (RUN / "responses.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != "scan":
                continue
            out_tok = float(row["output_tokens"] or 0)
            r_raw = row.get("reasoning_tokens", "")
            r_tok = float(r_raw) if r_raw not in ("", None) else float("nan")
            per_key[row["run_key"]].append((int(row["input_count"]), out_tok, r_tok))
    return per_key


BUDGET_RUN = ROOT / "results" / "budget-32k"


def budget_rescans() -> dict[str, tuple[int, list[tuple[int, float, float]]]]:
    """Reasoning configurations rescanned at a 32,768-token cap.

    Under the catalogue's fixed 8,192-token cap sixteen verified pairs have a
    reasoning member whose scan ended on budget exhaustion rather than on wrong
    answers. Those configurations were rescanned with the cap raised fourfold
    and everything else unchanged (scripts/run_budget_rerun.py); the pair
    analysis uses that measurement, and its per-probe tokens, in place of the
    bound. Returns {model: (cc at 32k, [(length, completion tokens,
    reasoning tokens)])}.
    """
    import json

    from counting.analysis.capacity import fit_transition

    out: dict[str, tuple[int, list[tuple[int, float, float]]]] = {}
    for trials in sorted(BUDGET_RUN.glob("*/trials/*.jsonl")):
        recs = [json.loads(line) for line in trials.open(encoding="utf-8") if line.strip()]
        scan = sorted(
            [r for r in recs if r.get("phase") == "scan" and r.get("response_status") != "error"],
            key=lambda r: r["center_length"],
        )
        if not scan:
            continue
        fit = fit_transition(
            [r["center_length"] for r in scan],
            [bool(r.get("exact_match")) for r in scan],
            [r.get("parsed") is None for r in scan],
        )
        if not fit.scc:
            continue
        model = recs[0]["model"].split("#", 1)[0]
        probes = [
            (int(r["center_length"]), float(r.get("output_tokens") or 0),
             float(r["reasoning_tokens"]) if r.get("reasoning_tokens") not in (None, "") else float("nan"))
            for r in scan
        ]
        out[model] = (int(fit.scc), probes)
    return out


def _classify(rtok_think: float, rtok_nothink: float) -> str:
    if np.isnan(rtok_think) or np.isnan(rtok_nothink):
        return "unreported"
    if rtok_think < 10:
        return "no reasoning"
    if rtok_nothink >= 10 and rtok_nothink / rtok_think >= 0.05:
        return "knob ignored"
    return "verified"


def hybrid_pairs() -> list[Pair]:
    cc: dict[str, dict[str, int]] = defaultdict(dict)
    with (RUN / "scc.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row["run_key"]
            if "#" not in key:
                continue
            model, variant = key.rsplit("#", 1)
            cc[model][variant] = int(row["scc"])

    tokens = _probe_tokens()
    # Configurations remeasured at the 32k cap carry their run of origin in the
    # registry; the pair table marks them.
    rescanned: set[str] = set()
    with (RUN / "registry.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["run_key"].endswith("#think") and row["run"].startswith("budget-32k"):
                rescanned.add(row["run_key"].rsplit("#", 1)[0])
    pairs: list[Pair] = []
    for model, variants in sorted(cc.items()):
        if "think" not in variants or "nothink" not in variants:
            continue
        # Compare token spend over the range both configurations count reliably,
        # so a higher-capacity variant is not charged for its longer probes.
        shared = min(variants["think"], variants["nothink"])
        stats = {}
        for variant in ("think", "nothink"):
            probes = [p for p in tokens.get(f"{model}#{variant}", []) if p[0] <= shared]
            if not probes:
                break
            out_values = [p[1] for p in probes if p[1] > 0]
            r_values = [p[2] for p in probes if not np.isnan(p[2])]
            stats[variant] = (
                float(np.mean(out_values)) if out_values else float("nan"),
                float(np.mean(r_values)) if r_values else float("nan"),
            )
        if len(stats) != 2:
            continue
        pairs.append(Pair(
            model=model,
            provider=infer_provider_family(model),
            cc_think=variants["think"],
            cc_nothink=variants["nothink"],
            tokens_think=stats["think"][0],
            tokens_nothink=stats["nothink"][0],
            rtok_think=stats["think"][1],
            rtok_nothink=stats["nothink"][1],
            classification=_classify(stats["think"][1], stats["nothink"][1]),
            rescanned=model in rescanned,
        ))
    return pairs


# -- compatibility shims for landscape_triptych ------------------------------

def load_payloads() -> list[Pair]:
    return hybrid_pairs()


def pair_records(pairs: list[Pair]) -> list[dict[str, object]]:
    return [
        {
            "label": p.model.split("/")[-1],
            "delta_scc": p.delta_cc,
            "token_multiplier": p.token_multiplier,
            "provider": p.provider,
            "color": PROVIDER_COLORS.get(p.provider, "#888888"),
        }
        for p in pairs
        if p.classification == "verified" and np.isfinite(p.token_multiplier)
    ]


def multiplier_formatter(value: float, _: object) -> str:
    if value < 1:
        return ""
    if abs(value - round(value)) < 1e-8:
        return f"{int(round(value))}x"
    return f"{value:g}x"


# -- figures -----------------------------------------------------------------

def make_pair_figure(pairs: list[Pair]) -> None:
    shown = [p for p in pairs if p.classification == "verified" and np.isfinite(p.token_multiplier)]
    fig, ax = plt.subplots(figsize=(3.4, 3.1), constrained_layout=True)
    ax.axvline(0, color="#b8b8b8", lw=1.0, ls="--", zorder=1)
    ax.axhline(1.0, color="#b8b8b8", lw=1.0, ls="--", zorder=1)
    for p in shown:
        ax.scatter([p.delta_cc], [p.token_multiplier], s=22,
                   color=PROVIDER_COLORS.get(p.provider, "#888888"),
                   edgecolors="white", linewidths=0.5, alpha=.9, zorder=3)
    median_delta = float(np.median([p.delta_cc for p in shown]))
    median_mult = float(np.median([p.token_multiplier for p in shown]))
    ax.scatter([median_delta], [median_mult], s=70, marker="D", color=BASELINE_DARK,
               edgecolors="white", linewidths=0.9, zorder=5)
    ax.annotate(f"median\n{median_delta:+.0f} at {median_mult:.1f}x",
                (median_delta, median_mult), xytext=(8, 8),
                textcoords="offset points", fontsize=7.0, color=BASELINE_DARK)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(multiplier_formatter))
    ax.set_xlabel(r"$\Delta$CC (reasoning on $-$ off)", fontsize=9)
    ax.set_ylabel("token multiplier, same lengths", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.18, which="both", zorder=0)
    set_spine_width(ax, 0.9)
    fig.savefig(OUTPUT_PDF, transparent=True)
    fig.savefig(OUTPUT_PDF.with_suffix(".png"), dpi=250, transparent=True)
    plt.close(fig)
    print(f"wrote {OUTPUT_PDF}  ({len(shown)} verified pairs)")


def make_token_figure() -> None:
    """Completion tokens per probe near the boundary, against CC, every measurement."""
    cc = {}
    with (RUN / "scc.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cc[row["run_key"]] = int(row["scc"])
    tokens = _probe_tokens()
    xs, ys, colours = [], [], []
    for key, probes in tokens.items():
        boundary = cc.get(key, 0)
        if boundary < 10:
            continue
        near = [p[1] for p in probes if 0.8 * boundary <= p[0] <= boundary and p[1] > 0]
        if not near:
            continue
        xs.append(boundary)
        ys.append(float(np.mean(near)))
        colours.append(PROVIDER_COLORS.get(infer_provider_family(key.split("#")[0]), "#888888"))
    fig, ax = plt.subplots(figsize=(3.4, 3.1), constrained_layout=True)
    ax.scatter(xs, ys, s=14, c=colours, alpha=.75, edgecolors="white",
               linewidths=0.3, zorder=3)
    # The floor: a bare-integer reply is ~2 tokens regardless of magnitude
    # (multi-digit numbers are one or two tokens). Everything above it is
    # deliberation.
    ax.axhline(2, color="#666666", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.annotate("bare-integer floor", (max(xs), 2.3), ha="right", va="bottom",
                fontsize=7.0, color="#666666")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("CC", fontsize=9)
    ax.set_ylabel("completion tokens per probe near CC", fontsize=9)
    ax.tick_params(labelsize=8, which="both")
    ax.grid(alpha=0.18, which="both", zorder=0)
    set_spine_width(ax, 0.9)
    fig.savefig(TOKENS_PDF, transparent=True)
    fig.savefig(TOKENS_PDF.with_suffix(".png"), dpi=250, transparent=True)
    plt.close(fig)
    print(f"wrote {TOKENS_PDF}  ({len(xs)} measurements)")


# -- appendix table ----------------------------------------------------------

def write_table(pairs: list[Pair]) -> None:
    def fmt(value: float) -> str:
        return "--" if np.isnan(value) else f"{value:,.0f}"

    def name(p: Pair) -> str:
        return p.model.split("/")[-1].replace("_", "\\_")

    verified = sorted((p for p in pairs if p.classification == "verified"),
                      key=lambda p: (p.provider, p.model))
    anomalies = sorted((p for p in pairs if p.classification != "verified"),
                       key=lambda p: (p.classification, p.provider, p.model))
    listed = ", ".join(
        f"\\texttt{{{name(p)}}}" + ("$^{\\dagger}$" if p.rescanned else "") for p in verified
    )
    lines = [
        "% Generated by counting.analysis.reasoning_pairs -- do not edit by hand.",
        "{\\scriptsize\\noindent " + listed + "."
        + (" $^{\\dagger}$reasoning member measured at a 32,768-token cap (Table~\\ref{tab:budget32k})." if any(p.rescanned for p in verified) else "")
        + "\\par}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Hybrid pairs excluded from the matched comparison. Token",
        "columns give the mean completion tokens per probe over the lengths that",
        "both configurations count reliably. \\emph{Knob ignored} marks",
        "providers that apply reasoning regardless of the request, so the pair",
        "amounts to two measurements of one configuration; \\emph{no reasoning}",
        "marks pairs whose reasoning-enabled configuration spent no reasoning",
        "tokens either.}",
        "\\label{tab:pair_anomalies}",
        "\\begin{tabular}{@{}l r r r r r l@{}}",
        "\\toprule",
        "Model & CC$_{\\text{off}}$ & CC$_{\\text{on}}$ & $\\Delta$CC &"
        " tok$_{\\text{off}}$ & tok$_{\\text{on}}$ & Class \\\\",
        "\\midrule",
    ]
    for p in anomalies:
        lines.append(
            f"\\texttt{{{name(p)}}} & {p.cc_nothink} & {p.cc_think} & {p.delta_cc:+d} & "
            f"{fmt(p.tokens_nothink)} & {fmt(p.tokens_think)} & {p.classification} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    TABLE_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLE_TEX}  ({len(verified)} verified listed, "
          f"{len(anomalies)} anomalies tabled)")


def main() -> int:
    pairs = hybrid_pairs()
    counts = defaultdict(int)
    for p in pairs:
        counts[p.classification] += 1
    print(f"{len(pairs)} hybrid pairs: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    make_pair_figure(pairs)
    make_token_figure()
    write_table(pairs)
    verified = [p for p in pairs if p.classification == "verified"]
    deltas = sorted(p.delta_cc for p in verified)
    mults = sorted(p.token_multiplier for p in verified if np.isfinite(p.token_multiplier))
    print(f"  verified pairs: {len(verified)}   delta-CC median {np.median(deltas):+.0f} "
          f"(IQR {deltas[len(deltas)//4]:+d}..{deltas[3*len(deltas)//4]:+d})   "
          f"multiplier median {np.median(mults):.1f}x")
    print(f"  delta-CC <= 0: {sum(1 for d in deltas if d <= 0)}/{len(deltas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
