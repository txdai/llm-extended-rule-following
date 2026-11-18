"""Replicate scans through the transition region (decoding-stability study).

    python -m counting.analysis.replicates

Reads results/temperature-replicates/rep-<k>/<slug>/trials/*.jsonl, refits
every replicate with the paper's two-regime estimator, and prints one line
per model: the catalogue CC, each replicate's fitted boundary with its 95%
profile interval, the between-replicate spread, and how many replicates fall
inside the catalogue interval.  Also writes results/temperature-replicates/
summary.csv for the appendix.
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from counting.analysis.capacity import fit_transition
from counting.paths import REPO_ROOT as ROOT

RUN_DIR = ROOT / "results" / "temperature-replicates"
CATALOGUE = ROOT / "results" / "counting-data"

# (model id, variant, short label, decoding control)
REPLICATE_MODELS = [
    ("anthropic/claude-sonnet-5", "nothink", "claude-sonnet-5", "locked"),
    ("openai/gpt-5.6-sol",        "think",   "gpt-5.6-sol",     "locked"),
    ("openai/gpt-5",              "",        "gpt-5",           "locked"),
    ("openai/o3",                 "think",   "o3",              "locked"),
    ("openai/gpt-5.2",            "think",   "gpt-5.2",         "locked"),
    ("anthropic/claude-opus-5",   "nothink", "claude-opus-5",   "ignored"),
    ("openai/gpt-5.4",            "nothink", "gpt-5.4",         "locked"),
    ("anthropic/claude-fable-5",  "",        "claude-fable-5",  "locked"),
    ("google/gemini-3.7-flash",   "",        "gemini-3.7-flash", "ignored"),
    ("anthropic/claude-sonnet-4.6", "nothink", "claude-sonnet-4.6", "pinned"),
]


def slug_of(model: str) -> str:
    return model.split("/")[-1].replace(".", "p")


@dataclass
class Replicate:
    rep: int
    lengths: list[int]
    correct: list[bool]
    tau: int | None
    ci_low: int | None
    ci_high: int | None
    resolved: bool


@dataclass
class ModelReplicates:
    model: str
    variant: str
    label: str
    control: str
    cc: int
    cc_low: int
    cc_high: int
    replicates: list[Replicate] = field(default_factory=list)

    @property
    def taus(self) -> list[int]:
        return [r.tau for r in self.replicates if r.tau]


def catalogue_cc() -> dict[tuple[str, str], tuple[int, int, int]]:
    out = {}
    with (CATALOGUE / "scc.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                out[(row["model"], row["variant"])] = (
                    int(row["scc"]), int(row["scc_ci_low"]), int(row["scc_ci_high"]))
            except ValueError:
                continue
    return out


def read_replicate(path: Path, rep: int) -> Replicate | None:
    lengths, correct, errors = [], [], []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") != "scan" or row.get("response_status") == "error":
            continue
        lengths.append(int(row["center_length"]))
        correct.append(bool(row.get("exact_match")))
        errors.append(row.get("parsed") is None)
    if not lengths:
        return None
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    lengths = [lengths[i] for i in order]
    correct = [correct[i] for i in order]
    errors = [errors[i] for i in order]
    fit = fit_transition(lengths, correct, errors)
    return Replicate(rep, lengths, correct, fit.scc, fit.ci_low, fit.ci_high,
                     bool(fit.scc))


def load() -> list[ModelReplicates]:
    cc = catalogue_cc()
    out = []
    for model, variant, label, control in REPLICATE_MODELS:
        c, lo, hi = cc[(model, variant)]
        mr = ModelReplicates(model, variant, label, control, c, lo, hi)
        for rep_dir in sorted(RUN_DIR.glob("rep-*")):
            rep = int(rep_dir.name.split("-")[1])
            run_dir = rep_dir / slug_of(model)
            for path in sorted((run_dir / "trials").glob("*.jsonl")):
                # A scan is complete once the CLI has written its model summary;
                # a fit on a still-running record would place the boundary at
                # wherever the record happens to end.
                if not (run_dir / "models" / path.with_suffix(".json").name).exists():
                    continue
                r = read_replicate(path, rep)
                if r is not None:
                    mr.replicates.append(r)
        out.append(mr)
    return out


def summarize(models: list[ModelReplicates]) -> list[dict]:
    rows = []
    for m in models:
        taus = m.taus
        inside = sum(1 for r in m.replicates if r.tau and m.cc_low <= r.tau <= m.cc_high)
        rows.append({
            "model": m.model, "variant": m.variant, "control": m.control,
            "catalogue_cc": m.cc, "catalogue_ci": f"[{m.cc_low}, {m.cc_high}]",
            "n_replicates": len(m.replicates), "n_resolved": len(taus),
            "taus": " ".join(str(t) for t in taus),
            "tau_median": statistics.median(taus) if taus else "",
            "tau_range": f"{min(taus)}-{max(taus)}" if taus else "",
            "tau_sd": round(statistics.pstdev(taus), 2) if len(taus) > 1 else "",
            "tau_sd_rel_cc": round(statistics.pstdev(taus) / m.cc, 4) if len(taus) > 1 else "",
            "median_rel_cc": round(statistics.median(taus) / m.cc, 3) if taus else "",
            "inside_catalogue_ci": inside,
        })
    return rows


def main() -> int:
    models = load()
    rows = summarize(models)
    out = RUN_DIR / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as sink:
        w = csv.DictWriter(sink, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']:30s} CC={r['catalogue_cc']:4d} {r['catalogue_ci']:12s} "
              f"reps={r['n_resolved']}/{r['n_replicates']} tau={r['taus']:22s} "
              f"sd={r['tau_sd']} inCI={r['inside_catalogue_ci']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
