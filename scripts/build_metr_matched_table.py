"""Regenerate paper/generated/metr_matched_table.tex from the merged benchmark table.

Rows: every model matched between the CC catalogue and the METR time-horizon
leaderboard (results/benchmark_data/counting_benchmark_merged.csv, written by
counting.analysis.figures.benchmark_correlations). The configuration column is
the leaderboard's own setting for the matched row (default, medium, high, a
16K thinking budget, or unknown); the CC is the larger of the model's two
configurations, as everywhere in the paper. ECI to one decimal.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "results" / "benchmark_data" / "counting_benchmark_merged.csv"
OUT = ROOT / "paper" / "generated" / "metr_matched_table.tex"
VENDOR = {"openai": "OpenAI", "anthropic": "Anthropic", "deepseek": "DeepSeek", "google": "Google", "qwen": "Qwen",
          "moonshotai": "Moonshot", "meta": "Meta"}


def configuration(version: str) -> str:
    m = re.search(r"_(\d+)[kK]$", version)
    if m:
        return f"{m.group(1)}K budget"
    m = re.search(r"_(low|medium|high|xhigh|unknown)$", version)
    return m.group(1) if m else "default"


def main() -> int:
    rows = list(csv.DictReader(MERGED.open()))
    eci = {r["counting_model"]: float(r["benchmark_score"]) for r in rows
           if r["benchmark_file"] == "epoch_capabilities_index.csv" and r["stable_count"] not in ("", "0")}
    metr = [r for r in rows if r["benchmark_file"] == "metr_time_horizons_external.csv" and r["stable_count"] not in ("", "0")]
    metr.sort(key=lambda r: (-float(r["stable_count"]), r["counting_model"]))
    folds = Counter(r["provider"] for r in metr if r["counting_model"] in eci)
    fold_text = ", ".join(f"{VENDOR.get(v, v)} {n}" for v, n in folds.most_common())
    lines = [
        "\\begin{table}[htbp]", "\\centering", "\\footnotesize",
        "\\caption{\\textbf{Models matched between the CC catalogue and the METR time-horizon leaderboard.} "
        "Entries are matched by the leaderboard's model version string, with dated checkpoints matched to the dated catalogue entry; "
        "the configuration column is the leaderboard's setting for the matched row, and where the leaderboard lists several settings for one model the medium or default row is used. "
        "The CC is the larger of the model's two reasoning configurations. The Epoch Capabilities Index (ECI) is given to one decimal; the correlations in the text use unrounded values. "
        f"Vendor folds for the held-out prediction: {fold_text}.}}",
        "\\label{tab:metr_matched}",
        "\\begin{tabular}{llrrr}", "\\toprule",
        "Model & Configuration & CC & Horizon (min) & ECI \\\\", "\\midrule",
    ]
    for r in metr:
        model = r["counting_model"].split("/", 1)[-1].replace("_", "\\_")
        e = eci.get(r["counting_model"])
        lines.append(f"\\texttt{{{model}}} & {configuration(r['benchmark_model_version'])} & {float(r['stable_count']):.0f} & "
                     f"{float(r['benchmark_score']):.1f} & {f'{e:.1f}' if e is not None else '--'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({len(metr)} rows, {sum(1 for r in metr if r['counting_model'] in eci)} with ECI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
