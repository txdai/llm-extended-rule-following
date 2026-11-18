#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from counting.analysis.figures.search_panels import PROVIDER_DISPLAY_NAMES, display_model_label, infer_provider_family


from counting.datastore import load_registry
from counting.paths import REPO_ROOT as ROOT
RUN_DIRS = (
    ROOT / "results" / "counting-stable-search",
    ROOT / "results" / "counting-stable-search-open-source",
)
OUTPUT_CSV = ROOT / "results" / "model_inventory_table.csv"
OUTPUT_TEX = ROOT / "paper" / "generated" / "model_inventory_table_si.tex"


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    model_name: str
    developer_family: str
    parameter_count: str
    nominal_context_window: str
    architecture: str
    measured_scc: int


def load_manifest(run_dir: Path) -> dict[str, dict[str, object]]:
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    models = payload.get("models")
    return models if isinstance(models, dict) else {}


def parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_architecture(total_params: float | None, active_params: float | None, description: str) -> str:
    lowered = description.lower()
    if total_params is not None and active_params is not None and total_params - active_params > 1e-9:
        return "MoE"
    if "mixture-of-experts" in lowered or re.search(r"\bmoe\b", lowered):
        return "MoE"
    if total_params is not None or active_params is not None:
        return "Dense"
    return "--"


def format_billions(value: float | None) -> str | None:
    if value is None:
        return None
    rounded = round(value, 3)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{int(round(rounded))}B"
    if rounded >= 100:
        return f"{rounded:.0f}B"
    if rounded >= 10:
        return f"{rounded:.1f}B"
    return f"{rounded:.2f}B".rstrip("0").rstrip(".") + "B"


def format_parameter_count(total_params: float | None, active_params: float | None) -> str:
    total_text = format_billions(total_params)
    active_text = format_billions(active_params)
    if total_text and active_text and total_text != active_text:
        return f"{total_text} total / {active_text} active"
    if total_text:
        return total_text
    if active_text:
        return f"{active_text} active"
    return "--"


def format_context_window(value: object) -> str:
    if value in (None, ""):
        return "--"
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return "--"
    return f"{integer:,}"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = []
    for char in text:
        escaped.append(replacements.get(char, char))
    return "".join(escaped)


def provider_name(model_id: str) -> str:
    provider = infer_provider_family(model_id)
    return PROVIDER_DISPLAY_NAMES.get(provider, provider.title())


def build_records() -> list[ModelRecord]:
    records: list[ModelRecord] = []
    seen: set[str] = set()

    for run_dir in RUN_DIRS:
        manifest_models = load_manifest(run_dir)
        for row in load_registry(run_dir):
            model_id = row.get("model", "")
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)

            if row.get("cc") in (None, ""):
                continue
            measured_scc = int(float(row["cc"]))

            manifest_entry = manifest_models.get(model_id, {})
            model_metadata = manifest_entry.get("model_metadata", {})
            parameter_metadata = manifest_entry.get("parameter_metadata", {})
            if not isinstance(model_metadata, dict):
                model_metadata = {}
            if not isinstance(parameter_metadata, dict):
                parameter_metadata = {}

            total_params = parse_float(row.get("total_params_billion"))
            active_params = parse_float(row.get("active_params_billion"))
            if total_params is None:
                total_params = parse_float(parameter_metadata.get("total_params_billion"))
            if active_params is None:
                active_params = parse_float(parameter_metadata.get("active_params_billion"))

            context_length = model_metadata.get("context_length")
            if context_length in (None, ""):
                top_provider = model_metadata.get("top_provider")
                if isinstance(top_provider, dict):
                    context_length = top_provider.get("context_length")

            records.append(
                ModelRecord(
                    model_id=model_id,
                    model_name=display_model_label(model_id),
                    developer_family=provider_name(model_id),
                    parameter_count=format_parameter_count(total_params, active_params),
                    nominal_context_window=format_context_window(context_length),
                    architecture=infer_architecture(
                        total_params,
                        active_params,
                        str(model_metadata.get("description") or ""),
                    ),
                    measured_scc=measured_scc,
                )
            )

    records.sort(key=lambda item: (-item.measured_scc, item.developer_family.lower(), item.model_name.lower()))
    return records


def write_csv(records: list[ModelRecord]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
            "model_id",
            "model_name",
            "developer_family",
            "parameter_count",
            "nominal_context_window",
            "architecture",
            "measured_scc",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
            {
                "model_id": record.model_id,
                "model_name": record.model_name,
                "developer_family": record.developer_family,
                "parameter_count": record.parameter_count,
                "nominal_context_window": record.nominal_context_window,
                "architecture": record.architecture,
                "measured_scc": record.measured_scc,
            }
            )


def write_tex(records: list[ModelRecord]) -> None:
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\LTleft}{0pt}")
    lines.append(r"\setlength{\LTright}{0pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\begin{longtable}{p{4.2cm}p{2.2cm}p{2.6cm}p{2.1cm}p{1.5cm}p{1.1cm}}")
    lines.append(
        r"\caption{\textbf{Table S1. Evaluated model inventory.} "
        r"Nominal context window and parameter metadata are taken from the run manifests when available. "
        r"The architecture label is inferred from published parameter metadata: models with distinct total and active "
        r"parameter counts are marked as MoE; models with only a single parameter count are marked as Dense; unavailable cases are left as ``--''.}"
        r"\label{tab:model_inventory} \\"
    )
    lines.append(r"\toprule")
    lines.append(r"Model Name & Developer/Family & Parameter Count & Nominal Context Window & Architecture & Measured SCC \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Model Name & Developer/Family & Parameter Count & Nominal Context Window & Architecture & Measured SCC \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{r}{Continued on next page} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for record in records:
        lines.append(
            " & ".join(
            [
                latex_escape(record.model_name),
                latex_escape(record.developer_family),
                latex_escape(record.parameter_count),
                latex_escape(record.nominal_context_window),
                latex_escape(record.architecture),
                str(record.measured_scc),
            ]
            )
            + r" \\"
        )

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    records = build_records()
    write_csv(records)
    write_tex(records)
    print(f"CSV: {OUTPUT_CSV}")
    print(f"LaTeX: {OUTPUT_TEX}")
    print(f"Rows: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
