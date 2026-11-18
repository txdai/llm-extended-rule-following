"""Response parsing and tier scoring.

The pass rule for a tier is deliberately the same one used for every run in the
paper, and changing it invalidates comparisons against existing results:

    a tier passes iff  parse_rate == 1.0  and  nMAE <= tolerance_fraction

where nMAE is the mean of |predicted - target| / target over parsed trials. The
paper calls this nMAE; the stored records call it `mape`. Both names refer to
this quantity -- the stored key is kept for compatibility with results/ written
before the refactor.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Iterable, Sequence

from .types import TrialResult

INTEGER_PATTERN = re.compile(r"-?\d+")


def parse_integer(text: str | None) -> int | None:
    """Return the last integer in the response, or None if there is none.

    Last rather than first: models that narrate before answering put the answer
    at the end, and the narration is full of intermediate counts.
    """
    if not text:
        return None
    matches = INTEGER_PATTERN.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def mean(values: Iterable[float]) -> float | None:
    materialised = [float(value) for value in values]
    if not materialised:
        return None
    return statistics.fmean(materialised)


def safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def score_trial(parsed: int | None, expected: int) -> dict[str, Any]:
    """Errors for a single trial. Unparsed responses contribute no error terms."""
    if parsed is None:
        return {
            "exact_match": False,
            "absolute_error": None,
            "squared_error": None,
            "relative_error": None,
        }
    error = parsed - expected
    return {
        "exact_match": parsed == expected,
        "absolute_error": float(abs(error)),
        "squared_error": float(error * error),
        "relative_error": abs(error) / expected if expected else None,
    }


def summarize_slice(results: Sequence[TrialResult]) -> dict[str, Any]:
    """Aggregate a set of trials. Shape matches the pre-refactor records."""
    total = len(results)
    parsed = [r for r in results if r.parsed is not None]

    def column(attr: str, source: Sequence[TrialResult] = results) -> list[float]:
        return [getattr(r, attr) for r in source if getattr(r, attr) is not None]

    input_tokens = column("input_tokens")
    output_tokens = column("output_tokens")
    total_tokens = column("total_tokens")
    reasoning_tokens = column("reasoning_tokens")
    cached_tokens = column("cached_input_tokens")

    return {
        "trials": total,
        "parse_rate": safe_divide(len(parsed), total),
        "success_rate": safe_divide(sum(r.exact_match for r in results), total),
        "mae": mean(column("absolute_error", parsed)),
        "mse": mean(column("squared_error", parsed)),
        "mape": mean(column("relative_error", parsed)),
        "mean_signed_error": mean([float(r.parsed - r.expected) for r in parsed]),
        "avg_latency_seconds": mean(column("latency_seconds")),
        "avg_input_tokens": mean(input_tokens),
        "avg_output_tokens": mean(output_tokens),
        "avg_total_tokens": mean(total_tokens),
        "avg_reasoning_tokens": mean(reasoning_tokens),
        "avg_cached_input_tokens": mean(cached_tokens),
        "sum_input_tokens": int(sum(input_tokens)),
        "sum_output_tokens": int(sum(output_tokens)),
        "sum_total_tokens": int(sum(total_tokens)),
        "sum_reasoning_tokens": int(sum(reasoning_tokens)),
        "sum_cached_input_tokens": int(sum(cached_tokens)),
    }


def tier_passed(summary: dict[str, Any], tolerance_fraction: float) -> bool:
    """The stability rule. See the module docstring before changing this."""
    parse_rate = summary.get("parse_rate")
    mape = summary.get("mape")
    return parse_rate == 1.0 and mape is not None and mape <= tolerance_fraction
