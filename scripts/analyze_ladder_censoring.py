#!/usr/bin/env python3
"""Fit the prediction relation without discarding censored ladder scans.

The latent execution boundary is modeled on the log scale as a Gaussian
linear function of the log CC-scaled prediction. Resolved scans contribute a
density, below-minimum scans a left-tail probability, and ceiling-stable scans
a right-tail probability. Nonmonotone mixed scans are not treated as ordinary
censoring; they enter a separate interval-inclusive sensitivity fit. Blank or
budget-dead scans carry no numerical boundary bound and are reported, but not
forced into either likelihood.

Outputs:
  results/ladder-full/censored_analysis.json
  results/ladder-full/model_summary.csv
  paper/generated/ladder_model_brackets.tex
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import chi2, norm, spearmanr, truncnorm

REPO = Path(__file__).resolve().parents[1]
FITS = REPO / "results" / "ladder-full" / "fits.csv"
ORIGINAL_FITS = REPO / "results" / "ladder-full" / "fits-original.csv"
OUT = REPO / "results" / "ladder-full" / "censored_analysis.json"
MODEL_OUT = REPO / "results" / "ladder-full" / "model_summary.csv"
LATEX_OUT = REPO / "paper" / "generated" / "ladder_model_brackets.tex"
SEED = 20260901
IMPUTATIONS = 5000
CLUSTER_BOOTSTRAPS = 500


def _logdiffexp(a: float, b: float) -> float:
    """log(exp(a)-exp(b)) for a >= b, with a finite floor."""
    if b >= a:
        return math.log(np.finfo(float).tiny)
    return a + math.log1p(-math.exp(b - a))


def _log_interval_probability(low_z: float, high_z: float) -> float:
    if math.isinf(low_z) and low_z < 0:
        return float(log_ndtr(high_z))
    if math.isinf(high_z) and high_z > 0:
        return float(log_ndtr(-low_z))
    # Work on whichever tail avoids subtracting two values numerically equal
    # to one.
    if high_z <= 0:
        return _logdiffexp(float(log_ndtr(high_z)), float(log_ndtr(low_z)))
    return _logdiffexp(float(log_ndtr(-low_z)), float(log_ndtr(-high_z)))


def _neg_log_likelihood(
    params: np.ndarray, x: np.ndarray, low: np.ndarray, high: np.ndarray
) -> float:
    alpha, beta, log_sigma = params
    sigma = math.exp(float(log_sigma))
    mu = alpha + beta * x
    ll = 0.0
    for mean, lo, hi in zip(mu, low, high):
        if lo == hi:
            ll += float(norm.logpdf(lo, loc=mean, scale=sigma))
        else:
            lo_z = (lo - mean) / sigma
            hi_z = (hi - mean) / sigma
            ll += _log_interval_probability(float(lo_z), float(hi_z))
    return -ll


def _fit(
    x: np.ndarray, low: np.ndarray, high: np.ndarray, beta_fixed: float | None = None
) -> tuple[np.ndarray, float]:
    exact = np.isfinite(low) & np.isfinite(high) & (low == high)
    if exact.sum() >= 2:
        slope, intercept = np.polyfit(x[exact], low[exact], 1)
        residual = low[exact] - (intercept + slope * x[exact])
        sigma = max(float(np.std(residual)), 0.1)
    else:
        intercept, slope, sigma = 0.0, 1.0, 1.0

    if beta_fixed is None:
        start = np.array([intercept, slope, math.log(sigma)])
        result = minimize(
            _neg_log_likelihood,
            start,
            args=(x, low, high),
            method="L-BFGS-B",
            bounds=((-20, 20), (-5, 5), (-4, 4)),
        )
        if not result.success:
            raise RuntimeError(f"censored regression failed: {result.message}")
        return np.asarray(result.x), -float(result.fun)

    def objective(reduced: np.ndarray) -> float:
        return _neg_log_likelihood(
            np.array([reduced[0], beta_fixed, reduced[1]]), x, low, high
        )

    result = minimize(
        objective,
        np.array([intercept, math.log(sigma)]),
        method="L-BFGS-B",
        bounds=((-20, 20), (-4, 4)),
    )
    if not result.success:
        raise RuntimeError(f"null censored regression failed: {result.message}")
    return np.array([result.x[0], beta_fixed, result.x[1]]), -float(result.fun)


def _as_log_bound(value: str, default: float) -> float:
    return math.log(float(value)) if value else default


def _format_limited(row: dict[str, str], floor: float) -> bool:
    """True when the scan's boundary is below `floor` objects.

    A model that is already wrong at a handful of objects is not reporting an
    execution capacity; it is failing to execute the task at all, the same way
    a pre-release model fails ARC-AGI-2 on format rather than on capacity.
    Resolved scans are judged on the fitted boundary, one-sided left-censored
    and interval scans on the upper bound the scan actually establishes.
    Right-censored scans are never format-limited.
    """
    if row["censoring"] == "exact":
        return float(row["boundary"]) < floor
    if row["censoring"] in {"left", "interval"}:
        return bool(row.get("bound_high")) and float(row["bound_high"]) <= floor
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", type=Path, default=FITS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model-out", type=Path, default=MODEL_OUT)
    ap.add_argument("--latex-out", type=Path, default=LATEX_OUT)
    ap.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        metavar="SCAN",
        help="drop every scan of this task, e.g. a task whose prompt admits a "
             "second reading of the rule; repeatable",
    )
    ap.add_argument(
        "--min-boundary",
        type=float,
        default=0.0,
        help="drop scans whose established boundary falls below this many "
             "objects, as format failures rather than capacity measurements",
    )
    args = ap.parse_args()
    all_rows = list(csv.DictReader(args.fits.open(encoding="utf-8")))
    if not ORIGINAL_FITS.exists():
        raise FileNotFoundError(
            f"{ORIGINAL_FITS} is required; generate it with "
            "scripts/fit_full_ladder.py --original-only --output "
            "results/ladder-full/fits-original.csv"
        )
    original_rows = list(csv.DictReader(ORIGINAL_FITS.open(encoding="utf-8")))
    included = [r for r in all_rows if r["analysis_included"].lower() == "true"]
    excluded_tasks = set(args.exclude_task)
    task_dropped = [r for r in included if r["scan"] in excluded_tasks]
    if excluded_tasks:
        included = [r for r in included if r["scan"] not in excluded_tasks]
        original_rows = [r for r in original_rows if r["scan"] not in excluded_tasks]
    dropped: list[dict[str, str]] = []
    if args.min_boundary > 0:
        dropped = [r for r in included if _format_limited(r, args.min_boundary)]
        keep = {(r["scan"], r["model"]) for r in included} - {
            (r["scan"], r["model"]) for r in dropped
        }
        included = [r for r in included if (r["scan"], r["model"]) in keep]
        original_rows = [
            r for r in original_rows if (r["scan"], r["model"]) in keep
        ]
    informative_all = [
        r
        for r in included
        if r.get("predicted")
        and r.get("censoring") in {"exact", "left", "right", "interval"}
    ]
    # Nonmonotone mixed scans are not ordinary censoring: no supported
    # transition orders their correct and incorrect probes. Keep them in the
    # table/figure and an interval-inclusive sensitivity fit, but do not force
    # them into the primary one-sided censored likelihood.
    informative = [r for r in informative_all if r["censoring"] != "interval"]
    x = np.array([math.log(float(r["predicted"])) for r in informative])
    low = np.array([_as_log_bound(r.get("bound_low", ""), -math.inf) for r in informative])
    high = np.array([_as_log_bound(r.get("bound_high", ""), math.inf) for r in informative])

    params, loglike = _fit(x, low, high)
    null_params, null_loglike = _fit(x, low, high, beta_fixed=0.0)
    lr = max(0.0, 2.0 * (loglike - null_loglike))
    p_value = float(chi2.sf(lr, 1))
    alpha, beta, log_sigma = (float(v) for v in params)
    sigma = math.exp(log_sigma)

    x_all = np.array([math.log(float(r["predicted"])) for r in informative_all])
    low_all = np.array(
        [_as_log_bound(r.get("bound_low", ""), -math.inf) for r in informative_all]
    )
    high_all = np.array(
        [_as_log_bound(r.get("bound_high", ""), math.inf) for r in informative_all]
    )
    inclusive_params, inclusive_loglike = _fit(x_all, low_all, high_all)
    _, inclusive_null_loglike = _fit(x_all, low_all, high_all, beta_fixed=0.0)
    inclusive_lr = max(0.0, 2.0 * (inclusive_loglike - inclusive_null_loglike))

    resolved = [r for r in informative if r["censoring"] == "exact"]
    resolved_rho = float(
        spearmanr(
            [math.log(float(r["predicted"])) for r in resolved],
            [math.log(float(r["boundary"])) for r in resolved],
        ).statistic
    )
    median_ratio = float(
        np.median([float(r["boundary"]) / float(r["predicted"]) for r in resolved])
    )

    # A transparent nonparametric sensitivity check: substitute each observed
    # one-sided censoring limit.
    limit_values = []
    for lo, hi in zip(low, high):
        if lo == hi:
            limit_values.append(lo)
        elif math.isinf(lo):
            limit_values.append(hi)
        elif math.isinf(hi):
            limit_values.append(lo)
        else:
            limit_values.append((lo + hi) / 2.0)
    limit_rho = float(spearmanr(x, limit_values).statistic)

    # Model-based rank sensitivity: draw every latent censored boundary from
    # its fitted truncated distribution and recompute Spearman's rho.  Exact
    # boundaries remain fixed.  This is an imputation range, not a sampling CI.
    rng = np.random.default_rng(SEED)
    rhos = np.empty(IMPUTATIONS)
    mu = alpha + beta * x
    for draw in range(IMPUTATIONS):
        y = np.empty(len(x))
        for i, (mean, lo, hi) in enumerate(zip(mu, low, high)):
            if lo == hi:
                y[i] = lo
            else:
                a = (lo - mean) / sigma
                b = (hi - mean) / sigma
                y[i] = float(truncnorm.rvs(a, b, loc=mean, scale=sigma, random_state=rng))
        rhos[draw] = float(spearmanr(x, y).statistic)

    # The six tasks within a model are correlated. Resample whole models,
    # rather than individual scans, to describe uncertainty in the censored
    # regression slope without pretending that all measurements are
    # independent. Duplicate model draws act as cluster weights in the
    # bootstrap likelihood.
    informative_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in informative:
        informative_by_model[row["model"]].append(row)
    bootstrap_models = sorted(informative_by_model)
    bootstrap_rng = np.random.default_rng(SEED + 1)
    bootstrap_slopes = []
    for _ in range(CLUSTER_BOOTSTRAPS):
        sampled_rows = []
        for model in bootstrap_rng.choice(
            bootstrap_models, size=len(bootstrap_models), replace=True
        ):
            sampled_rows.extend(informative_by_model[str(model)])
        sampled_x = np.array(
            [math.log(float(r["predicted"])) for r in sampled_rows]
        )
        sampled_low = np.array(
            [
                _as_log_bound(r.get("bound_low", ""), -math.inf)
                for r in sampled_rows
            ]
        )
        sampled_high = np.array(
            [
                _as_log_bound(r.get("bound_high", ""), math.inf)
                for r in sampled_rows
            ]
        )
        try:
            sampled_params, _ = _fit(sampled_x, sampled_low, sampled_high)
        except RuntimeError:
            continue
        bootstrap_slopes.append(float(sampled_params[1]))
    bootstrap_slopes_array = np.asarray(bootstrap_slopes)

    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in included:
        by_model[row["model"]].append(row)
    original_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in original_rows:
        if row["analysis_included"].lower() == "true":
            original_by_model[row["model"]].append(row)

    def class_tuple(rows: list[dict[str, str]]) -> str:
        counts = Counter(r["class"] for r in rows)
        return "/".join(
            str(counts[name])
            for name in (
                "resolved",
                "below-start",
                "ceiling-stable",
                "unresolved-mixed",
                "blank-dead",
            )
        )

    model_rows = []
    for model, rows in sorted(by_model.items()):
        counts = Counter(r["class"] for r in rows)
        ratios = [float(r["rel"]) for r in rows if r.get("rel")]
        model_rows.append(
            {
                "model": model,
                "original_R_L_U_M_B": class_tuple(original_by_model[model]),
                "expanded_R_L_U_M_B": class_tuple(rows),
                "resolved": counts["resolved"],
                "below_minimum": counts["below-start"],
                "above_maximum": counts["ceiling-stable"],
                "mixed": counts["unresolved-mixed"],
                "blank_dead": counts["blank-dead"],
                "median_resolved_ratio": round(float(np.median(ratios)), 4) if ratios else "",
                "min_resolved_ratio": round(min(ratios), 4) if ratios else "",
                "max_resolved_ratio": round(max(ratios), 4) if ratios else "",
            }
        )
    with args.model_out.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)

    latex = [
        r"\begin{longtable}{@{}lccr@{}}",
        r"\caption{\textbf{Model-level bracket sensitivity.} Counts among the six ladder tasks before and after extending scans down to the task minimum and up to $4\times$ the CC-scaled prediction. Each tuple is resolved/left-censored/right-censored/mixed/blank (R/L/U/M/B); the last column is the median measured/predicted ratio among resolved expanded scans.}\label{tab:ladder_model_brackets}\\",
        r"\toprule",
        r"Model & Original R/L/U/M/B & Expanded R/L/U/M/B & Median ratio \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Model & Original R/L/U/M/B & Expanded R/L/U/M/B & Median ratio \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for row in model_rows:
        ratio = row["median_resolved_ratio"] or r"--"
        display_model = re.sub(r"(?<=\d)p(?=\d)", ".", row["model"])
        latex.append(
            rf"\texttt{{{display_model}}} & {row['original_R_L_U_M_B']} & "
            rf"{row['expanded_R_L_U_M_B']} & {ratio} \\"
        )
    latex.append(r"\end{longtable}")
    args.latex_out.write_text("\n".join(latex) + "\n", encoding="utf-8")

    class_counts = Counter(r["class"] for r in included)
    original_included = [
        r for r in original_rows if r["analysis_included"].lower() == "true"
    ]
    original_class_counts = Counter(r["class"] for r in original_included)
    original_by_key = {(r["scan"], r["model"]): r for r in original_included}
    transition_counts = Counter(
        (original_by_key[(r["scan"], r["model"])]["class"], r["class"])
        for r in included
    )
    payload = {
        "analysis_included": len(included),
        "excluded_tasks": {
            "tasks": sorted(excluded_tasks),
            "dropped": len(task_dropped),
            "dropped_by_class": dict(Counter(r["class"] for r in task_dropped)),
        },
        "format_limit": {
            "min_boundary": args.min_boundary,
            "dropped": len(dropped),
            "dropped_by_class": dict(Counter(r["class"] for r in dropped)),
            "dropped_by_model": dict(Counter(r["model"] for r in dropped)),
            "dropped_scans": sorted(
                f"{r['scan']}/{r['model']}" for r in dropped
            ),
        },
        "probe_counts": {
            "original": sum(int(r["n_probes"]) for r in original_included),
            "expanded": sum(int(r["n_probes"]) for r in included),
            "added": (
                sum(int(r["n_probes"]) for r in included)
                - sum(int(r["n_probes"]) for r in original_included)
            ),
        },
        "original_class_counts": dict(sorted(original_class_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "class_transitions": {
            f"{before} -> {after}": count
            for (before, after), count in sorted(transition_counts.items())
        },
        "informative_bound_count": len(informative),
        "mixed_interval_sensitivity_count": len(informative_all) - len(informative),
        "resolved_only": {
            "n": len(resolved),
            "spearman_rho": resolved_rho,
            "median_boundary_prediction_ratio": median_ratio,
        },
        "censor_limit_sensitivity": {
            "n": len(informative),
            "spearman_rho": limit_rho,
            "substitution_rule": "resolved boundary or observed one-sided censoring limit",
        },
        "censored_gaussian_regression": {
            "n": len(informative),
            "intercept": alpha,
            "slope": beta,
            "residual_sigma_log": sigma,
            "log_likelihood": loglike,
            "null_log_likelihood": null_loglike,
            "likelihood_ratio": lr,
            "p_value_beta_zero": p_value,
            "outcome_scale": "natural log boundary",
            "predictor_scale": "natural log CC-scaled prediction",
        },
        "model_cluster_bootstrap": {
            "models": len(bootstrap_models),
            "draws_requested": CLUSTER_BOOTSTRAPS,
            "draws_completed": len(bootstrap_slopes),
            "seed": SEED + 1,
            "slope_median": float(np.median(bootstrap_slopes_array)),
            "slope_p2_5": float(np.quantile(bootstrap_slopes_array, 0.025)),
            "slope_p97_5": float(np.quantile(bootstrap_slopes_array, 0.975)),
            "note": "nonparametric bootstrap resampling models with all six task scans as a cluster",
        },
        "mixed_interval_inclusive_regression_sensitivity": {
            "n": len(informative_all),
            "intercept": float(inclusive_params[0]),
            "slope": float(inclusive_params[1]),
            "residual_sigma_log": math.exp(float(inclusive_params[2])),
            "log_likelihood": inclusive_loglike,
            "null_log_likelihood": inclusive_null_loglike,
            "likelihood_ratio": inclusive_lr,
            "p_value_beta_zero": float(chi2.sf(inclusive_lr, 1)),
        },
        "censored_multiple_imputation": {
            "n": len(informative),
            "draws": IMPUTATIONS,
            "seed": SEED,
            "spearman_rho_median": float(np.median(rhos)),
            "spearman_rho_p2_5": float(np.quantile(rhos, 0.025)),
            "spearman_rho_p97_5": float(np.quantile(rhos, 0.975)),
            "note": "conditional imputation range under the censored Gaussian model, not a sampling confidence interval",
        },
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {OUT}")
    print(f"wrote {MODEL_OUT}")
    print(f"wrote {LATEX_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
