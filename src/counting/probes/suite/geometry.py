"""Offline geometry analysis of cached residual states.

Everything here runs on the `collect` cache; no model, no GPU.

The paper's original fit (`fit_counter_direction`) is a min-norm least-squares
readout. With hidden size far above the number of stable counts that fit
interpolates its training points exactly, so in-sample R^2 is vacuous. The
honest quantities are cross-validated: a ridge readout whose penalty is chosen
by K-fold CV on the stable regime, and whose held-out predictions give

  cv_rmse            how precisely the count is linearly readable (in items)
  resolution curve   local noise of the held-out readout vs n; where the
                     per-item step (always 1 count) sinks below the noise is
                     the capacity the *linear channel alone* could support
  extrapolation      the stable-regime fit applied beyond the boundary --
                     saturation/collapse of the coordinate in the failed regime

plus the paper-consistent AR(1) rho/delta, PCA spectra, and the overlay of
teacher-forced margin collapse against the behavioral SCC.

    python -m counting.probes.suite.geometry \
        --collect-dir results/suite/gpt-oss-20b/collect \
        --scan-dir results/suite/gpt-oss-20b/scan-final \
        --output results/suite/gpt-oss-20b/geometry.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..common.directions import fit_counter_direction, fit_principal_components
from .util import correctness_by_count

LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


# -- ridge readout (dual form: hidden >> samples) -----------------------------

def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (direction, x_center, y_center) for ridge regression of y on x.

    Solved in sample space: d = Xc^T (Xc Xc^T + lam*tr/n I)^-1 yc, so the cost
    is O(n^3) regardless of hidden size. lam is relative to the mean kernel
    eigenvalue, which keeps the grid meaningful across layers and models.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_center = x.mean(axis=0)
    y_center = float(y.mean())
    xc = x - x_center
    kernel = xc @ xc.T
    scale = max(float(np.trace(kernel)) / len(y), 1e-12)
    alpha = np.linalg.solve(kernel + lam * scale * np.eye(len(y)), y - y_center)
    return xc.T @ alpha, x_center, y_center


def ridge_predict(x: np.ndarray, direction: np.ndarray, x_center: np.ndarray, y_center: float) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - x_center) @ direction + y_center


def cv_predictions(x: np.ndarray, y: np.ndarray, lam: float, k: int = 5, seed: int = 0) -> np.ndarray:
    """Held-out ridge predictions for every sample via K-fold CV."""
    n = len(y)
    order = np.random.default_rng(seed).permutation(n)
    out = np.full(n, np.nan)
    for fold in range(k):
        test = order[fold::k]
        train = np.setdiff1d(order, test)
        direction, xc, yc = ridge_fit(x[train], y[train], lam)
        out[test] = ridge_predict(x[test], direction, xc, yc)
    return out


def choose_lambda(x: np.ndarray, y: np.ndarray, grid=LAMBDA_GRID, k: int = 5) -> tuple[float, float]:
    """Pick the ridge penalty with the best held-out R^2. Returns (lam, cv_r2)."""
    best_lam, best_r2 = grid[0], -np.inf
    var = float(np.var(y))
    for lam in grid:
        pred = cv_predictions(x, y, lam, k=k)
        r2 = 1.0 - float(np.mean((pred - y) ** 2)) / max(var, 1e-12)
        if r2 > best_r2:
            best_lam, best_r2 = lam, r2
    return best_lam, best_r2


# -- resolution curve ----------------------------------------------------------

def resolution_curve(counts: np.ndarray, cv_pred: np.ndarray, window: int = 5) -> list[dict]:
    """Local readout noise and per-item discriminability along the stable regime.

    The step between adjacent counts is 1 (in count units, because the readout
    predicts the count directly), so d'(n) = 1 / local_sigma(n) where sigma is
    the rolling std of the held-out residual.
    """
    residual = cv_pred - counts
    rows = []
    for i, n in enumerate(counts):
        lo, hi = max(0, i - window), min(len(counts), i + window + 1)
        sigma = float(np.std(residual[lo:hi]))
        rows.append(
            {
                "count": int(n),
                "sigma": round(sigma, 4),
                "d_prime": round(1.0 / sigma, 4) if sigma > 1e-9 else float("inf"),
            }
        )
    return rows


def first_crossing(rows: list[dict], threshold: float) -> int | None:
    """First count where d' falls below threshold and stays below for 3 points."""
    for i, row in enumerate(rows):
        if row["d_prime"] < threshold and all(
            r["d_prime"] < threshold for r in rows[i : i + 3]
        ):
            return row["count"]
    return None


# -- per-site analysis ---------------------------------------------------------

def analyse_site(
    states: np.ndarray, counts: np.ndarray, stable: np.ndarray, scc: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Full geometry battery for one (layer, position) site.

    Returns (metrics, extrapolated count-units projection for all counts,
    held-out CV prediction aligned to `counts` with NaN outside the stable set).
    """
    x = states.astype(np.float32)
    x_stable, y_stable = x[stable], counts[stable].astype(np.float64)

    lam, cv_r2 = choose_lambda(x_stable, y_stable)
    cv_pred_stable = cv_predictions(x_stable, y_stable, lam)
    cv_rmse = float(np.sqrt(np.mean((cv_pred_stable - y_stable) ** 2)))

    direction, x_center, y_center = ridge_fit(x_stable, y_stable, lam)
    extrapolated = ridge_predict(x, direction, x_center, y_center)

    curve = resolution_curve(counts[stable], cv_pred_stable)
    cv_aligned = np.full(len(counts), np.nan)
    cv_aligned[stable] = cv_pred_stable

    # Paper-consistent min-norm fit for AR(1) structure.
    paper_fit = fit_counter_direction(x, counts.astype(np.float32), stable)

    # Beyond-regime bands: does the extrapolated coordinate keep advancing,
    # saturate, or collapse once behavior fails?
    bands = {}
    for lo_f, hi_f in ((1.0, 1.2), (1.2, 1.6), (1.6, 2.2)):
        band = (counts > scc * lo_f) & (counts <= scc * hi_f)
        if band.sum() >= 3:
            err = extrapolated[band] - counts[band]
            slope = float(np.polyfit(counts[band], extrapolated[band], 1)[0]) if band.sum() >= 4 else float("nan")
            bands[f"{lo_f:.1f}-{hi_f:.1f}x"] = {
                "n": int(band.sum()),
                "mean_error": round(float(err.mean()), 2),
                "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
                "local_slope": round(slope, 3),
            }

    _, evr = fit_principal_components(x_stable, n_components=8)

    metrics = {
        "lambda": lam,
        "cv_r2": round(cv_r2, 4),
        "cv_rmse_items": round(cv_rmse, 3),
        "rho": round(paper_fit.rho, 4),
        "delta": round(paper_fit.delta, 4),
        "resolution": {
            "curve": curve,
            "capacity_dprime_2": first_crossing(curve, 2.0),
            "capacity_dprime_1": first_crossing(curve, 1.0),
        },
        "beyond_regime": bands,
        "pca_evr_top8": [round(float(v), 4) for v in evr],
    }
    return metrics, extrapolated, cv_aligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--collect-dir", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads((args.collect_dir / "meta.json").read_text())
    margins = json.loads((args.collect_dir / "margins.json").read_text())
    cache = np.load(args.collect_dir / "activations.npz")
    counts = cache["counts"].astype(np.int64)
    scc = int(meta["scc"])

    correct = correctness_by_count(args.scan_dir)
    stable = np.array([c <= scc and correct.get(int(c), False) for c in counts])
    print(f"{meta['name']}: {stable.sum()} stable counts of {len(counts)} (scc={scc})")

    margin_by_count = {m["count"]: m for m in margins}
    first_margin_fail = next((m["count"] for m in margins if m["min_gap"] < 0), None)
    first_greedy_fail = next((m["count"] for m in margins if not m["greedy_exact"]), None)

    sites: dict[str, dict] = {}
    projection_rows: list[dict] = []
    for key in cache.files:
        if key == "counts":
            continue
        label, position = key.split("|")
        metrics, extrapolated, cv_aligned = analyse_site(cache[key], counts, stable, scc)
        sites[key] = metrics
        for i, c in enumerate(counts):
            projection_rows.append(
                {
                    "site": label,
                    "position": position,
                    "count": int(c),
                    "stable": bool(stable[i]),
                    "extrapolated_count": round(float(extrapolated[i]), 3),
                    "cv_pred": round(float(cv_aligned[i]), 3) if np.isfinite(cv_aligned[i]) else "",
                    "min_gap": margin_by_count.get(int(c), {}).get("min_gap"),
                }
            )

    result = {
        "name": meta["name"],
        "scc": scc,
        "n_counts": len(counts),
        "n_stable": int(stable.sum()),
        "margin_collapse": {
            "first_negative_min_gap": first_margin_fail,
            "first_greedy_divergence": first_greedy_fail,
            "behavioral_scc": scc,
        },
        "sites": sites,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))

    csv_path = args.output.with_suffix(".projections.csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(projection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(projection_rows)
    print(f"wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
