"""Subspace-readout and decode-editing checks on synthetic data; no checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from counting.probes.suite.editing import ablate_to_mean, project_delta, rotate_pair
from counting.probes.suite.subspace import (
    ambient_random_basis,
    count_folds,
    digit_readouts,
    grouped_cv,
    orthonormal_rows,
    periodic_readouts,
    principal_cosines,
    r2,
    random_bases,
    residualize,
    ridge_multi,
    trajectory_basis,
)


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        raise AssertionError(label)


def synthetic(n_counts: int = 40, per_count: int = 6, hidden: int = 64, digit_code: bool = True, seed: int = 0):
    """States with a linear count axis, optionally a period-10 code, plus noise."""
    rng = np.random.default_rng(seed)
    counts = np.repeat(np.arange(1, n_counts + 1), per_count)
    # Noise 0.15: at 0.3 the held-out count's own noise centroid (6 samples,
    # 62 noise dims) is as large as the class separation and even a
    # nearest-centroid rule reaches only 0.78; that is a hard instance, not a
    # solver defect (the dual ridge matches the primal to 1e-15).
    x = rng.standard_normal((len(counts), hidden)) * 0.15
    x[:, 0] += counts * 0.5
    if digit_code:
        x[:, 1] += 3 * np.cos(2 * np.pi * counts / 10)
        x[:, 2] += 3 * np.sin(2 * np.pi * counts / 10)
    return x.astype(np.float32), counts


def test_ridge_and_folds() -> None:
    x, counts = synthetic()
    folds = count_folds(counts)
    for c in np.unique(counts):
        check(f"count {c} in one fold", len(np.unique(folds[counts == c])) == 1)
    pred = grouped_cv(x, counts.astype(float), folds, 1e-2)
    check("grouped CV reads the count", r2(pred, counts) > 0.95, f"r2={r2(pred, counts):.3f}")
    w, xc, yc = ridge_multi(x, counts.astype(float), 1e-2)
    check("ridge direction points along the count axis", abs(w[0, 0]) > 5 * np.abs(w[3:, 0]).max())


def test_periodic_and_digits() -> None:
    x, counts = synthetic(digit_code=True)
    folds = count_folds(counts)
    n_hat = grouped_cv(x, counts.astype(float), folds, 1e-2)
    results, fits = periodic_readouts(x, counts.astype(float), folds, n_hat)
    check("period 10 readable when present", results["10"]["readable"], str(results["10"]))
    check("null flag reported", "beyond_null" in results["10"])
    check("period 20 untested with too few cycles", results["20"]["tested"] is False)
    check("period 10 fit has two directions", fits[10]["w"].shape == (2, x.shape[1]))
    x_resid = residualize(x, counts.astype(float))
    check("residualizing removes the count trend", abs(np.corrcoef(x_resid[:, 0], counts)[0, 1]) < 0.05)
    metrics, basis = digit_readouts(x_resid, counts % 10, counts, folds, seed=0)
    check("last digit decodable after residualizing", metrics["accuracy"] > 0.8, str(metrics))
    check("permutation null near chance", metrics["perm_mean"] < 0.3, str(metrics["perm_mean"]))
    check("digit basis is orthonormal", np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-4))
    raw, _ = digit_readouts(x, counts % 10, counts, folds, seed=0)
    check("raw LDA readout also decodes the digit (count axis is within-class spread)", raw["accuracy"] > 0.8, str(raw["accuracy"]))

    smooth, counts2 = synthetic(digit_code=False)
    folds2 = count_folds(counts2)
    metrics2, _ = digit_readouts(smooth, counts2 % 10, counts2, folds2, seed=0)
    check("smooth code confuses neighbours", metrics2["neighbour_confusion"] > metrics2["accuracy"], str(metrics2))
    metrics3, _ = digit_readouts(residualize(smooth, counts2.astype(float)), counts2 % 10, counts2, folds2, seed=0)
    check("no digit code in residualized smooth data", metrics3["accuracy"] < 0.3, str(metrics3["accuracy"]))


def test_bases() -> None:
    x, counts = synthetic()
    t_basis, evr, means = trajectory_basis(x, counts)
    check("trajectory rank below count of counts", t_basis.shape[0] <= len(np.unique(counts)) - 1)
    check("trajectory PC1 is the count axis", abs(t_basis[0, 0]) > 0.9, f"{t_basis[0, 0]:.3f}")
    draws = random_bases(t_basis[:8], 2, seed=7)
    for r in draws:
        check("random basis orthonormal", np.allclose(r @ r.T, np.eye(8), atol=1e-4))
        residual = r - (r @ t_basis[:8].T) @ t_basis[:8]
        check("random basis inside span(T)", float(np.abs(residual).max()) < 1e-4)
    check("random draws are deterministic", np.allclose(random_bases(t_basis[:8], 1, 7)[0], draws[0]))
    check("random draws differ", not np.allclose(draws[0], draws[1]))
    amb = ambient_random_basis(64, 4, seed=1)
    check("ambient basis orthonormal", np.allclose(amb @ amb.T, np.eye(4), atol=1e-5))
    cos = principal_cosines(t_basis[:2], t_basis[:2])
    check("principal cosines of a basis with itself are 1", np.allclose(cos, 1.0, atol=1e-4), str(cos))
    check("orthonormal_rows drops dependent rows", orthonormal_rows(np.stack([t_basis[0], t_basis[0] * 2])).shape[0] == 1)


def test_edits() -> None:
    x, counts = synthetic()
    base, donor = x[10], x[200]
    delta = donor - base
    t_basis, _, _ = trajectory_basis(x, counts)
    full = project_delta(base, delta, orthonormal_rows(np.stack([base, donor, np.ones_like(base)])), False)(base)
    check("full-rank projection reproduces the donor", np.allclose(full, donor, atol=1e-3))
    one = project_delta(base, delta, t_basis[:1], False)(base)
    comp = project_delta(base, delta, t_basis[:1], True)(base)
    check("projection and complement sum to the donor", np.allclose(one + comp - base, donor, atol=1e-3))

    mean = x.mean(axis=0)
    fn = ablate_to_mean(t_basis[:3], mean)
    once, twice = fn(base), fn(fn(base))
    check("mean ablation is idempotent", np.allclose(once, twice, atol=1e-4))
    check("ablation matches the mean inside the subspace",
          np.allclose(t_basis[:3] @ once, t_basis[:3] @ mean, atol=1e-3))

    folds = count_folds(counts)
    n_hat = grouped_cv(x, counts.astype(float), folds, 1e-2)
    _, fits = periodic_readouts(x, counts.astype(float), folds, n_hat)
    w, xc, yc = fits[10]["w"], fits[10]["xc"], fits[10]["yc"]
    h = x[counts == 13][0]
    before = (h - xc) @ w.T + yc
    after = (rotate_pair(w, xc, yc, 2 * np.pi * 3 / 10)(h) - xc) @ w.T + yc
    angle_before, angle_after = np.arctan2(before[1], before[0]), np.arctan2(after[1], after[0])
    shift = (angle_after - angle_before) % (2 * np.pi)
    check("pair rotation advances the readout phase by +3 counts",
          abs(shift - 2 * np.pi * 3 / 10) < 1e-3, f"shift={shift:.4f}")
    w2, xc2, yc2 = fits[2]["w"], fits[2]["xc"], fits[2]["yc"]
    check("period-2 fit has one (parity) row", w2.shape[0] == 1)
    parity_before = (h - xc2) @ w2.T + yc2
    parity_after = (rotate_pair(w2, xc2, yc2, np.pi)(h) - xc2) @ w2.T + yc2
    check("parity rotation by pi negates the readout", np.allclose(parity_after, -parity_before, atol=1e-3))


def main() -> None:
    for test in (test_ridge_and_folds, test_periodic_and_digits, test_bases, test_edits):
        print(test.__name__)
        test()
    print("all subspace checks passed")


if __name__ == "__main__":
    main()
