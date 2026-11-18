"""Fitting and using the linear "progress coordinate".

The central mechanistic claim of the paper is that a model's position in a count
is linearly readable off the residual stream inside the stable regime. That claim
rests on one fit, implemented here and shared by every probe:

    fit_counter_direction(states, counts, fit_mask)

`fit_mask` selects the *successful* trials. Fitting on those and then projecting
everything -- successes and failures alike -- is what makes the divergence past
the boundary meaningful. Fitting on all trials would let the failures drag the
direction toward whatever the collapsed state looks like and hide the effect.

Pure numpy: no torch, no model, so it is testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CounterDirection:
    """A fitted 1-D progress coordinate."""

    center: np.ndarray
    """Mean of the fitted states; subtracted before projecting."""

    direction: np.ndarray
    """Unnormalised least-squares direction from state to count."""

    projected: np.ndarray
    """Projection of every supplied state, fitted or not."""

    rho: float
    """Autoregressive slope of the projected series, z[t+1] ~ rho*z[t] + delta."""

    delta: float

    fit_slope: float
    """Slope of the count -> projection line fitted over the fitted points.

    Note the direction: `projection ~= fit_intercept + fit_slope * count`. The
    clamping experiment needs to turn a count into the projection it *should*
    have, so this is the orientation stored. Use `count_from_projection` for the
    inverse.
    """

    fit_intercept: float

    def project(self, states: np.ndarray) -> np.ndarray:
        """Project new states onto the coordinate."""
        return ((states - self.center.reshape(1, -1)) @ self.direction).astype(np.float32)

    def target_projection(self, count: int | np.ndarray) -> np.ndarray:
        """The projection a given count *should* have, extrapolated from the fit.

        This is what counter-direction clamping drives a failed state toward.
        """
        return np.asarray(
            self.fit_intercept + self.fit_slope * np.asarray(count, dtype=np.float32),
            dtype=np.float32,
        )

    def count_from_projection(self, projection: float | np.ndarray) -> np.ndarray:
        """Read a count back off the coordinate. Inverse of `target_projection`."""
        if abs(self.fit_slope) < 1e-12:
            return np.full_like(np.asarray(projection, dtype=np.float32), np.nan)
        return (
            (np.asarray(projection, dtype=np.float32) - self.fit_intercept) / self.fit_slope
        ).astype(np.float32)

    def as_dict(self) -> dict[str, object]:
        return {
            "center": self.center.tolist(),
            "direction": self.direction.tolist(),
            "rho": self.rho,
            "delta": self.delta,
            "fit_slope": self.fit_slope,
            "fit_intercept": self.fit_intercept,
        }


def fit_counter_direction(
    states: np.ndarray,
    counts: np.ndarray,
    fit_mask: np.ndarray | None = None,
) -> CounterDirection:
    """Least-squares direction from residual state to count.

    states     (n_trials, hidden_size)
    counts     (n_trials,)
    fit_mask   boolean (n_trials,) selecting the trials to fit on; default all.
    """
    states = np.asarray(states, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.float32)
    if fit_mask is None:
        fit_mask = np.ones(len(counts), dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if fit_mask.sum() < 2:
        raise ValueError("need at least two fitted points to fit a counter direction")

    fit_x = states[fit_mask]
    fit_y = counts[fit_mask]
    center = fit_x.mean(axis=0, keepdims=True)
    direction, *_ = np.linalg.lstsq(fit_x - center, fit_y - float(fit_y.mean()), rcond=None)
    projected = ((states - center) @ direction).astype(np.float32)

    # Autoregressive structure of the projected series: rho near 1 means the
    # coordinate advances by a roughly constant step per item, which is the
    # analog, unary-like geometry the paper describes.
    z0, z1 = projected[:-1], projected[1:]
    if len(z0) >= 2:
        design = np.column_stack([z0, np.ones_like(z0)])
        rho, delta = np.linalg.lstsq(design, z1, rcond=None)[0]
    else:
        rho, delta = float("nan"), float("nan")

    # projection = intercept + slope * count, over the fitted points only.
    # This orientation (count -> projection) is what the clamping experiment
    # extrapolates: given a failed count, what projection should it have had.
    design = np.column_stack([fit_y, np.ones_like(fit_y)])
    fit_slope, fit_intercept = np.linalg.lstsq(design, projected[fit_mask], rcond=None)[0]

    return CounterDirection(
        center=center.reshape(-1).astype(np.float32),
        direction=direction.astype(np.float32),
        projected=projected,
        rho=float(rho),
        delta=float(delta),
        fit_slope=float(fit_slope),
        fit_intercept=float(fit_intercept),
    )


def fit_principal_components(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """PCA coordinates and explained-variance ratios, via SVD."""
    x = np.asarray(x, dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    explained = (s * s) / max(float((s * s).sum()), 1e-8)
    return coords.astype(np.float32), explained[:n_components].astype(np.float32)


def project_position_window(
    window: np.ndarray, center: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Project a (trials, positions, hidden) window onto a direction."""
    centered = np.asarray(window, dtype=np.float32) - center.reshape(1, 1, -1)
    return np.tensordot(centered, direction, axes=([2], [0])).astype(np.float32)


# -- correlation helpers -----------------------------------------------------

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    return float((xc * yc).sum() / denominator) if denominator > 1e-12 else 0.0


def pearson_by_feature(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Correlation of every feature column with `target`, vectorised.

    features (n_samples, n_features); returns (n_features,). Constant columns --
    which is most of an SAE dictionary on any given prompt -- yield 0.0 rather
    than NaN, so downstream ranking does not have to special-case them.
    """
    features = np.asarray(features, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    fc = features - features.mean(axis=0, keepdims=True)
    tc = target - target.mean()
    numerator = fc.T @ tc
    denominator = np.sqrt((fc * fc).sum(axis=0) * float((tc * tc).sum()))
    out = np.zeros_like(numerator, dtype=np.float32)
    valid = denominator > 1e-12
    out[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return out


def monotonicity_score(series: np.ndarray) -> float:
    """Fraction of consecutive steps that move in the series' dominant direction.

    1.0 is a strictly monotone feature; 0.5 is a coin flip. Used to separate
    accumulator-like SAE features from oscillating ones.
    """
    series = np.asarray(series, dtype=np.float32)
    if len(series) < 2:
        return float("nan")
    steps = np.diff(series)
    nonzero = steps[steps != 0]
    if nonzero.size == 0:
        return 0.0
    return float(max((nonzero > 0).mean(), (nonzero < 0).mean()))
