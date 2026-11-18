"""Offline subspace and periodic-feature readouts of the decode-point state.

Runs on the caches; no model, no GPU. For every hooked site it fits, on the
stable regime of the multi-prompt cache (`collect_multi.py`) with folds that
hold out whole counts:

  linear       ridge readout of n (reference) and its agreement with the
               paper's min-norm direction fitted on the single-prompt cache
  periodic     ridge readouts of cos/sin(2 pi n / P) for periods with at least
               three cycles inside the regime, against a smoothness null (the
               same target computed from the held-out linear readout)
  digits       ten-way last-digit and tens-digit one-vs-rest ridge classifiers,
               with a count-level permutation null and the neighbour-confusion
               rate (a smooth analog code mistakes a digit for its neighbours)
  trajectory   PCA of the count-conditioned mean states (rank <= counts - 1)

and writes orthonormal hypothesis bases for the causal script
(`subspace_intervene.py`):

  L   the paper's linear count direction (k = 1)
  F   the readable periodic directions
  G   the digit-class directions
  T   the trajectory principal components, full rank (slice for T_k)
  U   the decoder's answer-token unembedding subspace (needs `unembed.npz`)
  R   random subspaces inside span(T), regenerated from the seed (`random_bases`)

    python -m counting.probes.suite.subspace \
        --collect-dir results/suite/gpt-oss-20b/collect-multi \
        --single-collect-dir results/suite/gpt-oss-20b/collect \
        --scan-dir results/suite/gpt-oss-20b/scan-final \
        --interventions-dir results/suite/gpt-oss-20b/interventions \
        --output-dir results/suite/gpt-oss-20b/subspace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..common.directions import fit_counter_direction
from .util import correctness_by_count, scanned_scc

K_GRID = (1, 2, 4, 8, 16, 32)
PERIODS = (2, 5, 10, 20)
MIN_CYCLES = 2.5  # gpt-oss-20b has 27 stable counts: P = 10 gives 2.7 cycles
READABLE_R2 = 0.5
READABLE_MARGIN = 0.3
# Wider than geometry.LAMBDA_GRID: with several prompts per count, a weak ridge
# memorises each training count's centroid in the noise dimensions and fails
# on held-out counts; the held-out score has to be free to pick a strong penalty.
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
N_PERMUTATIONS = 20


# -- ridge in dual form, multi-target, grouped folds ---------------------------

def ridge_multi(x: np.ndarray, y: np.ndarray, lam: float, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (W, x_center, y_center) with predictions (x - x_center) @ W + y_center.

    Dual form: cost O(n^3) regardless of hidden size; `lam` is relative to the
    mean kernel eigenvalue, as in `geometry.ridge_fit`. With sample `weights`
    (mean 1) the fit minimises sum_i w_i (y_i - f(x_i))^2 + lam |W|^2, via
    W = Xc^T (Xc Xc^T + lam * scale * diag(1/w))^-1 (y - y_center), with
    weighted centering.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if weights is None:
        weights = np.ones(len(y))
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights * (len(weights) / weights.sum())
    x_center = (weights[:, None] * x).sum(axis=0) / len(y)
    y_center = (weights[:, None] * y).sum(axis=0) / len(y)
    xc = x - x_center
    kernel = xc @ xc.T
    scale = max(float(np.trace(kernel)) / len(y), 1e-12)
    alpha = np.linalg.solve(kernel + lam * scale * np.diag(1.0 / weights), y - y_center)
    return xc.T @ alpha, x_center, y_center


def count_folds(counts: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per sample such that every sample of a count shares a fold."""
    unique = np.unique(counts)
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of_count = {int(unique[i]): int(pos % k) for pos, i in enumerate(order)}
    return np.array([fold_of_count[int(c)] for c in counts])


def grouped_cv(x: np.ndarray, y: np.ndarray, folds: np.ndarray, lam: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    squeeze = y.ndim == 1
    if squeeze:
        y = y[:, None]
    out = np.full(y.shape, np.nan)
    for fold in np.unique(folds):
        test = folds == fold
        w, xc, yc = ridge_multi(x[~test], y[~test], lam)
        out[test] = (np.asarray(x[test], dtype=np.float64) - xc) @ w + yc
    return out[:, 0] if squeeze else out


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    var = float(np.var(y))
    return 1.0 - float(np.mean((pred - y) ** 2)) / max(var, 1e-12)


def choose_lambda_grouped(x: np.ndarray, y: np.ndarray, folds: np.ndarray, grid=LAMBDA_GRID) -> tuple[float, float, np.ndarray]:
    best = (grid[0], -np.inf, None)
    for lam in grid:
        pred = grouped_cv(x, y, folds, lam)
        score = r2(pred, y)
        if score > best[1]:
            best = (lam, score, pred)
    return best


# -- bases ---------------------------------------------------------------------

def orthonormal_rows(vectors: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """Orthonormal basis (rows) of the row space of `vectors`."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.size == 0:
        return np.zeros((0, vectors.shape[-1]), dtype=np.float32)
    _, s, vt = np.linalg.svd(vectors, full_matrices=False)
    keep = s > rel_tol * max(float(s[0]), 1e-30)
    return vt[keep].astype(np.float32)


def trajectory_basis(x: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA of count-conditioned means: (components rows, explained ratio, means)."""
    unique = np.unique(counts)
    means = np.stack([x[counts == c].mean(axis=0) for c in unique]).astype(np.float64)
    centered = means - means.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    keep = s > 1e-6 * max(float(s[0]), 1e-30)
    explained = (s * s) / max(float((s * s).sum()), 1e-30)
    return vt[keep].astype(np.float32), explained[keep].astype(np.float32), means.astype(np.float32)


def random_bases(span: np.ndarray, n_draws: int, seed: int) -> list[np.ndarray]:
    """Random orthonormal bases of span(rows of `span`); slice rows for R_k.

    Deterministic in (seed, draw): the causal script regenerates them rather
    than reading them from disk.
    """
    rank = span.shape[0]
    out = []
    for draw in range(n_draws):
        gauss = np.random.default_rng(seed + 1000 * draw).standard_normal((rank, rank))
        q, _ = np.linalg.qr(gauss)
        out.append((q.T @ span.astype(np.float64)).astype(np.float32))
    return out


def ambient_random_basis(hidden: int, k: int, seed: int) -> np.ndarray:
    gauss = np.random.default_rng(seed).standard_normal((hidden, k))
    q, _ = np.linalg.qr(gauss)
    return q.T.astype(np.float32)


def principal_cosines(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Cosines of the principal angles between two orthonormal row bases."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return []
    s = np.linalg.svd(a.astype(np.float64) @ b.astype(np.float64).T, compute_uv=False)
    return [round(float(v), 4) for v in np.clip(s, 0, 1)]


# -- readouts ------------------------------------------------------------------

def residualize(x: np.ndarray, n: np.ndarray, degree: int = 3) -> np.ndarray:
    """Remove the best polynomial trend in n (degree 3) from every coordinate.

    With folds that hold out whole counts, a readout of any function of n can
    lean on the smooth count axis: the training counts of a digit class have a
    different mean count from the held-out one. Removing a cubic trend, which
    cannot represent n mod 10 over several cycles, closes that leak; a genuine
    periodic or digit code survives it.
    """
    x = np.asarray(x, dtype=np.float64)
    z = (n - n.mean()) / max(float(n.std()), 1e-9)
    design = np.column_stack([z ** d for d in range(degree + 1)])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    return (x - design @ coef).astype(np.float32)


def periodic_readouts(x: np.ndarray, n: np.ndarray, folds: np.ndarray, n_hat: np.ndarray, x_resid: np.ndarray | None = None) -> tuple[dict, dict]:
    n_distinct = len(np.unique(n))
    results: dict[str, dict] = {}
    fits: dict[int, dict] = {}
    for period in PERIODS:
        cycles = n_distinct / period
        entry: dict = {"cycles": round(cycles, 1), "tested": cycles >= MIN_CYCLES}
        if not entry["tested"]:
            results[str(period)] = entry
            continue
        # For P = 2 the sine phase is identically zero on integers: only the
        # cosine (parity) phase exists, and the fitted pair has one row.
        phases = [np.cos] if period == 2 else [np.cos, np.sin]
        lams, direct, null, resid = [], [], [], []
        for f in phases:
            y = f(2 * np.pi * n / period)
            lam, score, _ = choose_lambda_grouped(x, y, folds)
            lams.append(lam)
            direct.append(round(score, 4))
            null.append(round(r2(f(2 * np.pi * n_hat / period), y), 4))
            if x_resid is not None:
                resid.append(round(choose_lambda_grouped(x_resid, y, folds)[1], 4))
        if resid:
            entry.update({"r2_cos_resid": resid[0], "r2_sin_resid": resid[1] if len(resid) > 1 else None})
        # `readable` is the absolute criterion (both phases held-out R^2 >= 0.5);
        # `beyond_null` additionally asks that the readout beat the smoothness
        # null by a margin -- a precise linear readout makes any smooth function
        # of n readable, so the two flags are reported separately.
        entry.update({
            "r2_cos": direct[0], "r2_sin": direct[1] if len(direct) > 1 else None,
            "null_cos": null[0], "null_sin": null[1] if len(null) > 1 else None,
            "readable": bool(min(direct) >= READABLE_R2),
            "beyond_null": bool(min(np.subtract(direct, null)) >= READABLE_MARGIN),
        })
        results[str(period)] = entry
        targets = np.column_stack([f(2 * np.pi * n / period) for f in phases])
        w, xc, yc = ridge_multi(x, targets, float(np.mean(lams)))
        fits[period] = {"w": w.T.astype(np.float32), "xc": xc.astype(np.float32), "yc": yc.astype(np.float32)}
    return results, fits


LDA_DIMS = 200
LDA_SHRINKAGE = (1e-2, 1e-1, 1.0, 10.0)


def lda_prepare(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA of a training set: (x_center, components (m, hidden), scores (n, m))."""
    x = np.asarray(x, dtype=np.float64)
    x_center = x.mean(axis=0)
    xc = x - x_center
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    m = min(LDA_DIMS, int((s > 1e-8 * s[0]).sum()))
    comps = vt[:m]
    return x_center, comps, xc @ comps.T


def lda_scores(z: np.ndarray, labels: np.ndarray, classes: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    """Shrinkage LDA in PCA coordinates: (class directions (classes, m), biases).

    score_c(z) = z @ directions[c] + bias[c]. Whitening uses the *within-class*
    covariance (a one-vs-rest ridge whitens by the total covariance, which
    shrinks exactly the between-class directions that carry a digit code),
    shrunk toward its mean eigenvalue by `shrink`.
    """
    m = z.shape[1]
    means = np.stack([z[labels == c].mean(axis=0) for c in classes])
    within = np.zeros((m, m))
    for i, c in enumerate(classes):
        d = z[labels == c] - means[i]
        within += d.T @ d
    within /= max(len(z) - len(classes), 1)
    within += shrink * (np.trace(within) / m) * np.eye(m)
    inv_means = np.linalg.solve(within, means.T).T
    return inv_means, -0.5 * np.einsum("ij,ij->i", inv_means, means)


def lda_fit(x: np.ndarray, labels: np.ndarray, classes: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LDA on the full set; returns (directions in state space, bias, x_center, components)."""
    x_center, comps, z = lda_prepare(x)
    inv_means, bias = lda_scores(z, labels, classes, shrink)
    return inv_means @ comps, bias, x_center, comps


def digit_readouts(x: np.ndarray, labels: np.ndarray, counts: np.ndarray, folds: np.ndarray, seed: int) -> tuple[dict, np.ndarray]:
    """Shrinkage-LDA digit classifier with count-held-out folds; returns (metrics, directions)."""
    classes = np.unique(labels)
    if len(classes) < 3:
        return {"tested": False, "n_classes": int(len(classes))}, np.zeros((0, x.shape[1]), np.float32)

    # The PCA of each training fold does not depend on the labels, so it is
    # shared across shrinkage values and permutation draws.
    prepared = {}
    for fold in np.unique(folds):
        test = folds == fold
        x_center, comps, z_train = lda_prepare(x[~test])
        prepared[fold] = (test, z_train, (np.asarray(x[test], dtype=np.float64) - x_center) @ comps.T)

    def accuracy(lab: np.ndarray, shrink: float) -> tuple[float, np.ndarray]:
        pred = np.empty(len(lab), dtype=classes.dtype)
        for fold, (test, z_train, z_test) in prepared.items():
            present = np.unique(lab[~test])
            inv_means, bias = lda_scores(z_train, lab[~test], present, shrink)
            pred[test] = present[(z_test @ inv_means.T + bias).argmax(axis=1)]
        return float((pred == lab).mean()), pred

    best_lam, best_acc, best_pred = LDA_SHRINKAGE[0], -1.0, None
    for lam in LDA_SHRINKAGE:
        acc, pred = accuracy(labels, lam)
        if acc > best_acc:
            best_lam, best_acc, best_pred = lam, acc, pred

    # Permutation null at the count level: every prompt of a count gets the
    # label of another count, so the within-count structure is preserved.
    unique_counts = np.unique(counts)
    label_of_count = {int(c): int(labels[counts == c][0]) for c in unique_counts}
    rng = np.random.default_rng(seed)
    perms = []
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(unique_counts)
        mapping = {int(c): label_of_count[int(s)] for c, s in zip(unique_counts, shuffled)}
        perms.append(accuracy(np.array([mapping[int(c)] for c in counts]), best_lam)[0])

    modulus = 10 if len(classes) == 10 else None
    if modulus:
        diff = (best_pred.astype(int) - labels.astype(int)) % modulus
        neighbour = float(np.mean((diff == 1) | (diff == modulus - 1)))
    else:
        neighbour = float(np.mean(np.abs(best_pred.astype(int) - labels.astype(int)) == 1))

    directions, _, _, _ = lda_fit(x, labels, classes, best_lam)
    metrics = {
        "tested": True, "n_classes": int(len(classes)), "shrinkage": best_lam,
        "accuracy": round(best_acc, 4),
        "chance": round(float(np.bincount(labels.astype(int)).max()) / len(labels), 4),
        "perm_mean": round(float(np.mean(perms)), 4), "perm_max": round(float(np.max(perms)), 4),
        "neighbour_confusion": round(neighbour, 4),
    }
    return metrics, orthonormal_rows(directions)


def restricted_logit_lens(states: np.ndarray, counts: np.ndarray, unembed: dict) -> dict:
    """Margin of the correct first answer token against the other answer tokens.

    `rows` are the gamma-scaled unembedding rows of the answer tokens, so the
    logit of token t is rms-normalised(h) . rows[t]; the final-norm epsilon is
    negligible at these norms.
    """
    rows = unembed["rows"].astype(np.float64)
    ids = unembed["token_ids"]
    first = unembed["first_token_by_count"]
    x = states.astype(np.float64)
    normed = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6)
    logits = normed @ rows.T
    margins = []
    for i, c in enumerate(counts):
        tid = int(first[int(c)]) if int(c) < len(first) else -1
        if tid < 0 or tid not in ids:
            continue
        j = int(np.where(ids == tid)[0][0])
        other = np.delete(logits[i], j).max()
        margins.append(float(logits[i, j] - other))
    margins = np.array(margins)
    return {
        "n": int(len(margins)),
        "fraction_positive": round(float((margins > 0).mean()), 4) if len(margins) else None,
        "mean_margin": round(float(margins.mean()), 4) if len(margins) else None,
    }


# -- driver --------------------------------------------------------------------

def load_stable(collect_dir: Path, scan_dir: Path) -> dict:
    cache = np.load(collect_dir / "activations.npz")
    counts = cache["counts"].astype(np.int64)
    scc = scanned_scc(scan_dir)
    if "greedy_exact" in cache.files:
        stable = cache["greedy_exact"].astype(bool) & (counts <= scc)
        multi = True
    else:
        correct = correctness_by_count(scan_dir)
        stable = np.array([c <= scc and correct.get(int(c), False) for c in counts])
        multi = False
    hooks = [k.split("|")[0] for k in cache.files if k.endswith("|decode_point")]
    return {"cache": cache, "counts": counts, "stable": stable, "scc": scc, "hooks": hooks, "multi": multi}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--collect-dir", type=Path, required=True, help="multi-prompt cache (or single)")
    parser.add_argument("--single-collect-dir", type=Path, help="the item-'a' cache used by the paper")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--interventions-dir", type=Path, help="for the chimera base and donors")
    parser.add_argument("--unembed", type=Path, help="unembed.npz written by subspace_intervene")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_stable(args.collect_dir, args.scan_dir)
    single = load_stable(args.single_collect_dir, args.scan_dir) if args.single_collect_dir else None
    unembed = dict(np.load(args.unembed)) if args.unembed and args.unembed.exists() else None
    iv_meta = (
        json.loads((args.interventions_dir / "meta.json").read_text())
        if args.interventions_dir and (args.interventions_dir / "meta.json").exists()
        else None
    )

    counts, stable = data["counts"], data["stable"]
    ns = counts[stable].astype(np.float64)
    folds = count_folds(counts[stable], seed=0)
    print(
        f"{args.collect_dir}: {int(stable.sum())} stable samples over "
        f"{len(np.unique(ns))} counts (scc={data['scc']}, multi={data['multi']})"
    )

    readouts: dict = {
        "scc": data["scc"], "multi": data["multi"], "n_stable": int(stable.sum()),
        "n_counts": int(len(np.unique(ns))), "k_grid": list(K_GRID), "seed": args.seed, "sites": {},
    }
    bases: dict[str, np.ndarray] = {}

    for hook in data["hooks"]:
        x = data["cache"][f"{hook}|decode_point"].astype(np.float32)[stable]
        site: dict = {}

        # Linear reference and the paper's direction.
        lam, cv_r2, n_hat = choose_lambda_grouped(x, ns, folds)
        w_lin, _, _ = ridge_multi(x, ns, lam)
        w_lin = w_lin[:, 0]
        site["linear"] = {
            "lambda": lam, "cv_r2": round(cv_r2, 4),
            "cv_rmse_items": round(float(np.sqrt(np.mean((n_hat - ns) ** 2))), 3),
        }
        if single is not None and f"{hook}|decode_point" in single["cache"].files:
            s_states = single["cache"][f"{hook}|decode_point"].astype(np.float32)
            paper = fit_counter_direction(s_states, single["counts"].astype(np.float32), single["stable"])
            u = paper.direction.astype(np.float64)
            site["linear"]["cos_multi_ridge_vs_paper_u"] = round(
                float(abs(w_lin @ u) / (np.linalg.norm(w_lin) * np.linalg.norm(u))), 4
            )
            bases[f"{hook}|L"] = (u / np.linalg.norm(u)).astype(np.float32)[None, :]
            bases[f"{hook}|mu_single"] = s_states[single["stable"]].mean(axis=0).astype(np.float32)
        else:
            bases[f"{hook}|L"] = (w_lin / np.linalg.norm(w_lin)).astype(np.float32)[None, :]
        bases[f"{hook}|L_multi"] = (w_lin / np.linalg.norm(w_lin)).astype(np.float32)[None, :]
        bases[f"{hook}|mu_multi"] = x.mean(axis=0).astype(np.float32)

        # Periodic features and the F basis (fitted on the raw states so the
        # rotation edit acts on the readout the state actually carries).
        x_resid = residualize(x, ns)
        site["periodic"], fits = periodic_readouts(x, ns, folds, n_hat, x_resid)
        readable = [p for p in PERIODS if site["periodic"][str(p)].get("readable")]
        f_rows = [fits[p]["w"] for p in readable]
        bases[f"{hook}|F"] = orthonormal_rows(np.concatenate(f_rows)) if f_rows else np.zeros((0, x.shape[1]), np.float32)
        site["periodic"]["readable_periods"] = readable
        for p, fit in fits.items():
            bases[f"{hook}|P{p}_w"] = fit["w"]
            bases[f"{hook}|P{p}_xc"] = fit["xc"]
            bases[f"{hook}|P{p}_yc"] = fit["yc"]

        # Digit classes: raw states carry the neighbour-confusion signature of
        # a smooth code; residualized states give the digit code beyond the
        # smooth trend, and the G basis comes from that fit.
        last_raw, _ = digit_readouts(x, (ns.astype(int) % 10), ns.astype(int), folds, args.seed)
        last_res, g_last = digit_readouts(x_resid, (ns.astype(int) % 10), ns.astype(int), folds, args.seed)
        tens_raw, _ = digit_readouts(x, (ns.astype(int) // 10), ns.astype(int), folds, args.seed)
        tens_res, g_tens = digit_readouts(x_resid, (ns.astype(int) // 10), ns.astype(int), folds, args.seed)
        site["digits"] = {"last": last_raw, "last_resid": last_res, "tens": tens_raw, "tens_resid": tens_res}
        bases[f"{hook}|G"] = orthonormal_rows(np.concatenate([g_last, g_tens])) if len(g_last) + len(g_tens) else np.zeros((0, x.shape[1]), np.float32)
        bases[f"{hook}|G_last"] = g_last

        # Trajectory subspace.
        t_basis, evr, means = trajectory_basis(x, ns.astype(int))
        bases[f"{hook}|T"] = t_basis
        site["trajectory"] = {
            "rank": int(t_basis.shape[0]),
            "evr_top16": [round(float(v), 4) for v in evr[:16]],
            "cum_evr": {str(k): round(float(evr[:k].sum()), 4) for k in K_GRID if k <= len(evr)},
        }

        # Decoder subspace and principal angles.
        if unembed is not None:
            u_basis = orthonormal_rows(unembed["rows"])
            bases[f"{hook}|U"] = u_basis
            site["decoder"] = {
                "rank_U": int(u_basis.shape[0]),
                "cos_L_U": principal_cosines(bases[f"{hook}|L"], u_basis),
                "cos_F_U": principal_cosines(bases[f"{hook}|F"], u_basis),
                "cos_G_U": principal_cosines(bases[f"{hook}|G"], u_basis),
                "cos_T8_U": principal_cosines(t_basis[:8], u_basis),
                "cos_L_T8": principal_cosines(bases[f"{hook}|L"], t_basis[:8]),
            }
            if single is not None and f"{hook}|decode_point" in single["cache"].files:
                s_states = single["cache"][f"{hook}|decode_point"].astype(np.float32)
                site["decoder"]["logit_lens_stable"] = restricted_logit_lens(
                    s_states[single["stable"]], single["counts"][single["stable"]], unembed
                )

        # Chord diagnostics against the paper's donors.
        if single is not None and iv_meta is not None and f"{hook}|decode_point" in single["cache"].files:
            s_states = single["cache"][f"{hook}|decode_point"].astype(np.float64)
            s_counts = single["counts"]
            base_idx = int(np.where(s_counts == iv_meta["base_count"])[0][0])
            donor_idx = [int(np.where(s_counts == d)[0][0]) for d in iv_meta["donor_counts"]]
            deltas = s_states[donor_idx] - s_states[base_idx]
            total = (deltas ** 2).sum(axis=1)
            u = bases[f"{hook}|L"][0].astype(np.float64)
            site["chord"] = {
                "energy_in_L": round(float(np.mean((deltas @ u) ** 2 / total)), 4),
                "energy_in_T": {
                    str(k): round(float(np.mean(((deltas @ t_basis[:k].T.astype(np.float64)) ** 2).sum(axis=1) / total)), 4)
                    for k in list(K_GRID) + [t_basis.shape[0]] if k <= t_basis.shape[0]
                },
            }
            if f"{hook}|U" in bases:
                ub = bases[f"{hook}|U"].astype(np.float64)
                site["chord"]["energy_in_U"] = round(float(np.mean(((deltas @ ub.T) ** 2).sum(axis=1) / total)), 4)
            if bases[f"{hook}|G"].shape[0]:
                gb = bases[f"{hook}|G"].astype(np.float64)
                site["chord"]["energy_in_G"] = round(float(np.mean(((deltas @ gb.T) ** 2).sum(axis=1) / total)), 4)

        readouts["sites"][hook] = site
        d, dr = site["digits"]["last"], site["digits"]["last_resid"]
        print(
            f"  {hook}: linear R2 {cv_r2:.3f}; last digit raw {d.get('accuracy')} "
            f"(neighbour {d.get('neighbour_confusion')}) resid {dr.get('accuracy')} "
            f"(perm {dr.get('perm_mean')}); readable periods {readable}; T rank {t_basis.shape[0]}"
        )

    (args.output_dir / "readouts.json").write_text(json.dumps(readouts, indent=1))
    np.savez_compressed(args.output_dir / "bases.npz", **bases)
    print(f"wrote {args.output_dir / 'readouts.json'} and bases.npz ({len(bases)} arrays)")


if __name__ == "__main__":
    main()
