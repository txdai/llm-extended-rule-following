"""Capacity as a fitted transition, not a thresholded one.

The scan records, for every length, whether the model answered exactly. The
first wrong answer is a weak summary of that record: 285 of 297 measurements
keep answering correctly above it, and as an extreme statistic it moves with
sampling density -- probe twice as densely and the first slip can only arrive
earlier. A capacity that depends on how hard you looked is not a property of
the model. Threshold repairs (accuracy floor over a window, k misses in a row)
fix that at the price of tuned constants the data cannot justify.

This module fits the record instead, under the simplest model that says
"stable, then not": accuracy p1 up to a transition tau, accuracy p2 above it,
all three estimated by maximum likelihood, with one structural requirement --
the regimes straddle one half. Stable means mostly exact; broken means mostly
not. SCC is the fitted tau: the length at which exact counting stops being the
model's majority behaviour.

One half is not a tuned threshold in the sense 90%-over-20 is. It is the only
distinguished accuracy for a binary record, the point symmetric under swapping
the labels; every other floor needs a justification, this one needs none. The
constraint also does real work: an unconstrained fit will happily split a
perfect prefix from a 95%-accurate tail, because with enough probes that drop
is statistically unmissable -- and would report a reliability micro-break as
the capacity. Requiring the cut to separate mostly-right from mostly-wrong
forces it to the collapse, and on curves with both a micro-break and a
collapse it selects the collapse automatically.

    - An isolated slip below tau costs the fit less than moving tau, so it
      lowers p1 slightly instead of truncating the capacity.
    - Sporadic correct answers above tau raise p2 instead of extending it.
    - Denser probing narrows the confidence interval; it does not move tau.

There are no tuned thresholds. The two conventions are textbook ones: the
transition is declared by BIC -- its two extra parameters must buy a
log-likelihood gain above ln(n) -- and the interval on tau is the 95% profile
set. BIC rather than a fixed bar also disposes of the classic changepoint edge
artifact, where a stray failure at the tail of a long record can be split off
as a spurious two-probe "regime"; the fitted transitions in this dataset clear
the BIC margin by an order of magnitude (the script prints the distribution),
so the choice of convention is checkable rather than load-bearing.

When no transition clears the bar the record is a single regime: a model still
stable when probing stopped (censored at its last exact answer, capacity is a
lower bound) or one that was never stable -- p_stable carries the difference
either way, which is exactly the information a bare scalar hides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# chi^2_1 at 95%, halved: the bar a profile interval would ordinarily use.
# The changepoint is a discrete parameter, so this bar is not calibrated (it
# covers the true cut ~86% of the time under the fitted model); it is kept as
# the floor of the simulation-calibrated bar below.
PROFILE_BAR = 1.9207
CALIBRATION_REPS = 200
CALIBRATION_SEED = 7


def calibrated_bar(n: int, k_hat: int, p1: float, p2: float,
                   reps: int = CALIBRATION_REPS, seed: int = CALIBRATION_SEED) -> float:
    """Simulation-calibrated log-likelihood bar for the 95% interval on tau.

    Draws `reps` synthetic records of `n` probes from the fitted two-regime
    model (accuracy p1 through the k_hat-th probe, p2 after it), records how
    far the best admissible cut's log-likelihood exceeds the true cut's, and
    returns the 95th percentile of that excess, floored at PROFILE_BAR. Every
    admissible cut within this bar of the maximum forms the interval; a check
    of its coverage, including under the sequential stopping rule, is in
    scripts/check_interval_calibration.py.
    """
    import random
    rng = random.Random(seed)
    drops: list[float] = []
    for _ in range(reps):
        prefix = [0]
        for i in range(n):
            prefix.append(prefix[-1] + (1 if rng.random() < (p1 if i < k_hat else p2) else 0))
        total = prefix[n]
        best, ll_true = -math.inf, None
        for k in range(1, n):
            below, above = prefix[k], total - prefix[k]
            if below / k <= 0.5 or above / (n - k) >= 0.5:
                continue
            ll = _bernoulli_ll(below, k) + _bernoulli_ll(above, n - k)
            if ll > best:
                best = ll
            if k == k_hat:
                ll_true = ll
        drops.append(math.inf if ll_true is None else best - ll_true)
    finite = sorted(d for d in drops if d < math.inf)
    if not finite:
        return PROFILE_BAR
    q = finite[min(len(finite) - 1, math.ceil(0.95 * len(drops)) - 1)]
    return max(PROFILE_BAR, q)


def _bernoulli_ll(successes: int, n: int) -> float:
    """Log-likelihood of `successes` in `n` at the empirical rate."""
    if n == 0 or successes == 0 or successes == n:
        return 0.0
    p = successes / n
    return successes * math.log(p) + (n - successes) * math.log(1.0 - p)


def wilson_upper(successes: int, n: int, z: float = 1.96) -> float:
    """Upper end of the Wilson score interval for a binomial rate.

    Used to ask whether the broken regime is *established* rather than merely
    estimated: with three probes above the transition, p_after can read 0.0 and
    still be consistent with a model that is mostly right up there.
    """
    if n <= 0:
        return 1.0
    p = successes / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + spread) / (1 + z * z / n)


def sufficiency(t: "Transition", n_above: int, correct_above: int) -> str:
    """Whether the probed range supports the fit it produced.

        ok               a transition is declared and its broken regime is
                         established -- the record settles the capacity
        censored         no transition, and the model was still mostly exact
                         when probing stopped: SCC is a lower bound and more
                         probing would raise it
        never_stable     no transition because no stable regime exists -- the
                         model was already mostly inexact at the shortest
                         lengths. Settled, not shallow: probing further would
                         only buy more failures
        budget_bound     the observed collapse is the output cap, not counting;
                         probing further will not fix it, a larger budget might
        tail_unresolved  a transition is declared but too few probes sit above
                         it to establish the broken regime

    The threshold is the same one half the fit itself is built on: the tail is
    resolved when the accuracy above the transition is *significantly* below a
    half, not merely estimated below it.
    """
    if t.censored:
        # A censored record means "no collapse found". That is a shallow sweep
        # only if the model was actually counting; if it was already mostly
        # wrong at the start there is no capacity to find, and calling it
        # censored would send a broken model back for more probing.
        return "censored" if t.p_stable > 0.5 else "never_stable"
    if t.bound_by == "budget":
        return "budget_bound"
    return "tail_unresolved" if wilson_upper(correct_above, n_above) >= 0.5 else "ok"


@dataclass(frozen=True)
class Transition:
    scc: int                 # last length of the stable regime (tau-hat)
    ci_low: int              # 95% profile interval on tau
    ci_high: int
    p_stable: float          # accuracy within the stable regime
    p_after: float | None    # accuracy above the transition; None if censored
    censored: bool           # True: no transition in the probed range
    delta_ll: float          # two-regime fit minus single-regime fit
    n_probes: int
    bound_by: str            # mistake | budget | range_end
    bar: float = PROFILE_BAR  # log-likelihood bar that defined the interval


def fit_transition(
    lengths: list[int],
    correct: list[bool],
    error: list[bool] | None = None,
    calibrate: bool = True,
    reps: int = CALIBRATION_REPS,
) -> Transition:
    """Fit the two-regime model to one measurement's curve.

    `lengths` must be sorted ascending; `correct` marks an exact answer and an
    unparseable response counts as incorrect (a model that cannot answer within
    the protocol's budget is not stably counting under the protocol). `error`
    marks which incorrect probes were unparseable, so the transition can be
    attributed to wrong answers or to the budget.
    """
    n = len(lengths)
    if n == 0:
        raise ValueError("empty curve")
    error = error or [False] * n

    prefix = [0]
    for value in correct:
        prefix.append(prefix[-1] + (1 if value else 0))
    total = prefix[n]

    if total == 0:
        return Transition(scc=max(0, lengths[0] - 1), ci_low=max(0, lengths[0] - 1),
                          ci_high=max(0, lengths[0] - 1), p_stable=0.0, p_after=None,
                          censored=False, delta_ll=0.0, n_probes=n,
                          bound_by=_attribution(correct, error, 0))

    ll_single = _bernoulli_ll(total, n)

    # Profile the likelihood over every cut with both sides non-empty.
    best_k, best_ll, valid = 0, -math.inf, []
    for k in range(1, n):
        below = prefix[k]
        above = total - below
        # The cut must separate a mostly-exact regime from a mostly-inexact
        # one; see the module docstring for why one half and nothing else.
        if below / k <= 0.5 or above / (n - k) >= 0.5:
            continue
        ll = _bernoulli_ll(below, k) + _bernoulli_ll(above, n - k)
        valid.append((k, ll))
        if ll > best_ll:
            best_k, best_ll = k, ll

    if not valid or best_ll - ll_single <= math.log(n):  # BIC: 2 params, ln n
        # One regime. Stable throughout (censored) or never stable; p_stable
        # tells them apart, and the capacity is the last exact answer.
        last_ok = max(length for length, ok in zip(lengths, correct) if ok)
        return Transition(scc=last_ok, ci_low=last_ok, ci_high=lengths[-1],
                          p_stable=total / n, p_after=None, censored=True,
                          delta_ll=max(0.0, best_ll - ll_single), n_probes=n,
                          bound_by=_attribution(correct, error, len(lengths)
                                                if lengths[-1] == last_ok else
                                                lengths.index(last_ok) + 1))

    below = prefix[best_k]
    bar = PROFILE_BAR
    if calibrate:
        bar = calibrated_bar(n, best_k, below / best_k, (total - below) / (n - best_k), reps=reps)
    in_ci = [k for k, ll in valid if best_ll - ll <= bar]
    return Transition(
        scc=lengths[best_k - 1],
        ci_low=lengths[min(in_ci) - 1], ci_high=lengths[max(in_ci) - 1],
        p_stable=below / best_k, p_after=(total - below) / (n - best_k),
        censored=False, delta_ll=best_ll - ll_single, n_probes=n,
        bound_by=_attribution(correct, error, best_k), bar=bar,
    )


def transition_profile(lengths: list[int], correct: list[bool]):
    """The likelihood profile the fit maximises, for plotting or inspection.

    Returns `(cuts, ll_single, bar)`: `cuts` is a list of `(length, ll)` over
    every admissible cut, `ll_single` the single-regime fit it must beat, and
    `bar` the BIC threshold. Exposed so a figure of the profile is drawn from
    the same arithmetic as the estimate rather than a second implementation of
    it that can drift.
    """
    n = len(lengths)
    prefix = [0]
    for value in correct:
        prefix.append(prefix[-1] + (1 if value else 0))
    total = prefix[n]
    cuts = []
    for k in range(1, n):
        below, above = prefix[k], total - prefix[k]
        if below / k <= 0.5 or above / (n - k) >= 0.5:
            continue
        cuts.append((lengths[k - 1], _bernoulli_ll(below, k) + _bernoulli_ll(above, n - k)))
    return cuts, _bernoulli_ll(total, n), math.log(n)


def _attribution(correct: list[bool], error: list[bool], k: int) -> str:
    """What ended the stable regime: wrong answers, or the token budget."""
    tail_wrong = [error[i] for i in range(k, len(correct)) if not correct[i]]
    if not tail_wrong:
        return "range_end"
    return "budget" if sum(tail_wrong) * 2 > len(tail_wrong) else "mistake"


def fit_step01(lengths: list[int], correct: list[bool]) -> int:
    """The zero-one twin: the cut minimizing misfit to an ideal step.

    An ideal counter is exact to tau and never above; pick the tau whose
    idealization the record most resembles (fewest disagreeing probes). Same
    idea as the likelihood fit under 0-1 loss; kept as a robustness check
    because it can be stated in one sentence.
    """
    n = len(lengths)
    prefix = [0]
    for value in correct:
        prefix.append(prefix[-1] + (1 if value else 0))
    best_k, best_cost = 0, math.inf
    for k in range(0, n + 1):
        wrong_below = k - prefix[k]
        correct_above = prefix[n] - prefix[k]
        cost = wrong_below + correct_above
        if cost < best_cost:
            best_k, best_cost = k, cost
    return lengths[best_k - 1] if best_k else max(0, lengths[0] - 1)


__all__ = ["Transition", "fit_transition", "fit_step01", "sufficiency",
           "transition_profile", "wilson_upper", "PROFILE_BAR"]
