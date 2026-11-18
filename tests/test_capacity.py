"""The fitted transition must ignore what first-mistake keys on, and hold
still when the sampling density changes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from counting.analysis.capacity import (
    fit_step01, fit_transition, sufficiency, transition_profile,
)


def curve(spec: str):
    """'T'*40 + 'F'*40 style spec -> (lengths, correct, error)."""
    correct = [c == "T" for c in spec]
    error = [c == "E" for c in spec]           # E: unparseable, counts incorrect
    correct = [c == "T" for c in spec]
    return list(range(1, len(spec) + 1)), correct, error


def test_clean_step():
    lengths, ok, err = curve("T" * 40 + "F" * 60)
    t = fit_transition(lengths, ok, err)
    assert t.scc == 40 and not t.censored
    assert t.ci_low == 40 and t.ci_high == 40
    assert t.p_stable == 1.0 and t.p_after == 0.0
    assert t.bound_by == "mistake"
    assert fit_step01(lengths, ok) == 40


def test_isolated_slip_does_not_truncate():
    # One wrong answer at 41 in an otherwise perfect run of 100: first-mistake
    # capacity says 40; the fit says the record is one stable regime.
    lengths, ok, err = curve("T" * 40 + "F" + "T" * 59)
    t = fit_transition(lengths, ok, err)
    assert t.censored and t.scc == 100
    assert t.p_stable > 0.98


def test_sporadic_recovery_does_not_extend():
    # Correct to 40, then only one answer in four correct: the stray successes
    # above must raise p_after, not move the transition.
    tail = ("FFFT" * 15)
    lengths, ok, err = curve("T" * 40 + tail)
    t = fit_transition(lengths, ok, err)
    assert 38 <= t.scc <= 44, t.scc
    assert t.p_after and 0.15 < t.p_after < 0.35


def test_budget_bound_tail():
    # Perfect until the token budget dies: transition exists and is the
    # budget's, not the model's.
    lengths, ok, err = curve("T" * 480 + "E" * 10)
    t = fit_transition(lengths, ok, err)
    assert t.scc == 480 and t.bound_by == "budget"


def test_never_correct():
    lengths, ok, err = curve("F" * 30)
    t = fit_transition(lengths, ok, err)
    assert t.scc == 0 and t.p_stable == 0.0


def test_all_correct_is_censored():
    lengths, ok, err = curve("T" * 50)
    t = fit_transition(lengths, ok, err)
    assert t.censored and t.scc == 50 and t.bound_by == "range_end"


def test_sampling_invariance():
    # The decisive property. Probing every 2nd or 5th length must not move the
    # fitted transition; first-mistake, an extreme statistic, drifts by design.
    spec = "T" * 197 + "FTTFT" + "FFTF" * 40
    lengths, ok, err = curve(spec)
    dense = fit_transition(lengths, ok, err).scc
    for step in (2, 5):
        sub = fit_transition(lengths[::step], ok[::step], err[::step]).scc
        assert abs(sub - dense) <= 2 * step, (step, dense, sub)


def test_zero_one_twin_agrees():
    lengths, ok, err = curve("T" * 120 + "TFFT" * 3 + "F" * 80)
    assert abs(fit_step01(lengths, ok) - fit_transition(lengths, ok, err).scc) <= 4


def test_micro_break_is_not_a_transition():
    # Perfect to 60, then 95% accurate to 360: statistically detectable, but
    # counting never stops being the majority behaviour -- no transition.
    tail = ("T" * 19 + "F") * 15
    lengths, ok, err = curve("T" * 60 + tail)
    t = fit_transition(lengths, ok, err)
    assert t.censored, (t.scc, t.p_after)
    assert t.scc == 360 or ok[t.scc - 1]


def test_collapse_outranks_micro_break():
    # 1.0 to 100, 0.75 to 300, 0.1 after: the capacity is the collapse at 300,
    # not the reliability dip at 100.
    mid = ("TTTF" * 50)
    tail = ("FFFFFFFFFT" * 10)
    lengths, ok, err = curve("T" * 100 + mid + tail)
    t = fit_transition(lengths, ok, err)
    assert 290 <= t.scc <= 310, t.scc
    assert t.p_stable > 0.8 and t.p_after < 0.2


# -- the criterion as a stop rule ----------------------------------------

def _run_scan(capacity, tail_accuracy, **kw):
    """Drive a LinearScan against a synthetic model, return (strategy, lengths)."""
    import random
    from counting.search import build_strategy
    from counting.types import TierVerdict

    kw.setdefault("stop", 4000)
    strategy = build_strategy("scan", start=1, **kw)
    rng = random.Random(7)
    asked = []
    while True:
        length = strategy.next_center()
        if length is None:
            return strategy, asked
        asked.append(length)
        ok = length <= capacity or rng.random() < tail_accuracy
        strategy.record(TierVerdict(
            center_length=length, phase="scan", evaluation_index=len(asked),
            sample_low=length, sample_high=length, passed=ok, trial_count=1,
            parse_rate=1.0, exact_match_rate=1.0 if ok else 0.0,
            mape=0.0, mae=0.0, summary={"predicted": length if ok else length - 1},
        ))


def test_resolved_rule_scans_past_the_old_stop():
    # Capacity 300 with a stray slip at 40: the multiple rule quits at 100
    # (the floor) and censors the model; the criterion rule finds 300.
    import random
    from counting.search import build_strategy
    from counting.types import TierVerdict

    def drive(**kw):
        s = build_strategy("scan", start=1, stop=4000, **kw)
        rng = random.Random(3)
        while True:
            n = s.next_center()
            if n is None:
                return s
            ok = (n != 40) and (n <= 300 or rng.random() < 0.15)
            s.record(TierVerdict(
                center_length=n, phase="scan", evaluation_index=n, sample_low=n,
                sample_high=n, passed=ok, trial_count=1, parse_rate=1.0,
                exact_match_rate=1.0 if ok else 0.0, mape=0.0, mae=0.0,
                summary={"predicted": n if ok else n - 1}))

    old = drive(stop_at_first_mistake_multiple=2.0, min_stop_length=100)
    new = drive(stop_at_first_mistake_multiple=None, stop_when_resolved=True)
    assert old.result()["highest_length_probed"] == 100
    assert new.result()["stopped_early"] == "criterion_resolved"

    hi = new.result()["highest_length_probed"]
    assert 300 < hi < 500, hi
    lengths = [v.center_length for v in new.verdicts]
    correct = [v.exact_match_rate == 1.0 for v in new.verdicts]
    fit = fit_transition(lengths, correct, [False] * len(lengths))
    assert 290 <= fit.scc <= 310, fit.scc
    assert not fit.censored


def test_resolved_rule_does_not_stop_while_stable():
    # Still mostly exact everywhere: nothing to declare, so it must run to the
    # ceiling rather than stopping on a phantom transition.
    strategy, asked = _run_scan(600, 0.0, stop=250,
                                stop_at_first_mistake_multiple=None,
                                stop_when_resolved=True)
    assert strategy.result()["stopped_early"] is None
    assert max(asked) == 250


def test_resolved_rule_still_honours_consecutive_errors():
    from counting.search import build_strategy
    from counting.types import TierVerdict

    s = build_strategy("scan", start=1, stop=4000,
                       stop_at_first_mistake_multiple=None, stop_when_resolved=True)
    while True:
        n = s.next_center()
        if n is None:
            break
        err = n > 30
        s.record(TierVerdict(
            center_length=n, phase="scan", evaluation_index=n, sample_low=n,
            sample_high=n, passed=not err, trial_count=1,
            parse_rate=0.0 if err else 1.0, exact_match_rate=0.0 if err else 1.0,
            mape=0.0, mae=0.0, summary={"predicted": None if err else n}))
    assert s.result()["stopped_early"] == "consecutive_errors"
    assert s.result()["highest_length_probed"] == 40


def test_never_stable_is_not_censored():
    # Mostly wrong from length 1: no transition, but nothing to extend either.
    lengths, ok, err = curve(("FFFFT" * 12))
    t = fit_transition(lengths, ok, err)
    assert t.censored and t.p_stable < 0.5
    assert sufficiency(t, 0, 0) == "never_stable"


def test_censored_needs_a_stable_regime():
    lengths, ok, err = curve("T" * 80)
    t = fit_transition(lengths, ok, err)
    assert sufficiency(t, 0, 0) == "censored"


def test_profile_matches_the_fit():
    """The exposed profile must peak where the fit puts the transition."""
    lengths, ok, err = curve("T" * 60 + "TFFT" * 4 + "F" * 70)
    t = fit_transition(lengths, ok, err)
    cuts, ll_single, bar = transition_profile(lengths, ok)
    best_len, best_ll = max(cuts, key=lambda c: c[1])
    assert best_len == t.scc
    assert abs((best_ll - ll_single) - t.delta_ll) < 1e-9
    assert best_ll - ll_single > bar          # this one is declared
    inside = [c for c in cuts if best_ll - c[1] <= 1.9207]
    assert min(c[0] for c in inside) == t.ci_low
    assert max(c[0] for c in inside) == t.ci_high
