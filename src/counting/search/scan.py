"""Linear scan -- one probe per length, starting at 1.

No tiers, no sampling, no jitter, no tolerance band. Length 1 is asked once,
then 2, then 3, and so on. Every response is kept.

Two outcomes are tracked separately, because they mean different things:

    error     the reply contained no parseable integer. Usually the model ran out
              of output budget, looped, or returned nothing. This is a failure of
              the *harness contract*, not necessarily of counting.
    mistake   the reply parsed but the number was wrong. This is the thing being
              measured: the first mistake is the counting capacity boundary.

Two stop conditions, whichever comes first:

    stop_after_consecutive_errors     N unparsed replies in a row (default 10).
                                      Consecutive rather than cumulative so a
                                      transient provider hiccup mid-scan does not
                                      end an otherwise healthy run.
    stop_at_first_mistake_multiple    stop once the length reaches this multiple
                                      of the first mistake (default 2.0). Enough
                                      to see the shape of the collapse and the
                                      attractors it falls into, without paying
                                      for the flat tail.
    stop_when_resolved                stop once the fitted transition (see
                                      analysis/capacity.py) is declared and the
                                      regime above it is established. This is
                                      the criterion-native rule: it stops when
                                      the record settles the capacity, rather
                                      than at a fixed multiple of a landmark the
                                      criterion does not use. Pair it with
                                      stop_at_first_mistake_multiple = null.
    min_stop_length                   never let the multiple stop the scan below
                                      this length. Failures near the start are
                                      frequently isolated slips rather than a
                                      boundary: a model that first errs at 2 and
                                      then answers correctly to 50 would, under
                                      the bare multiple, stop at 4 and be
                                      recorded as capacity 1. The floor buys
                                      enough of the curve to tell the two apart.
                                      It does not delay the consecutive-error
                                      stop, so a model returning nothing still
                                      costs ten probes rather than a hundred.

Note that `stop_at_first_mistake_multiple` is defined against the first mistake,
which the fitted criterion deliberately does not use. Scanning to 2x the first
mistake therefore leaves headroom above the *wrong* landmark, and censors
exactly the models whose transition sits far above their first slip. That is
what `stop_when_resolved` exists to fix.

`stable_count` is `first_mistake - 1`: the last length answered exactly before
the first wrong answer. A scan that never makes a mistake reports the last length
reached and is marked unbounded.
"""

from __future__ import annotations

from typing import Any

from ..analysis.capacity import fit_transition, sufficiency
from .base import SearchStrategy


class LinearScan(SearchStrategy):
    name = "scan"
    description = "Probe every length from 1 upward, one request each."

    def __init__(
        self,
        start: int = 1,
        stop: int = 100000,
        step: int = 1,
        stop_after_consecutive_errors: int | None = 10,
        stop_at_first_mistake_multiple: float | None = 2.0,
        min_stop_length: int = 0,
        stop_when_resolved: bool = False,
        **options: Any,
    ) -> None:
        super().__init__(
            start=start,
            stop=stop,
            step=step,
            stop_after_consecutive_errors=stop_after_consecutive_errors,
            stop_at_first_mistake_multiple=stop_at_first_mistake_multiple,
            min_stop_length=min_stop_length,
            stop_when_resolved=stop_when_resolved,
            **options,
        )
        self.start = int(start)
        self.stop = int(stop)
        self.step = max(1, int(step))
        self.stop_after_consecutive_errors = stop_after_consecutive_errors
        self.stop_at_first_mistake_multiple = stop_at_first_mistake_multiple
        self.min_stop_length = int(min_stop_length)
        self.stop_when_resolved = bool(stop_when_resolved)

        self._next = self.start
        self._consecutive_errors = 0
        self._first_mistake: int | None = None
        self._first_error: int | None = None
        self._error_count = 0
        self._mistake_count = 0
        self._correct_count = 0
        self._last_correct: int | None = None
        self._stopped: str | None = None

    # -- driver ------------------------------------------------------------

    def next_center(self) -> int | None:
        if self._stopped is not None or self._next > self.stop:
            return None
        # Checked before issuing, so the length that would breach the ceiling is
        # never requested rather than requested and then discarded.
        if (
            self.stop_at_first_mistake_multiple is not None
            and self._first_mistake is not None
            and self._next > self.mistake_stop_length()
        ):
            self._stopped = "past_first_mistake_multiple"
            return None
        if self.stop_when_resolved and self._criterion_resolved():
            self._stopped = "criterion_resolved"
            return None
        return self._next

    def _criterion_resolved(self) -> bool:
        """True once the record settles the capacity under the fitted criterion.

        Checked before each probe. The fit is O(n) and a probe is a network
        round trip, so re-fitting every time costs nothing worth saving and
        stops on exactly the probe that resolves the tail.
        """
        # A transition needs a mostly-inexact regime to separate from; with no
        # wrong answers yet there is nothing that could be declared.
        if self._mistake_count + self._error_count < 3:
            return False
        lengths, correct, errors = [], [], []
        for verdict in sorted(self.verdicts, key=lambda v: v.center_length):
            outcome = classify(verdict)
            lengths.append(verdict.center_length)
            correct.append(outcome == "correct")
            errors.append(outcome == "error")
        fit = fit_transition(lengths, correct, errors)
        above = [(length, ok) for length, ok in zip(lengths, correct) if length > fit.scc]
        return sufficiency(fit, len(above), sum(1 for _, ok in above if ok)) == "ok"

    def mistake_stop_length(self) -> float:
        """Highest length the mistake rule allows, once a mistake has happened."""
        assert self._first_mistake is not None
        return max(
            float(self.min_stop_length),
            self._first_mistake * float(self.stop_at_first_mistake_multiple or 0),
        )

    def record(self, verdict) -> None:  # type: ignore[override]
        super().record(verdict)
        length = verdict.center_length
        self._next = length + self.step

        outcome = classify(verdict)
        if outcome == "error":
            self._error_count += 1
            self._consecutive_errors += 1
            if self._first_error is None:
                self._first_error = length
            limit = self.stop_after_consecutive_errors
            if limit is not None and self._consecutive_errors >= limit:
                self._stopped = "consecutive_errors"
            return

        self._consecutive_errors = 0
        if outcome == "mistake":
            self._mistake_count += 1
            if self._first_mistake is None or length < self._first_mistake:
                self._first_mistake = length
        else:
            self._correct_count += 1
            if self._last_correct is None or length > self._last_correct:
                self._last_correct = length

    def phase_label(self) -> str:
        return "scan"

    # -- resume ------------------------------------------------------------

    def resume_from(self, verdicts) -> None:
        """Replay already-recorded probes so the counters reflect them.

        Used when a scan is restarted from its checkpoint: without this the
        consecutive-error count and the first mistake would reset, and the scan
        would run past the point where it should have stopped.
        """
        for verdict in sorted(verdicts, key=lambda v: v.center_length):
            self.record(verdict)

    # -- summary -----------------------------------------------------------

    def result(self) -> dict[str, Any]:
        if self._first_mistake is not None:
            stable_count = self._first_mistake - 1
        else:
            stable_count = self._last_correct or 0

        summary = self.base_result(stable_count, self._first_mistake)
        # The ladder's error bar expresses search resolution. A scan has none to
        # express: it tested every length, so the boundary sits exactly between
        # the last correct answer and the first mistake.
        summary["stable_error_bar"] = {
            "lower": 0,
            "upper": 1 if self._first_mistake is not None else 0,
        }
        summary.update(
            {
                "first_mistake": self._first_mistake,
                "first_error": self._first_error,
                "error_count": self._error_count,
                "mistake_count": self._mistake_count,
                "correct_count": self._correct_count,
                "last_correct": self._last_correct,
                "lengths_probed": len(self.verdicts),
                "highest_length_probed": max(
                    (v.center_length for v in self.verdicts), default=None
                ),
                "stopped_early": self._stopped,
                "consecutive_errors_at_stop": self._consecutive_errors,
                "stop_after_consecutive_errors": self.stop_after_consecutive_errors,
                "stop_at_first_mistake_multiple": self.stop_at_first_mistake_multiple,
                "min_stop_length": self.min_stop_length,
                "stop_when_resolved": self.stop_when_resolved,
                "curve": [
                    {
                        "length": v.center_length,
                        "outcome": classify(v),
                        "predicted": (v.summary or {}).get("predicted"),
                    }
                    for v in sorted(self.verdicts, key=lambda v: v.center_length)
                ],
            }
        )
        return summary


def classify(verdict) -> str:
    """One probe's outcome: "correct", "mistake", or "error"."""
    if not verdict.parse_rate:
        return "error"
    return "correct" if verdict.exact_match_rate else "mistake"
