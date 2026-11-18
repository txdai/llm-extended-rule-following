"""Dense sweep -- evaluate a whole grid instead of hunting for the boundary.

The ladder answers "where does it break?" in a handful of tiers. A dense sweep
answers "what does breaking look like?": it evaluates every point on a grid, so
you get the full accuracy-versus-length curve, the shape of the transition, and
the post-collapse attractor distribution, rather than a single threshold.

Grid, in order of precedence:

    points: [32, 64, 96, ...]        explicit list, used verbatim
    spacing: linear                  start .. stop inclusive, every `step`
    spacing: log                     `num` points geometric from start to stop

Because it is far more expensive than the ladder, three guards are available:

    stop_after_consecutive_failures   stop once the model has failed N tiers in a
                                      row; the curve past that is flat anyway
    stop_beyond_first_failure_ratio   stop once the length under test exceeds the
                                      first failing length by this fraction. 1.0
                                      means "stop at twice the first error",
                                      which captures the shape of the transition
                                      without paying for the flat tail
    max_tiers                         hard ceiling on tiers evaluated

All three are off by default -- an unguarded sweep evaluates the whole grid.
When several are set, whichever fires first wins, and `stopped_early` records
which one it was.

`stable_count` is defined the same way as for the ladder (the largest passing
length below the first failure) so SCC numbers stay comparable across strategies.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .base import SearchStrategy


def build_grid(
    points: Sequence[int] | None = None,
    start: int = 32,
    stop: int = 2000,
    step: int = 32,
    spacing: str = "linear",
    num: int = 32,
) -> list[int]:
    """Materialise the grid of center lengths, deduplicated and ascending."""
    if points:
        grid = [int(p) for p in points]
    elif spacing == "log":
        if start <= 0 or stop <= 0:
            raise ValueError("log spacing needs start and stop to be positive")
        if num < 2:
            raise ValueError("log spacing needs num >= 2")
        ratio = (stop / start) ** (1.0 / (num - 1))
        grid = [int(round(start * ratio**index)) for index in range(num)]
    elif spacing == "linear":
        if step <= 0:
            raise ValueError("linear spacing needs a positive step")
        grid = list(range(int(start), int(stop) + 1, int(step)))
    else:
        raise ValueError(f"unknown spacing {spacing!r}; expected 'linear' or 'log'")

    return sorted({value for value in grid if value > 0})


class DenseSearch(SearchStrategy):
    name = "dense"
    description = "Evaluate every point on a grid; reports the whole curve."

    def __init__(
        self,
        points: Sequence[int] | None = None,
        start: int = 32,
        stop: int = 2000,
        step: int = 32,
        spacing: str = "linear",
        num: int = 32,
        stop_after_consecutive_failures: int | None = None,
        stop_beyond_first_failure_ratio: float | None = None,
        max_tiers: int | None = None,
        step_fraction: float = 0.1,
        **options: Any,
    ) -> None:
        super().__init__(
            points=list(points) if points else None,
            start=start,
            stop=stop,
            step=step,
            spacing=spacing,
            num=num,
            stop_after_consecutive_failures=stop_after_consecutive_failures,
            stop_beyond_first_failure_ratio=stop_beyond_first_failure_ratio,
            max_tiers=max_tiers,
            step_fraction=step_fraction,
            **options,
        )
        self.grid = build_grid(points, start, stop, step, spacing, num)
        self.stop_after_consecutive_failures = stop_after_consecutive_failures
        self.stop_beyond_first_failure_ratio = stop_beyond_first_failure_ratio
        self.max_tiers = max_tiers
        self._index = 0
        self._consecutive_failures = 0
        self._first_failure: int | None = None
        self._stopped_early: str | None = None

    def next_center(self) -> int | None:
        if self._stopped_early is not None:
            return None
        if self._index >= len(self.grid):
            return None
        if self.max_tiers is not None and self._index >= self.max_tiers:
            self._stopped_early = "max_tiers"
            return None

        center = self.grid[self._index]
        # Checked here rather than in record() so the tier that would breach the
        # ceiling is never evaluated, instead of being evaluated and discarded.
        if (
            self.stop_beyond_first_failure_ratio is not None
            and self._first_failure is not None
            and center > self._first_failure * (1.0 + self.stop_beyond_first_failure_ratio)
        ):
            self._stopped_early = "beyond_first_failure"
            return None
        return center

    def record(self, verdict) -> None:  # type: ignore[override]
        super().record(verdict)
        self._index += 1
        if verdict.passed:
            self._consecutive_failures = 0
            return

        self._consecutive_failures += 1
        if self._first_failure is None or verdict.center_length < self._first_failure:
            self._first_failure = verdict.center_length

        limit = self.stop_after_consecutive_failures
        if limit is not None and self._consecutive_failures >= limit:
            self._stopped_early = "consecutive_failures"

    def phase_label(self) -> str:
        return "sweep"

    def result(self) -> dict[str, Any]:
        passed = self.passed_centers
        failed = self.failed_centers
        first_fail = min(failed) if failed else None

        # Same definition as the ladder: the largest length that passed while
        # still sitting below the first failure. Passes above the first failure
        # are real data but do not extend the stable regime.
        if first_fail is None:
            stable_count = max(passed) if passed else 0
        else:
            below = [center for center in passed if center < first_fail]
            stable_count = max(below) if below else 0

        summary = self.base_result(stable_count, first_fail)
        summary.update(
            {
                "grid": self.grid,
                "grid_size": len(self.grid),
                "tiers_evaluated": len(self.verdicts),
                "stopped_early": self._stopped_early,
                "consecutive_failures_at_stop": self._consecutive_failures,
                "stop_beyond_first_failure_ratio": self.stop_beyond_first_failure_ratio,
                "passes_above_first_failure": sorted(
                    center for center in passed if first_fail is not None and center > first_fail
                ),
                "curve": [
                    {
                        "center_length": v.center_length,
                        "passed": v.passed,
                        "parse_rate": v.parse_rate,
                        "exact_match_rate": v.exact_match_rate,
                        "mape": v.mape,
                        "mae": v.mae,
                    }
                    for v in self.verdicts
                ],
            }
        )
        return summary
