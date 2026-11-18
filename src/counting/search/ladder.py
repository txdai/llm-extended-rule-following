"""Adaptive randomized ladder -- the "skipping stones" search.

Two phases, exactly as reported in the paper:

    expansion   start at `initial_length`, double while tiers pass
    refinement  bisect between the highest pass and the lowest failure until the
                bracket is narrower than `step_fraction` of its midpoint

A model that fails the very first tier gets no lower bound to bisect from, so the
refinement is skipped and its SCC is recorded as 0 (reported as "<initial_length"
in the paper).

Cheap -- about log2(L) + log2(1/step_fraction) tiers -- which is why it was used
for the 126-model sweep. Use `dense` instead when you want the shape of the
failure curve rather than just where it starts.
"""

from __future__ import annotations

import math
from typing import Any

from .base import SearchStrategy


class LadderSearch(SearchStrategy):
    name = "ladder"
    description = "Double while tiers pass, then bisect to the boundary."

    def __init__(
        self,
        initial_length: int = 32,
        max_length: int = 20000,
        step_fraction: float = 0.1,
        growth_factor: float = 2.0,
        **options: Any,
    ) -> None:
        super().__init__(
            initial_length=initial_length,
            max_length=max_length,
            step_fraction=step_fraction,
            growth_factor=growth_factor,
            **options,
        )
        self.initial_length = int(initial_length)
        self.max_length = int(max_length)
        self.step_fraction = float(step_fraction)
        self.growth_factor = float(growth_factor)

        self.phase = "expansion"
        self.low_pass: int | None = None
        self.first_fail: int | None = None
        self._next: int | None = self.initial_length

    # -- driver ------------------------------------------------------------

    def next_center(self) -> int | None:
        return self._next

    def record(self, verdict) -> None:  # type: ignore[override]
        super().record(verdict)
        center = verdict.center_length

        if self.phase == "expansion":
            if verdict.passed:
                self.low_pass = center
                if center >= self.max_length:
                    self._enter_refinement()
                    return
                nxt = min(self.max_length, int(center * self.growth_factor))
                self._next = None if nxt == center else nxt
                if self._next is None:
                    self._enter_refinement()
                return
            self.first_fail = center
            self._enter_refinement()
            return

        # refinement
        if verdict.passed:
            self.low_pass = center
        else:
            self.first_fail = center
        self._next = self._refinement_candidate()

    def _enter_refinement(self) -> None:
        self.phase = "refinement"
        self._next = self._refinement_candidate()

    def _refinement_candidate(self) -> int | None:
        """Midpoint of the current bracket, or None once it is tight enough."""
        if self.first_fail is None or self.low_pass is None or self.low_pass <= 0:
            return None
        low, high = self.low_pass, self.first_fail
        if self._precise_enough(low, high):
            return None
        return max(low + 1, min(high - 1, round((low + high) / 2)))

    def _precise_enough(self, low: int, high: int) -> bool:
        if high <= low:
            return True
        midpoint = max(1, math.ceil((low + high) / 2))
        return (high - low) <= max(1, math.ceil(self.step_fraction * midpoint))

    def phase_label(self) -> str:
        return self.phase

    # -- summary -----------------------------------------------------------

    def result(self) -> dict[str, Any]:
        stable_count = self.low_pass or 0
        summary = self.base_result(stable_count, self.first_fail)
        summary.update(
            {
                "initial_length": self.initial_length,
                "max_length": self.max_length,
                "search_step_fraction": self.step_fraction,
            }
        )
        return summary
