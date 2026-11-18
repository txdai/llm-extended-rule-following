"""Search-strategy interface.

A strategy decides which sequence lengths to spend trials on. The runner drives
it as a simple loop:

    while (center := strategy.next_center()) is not None:
        verdict = evaluate(center)
        strategy.record(verdict)
    summary = strategy.result()

That is the entire contract, so a strategy never touches a provider, a task, or
the filesystem. Two ship with the harness:

    ladder  doubling expansion then bisection -- few tiers, finds the boundary
    dense   a full grid -- many tiers, gives the whole error curve

To add one: subclass, implement the three methods, register in __init__.py, and
add configs/strategies/<name>.yaml.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from ..types import TierVerdict


class SearchStrategy(ABC):
    """Chooses the tiers to evaluate and summarises what came back."""

    name: str = "search"
    description: str = ""

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.verdicts: list[TierVerdict] = []

    @abstractmethod
    def next_center(self) -> int | None:
        """The next center length to evaluate, or None when the search is done."""

    def record(self, verdict: TierVerdict) -> None:
        """Take in the outcome of the tier just evaluated."""
        self.verdicts.append(verdict)

    @abstractmethod
    def result(self) -> dict[str, Any]:
        """Summarise the search. Must include `stable_count`."""

    # -- shared summary pieces ---------------------------------------------

    @property
    def passed_centers(self) -> list[int]:
        return [v.center_length for v in self.verdicts if v.passed]

    @property
    def failed_centers(self) -> list[int]:
        return [v.center_length for v in self.verdicts if not v.passed]

    def base_result(self, stable_count: int, unstable_upper_bound: int | None) -> dict[str, Any]:
        """The fields every strategy reports, so downstream analysis is uniform.

        `stable_error_bar` brackets the true boundary: the lower arm reflects the
        resolution the search settled at, and the upper arm reaches to the lowest
        length that actually failed.
        """
        step_fraction = float(self.options.get("step_fraction", 0.1))
        lower = 0 if stable_count <= 0 else max(1, math.floor(stable_count * step_fraction))
        if unstable_upper_bound is not None:
            upper = max(1, unstable_upper_bound - stable_count)
        else:
            upper = max(1, math.ceil(stable_count * step_fraction))
        return {
            "strategy": self.name,
            "stable_count": stable_count,
            "stable_lower_bound": stable_count,
            "unstable_upper_bound": unstable_upper_bound,
            "bounded": unstable_upper_bound is not None,
            "stable_error_bar": {"lower": lower, "upper": upper},
            "evaluations": [v.as_record() for v in self.verdicts],
            "evaluation_count": len(self.verdicts),
        }
