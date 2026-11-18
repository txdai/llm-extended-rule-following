"""Rung 3 of the execution ladder: 1D rule execution with count probes.

The stimulus is the rung-2 digit row plus an explicitly stated transformation
rule; the probe asks for the count of the target digit *after* the rule is
applied. The transformed row never exists in the context, so the probed
quantity must be derived by executing the rule internally while reading.

Rules (all count-changing, so no probe is answerable from the untransformed
row; permutation rules like mirror or shift are excluded by construction):

    grow_run          every run of 7s gains one cell at each end;
                      answer = cells(7) + 2 * runs(7)
    recolor_merge     every 3 becomes a 7; answer = cells(7) + cells(3)
    erase_singletons  every run of length exactly one is deleted;
                      answer = cells(7) - singleton runs of 7

Every builder chooses its run structure so the post-transform answer equals
the scanned load exactly, keeping the fitted boundary in the same unit as CC
and the rung-2 capacities. Each stimulus is verified against an independent
simulator of the rule before it is returned.
"""

from __future__ import annotations

import random
from typing import Any

from ..types import Stimulus
from .base import Task
from .run_structure import SYSTEM_RUNG, TARGET_DIGIT, _parts

MERGE_DIGIT = "3"
FILLERS = [d for d in "0123456789" if d not in (TARGET_DIGIT, MERGE_DIGIT)]

RULES = {
    "grow_run": (
        f"every run of consecutive {TARGET_DIGIT}s grows by one cell at each "
        f"end, overwriting the neighboring cells"
    ),
    "recolor_merge": (
        f"every cell containing {MERGE_DIGIT} becomes {TARGET_DIGIT}"
    ),
    "erase_singletons": (
        "every run of exactly one cell (any digit) is deleted from the row"
    ),
}


def simulate(rule: str, cells: list[str]) -> int:
    """Independent reference implementation; returns the post-rule 7-count."""
    if rule == "grow_run":
        out = list(cells)
        for i, c in enumerate(cells):
            if c == TARGET_DIGIT:
                continue
            left = i > 0 and cells[i - 1] == TARGET_DIGIT
            right = i + 1 < len(cells) and cells[i + 1] == TARGET_DIGIT
            if left or right:
                out[i] = TARGET_DIGIT
        return out.count(TARGET_DIGIT)
    if rule == "recolor_merge":
        return sum(1 for c in cells if c in (TARGET_DIGIT, MERGE_DIGIT))
    if rule == "erase_singletons":
        runs: list[tuple[str, int]] = []
        for c in cells:
            if runs and runs[-1][0] == c:
                runs[-1] = (c, runs[-1][1] + 1)
            else:
                runs.append((c, 1))
        return sum(n for c, n in runs if c == TARGET_DIGIT and n > 1)
    raise ValueError(rule)


class RuleExecutionTask(Task):
    """Ladder rung 3: count probe over a stated 1D transformation."""

    name = "rule_execution"
    level = 7
    description = "Single-integer count probe over an explicitly stated 1D rule."

    def __init__(self, rule: str = "grow_run", **options: Any) -> None:
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}; known: {sorted(RULES)}")
        super().__init__(rule=rule, **options)
        self.rule = rule

    # -- builders ----------------------------------------------------------
    # Each returns the full cell list; separators between special runs use
    # filler digits only, in blocks of >= 3 cells so grow_run never collides
    # and erase_singletons keeps its bookkeeping local to each run.

    def _filler_block(self, rng: random.Random, min_cells: int = 3) -> list[str]:
        block: list[str] = []
        previous = None
        while len(block) < min_cells or rng.random() < 0.4:
            digit = rng.choice([d for d in FILLERS if d != previous])
            width = rng.randint(2, 5)
            block.extend(digit * width)
            previous = digit
        return block

    def _assemble(self, special_runs: list[list[str]], rng: random.Random) -> list[str]:
        cells = self._filler_block(rng)
        for run in special_runs:
            cells.extend(run)
            cells.extend(self._filler_block(rng))
        return cells

    def _build_grow(self, n: int, rng: random.Random) -> list[str]:
        # n = cells(7) + 2 * runs(7): runs in [ceil(n/6), floor(n/3)],
        # lengths 1-4 summing to n - 2*runs.
        lo, hi = -(-n // 6), n // 3
        r = rng.randint(lo, hi)
        lengths = self._fixed_parts(n - 2 * r, r, 1, 4, rng)
        return self._assemble([[TARGET_DIGIT] * k for k in lengths], rng)

    def _build_merge(self, n: int, rng: random.Random) -> list[str]:
        n7 = rng.randint(0, n)
        runs = [[TARGET_DIGIT] * k for k in _parts(n7, 1, 4, rng)]
        runs += [[MERGE_DIGIT] * k for k in _parts(n - n7, 1, 4, rng)]
        rng.shuffle(runs)
        return self._assemble(runs, rng)

    def _build_erase(self, n: int, rng: random.Random) -> list[str]:
        survivors = [[TARGET_DIGIT] * k for k in self._fixed_parts_free(n, 2, 4, rng)]
        singles = [[TARGET_DIGIT]] * rng.randint(1, 4)
        runs = survivors + singles
        rng.shuffle(runs)
        return self._assemble(runs, rng)

    @staticmethod
    def _fixed_parts(total: int, count: int, lo: int, hi: int,
                     rng: random.Random) -> list[int]:
        """Split `total` into exactly `count` parts within [lo, hi]."""
        parts = [lo] * count
        remaining = total - lo * count
        while remaining > 0:
            i = rng.randrange(count)
            if parts[i] < hi:
                parts[i] += 1
                remaining -= 1
        return parts

    @staticmethod
    def _fixed_parts_free(total: int, lo: int, hi: int,
                          rng: random.Random) -> list[int]:
        """Split `total` into parts within [lo, hi], any count (total >= lo).

        Whenever the remainder exceeds `hi`, the drawn part is capped at
        `remaining - lo`, so the remainder never falls below `lo` and the
        final part is always in range.
        """
        parts: list[int] = []
        remaining = total
        while remaining > hi:
            part = rng.randint(lo, min(hi, remaining - lo))
            parts.append(part)
            remaining -= part
        parts.append(remaining)
        return parts

    MIN_LOAD = {"grow_run": 3, "recolor_merge": 1, "erase_singletons": 2}

    def build(self, target: int, rng: random.Random) -> Stimulus:
        n = max(target, self.MIN_LOAD[self.rule])
        builder = {"grow_run": self._build_grow,
                   "recolor_merge": self._build_merge,
                   "erase_singletons": self._build_erase}[self.rule]
        cells = builder(n, rng)
        answer = simulate(self.rule, cells)
        assert answer == n, (self.rule, target, answer)
        question = (
            f"Apply this rule to the row below: {RULES[self.rule]}. "
            f"After applying the rule, how many cells contain the digit "
            f"{TARGET_DIGIT}?"
        )
        return Stimulus(
            target=answer,
            system=SYSTEM_RUNG,
            user=f"{question}\n\n{' '.join(cells)}",
            metadata={"rule": self.rule, "row_cells": len(cells)},
        )
