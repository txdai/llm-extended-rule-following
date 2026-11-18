"""Rung 2 of the execution ladder: run structure of a 1D cell row.

The stimulus is a single row of space-separated digits -- the serialization the
grid rungs will use -- and the reply is one integer, so the fitted capacity is
in the same unit as CC: the quantity the model must have exactly tracked. Two
probes:

    runs         the row contains exactly N maximal runs of equal digits, each
                 1-4 cells long; report the number of runs. The answer scales
                 with row length, as in baseline counting.
    color_total  cells with the target digit total exactly N, split into short
                 runs buried among distractor runs whose number and lengths
                 vary widely, so total row length is decorrelated from the
                 answer; report how many cells hold the target digit. This
                 closes the length-percept route entirely.

Both probes are global (no partial evaluation answers them) and neither is a
conserved quantity of anything simpler than the row itself. Digits and single
spaces keep the tokenizer regime of the baseline assay.
"""

from __future__ import annotations

import random
from typing import Any

from ..types import Stimulus
from .base import Task

SYSTEM_RUNG = (
    "You follow instructions exactly. Reply with one integer and nothing else."
)

TARGET_DIGIT = "7"
DIGITS = "0123456789"
NON_TARGET = [d for d in DIGITS if d != TARGET_DIGIT]


def _parts(total: int, lo: int, hi: int, rng: random.Random) -> list[int]:
    """Split `total` into parts each within [lo, hi]."""
    parts: list[int] = []
    remaining = total
    while remaining > 0:
        part = rng.randint(lo, min(hi, remaining))
        parts.append(part)
        remaining -= part
    return parts


class RunStructureTask(Task):
    """Ladder rung 2: single-integer probes over runs in a digit row."""

    name = "run_structure"
    level = 6
    description = "Count runs, or cells of one digit, in a 1D row of digits."

    def __init__(self, probe: str = "runs", **options: Any) -> None:
        if probe not in ("runs", "color_total"):
            raise ValueError(f"unknown probe {probe!r}")
        super().__init__(probe=probe, **options)
        self.probe = probe

    def build(self, target: int, rng: random.Random) -> Stimulus:
        if self.probe == "runs":
            row, answer = self._build_runs(target, rng)
            question = (
                "How many runs of consecutive equal digits does this row "
                "contain?"
            )
        else:
            row, answer = self._build_color_total(target, rng)
            question = f'How many cells contain the digit {TARGET_DIGIT}?'
        assert answer == target
        cells = row.split(" ")
        return Stimulus(
            target=answer,
            system=SYSTEM_RUNG,
            user=f"{question}\n\n{row}",
            metadata={
                "probe": self.probe,
                "row_cells": len(cells),
                "row_runs": self._count_runs(cells),
            },
        )

    @staticmethod
    def _count_runs(cells: list[str]) -> int:
        return sum(1 for i, c in enumerate(cells) if i == 0 or c != cells[i - 1])

    def _build_runs(self, target: int, rng: random.Random) -> tuple[str, int]:
        cells: list[str] = []
        previous = None
        for _ in range(target):
            digit = rng.choice([d for d in DIGITS if d != previous])
            cells.extend(digit * rng.randint(1, 4))
            previous = digit
        return " ".join(cells), self._count_runs(cells)

    def _build_color_total(self, target: int, rng: random.Random) -> tuple[str, int]:
        target_runs = _parts(target, 1, 4, rng)

        def distractor_block(min_runs: int) -> list[str]:
            block: list[str] = []
            previous = TARGET_DIGIT
            for _ in range(rng.randint(min_runs, 3)):
                digit = rng.choice([d for d in NON_TARGET if d != previous])
                block.extend(digit * rng.randint(1, 6))
                previous = digit
            return block

        cells: list[str] = distractor_block(0)
        for i, run in enumerate(target_runs):
            if i > 0 or cells:
                cells.extend(distractor_block(1))
            cells.extend(TARGET_DIGIT * run)
        cells.extend(distractor_block(0))
        return " ".join(cells), cells.count(TARGET_DIGIT)
