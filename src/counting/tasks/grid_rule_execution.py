"""Rung 4 of the execution ladder: 2D grid rule execution with count probes.

The stimulus is an H x W grid of digits, one row per line, cells separated by
spaces -- the serialization ARC grids use in text. A transformation rule is
stated, and the probe asks for the count of the target digit after the rule is
applied. The transformed grid never exists in the context.

Rules (2D analogues of rung 3):

    grow_region       every connected region of 7s grows one cell in each of
                      the four directions; regions are axis-aligned rectangles
                      placed with clearance, so a region of h x w contributes
                      h*w + 2*(h+w) cells after growth
    recolor_merge     every 3 becomes a 7; answer = cells(7) + cells(3)
    erase_singletons  every single-cell region (no horizontal or vertical
                      neighbor with the same digit) is deleted;
                      answer = cells of 7 in regions of two or more cells

Builders choose region sizes so the post-rule answer equals the scanned load
exactly; every grid is verified by an independent simulator (true dilation and
connected components, not the builder's rectangle arithmetic) before it is
returned. Background is a uniform 0 field with distractor rectangles of other
digits, the visual grammar of an ARC task.
"""

from __future__ import annotations

import random
from typing import Any

from ..types import Stimulus
from .base import Task
from .run_structure import SYSTEM_RUNG, TARGET_DIGIT

MERGE_DIGIT = "3"
BACKGROUND = "0"
DISTRACTORS = [d for d in "123456789" if d not in (TARGET_DIGIT, MERGE_DIGIT)]

GAP = 3       # min cells between rectangles: grown regions never touch
MARGIN = 2    # min cells to the grid edge: growth never clips

RULES = {
    "grow_region": (
        f"every connected region of {TARGET_DIGIT}s (cells joined horizontally "
        f"or vertically) grows by one cell in each of the four directions, "
        f"overwriting the cells it grows into"
    ),
    "recolor_merge": (
        f"every cell containing {MERGE_DIGIT} becomes {TARGET_DIGIT}"
    ),
    "erase_singletons": (
        "every region consisting of a single cell (no horizontal or vertical "
        "neighbor with the same digit) is deleted from the grid"
    ),
}

# grow_region part table: rectangle h x w contributes t = h*w + 2*(h+w).
GROW_PARTS = sorted({(h * w + 2 * (h + w), h, w)
                     for h in range(1, 5) for w in range(1, 5)})
GROW_VALUES = sorted({t for t, _, _ in GROW_PARTS})

# recolor / erase parts: rectangle areas from dims 1..3.
AREA_DIMS = {1: [(1, 1)], 2: [(1, 2), (2, 1)], 3: [(1, 3), (3, 1)],
             4: [(2, 2)], 6: [(2, 3), (3, 2)], 9: [(3, 3)]}


def simulate(rule: str, grid: list[list[str]]) -> int:
    """Independent reference implementation on the parsed grid."""
    h, w = len(grid), len(grid[0])
    if rule == "grow_region":
        out = {(r, c) for r in range(h) for c in range(w)
               if grid[r][c] == TARGET_DIGIT}
        for r in range(h):
            for c in range(w):
                if grid[r][c] == TARGET_DIGIT:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w:
                            out.add((rr, cc))
        return len(out)
    if rule == "recolor_merge":
        return sum(row.count(TARGET_DIGIT) + row.count(MERGE_DIGIT)
                   for row in grid)
    if rule == "erase_singletons":
        seen: set[tuple[int, int]] = set()
        total = 0
        for r in range(h):
            for c in range(w):
                if (r, c) in seen:
                    continue
                digit = grid[r][c]
                stack, comp = [(r, c)], []
                seen.add((r, c))
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and (nr, nc) not in seen
                                and grid[nr][nc] == digit):
                            seen.add((nr, nc))
                            stack.append((nr, nc))
                if digit == TARGET_DIGIT and len(comp) > 1:
                    total += len(comp)
        return total
    raise ValueError(rule)


def _decompose(total: int, values: list[int], rng: random.Random) -> list[int]:
    """Random decomposition of `total` into parts drawn from `values`."""
    reachable = [False] * (total + 1)
    reachable[0] = True
    for n in range(1, total + 1):
        reachable[n] = any(n >= v and reachable[n - v] for v in values)
    if not reachable[total]:
        raise ValueError(f"{total} not representable from {values}")
    parts: list[int] = []
    remaining = total
    while remaining:
        options = [v for v in values if v <= remaining and reachable[remaining - v]]
        part = rng.choice(options)
        parts.append(part)
        remaining -= part
    return parts


class GridRuleExecutionTask(Task):
    """Ladder rung 4: count probe over a stated rule on a 2D digit grid."""

    name = "grid_rule_execution"
    level = 8
    description = "Single-integer count probe over an explicitly stated 2D grid rule."

    MIN_LOAD = {"grow_region": 10, "recolor_merge": 1, "erase_singletons": 2}

    def __init__(self, rule: str = "grow_region", **options: Any) -> None:
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}; known: {sorted(RULES)}")
        super().__init__(rule=rule, **options)
        self.rule = rule

    # -- rectangle selection ----------------------------------------------

    def _rects(self, n: int, rng: random.Random) -> list[tuple[int, int, str]]:
        rects: list[tuple[int, int, str]] = []
        if self.rule == "grow_region":
            for t in _decompose(n, GROW_VALUES, rng):
                _, h, w = rng.choice([p for p in GROW_PARTS if p[0] == t])
                rects.append((h, w, TARGET_DIGIT))
        elif self.rule == "recolor_merge":
            n7 = rng.randint(0, n)
            for total, digit in ((n7, TARGET_DIGIT), (n - n7, MERGE_DIGIT)):
                if total:
                    for a in _decompose(total, list(AREA_DIMS), rng):
                        h, w = rng.choice(AREA_DIMS[a])
                        rects.append((h, w, digit))
        else:  # erase_singletons
            for a in _decompose(n, [2, 3, 4, 6, 9], rng):
                h, w = rng.choice(AREA_DIMS[a])
                rects.append((h, w, TARGET_DIGIT))
            rects.extend([(1, 1, TARGET_DIGIT)] * rng.randint(1, 4))
        # distractor rectangles, roughly one per three special rectangles
        for _ in range(max(1, len(rects) // 3)):
            h, w = rng.randint(1, 3), rng.randint(1, 3)
            rects.append((h, w, rng.choice(DISTRACTORS)))
        rng.shuffle(rects)
        return rects

    # -- placement ---------------------------------------------------------

    @staticmethod
    def _pack(rects: list[tuple[int, int, str]],
              rng: random.Random) -> list[list[str]]:
        total_area = sum(h * w for h, w, _ in rects)
        widest = max(w for _, w, _ in rects)
        width = max(widest + 2 * MARGIN,
                    int((total_area * 6) ** 0.5) + 2 * MARGIN)
        x, y, shelf_h = MARGIN, MARGIN, 0
        placed: list[tuple[int, int, int, int, str]] = []
        for h, w, digit in rects:
            if x + w + MARGIN > width:
                x = MARGIN
                y += shelf_h + GAP + rng.randint(0, 1)
                shelf_h = 0
            placed.append((y, x, h, w, digit))
            x += w + GAP + rng.randint(0, 2)
            shelf_h = max(shelf_h, h)
        height = y + shelf_h + MARGIN
        grid = [[BACKGROUND] * width for _ in range(height)]
        for r, c, h, w, digit in placed:
            for i in range(h):
                for j in range(w):
                    grid[r + i][c + j] = digit
        return grid

    def build(self, target: int, rng: random.Random) -> Stimulus:
        n = max(target, self.MIN_LOAD[self.rule])
        grid = self._pack(self._rects(n, rng), rng)
        answer = simulate(self.rule, grid)
        assert answer == n, (self.rule, target, answer)
        question = (
            f"Apply this rule to the grid below: {RULES[self.rule]}. "
            f"After applying the rule, how many cells contain the digit "
            f"{TARGET_DIGIT}?"
        )
        rows = "\n".join(" ".join(row) for row in grid)
        return Stimulus(
            target=answer,
            system=SYSTEM_RUNG,
            user=f"{question}\n\n{rows}",
            metadata={"rule": self.rule,
                      "grid_h": len(grid), "grid_w": len(grid[0]),
                      "grid_cells": len(grid) * len(grid[0])},
        )
