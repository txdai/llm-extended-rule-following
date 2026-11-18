"""Search-strategy registry.

    from counting.search import build_strategy
    strategy = build_strategy("dense", start=32, stop=1024, step=32)
"""

from __future__ import annotations

from typing import Any

from .base import SearchStrategy
from .dense import DenseSearch, build_grid
from .ladder import LadderSearch
from .scan import LinearScan, classify

STRATEGIES: dict[str, type[SearchStrategy]] = {
    LadderSearch.name: LadderSearch,
    DenseSearch.name: DenseSearch,
    LinearScan.name: LinearScan,
}

ALIASES = {
    "linear": "scan",
    "scan_from_1": "scan",
    "adaptive": "ladder",
    "binary": "ladder",
    "sweep": "dense",
    "grid": "dense",
}


def build_strategy(name: str, **options: Any) -> SearchStrategy:
    key = ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if key not in STRATEGIES:
        known = ", ".join(sorted(STRATEGIES))
        raise KeyError(f"unknown search strategy {name!r}; known strategies: {known}")
    return STRATEGIES[key](**options)


__all__ = ["STRATEGIES", "SearchStrategy", "build_grid", "build_strategy", "classify"]
