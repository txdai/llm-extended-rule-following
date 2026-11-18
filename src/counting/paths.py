"""Canonical locations, so no module has to guess how deep it sits.

Every analysis and figure script imports from here instead of recomputing
`Path(__file__).parents[N]`. When the layout moves again, this is the one file
that changes.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS = REPO_ROOT / "configs"
RESULTS = REPO_ROOT / "results"
PAPER = REPO_ROOT / "paper"
FIGURES = PAPER / "figures"
GENERATED = PAPER / "generated"
SCRIPTS = REPO_ROOT / "scripts"

# API evaluation runs.
STABLE_SEARCH = RESULTS / "counting-stable-search"
STABLE_SEARCH_OPEN = RESULTS / "counting-stable-search-open-source"
STABLE_MODIFIED = RESULTS / "counting-stable-modified"
STABLE_NESTED = RESULTS / "counting-stable-nested"
BENCHMARK_DATA = RESULTS / "benchmark_data"

# Mechanistic probe runs, consolidated under one tree.
PROBES = RESULTS / "probes"
GEMMA_PROBES = PROBES / "gemma"
QWEN_PROBES = PROBES / "qwen"

GEMMA_COUNTING = GEMMA_PROBES / "gemma3-gemmascope2-counting"
GEMMA_MOTIF = GEMMA_PROBES / "gemma3-gemmascope2-motif-invariance"
QWEN_COUNTING = QWEN_PROBES / "qwen3.5-35b-a3b-counting"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "BENCHMARK_DATA",
    "CONFIGS",
    "FIGURES",
    "GEMMA_COUNTING",
    "GEMMA_MOTIF",
    "GEMMA_PROBES",
    "GENERATED",
    "PAPER",
    "PROBES",
    "QWEN_COUNTING",
    "QWEN_PROBES",
    "REPO_ROOT",
    "RESULTS",
    "SCRIPTS",
    "STABLE_MODIFIED",
    "STABLE_NESTED",
    "STABLE_SEARCH",
    "STABLE_SEARCH_OPEN",
    "ensure",
]
