"""Exact state tracking in language models -- evaluation harness and analysis.

Four independent axes, each behind one interface:

    providers/  which backend serves the model
    tasks/      which assay is being run (the complexity ladder)
    search/     which lengths to spend trials on (ladder or dense sweep)
    probes/     mechanistic analysis of open-weight checkpoints

`runner.Runner` composes the first three; `cli` wires a YAML config to a run.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
