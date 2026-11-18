"""Task interface.

A task turns a target integer into a prompt whose correct answer is that
integer. That is the whole contract, and it is what lets one search strategy and
one runner drive every assay in the paper.

Each task declares a `level`, its position on the complexity ladder:

    1  homogeneous counting        one symbol, one accumulator
    2  motif-perturbed counting    same accumulator, varying surface syntax
    3  nested key-path matching    parse + compare + accumulate
    4  dual-task counting          accumulate while executing a second task
    5  state-machine dual task     accumulate while running a machine of
                                   graded state complexity (levels 0-4)

Levels are advisory -- nothing dispatches on them -- but they keep the ladder
legible and let `counting list-tasks` print it in order. To add a rung:

    1. subclass Task, implement `build`
    2. register it in tasks/__init__.py
    3. add configs/tasks/<name>.yaml

The runner, the search strategies, the storage layer, and the analysis layer all
stay untouched.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from ..types import Stimulus

SYSTEM_INSTRUCTION = (
    "You count items exactly. Return only one integer with no words, "
    "punctuation, or explanation."
)


class Task(ABC):
    """One assay."""

    name: str = "task"
    level: int = 1
    description: str = ""

    def __init__(self, **options: Any) -> None:
        self.options = options

    @abstractmethod
    def build(self, target: int, rng: random.Random) -> Stimulus:
        """Return a stimulus whose correct answer is exactly `target`.

        `rng` is the run's seeded generator. Draw every random choice from it so
        a run reproduces from its config plus its seed.
        """

    def parse(self, text: str | None) -> int | None:
        """Extract the answer from the response.

        The default reads the last integer in the reply, which is right for any
        task that asks for a bare integer. Override it when the task asks for
        something structured -- the dual-task assay asks for JSON, so it reads
        the `count` field instead of trusting the last number on the line.
        """
        from ..metrics import parse_integer

        return parse_integer(text)

    def config_record(self) -> dict[str, Any]:
        """What to store in the run manifest so a run can be reconstructed."""
        return {"task": self.name, "level": self.level, **self.options}
