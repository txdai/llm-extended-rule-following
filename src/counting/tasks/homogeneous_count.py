"""Level 1-2: count a homogeneous sequence.

The baseline assay of the paper: N copies of one symbol joined by one delimiter,
and the model returns N. With `item`/`delimiter` left at their defaults this is
the exact stimulus every SCC number in the paper was measured on -- do not change
the defaults without renaming the task.

Varying `item` and `delimiter` gives the motif-perturbation sweep (level 2). The
motif variants that were previously listed in benchmark_modified_config.json now
live in configs/tasks/motif_count.yaml.
"""

from __future__ import annotations

import random
from typing import Any

from ..types import Stimulus
from .base import SYSTEM_INSTRUCTION, Task


class HomogeneousCountTask(Task):
    name = "homogeneous_count"
    level = 1
    description = "Count N repetitions of a single symbol joined by a fixed delimiter."

    def __init__(
        self,
        item: str = "a",
        delimiter: str = ", ",
        item_label: str | None = None,
        delimiter_label: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(
            item=item,
            delimiter=delimiter,
            item_label=item_label,
            delimiter_label=delimiter_label,
            **options,
        )
        self.item = item
        self.delimiter = delimiter
        self.item_label = item_label or item
        self.delimiter_label = delimiter_label

    def build(self, target: int, rng: random.Random) -> Stimulus:
        sequence = self.delimiter.join(self.item for _ in range(target))
        return Stimulus(
            target=target,
            system=SYSTEM_INSTRUCTION,
            user=(
                f'How many occurrences of "{self.item_label}" are in this sequence?'
                f"\n\n{sequence}"
            ),
            metadata={
                "item": self.item,
                "delimiter": self.delimiter,
                "item_label": self.item_label,
                "delimiter_label": self.delimiter_label,
            },
        )


class MotifCountTask(HomogeneousCountTask):
    """Level 2: the same accumulator under a perturbed surface form.

    Behaviourally identical to HomogeneousCountTask -- it exists so that motif
    sweeps are labelled as their own rung of the ladder in run records and in
    `counting list-tasks`, rather than silently overwriting baseline results.
    """

    name = "motif_count"
    level = 2
    description = "Homogeneous counting with a substituted symbol or delimiter."
