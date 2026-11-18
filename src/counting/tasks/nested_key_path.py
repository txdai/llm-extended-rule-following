"""Level 3: heterogeneous nested key-path matching.

Each record carries a KEY token and a PATH field with a token nested inside
alternating bracket geometries. A record matches when KEY equals the deepest
token in PATH, and the model returns the number of matches.

The target is the number of *matches*, not the number of records: the sequence is
padded with non-matching distractors so that record-counting is not a shortcut.
Distractor volume is `max(2, ceil(negative_fraction * target) + negative_bias)`,
matching the assay reported in the paper.

Ported from results/benchmark_nested.py; the record format, bracket cycle, token
pool, and per-target seeding are unchanged, so numbers stay comparable.
"""

from __future__ import annotations

import math
import random
from typing import Any

from ..types import Stimulus
from .base import Task

# The exact pool used for the published nested-assay numbers. Changing the pool
# or its order changes every generated sequence, so leave it alone unless you
# intend to invalidate comparisons with the reported results.
TOKEN_POOL = (
    "BX", "CU", "DV", "FW", "GY", "HZ", "JQ", "KR", "LM", "NP",
    "QS", "TV", "WX", "YZ", "PC", "RF", "SD", "MN", "JK", "UV",
)

SYSTEM_INSTRUCTION = (
    "Each record has a KEY token and a nested PATH field. "
    "A record counts as a match when the KEY token is identical to the deepest "
    "token inside PATH. Count how many matching records are in the following "
    "sequence. Return only one integer."
)


def build_path(leaf: str, fillers: list[str], style: int) -> str:
    """Cycle three bracket geometries so depth cues cannot be memorised."""
    left, right = fillers
    if style % 3 == 0:
        return f"[{left}<{{{leaf}}}>{right}]"
    if style % 3 == 1:
        return f"({left}[<{leaf}>]{right})"
    return f"<{left}({{{leaf}}}){right}>"


def _pick_distinct(rng: random.Random, count: int, forbidden: set[str]) -> list[str]:
    candidates = [token for token in TOKEN_POOL if token not in forbidden]
    if len(candidates) < count:
        raise ValueError("token pool too small for nested assay construction")
    return rng.sample(candidates, count)


class NestedKeyPathTask(Task):
    name = "nested_key_path"
    level = 3
    description = "Count records whose KEY equals the deepest token nested inside PATH."

    def __init__(
        self,
        negative_fraction: float = 0.5,
        negative_bias: int = 2,
        record_delimiter: str = " ; ",
        seed_namespace: str = "heterogeneous-nested-key-path-match",
        **options: Any,
    ) -> None:
        super().__init__(
            negative_fraction=negative_fraction,
            negative_bias=negative_bias,
            record_delimiter=record_delimiter,
            seed_namespace=seed_namespace,
            **options,
        )
        self.negative_fraction = negative_fraction
        self.negative_bias = negative_bias
        self.record_delimiter = record_delimiter
        # Prefix for the per-target seed. Defaults to the assay id used for the
        # published run so regenerated sequences match the reported results.
        self.seed_namespace = seed_namespace

    def build(self, target: int, rng: random.Random) -> Stimulus:
        negative_count = max(2, math.ceil(target * self.negative_fraction) + self.negative_bias)
        labels = [True] * target + [False] * negative_count

        # Seeded per target so a given target always yields the same sequence,
        # independent of how many tiers ran before it. This is what makes a dense
        # sweep and a ladder search agree on the same target.
        local_rng = random.Random(f"{self.seed_namespace}:{target}")
        local_rng.shuffle(labels)

        records: list[str] = []
        for index, is_positive in enumerate(labels):
            key_token = local_rng.choice(TOKEN_POOL)
            leaf_token = (
                key_token
                if is_positive
                else local_rng.choice([t for t in TOKEN_POOL if t != key_token])
            )
            side = _pick_distinct(local_rng, 2, {key_token, leaf_token})
            fillers = _pick_distinct(local_rng, 2, {key_token, leaf_token, *side})
            path = build_path(leaf_token, fillers, index)
            records.append(
                f"R{index:03d}{{KEY={key_token}; PATH={path}; SIDE=({side[0]}|{side[1]})}}"
            )

        return Stimulus(
            target=target,
            system=SYSTEM_INSTRUCTION,
            user=self.record_delimiter.join(records),
            metadata={
                "record_count": len(records),
                "negative_count": negative_count,
                "negative_fraction": self.negative_fraction,
                "negative_bias": self.negative_bias,
            },
        )
