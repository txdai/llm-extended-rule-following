"""Level 4: count while doing something else.

The interference experiment from the paper. The model must return a JSON object
carrying the count, and -- depending on the condition -- also the answer to a
second task occupying the same context:

    plain             count only; the floor for the other conditions
    irrelevant_code   count, with token-matched but inert code alongside
    irrelevant_text   count, with a token-matched prose passage alongside
    dual_counting     count 'a' markers and a second run of 'b' markers
    benchmark         count, and solve a real benchmark item (BBH, CRUXEval-O,
                      MATH-500, MMLU-Pro)

The controls are length-matched to the benchmark condition, which is the point of
the experiment: it separates the cost of *executing* a second task from the cost
of merely carrying more tokens.

Benchmark items are read from a manifest rather than downloaded, so a run is
reproducible and does not need the `datasets` package. Regenerate a manifest with
scripts/build_dual_task_manifest.py.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

from ..types import Stimulus
from .base import Task

SYSTEM_INSTRUCTION = (
    "You are being evaluated. Answer the requested task exactly. "
    "Return only a single JSON object with the requested keys. "
    "Do not include markdown, comments, hidden reasoning, or extra text."
)

CONTROL_WORDS = (
    "archive beacon corridor delta enamel fabric granite harbor index junction kernel "
    "lantern meadow nickel orbit parcel quartz rivet summit tablet umber velvet window "
    "yonder zephyr amber bracket cedar diagram ember fixture gallery horizon isotope"
).split()

CODE_NAMES = ["alpha", "beta", "gamma", "delta", "theta", "omega", "sigma", "kappa"]

CONDITIONS = ("plain", "irrelevant_code", "irrelevant_text", "dual_counting", "benchmark")


# -- token budgeting ---------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Token count via tiktoken when available, else a 4-chars-per-token estimate."""
    try:
        import tiktoken
    except ImportError:
        return max(1, math.ceil(len(text) / 4))
    try:
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001 - encoding lookup can fail offline
        return max(1, math.ceil(len(text) / 4))


def marker_sequence(count: int, marker: str = "a") -> str:
    return ", ".join(marker for _ in range(max(0, count)))


def _bisect_units(builder: Any, target_tokens: int) -> Any:
    """Smallest unit count whose rendered text reaches `target_tokens`."""
    target_tokens = max(1, int(target_tokens))
    units = 1
    while estimate_tokens(str(builder(units))) < target_tokens:
        units *= 2
    low, high = 1, units
    while low < high:
        mid = (low + high) // 2
        if estimate_tokens(str(builder(mid))) < target_tokens:
            low = mid + 1
        else:
            high = mid
    return low


def count_for_target_tokens(target_tokens: int) -> int:
    return _bisect_units(lambda n: marker_sequence(n, "a"), target_tokens)


def irrelevant_text(target_tokens: int, seed: str) -> str:
    def builder(sentence_count: int) -> str:
        rng = random.Random(f"{seed}:sentences:{sentence_count}")
        return " ".join(
            " ".join(rng.choice(CONTROL_WORDS) for _ in range(12)).capitalize()
            + f" note {index}."
            for index in range(sentence_count)
        )

    return builder(_bisect_units(builder, target_tokens))


def irrelevant_code(target_tokens: int, seed: str) -> str:
    def builder(line_count: int) -> str:
        rng = random.Random(f"{seed}:code:{line_count}")
        lines = ["def unused_transform(values):", "    total = 0"]
        for index in range(line_count):
            name = rng.choice(CODE_NAMES)
            value = rng.randint(2, 97)
            lines.append(f"    {name}_{index} = ({value} * {index + 3}) % {value + 11}")
            lines.append(f"    total += {name}_{index}")
        lines.append("    return total")
        return "\n".join(lines)

    return builder(_bisect_units(builder, target_tokens))


# -- response parsing --------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    """Pull the JSON object out of a reply, fenced or bare."""
    if not text:
        return None
    for candidate in _candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _candidates(text: str) -> list[str]:
    stripped = text.strip()
    found = [match.group(1) for match in _FENCE.finditer(stripped)]
    match = _OBJECT.search(stripped)
    if match:
        found.append(match.group(0))
    found.append(stripped)
    return found


# -- the task ----------------------------------------------------------------

class DualTaskCountTask(Task):
    name = "dual_task_count"
    level = 4
    description = "Count marker tokens while carrying, or executing, a second task."

    def __init__(
        self,
        condition: str = "plain",
        manifest: str | None = None,
        marker: str = "a",
        secondary_marker: str = "b",
        **options: Any,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(
                f"unknown dual-task condition {condition!r}; expected one of {CONDITIONS}"
            )
        super().__init__(
            condition=condition,
            manifest=manifest,
            marker=marker,
            secondary_marker=secondary_marker,
            **options,
        )
        self.condition = condition
        self.marker = marker
        self.secondary_marker = secondary_marker
        self._manifest_path = Path(manifest) if manifest else None
        self._items: list[dict[str, Any]] | None = None

    @property
    def items(self) -> list[dict[str, Any]]:
        """Benchmark items backing the length-matched and benchmark conditions."""
        if self._items is None:
            if self._manifest_path is None:
                raise ValueError(
                    f"dual-task condition {self.condition!r} needs a benchmark manifest; "
                    "set `manifest:` in the task config "
                    "(build one with scripts/build_dual_task_manifest.py)"
                )
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            self._items = payload["items"] if isinstance(payload, dict) else payload
            if not self._items:
                raise ValueError(f"manifest {self._manifest_path} contains no items")
        return self._items

    def build(self, target: int, rng: random.Random) -> Stimulus:
        primary = marker_sequence(target, self.marker)
        metadata: dict[str, Any] = {"condition": self.condition, "marker": self.marker}

        if self.condition == "plain":
            user = (
                "Count the marker sequence. Return JSON with exactly this schema: "
                f'{{"count": <number of {self.marker} markers>}}.\n\n'
                f"Marker sequence:\n{primary}"
            )
            return Stimulus(target, SYSTEM_INSTRUCTION, user, metadata)

        item = rng.choice(self.items)
        metadata["item_id"] = item.get("item_id")
        metadata["category"] = item.get("category")
        matched_tokens = estimate_tokens(item["prompt"])
        metadata["matched_tokens"] = matched_tokens
        seed = f"{self.condition}:{item.get('item_id')}:{target}"

        if self.condition == "benchmark":
            user = (
                "You must answer the task and count the marker sequence. Return JSON with "
                'exactly this schema: {"task_answer": <answer to Task>, '
                f'"count": <number of {self.marker} markers>}}.\n\n'
                f"Task:\n{item['prompt']}\n\n"
                f"Marker sequence:\n{primary}"
            )
            metadata["task_answer"] = item.get("answer")
            return Stimulus(target, SYSTEM_INSTRUCTION, user, metadata)

        if self.condition == "irrelevant_code":
            distractor = irrelevant_code(matched_tokens, seed)
            instruction = (
                "Count only the a markers in the marker sequence. Ignore the unrelated code."
            )
            schema = f'{{"count": <number of {self.marker} markers>}}'
            label = "Unrelated code"
        elif self.condition == "irrelevant_text":
            distractor = irrelevant_text(matched_tokens, seed)
            instruction = (
                "Count only the a markers in the marker sequence. Ignore the unrelated passage."
            )
            schema = f'{{"count": <number of {self.marker} markers>}}'
            label = "Unrelated passage"
        else:  # dual_counting
            secondary_count = count_for_target_tokens(matched_tokens)
            metadata["secondary_count"] = secondary_count
            distractor = marker_sequence(secondary_count, self.secondary_marker)
            instruction = "Count both marker sequences exactly."
            schema = (
                f'{{"count": <number of {self.marker} markers>, '
                f'"secondary_count": <number of {self.secondary_marker} markers>}}'
            )
            label = "Secondary marker sequence"

        user = (
            f"{instruction} Return JSON with exactly this schema: {schema}.\n\n"
            f"{label}:\n{distractor}\n\n"
            f"Marker sequence:\n{primary}"
        )
        return Stimulus(target, SYSTEM_INSTRUCTION, user, metadata)

    def parse(self, text: str | None) -> int | None:
        """Read `count` out of the JSON reply.

        Deliberately strict: falling back to "last integer in the response" would
        pick up the secondary count, or a number from the benchmark answer, and
        silently inflate accuracy in exactly the conditions under test.
        """
        payload = extract_json_object(text)
        if payload is None:
            return None
        value = payload.get("count")
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+", value)
            if match:
                return int(match.group(0))
        return None
