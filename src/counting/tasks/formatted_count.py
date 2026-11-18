"""Output-format and modular-arithmetic controls for homogeneous counting.

Tests whether the CC collapse boundary and post-boundary constant-offset streaks
(+-100, -50) shift when multi-digit BPE numeral emission is replaced by:
  - english_words:      lowercase English number words ('one hundred forty-two')
  - zero_padded_spaced: four space-separated single digits ('0 1 4 2')
  - mod_10:             single ones digit (count modulo 10, 0..9)
  - int_bare:           baseline bare integer ('142')
"""

from __future__ import annotations

import random
import re
from typing import Any

from ..metrics import parse_integer
from ..types import Stimulus
from .base import SYSTEM_INSTRUCTION, Task

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000}
_NUM_WORDS = frozenset(_ONES) | frozenset(_TENS) | frozenset(_SCALES) | {"and"}

_INT_TO_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_INT_TO_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
]


def int_to_english(n: int) -> str:
    if n < 0 or n >= 1_000_000:
        raise ValueError(f"out of range: {n}")
    if n < 20:
        return _INT_TO_ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _INT_TO_TENS[t] if r == 0 else f"{_INT_TO_TENS[t]}-{_INT_TO_ONES[r]}"
    if n < 1000:
        h, r = divmod(n, 100)
        prefix = f"{_INT_TO_ONES[h]} hundred"
        return prefix if r == 0 else f"{prefix} {int_to_english(r)}"
    th, r = divmod(n, 1000)
    prefix = f"{int_to_english(th)} thousand"
    return prefix if r == 0 else f"{prefix} {int_to_english(r)}"


def _eval_word_tokens(tokens: list[str]) -> int | None:
    filtered = [t for t in tokens if t != "and"]
    if not filtered:
        return None
    if filtered == ["zero"]:
        return 0
    total = 0
    current = 0
    for tok in filtered:
        if tok == "zero":
            return None
        if tok in _ONES:
            current += _ONES[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok == "hundred":
            if current == 0:
                current = 1
            current *= 100
        elif tok == "thousand":
            if current == 0:
                current = 1
            total += current * 1000
            current = 0
        else:
            return None
    return total + current


def parse_english_number(text: str | None) -> int | None:
    """Parse the last contiguous English number-word phrase; reject digit-only replies."""
    if not text:
        return None
    # Reject bare digit answers so models must actually emit words
    cleaned = text.strip().lower()
    if re.fullmatch(r"[\s\d,._+-]+", cleaned):
        return None
    words = re.findall(r"[a-z]+|\d+", cleaned.replace("-", " "))
    spans: list[list[str]] = []
    cur: list[str] = []
    for w in words:
        if w in _NUM_WORDS:
            cur.append(w)
        else:
            if any(t != "and" for t in cur):
                spans.append(cur)
            cur = []
    if any(t != "and" for t in cur):
        spans.append(cur)
    for span in reversed(spans):
        val = _eval_word_tokens(span)
        if val is not None:
            return val
    return None


_SPACED4_RE = re.compile(r"(?<!\d)(\d)[ \t]+(\d)[ \t]+(\d)[ \t]+(\d)(?!\d)")


def parse_zero_padded_spaced(text: str | None) -> int | None:
    """Parse four space-separated single digits 'd3 d2 d1 d0' -> integer."""
    if not text:
        return None
    matches = _SPACED4_RE.findall(text)
    if not matches:
        return None
    d3, d2, d1, d0 = (int(x) for x in matches[-1])
    return 1000 * d3 + 100 * d2 + 10 * d1 + d0


SYSTEM_WORDS = (
    "You count items exactly. Return the exact count written out in lowercase "
    "English words only (for example, 'forty-two' or 'one hundred twenty-three'), "
    "with no digits, punctuation, or explanation."
)
SYSTEM_PADDED = (
    "You count items exactly. Return the exact count as four space-separated "
    "digits padded with leading zeros (for example, '0 0 4 2' or '0 1 2 3'), "
    "with no words, punctuation, or explanation."
)
SYSTEM_MOD10 = (
    "You count items exactly. Return only the ones digit of the exact count "
    "(the count modulo 10), as a single digit from 0 to 9, with no words, "
    "punctuation, or explanation."
)


class FormattedCountTask(Task):
    name = "formatted_count"
    level = 1
    description = "Homogeneous counting under controlled output formats (words, zero-padded, mod 10)."

    def __init__(
        self,
        output_format: str = "int_bare",
        item: str = "a",
        delimiter: str = ", ",
        **options: Any,
    ) -> None:
        if output_format not in ("int_bare", "english_words", "zero_padded_spaced", "mod_10"):
            raise ValueError(f"unknown output_format {output_format!r}")
        super().__init__(output_format=output_format, item=item, delimiter=delimiter, **options)
        self.output_format = output_format
        self.item = item
        self.delimiter = delimiter

    def build(self, target: int, rng: random.Random) -> Stimulus:
        sequence = self.delimiter.join(self.item for _ in range(target))
        if self.output_format == "int_bare":
            system = SYSTEM_INSTRUCTION
            user = f'How many occurrences of "{self.item}" are in this sequence?\n\n{sequence}'
            expected = target
        elif self.output_format == "english_words":
            system = SYSTEM_WORDS
            user = (
                f'How many occurrences of "{self.item}" are in this sequence? '
                f'Write the exact count in lowercase English words only.\n\n{sequence}'
            )
            expected = target
        elif self.output_format == "zero_padded_spaced":
            system = SYSTEM_PADDED
            user = (
                f'How many occurrences of "{self.item}" are in this sequence? '
                f'Return four space-separated digits padded with leading zeros.\n\n{sequence}'
            )
            expected = target
        else:  # mod_10
            system = SYSTEM_MOD10
            user = (
                f'What is the number of occurrences of "{self.item}" in this sequence, '
                f'modulo 10 (the last digit of the exact count)?\n\n{sequence}'
            )
            expected = target % 10
        return Stimulus(
            target=expected,
            system=system,
            user=user,
            metadata={
                "output_format": self.output_format,
                "item": self.item,
                "delimiter": self.delimiter,
                "full_count": target,
            },
        )

    def parse(self, text: str | None) -> int | None:
        if self.output_format == "english_words":
            return parse_english_number(text)
        if self.output_format == "zero_padded_spaced":
            return parse_zero_padded_spaced(text)
        return parse_integer(text)
