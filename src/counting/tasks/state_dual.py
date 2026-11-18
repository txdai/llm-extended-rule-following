"""Dual-task counting against a state machine of graded complexity.

The subtask beside the count is an explicit machine the model must run. Its
load scales with the count: every level processes a stream of
m = max(3, round(load_fraction * N)) elements, one quarter of the counted
length by default, so at boundary-scale lengths the rival computation is
comparable in weight to the count itself rather than a token gesture. The five
levels then differ only in the kind of state the machine needs:

    0  none        m lines of inert text; the second answer is printed in the
                   question itself
    1  read        one lookup in a table of m word-to-number entries
    2  chase       follow a shuffled pointer chain of m links from a start
                   word to its end, and report the final word's number
    3  accumulate  one register updated by m signed increments; updates commute
    4  automaton   a small finite-state machine driven by m input words;
                   updates do not commute, so order must be tracked

A plain "hold this value" level is deliberately absent. In a transformer the
context is the memory: a value stated in the prompt can be attended to at
answer time, so holding it demands no state at all. Genuine state begins where
the answer appears nowhere in the context and must be derived. Level 1 needs
one dependent reference, level 2 a sequence of m dependent references, and
levels 3 and 4 a derived value that no reference can reach, without and with
order sensitivity.

Keys and symbols are common English words, hyphenated pairs of common words
once the pool is exhausted; the motif sweep showed capacity is sensitive to
unusual symbols, so rare tokens would confound level with syntax. All sampling
draws from the run's seeded generator.

Every level shares one layout and one answer format: two integers, the subtask
answer first and the count last, so the default last-integer parser scores the
counting side unchanged and the subtask answer is graded from the raw reply.
"""

from __future__ import annotations

import random
from typing import Any

from ..types import Stimulus
from .base import Task

# Common concrete nouns: familiar to every tokenizer, no rare-symbol effects.
WORDS = (
    "apple", "river", "chair", "music", "garden", "window", "bridge", "candle",
    "forest", "hammer", "island", "jacket", "kitchen", "ladder", "market",
    "needle", "orange", "pencil", "quarter", "rabbit", "silver", "table",
    "umbrella", "valley", "wagon", "yellow", "zebra", "anchor", "basket",
    "carpet", "dolphin", "engine", "feather", "guitar", "harbor", "iron",
    "jungle", "kettle", "lantern", "mirror",
)

SYSTEM_DUAL = (
    "You follow instructions exactly. Reply with two integers separated by a "
    "single space and nothing else: first the requested number, then the count."
)


class StateDualTask(Task):
    name = "state_dual"
    level = 5
    description = "Count while running a state machine of graded complexity (levels 0-4)."

    def __init__(
        self,
        machine_level: int = 0,
        load_fraction: float = 0.25,
        n_states: int = 4,
        n_symbols: int = 3,
        item: str = "a",
        delimiter: str = ", ",
        **options: Any,
    ) -> None:
        super().__init__(
            machine_level=machine_level, load_fraction=load_fraction,
            n_states=n_states, n_symbols=n_symbols,
            item=item, delimiter=delimiter, **options,
        )
        if machine_level not in (0, 1, 2, 3, 4):
            raise ValueError(f"machine_level must be 0-4, got {machine_level}")
        self.machine_level = machine_level
        self.load_fraction = float(load_fraction)
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.item = item
        self.delimiter = delimiter

    def _load(self, target: int) -> int:
        return max(3, round(self.load_fraction * target))

    def _words(self, count: int, rng: random.Random) -> list[str]:
        """`count` distinct keys: single common words, then hyphenated pairs."""
        if count <= len(WORDS):
            return rng.sample(WORDS, count)
        pairs = [f"{a}-{b}" for a in WORDS for b in WORDS if a != b]
        return list(rng.sample(WORDS, len(WORDS))) + rng.sample(pairs, count - len(WORDS))

    # -- level pieces -------------------------------------------------------

    def _level0(self, target: int, rng: random.Random):
        m = self._load(target)
        words = self._words(2 * m, rng)
        filler = "\n".join(
            f"The {words[2*i]} is near the {words[2*i+1]}." for i in range(m)
        )
        value = rng.randint(10, 99)
        preamble = "Some unrelated notes.\n" + filler
        return preamble, "", f"the number {value}", value

    def _level1(self, target: int, rng: random.Random):
        m = self._load(target)
        words = self._words(m, rng)
        values = [rng.randint(10, 99) for _ in words]
        key_index = rng.randrange(m)
        table = "\n".join(f"{w} = {v}" for w, v in zip(words, values))
        preamble = "Here is a table of words and numbers.\n" + table
        return preamble, "", f"the number for '{words[key_index]}'", values[key_index]

    def _level2(self, target: int, rng: random.Random):
        m = self._load(target)
        chain = self._words(m + 1, rng)
        links = [f"{chain[i]} points to {chain[i+1]}" for i in range(m)]
        rng.shuffle(links)                     # in order, the chain reads off the page
        value = rng.randint(10, 99)
        preamble = (
            "Here is a list of pointers.\n" + "\n".join(links)
            + f"\n{chain[-1]} = {value}"
        )
        phrase = (
            f"the number of the last word reached by following the pointers "
            f"from '{chain[0]}'"
        )
        return preamble, "", phrase, value

    def _level3(self, target: int, rng: random.Random):
        m = self._load(target)
        start = rng.randint(10, 99)
        total = start
        lines = []
        for _ in range(m):
            delta = rng.randint(1, 9)
            if rng.random() < 0.5 and total - delta >= 0:
                total -= delta
                lines.append(f"subtract {delta}")
            else:
                total += delta
                lines.append(f"add {delta}")
        preamble = f"A running total starts at {start}. Apply every update below, in order."
        updates = "Updates:\n" + "\n".join(lines)
        return preamble, updates, "the final total", total

    def _level4(self, target: int, rng: random.Random):
        m = self._load(target)
        symbols = self._words(self.n_symbols, rng)
        table = {
            (state, symbol): rng.randrange(self.n_states)
            for state in range(self.n_states)
            for symbol in symbols
        }
        rules = "\n".join(
            f"In state {state}, '{symbol}' moves to state {table[(state, symbol)]}."
            for state in range(self.n_states)
            for symbol in symbols
        )
        start = rng.randrange(self.n_states)
        stream = [rng.choice(symbols) for _ in range(m)]
        state = start
        for symbol in stream:
            state = table[(state, symbol)]
        preamble = (
            f"A machine has states 0 to {self.n_states - 1} and starts in state {start}. "
            "It follows these rules.\n" + rules
        )
        updates = "Inputs, in order: " + ", ".join(stream)
        return preamble, updates, "the machine's final state", state

    # -- assembly -----------------------------------------------------------

    def build(self, target: int, rng: random.Random) -> Stimulus:
        builders = (self._level0, self._level1, self._level2,
                    self._level3, self._level4)
        preamble, updates, phrase, answer = builders[self.machine_level](target, rng)
        sequence = self.delimiter.join(self.item for _ in range(target))
        parts = [part for part in (
            preamble,
            f"Count the '{self.item}' items in this sequence.",
            sequence,
            updates,
            f"Report {phrase} and the count.",
        ) if part]
        return Stimulus(
            target=target,
            system=SYSTEM_DUAL,
            user="\n\n".join(parts),
            metadata={
                "machine_level": self.machine_level,
                "subtask_answer": answer,
                "machine_load": self._load(target),
            },
        )


SYSTEM_SOLO = (
    "You follow instructions exactly. Reply with one integer and nothing else."
)


class StateSoloTask(StateDualTask):
    """One ladder level alone: the machine without the count.

    The scan variable is the machine load itself -- table size for level 1,
    chain length, update count, or input-stream length above it -- so the fitted
    transition is the level's own capacity, measured by the same estimator as
    counting. The correct answer is the machine's output, not the load, so the
    stimulus target diverges from the probed length by design; answers are
    two-digit values (or machine states) and never equal the load, so a model
    cannot score by counting the lines of its own subtask.
    """

    name = "state_solo"
    level = 5
    description = "One state-machine level alone; scan variable is machine load."

    def _load(self, target: int) -> int:
        return max(1, target)

    def build(self, target: int, rng: random.Random) -> Stimulus:
        builders = (self._level0, self._level1, self._level2,
                    self._level3, self._level4)
        preamble, updates, phrase, answer = builders[self.machine_level](target, rng)
        parts = [part for part in (preamble, updates, f"Report {phrase}.") if part]
        return Stimulus(
            target=answer,
            system=SYSTEM_SOLO,
            user="\n\n".join(parts),
            metadata={
                "machine_level": self.machine_level,
                "machine_load": self._load(target),
            },
        )
