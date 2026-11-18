"""Dual-task interference: counting against the calibrated ladder.

The scanned task is the assay's own counting stimulus (answer = N). A second,
fixed-size task rides in the prompt, and the model reports two integers: the
subtask answer first, then the count. Three load kinds:

    grid     a 2D grid with K regions of 7s and 3s plus the recolor-merge rule
             and its count probe -- the rung-4 task at fixed object load K,
             set per model as a fraction of its measured solo capacity
    inert    the matched-length control: a grid of the same construction with
             no 7s or 3s, and the first integer printed in the prompt, so the
             token volume of the load is present but no state is demanded
    pointer  the anchored control: a K-link pointer chain in common words
             (nameable state; CoT can carry it)

The counting boundary is fitted from the last integer (the runner's default
parser scores it against the stimulus target unchanged). The subtask answer
is recorded in metadata (`subtask_answer`) and graded post hoc from
`raw_response`, giving three boundaries per scan: count-correct,
subtask-correct, and both-correct.
"""

from __future__ import annotations

import random
from typing import Any

from ..types import Stimulus
from .base import Task
from .grid_rule_execution import BACKGROUND, DISTRACTORS, MERGE_DIGIT
from .run_structure import TARGET_DIGIT
from .state_dual import WORDS

SEMANTIC_QA = [('In classical music, the stringed instrument slightly larger than a violin and tuned a perfect fifth lower is the viola.', 'Which stringed instrument is tuned a perfect fifth lower than a violin?', 'viola', ('flute', 'timpani', 'oboe')), ('Photosynthesis in green plants primarily takes place inside specialized organelles called chloroplasts.', 'Inside which organelle does photosynthesis primarily occur in green plants?', 'chloroplast', ('ribosome', 'lysosome', 'golgi')), ('The Nile River flows northward through northeastern Africa before emptying into the Mediterranean Sea.', 'Into which body of water does the Nile River empty?', 'Mediterranean Sea', ('Bering Strait', 'Tasman Sea', 'Hudson Bay')), ('William Shakespeare wrote the tragedy Hamlet, which is set at Elsinore Castle in Denmark.', 'In which country is the play Hamlet set?', 'Denmark', ('Portugal', 'Peru', 'Nepal')), ('Solid carbon dioxide sublimates directly into gas at atmospheric pressure and is commonly called dry ice.', 'What is solid carbon dioxide commonly called?', 'dry ice', ('heavy water', 'quicklime', 'quartz')), ('Mercury is the innermost planet of the Solar System and completes an orbit faster than Earth.', 'Which planet is the innermost planet of the Solar System?', 'Mercury', ('Neptune', 'Saturn', 'Uranus')), ('In human physiology, hemoglobin is the iron-containing protein in red blood cells that transports oxygen.', 'Which protein in red blood cells transports oxygen?', 'hemoglobin', ('keratin', 'collagen', 'elastin')), ('The Rosetta Stone helped scholars decipher Egyptian hieroglyphs because it carried a decree in three scripts.', 'Which ancient writing system did the Rosetta Stone help decipher?', 'Egyptian hieroglyphs', ('Inca quipu', 'Norse runes', 'Ogham')), ('Tectonic plates meet along boundaries where earthquakes and volcanic activity frequently concentrate.', 'Along what geological features do earthquakes and volcanoes frequently concentrate?', 'tectonic plate boundaries', ('coral lagoons', 'sand dunes', 'glacial moraines')), ('The printing press with movable metal type in Europe was pioneered by Johannes Gutenberg in Mainz.', 'Who pioneered movable metal type printing in Europe?', 'Johannes Gutenberg', ('Vasco da Gama', 'Ferdinand Magellan', 'Marco Polo')), ('Penicillin was discovered by Alexander Fleming after observing mold inhibiting bacterial growth.', 'Who discovered penicillin from mold inhibiting bacteria?', 'Alexander Fleming', ('Louis Daguerre', 'Gregor Mendel', 'Niels Bohr')), ('In architecture, a flying buttress transfers roof thrust outward to an exterior pier, typical of Gothic cathedrals.', 'Which architectural style is famous for using flying buttresses on cathedrals?', 'Gothic', ('Bauhaus', 'Brutalist', 'Minimalist'))]

FILLER_SENTENCES = ['Coastal breezes often shift direction between afternoon warmth and evening cooling along calm harbors.', 'Botanical illustrations traditionally depict leaves, stems, and petals on pale archival paper.', 'Granite boulders weather slowly under alpine frost and summer meltwater across high ridges.', 'Pottery glazes vitrify in kilns to form smooth protective surfaces over fired earthenware.', 'Migratory terns follow seasonal wind corridors over open ocean basins between hemispheres.', 'Archival bindings use linen thread and acid-free board to preserve printed folios.', 'Tidal marshes filter suspended sediment through dense stands of salt-tolerant reeds.', 'Limestone caverns form gradually as weakly acidic groundwater dissolves carbonate rock.', 'Traditional sundials project a shadow from an angled gnomon onto an engraved dial plate.', 'Orchardists graft scion wood onto hardy rootstocks during late winter dormancy.']

SYSTEM_LOAD = (
    "You follow instructions exactly. Reply with two integers separated by a "
    "single space and nothing else: first the requested number, then the count."
)

GAP = 2
MARGIN = 1


class CountWithLoadTask(Task):
    """Counting scanned under a fixed calibrated load."""

    name = "count_with_load"
    level = 9
    description = "Count a's while carrying a fixed grid, inert, or pointer load."

    def __init__(self, load_kind: str = "grid", load_size: int = 26,
                 **options: Any) -> None:
        if load_kind not in ("grid", "inert", "pointer", "row2", "row3", "semantic"):
            raise ValueError(f"unknown load_kind {load_kind!r}")
        super().__init__(load_kind=load_kind, load_size=load_size, **options)
        self.load_kind = load_kind
        self.load_size = int(load_size)

    # -- grid load ---------------------------------------------------------

    def _grid(self, rng: random.Random, special: bool) -> tuple[str, int]:
        k = self.load_size
        rects = []
        for _ in range(k):
            digit = (rng.choice((TARGET_DIGIT, MERGE_DIGIT)) if special
                     else rng.choice(DISTRACTORS))
            rects.append((rng.randint(1, 3), rng.randint(1, 3), digit))
        for _ in range(max(1, k // 3)):
            rects.append((rng.randint(1, 3), rng.randint(1, 3),
                          rng.choice(DISTRACTORS)))
        rng.shuffle(rects)
        total_area = sum(h * w for h, w, _ in rects)
        width = max(3 + 2 * MARGIN, int((total_area * 5) ** 0.5) + 2 * MARGIN)
        x, y, shelf_h = MARGIN, MARGIN, 0
        placed = []
        for h, w, digit in rects:
            if x + w + MARGIN > width:
                x = MARGIN
                y += shelf_h + GAP
                shelf_h = 0
            placed.append((y, x, h, w, digit))
            x += w + GAP
            shelf_h = max(shelf_h, h)
        height = y + shelf_h + MARGIN
        grid = [[BACKGROUND] * width for _ in range(height)]
        for r, c, h, w, digit in placed:
            for i in range(h):
                for j in range(w):
                    grid[r + i][c + j] = digit
        answer = sum(row.count(TARGET_DIGIT) + row.count(MERGE_DIGIT)
                     for row in grid)
        return "\n".join(" ".join(row) for row in grid), answer

    # -- row loads (rung-2 / rung-3 tasks at fixed unit load) --------------

    def _row(self, rng: random.Random, merge: bool) -> tuple[str, int]:
        """A 1D row with exactly `load_size` special runs; answer = special cells.

        merge=False: rung-2 color-total (runs of 7 among distractor runs).
        merge=True: rung-3 recolor-merge (runs of 7 and 3; answer counts both).
        """
        k = self.load_size
        cells: list[str] = []

        def distract(min_runs: int) -> None:
            previous = cells[-1] if cells else None
            for _ in range(rng.randint(min_runs, 2)):
                digit = rng.choice([d for d in DISTRACTORS if d != previous])
                cells.extend(digit * rng.randint(1, 4))
                previous = digit

        distract(1)
        for i in range(k):
            digit = rng.choice((TARGET_DIGIT, MERGE_DIGIT)) if merge else TARGET_DIGIT
            cells.extend(digit * rng.randint(1, 4))
            distract(1)
        answer = cells.count(TARGET_DIGIT) + (cells.count(MERGE_DIGIT) if merge else 0)
        return " ".join(cells), answer

    # -- pointer load ------------------------------------------------------

    def _chain(self, rng: random.Random) -> tuple[str, str, int]:
        m = self.load_size
        pool = list(WORDS)
        rng.shuffle(pool)
        words = pool[:]
        while len(words) < m + 1:
            words = words + [f"{a}-{b}" for a in pool for b in pool]
        chain = words[: m + 1]
        links = [f"{chain[i]} points to {chain[i + 1]}" for i in range(m)]
        rng.shuffle(links)
        value = rng.randint(10, 99)
        preamble = ("Here is a list of pointers.\n" + "\n".join(links)
                    + f"\n{chain[-1]} = {value}")
        phrase = (f"the number of the last word reached by following the "
                  f"pointers from '{chain[0]}'")
        return preamble, phrase, value


    # -- non-executive semantic control ------------------------------------

    def _semantic(self, rng: random.Random) -> tuple[str, str, int]:
        ref_grid, _ = self._grid(random.Random(rng.randint(0, 10**9)), special=False)
        target_chars = len(ref_grid)
        fact, question, correct_opt, wrong_opts = SEMANTIC_QA[rng.randrange(len(SEMANTIC_QA))]
        options = [correct_opt.strip()] + [w.strip() for w in wrong_opts]
        rng.shuffle(options)
        answer = options.index(correct_opt.strip()) + 1
        sentences = [fact]
        while sum(len(s) + 1 for s in sentences) < target_chars:
            sentences.append(FILLER_SENTENCES[rng.randrange(len(FILLER_SENTENCES))])
        insert_pos = rng.randrange(len(sentences))
        sentences[0], sentences[insert_pos] = sentences[insert_pos], sentences[0]
        passage = " ".join(sentences)
        opt_lines = " ".join(f"({i+1}) {opt}" for i, opt in enumerate(options))
        preamble = (
            "First, read this short passage and answer the comprehension question.\n\n"
            f"{passage}\n\nQuestion: {question}\nOptions: {opt_lines}"
        )
        phrase = "the option number (1 to 4) for the passage question"
        return preamble, phrase, answer

    def build(self, target: int, rng: random.Random) -> Stimulus:
        if self.load_kind == "row2":
            row, answer = self._row(rng, merge=False)
            preamble = (
                f"First, a row task. Count how many cells in this row contain "
                f"the digit {TARGET_DIGIT}.\n\n{row}"
            )
            phrase = "the row answer"
        elif self.load_kind == "row3":
            row, answer = self._row(rng, merge=True)
            preamble = (
                f"First, a row task. Apply this rule to the row below: every "
                f"cell containing {MERGE_DIGIT} becomes {TARGET_DIGIT}. After "
                f"applying the rule, count how many cells contain the digit "
                f"{TARGET_DIGIT}.\n\n{row}"
            )
            phrase = "the row answer"
        elif self.load_kind == "grid":
            grid, answer = self._grid(rng, special=True)
            preamble = (
                f"First, a grid task. Apply this rule to the grid below: every "
                f"cell containing {MERGE_DIGIT} becomes {TARGET_DIGIT}. After "
                f"applying the rule, count how many cells contain the digit "
                f"{TARGET_DIGIT}.\n\n{grid}"
            )
            phrase = "the grid answer"
        elif self.load_kind == "inert":
            grid, _ = self._grid(rng, special=False)
            answer = rng.randint(10, 99)
            preamble = (
                f"First, note this grid; it needs no work. The first integer "
                f"to report is {answer}.\n\n{grid}"
            )
            phrase = "the integer given above"
        elif self.load_kind == "semantic":
            preamble, phrase, answer = self._semantic(rng)
        else:
            preamble, phrase, answer = self._chain(rng)

        sequence = ", ".join("a" for _ in range(target))
        user = (
            f"{preamble}\n\n"
            f'Second, count the occurrences of "a" in this sequence:\n\n'
            f"{sequence}\n\n"
            f"Report {phrase}, then the count."
        )
        return Stimulus(
            target=target,
            system=SYSTEM_LOAD,
            user=user,
            metadata={"load_kind": self.load_kind,
                      "load_size": self.load_size,
                      "subtask_answer": answer},
        )
