#!/usr/bin/env python3
"""Sample benchmark items into a manifest for the level-4 dual-task assay.

    python scripts/build_dual_task_manifest.py --output results/counting-dual-task/manifest.json

The dual-task conditions need real benchmark prompts: the `benchmark` arm asks
the model to solve one, and every length-matched control sizes its distractor to
that prompt's token count. Sampling is done once, up front, and written to a
manifest so a run is reproducible and does not re-download datasets -- point
`task.manifest` in configs/experiments/dual_task.yaml at the result.

Needs the dual-task extras:  pip install -e '.[dual-task]'

Datasets sampled (matching the paper): BBH reasoning tasks, CRUXEval-O,
MATH-500, and MMLU-Pro with mathematics and computer science excluded, since
those two categories overlap the counting task's own arithmetic demands.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEFAULT_DATASETS: dict[str, dict[str, Any]] = {
    "bbh_reasoning": {
        "path": "lukaemon/bbh",
        "split": "test",
        "tasks": [
            "date_understanding",
            "logical_deduction_three_objects",
            "tracking_shuffled_objects_three_objects",
            "temporal_sequences",
        ],
    },
    "cruxeval_o": {"path": "cruxeval-org/cruxeval", "split": "test"},
    "math500": {"path": "HuggingFaceH4/MATH-500", "split": "test"},
    "mmlu_pro": {
        "path": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "exclude_categories": ["math", "computer science"],
    },
}


def load_rows(path: str, split: str, name: str | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "this script needs the dual-task extras: pip install -e '.[dual-task]'"
        ) from exc
    dataset = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
    return [dict(row) for row in dataset]


# -- per-dataset item builders ----------------------------------------------

def build_mmlu(config: dict[str, Any]) -> list[dict[str, Any]]:
    excluded = {str(c).lower() for c in config.get("exclude_categories", [])}
    items = []
    for row in load_rows(config["path"], config.get("split", "test")):
        if str(row.get("category", "")).lower() in excluded:
            continue
        options = list(row["options"])
        answer = str(row.get("answer") or "").strip().upper()
        if not answer and row.get("answer_index") is not None:
            answer = LETTERS[int(row["answer_index"])]
        lines = [
            "Answer the following multiple-choice question. "
            "Return the option letter only in JSON.",
            "",
            f"Question: {row['question']}",
            "",
            "Options:",
            *(f"{LETTERS[i]}. {option}" for i, option in enumerate(options)),
        ]
        items.append(
            {
                "item_id": f"mmlu_pro:{row.get('question_id', len(items))}",
                "category": "mmlu_pro",
                "answer_type": "mcq_letter",
                "prompt": "\n".join(lines),
                "answer": answer,
                "source": config["path"],
            }
        )
    return items


def build_math500(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": f"math500:{row.get('unique_id', index)}",
            "category": "math500",
            "answer_type": "math",
            "prompt": (
                "Solve the following math problem. Return only the final answer in JSON.\n\n"
                f"Problem: {row['problem']}"
            ),
            "answer": str(row["answer"]),
            "source": config["path"],
        }
        for index, row in enumerate(load_rows(config["path"], config.get("split", "test")))
    ]


def build_cruxeval(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": f"cruxeval_o:{row.get('id', index)}",
            "category": "cruxeval_o",
            "answer_type": "python_literal",
            "prompt": (
                "Predict the exact Python return value for this function call. "
                "Return a Python literal as a JSON string.\n\n"
                f"Code:\n{row['code']}\n\nCall:\nf({row['input']})"
            ),
            "answer": str(row["output"]),
            "source": config["path"],
        }
        for index, row in enumerate(load_rows(config["path"], config.get("split", "test")))
    ]


def build_bbh(config: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for task in config.get("tasks") or ["date_understanding"]:
        rows = load_rows(config["path"], config.get("split", "test"), task)
        for index, row in enumerate(rows):
            items.append(
                {
                    "item_id": f"bbh_reasoning:{task}:{index}",
                    "category": "bbh_reasoning",
                    "answer_type": "short_text",
                    "prompt": (
                        "Answer the following reasoning problem. "
                        "Return only the final answer in JSON.\n\n"
                        f"{row.get('input', row.get('question'))}"
                    ),
                    "answer": str(row.get("target", "")),
                    "source": f"{config['path']}:{task}",
                }
            )
    return items


BUILDERS = {
    "bbh_reasoning": build_bbh,
    "cruxeval_o": build_cruxeval,
    "math500": build_math500,
    "mmlu_pro": build_mmlu,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="manifest path to write")
    parser.add_argument(
        "--sample-size", type=int, default=24, help="items sampled per category"
    )
    parser.add_argument("--seed", type=int, default=23, help="sampling seed")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(BUILDERS),
        default=sorted(BUILDERS),
    )
    args = parser.parse_args()

    selected: list[dict[str, Any]] = []
    for category in args.categories:
        print(f"loading {category} ...", file=sys.stderr)
        items = BUILDERS[category](DEFAULT_DATASETS[category])
        if len(items) < args.sample_size:
            raise SystemExit(
                f"{category} has {len(items)} items, need {args.sample_size}"
            )
        # Seeded per category so adding a category does not reshuffle the others.
        rng = random.Random(f"{args.seed}:{category}:manifest")
        chosen = sorted(rng.sample(items, args.sample_size), key=lambda item: item["item_id"])
        selected.extend(chosen)
        print(f"  sampled {len(chosen)} of {len(items)}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "sample_size": args.sample_size,
                "categories": args.categories,
                "items": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(selected)} items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
