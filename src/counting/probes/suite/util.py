"""Shared plumbing for the probe suite: tokenizer unwrapping, scan-output loading.

VLM checkpoints hand back an `AutoProcessor` whose `__call__` routes the first
positional argument to `images`, so anything that needs raw token ids has to
reach the wrapped tokenizer explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def underlying_tokenizer(tokenizer: Any) -> Any:
    return getattr(tokenizer, "tokenizer", tokenizer)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return list(underlying_tokenizer(tokenizer).encode(text, add_special_tokens=False))


def decode_tokens(tokenizer: Any, ids: Any) -> str:
    return underlying_tokenizer(tokenizer).decode(ids, skip_special_tokens=True)


def load_summary(scan_dir: Path | str) -> dict:
    return json.loads((Path(scan_dir) / "summary.json").read_text())


def load_records(scan_dir: Path | str) -> list[dict]:
    path = Path(scan_dir) / "records.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def correctness_by_count(scan_dir: Path | str) -> dict[int, bool]:
    return {r["length"]: bool(r["correct"]) for r in load_records(scan_dir)}


def scanned_scc(scan_dir: Path | str) -> int:
    return int(load_summary(scan_dir)["transition"]["scc"])
