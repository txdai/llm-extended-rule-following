#!/usr/bin/env python3
"""Harvest a stratified sample of stored reasoning traces for every model.

    OPENROUTER_API_KEY=... python scripts/harvest_reasoning_traces.py \
        --activity openrouter_activity_2026-08-26.csv --per-model 16

For each model in scc.csv with logged reasoning generations, sample rows
quantile-spaced by prompt tokens (which stands in for sequence length), fetch
the stored content, and write one JSONL per model under
results/harvested-traces/. Retrieval is free; the fetched input identifies the
probe, and records whose prompt is not a counting task are tagged rather than
dropped. Resumable: already-fetched generation ids are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "harvested-traces"
API = "https://openrouter.ai/api/v1/generation/content"


def fetch(gen_id: str, key: str) -> dict | None:
    url = f"{API}?{urllib.parse.urlencode({'id': gen_id})}"
    req = urllib.request.Request(url, headers={"authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode()).get("data")
    except Exception as exc:  # noqa: BLE001
        print(f"  {gen_id}: {str(exc)[:50]}", file=sys.stderr)
        return None


def parse_probe(user_message: str) -> tuple[int | None, bool]:
    """Return (sequence_length, is_counting_probe) parsed from the prompt.

    Counting probes ask 'How many occurrences of "X" are in this sequence?'
    followed by the sequence itself; the item is homogeneous, so its
    occurrence count is the probed length regardless of delimiter.
    """
    m = re.match(
        r'How many occurrences of "(.+?)" are in this sequence\?\n+(.*)$',
        user_message,
        re.S,
    )
    if not m:
        return None, False
    item, sequence = m.group(1), m.group(2)
    return sequence.count(item), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--per-model", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]

    ours = sorted({r["model"] for r in csv.DictReader(
        (REPO_ROOT / "results" / "counting-data" / "scc.csv").open())}, key=len, reverse=True)

    def match(slug: str) -> str | None:
        for m in ours:
            if slug.startswith(m):
                return m
        return None

    by_model: dict[str, list[dict]] = {}
    for r in csv.DictReader(args.activity.open(encoding="utf-8")):
        if int(r["tokens_reasoning"] or 0) <= 0:
            continue
        m = match(r["model_permaslug"])
        if m:
            by_model.setdefault(m, []).append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model, rows in sorted(by_model.items()):
        rows.sort(key=lambda r: int(r["tokens_prompt"] or 0))
        n = len(rows)
        take = min(args.per_model, n)
        picks = [rows[round(i * (n - 1) / max(take - 1, 1))] for i in range(take)]
        out = OUT_DIR / (model.replace("/", "__") + ".jsonl")
        done = set()
        if out.exists():
            for line in out.open():
                if line.strip():
                    done.add(json.loads(line)["generation_id"])
        for row in picks:
            if row["generation_id"] not in done:
                jobs.append((model, out, row))
    print(f"{len(by_model)} models, {len(jobs)} generations to fetch", file=sys.stderr)

    locks: dict[Path, object] = {}
    import threading
    def work(job):
        model, out, row = job
        data = fetch(row["generation_id"], key)
        time.sleep(0.25)
        if not data:
            return 0
        messages = (data.get("input") or {}).get("messages") or []
        user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        if not isinstance(user_msg, str):
            user_msg = "".join(
                part.get("text", "") for part in user_msg if isinstance(part, dict))
        seq_len, is_probe = parse_probe(user_msg)
        output = data.get("output") or {}
        record = {
            "generation_id": row["generation_id"],
            "created_at": row["created_at"],
            "model": model,
            "tokens_prompt": int(row["tokens_prompt"] or 0),
            "tokens_reasoning": int(row["tokens_reasoning"] or 0),
            "finish": row.get("finish_reason_normalized"),
            "sequence_length": seq_len,
            "counting_task": is_probe,
            "question": user_msg.split("\n")[0][:120],
            "completion": output.get("completion"),
            "reasoning": output.get("reasoning"),
        }
        lock = locks.setdefault(out, threading.Lock())
        with lock:
            with out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record) + "\n")
        return 1

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        got = sum(pool.map(work, jobs))
    print(f"fetched {got}/{len(jobs)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
