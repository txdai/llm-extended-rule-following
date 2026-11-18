#!/usr/bin/env python3
"""Harvest stored reasoning traces from OpenRouter's generation logs.

    python scripts/fetch_generation_traces.py --activity openrouter_activity_2026-08-26.csv \
        --model anthropic/claude-sonnet-5 --min-reasoning 1 --limit 20 \
        --out results/harvested-traces/sonnet-5.jsonl

Every generation this account has run is listed in the activity export with its
generation id. For generations made while input/output logging was enabled, the
content endpoint returns the full input messages, the completion, and the
reasoning stream. Retrieval is free of model cost and the fetched input makes
each record self-identifying, so no join against trial logs is needed: the
probed length is recomputed from the prompt itself.

The output is JSONL, one record per generation, and the script resumes: ids
already present in the output file are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
import os

API = "https://openrouter.ai/api/v1/generation/content"


def fetch(gen_id: str, key: str) -> dict | None:
    url = f"{API}?{urllib.parse.urlencode({'id': gen_id})}"
    req = urllib.request.Request(url, headers={"authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode()).get("data")
    except Exception as exc:  # noqa: BLE001 - a missing log is data, not a crash
        print(f"  {gen_id}: {str(exc)[:60]}", file=sys.stderr)
        return None


def sequence_length(user_message: str) -> int | None:
    """Items in the counted sequence, recomputed from the prompt itself."""
    best = 0
    for line in user_message.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"(?:\S+(?:, | \| | § | ))*\S+", stripped) and stripped.count(",") > best:
            best = stripped.count(",")
    return best + 1 if best else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--model", required=True, help="model_permaslug prefix")
    parser.add_argument("--min-reasoning", type=int, default=1)
    parser.add_argument("--min-prompt", type=int, default=0)
    parser.add_argument("--max-prompt", type=int, default=10**9)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=0.4)
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")

    done = set()
    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["generation_id"])

    rows = [r for r in csv.DictReader(args.activity.open(encoding="utf-8"))
            if r["model_permaslug"].startswith(args.model)
            and int(r["tokens_reasoning"] or 0) >= args.min_reasoning
            and args.min_prompt <= int(r["tokens_prompt"] or 0) <= args.max_prompt
            and r["generation_id"] not in done]
    rows.sort(key=lambda r: r["created_at"])
    print(f"{len(rows)} candidate generations; fetching up to {args.limit}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fetched = 0
    with args.out.open("a", encoding="utf-8") as sink:
        for row in rows:
            if fetched >= args.limit:
                break
            data = fetch(row["generation_id"], key)
            time.sleep(args.sleep)
            if not data:
                continue
            messages = (data.get("input") or {}).get("messages") or []
            user_msg = next((m.get("content", "") for m in messages
                             if m.get("role") == "user"), "")
            output = data.get("output") or {}
            sink.write(json.dumps({
                "generation_id": row["generation_id"],
                "created_at": row["created_at"],
                "model": row["model_permaslug"],
                "tokens_prompt": int(row["tokens_prompt"] or 0),
                "tokens_reasoning": int(row["tokens_reasoning"] or 0),
                "sequence_length": sequence_length(user_msg),
                "user_message_head": user_msg[:200],
                "completion": output.get("completion"),
                "reasoning": output.get("reasoning"),
            }) + "\n")
            sink.flush()
            fetched += 1
    print(f"fetched {fetched} -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
