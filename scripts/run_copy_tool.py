#!/usr/bin/env python3
"""Verbatim-copy-through-a-tool-call control.

    OPENROUTER_API_KEY=... python scripts/run_copy_tool.py

The model is asked to do no counting at all: it must call a tool with the
prompt's comma-separated sequence of a's copied verbatim. The number of a's
in the tool-call argument is compared with the number in the prompt. If the
copy loses or gains items beyond the model's CC, the sequence's exact
multiplicity was never represented -- ingestion, not arithmetic, is the
limit. Output: results/copy-tool/<slug>.jsonl, one record per probed length.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "copy-tool"
API = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    ("google/gemini-3.7-flash", "gemini-3p7-flash",
     {"reasoning": {"effort": "medium"}}, list(range(20, 801, 8))),
    ("anthropic/claude-sonnet-5", "claude-sonnet-5",
     {"reasoning": {"effort": "medium"}}, list(range(10, 301, 4))),
]

TOOL = {
    "type": "function",
    "function": {
        "name": "record_sequence",
        "description": "Record a sequence of items exactly as given.",
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "The exact comma-separated sequence, copied verbatim.",
                }
            },
            "required": ["sequence"],
        },
    },
}


def probe(model: str, extra: dict, n: int, key: str) -> dict:
    sequence = ", ".join("a" for _ in range(n))
    payload = {
        "model": model,
        "max_tokens": 12000,
        "tools": [TOOL],
        "tool_choice": {"type": "function", "function": {"name": "record_sequence"}},
        "messages": [
            {"role": "system",
             "content": "You copy text exactly. Never abbreviate or summarize."},
            {"role": "user",
             "content": "Call record_sequence with the exact sequence below, "
                        f"copied verbatim.\n\n{sequence}"},
        ],
        **extra,
    }
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={"authorization": f"Bearer {key}",
                 "content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode())
            msg = body["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                return {"n": n, "copied": None, "exact": False,
                        "detail": "no_tool_call",
                        "content": str(msg.get("content"))[:200]}
            args = calls[0]["function"]["arguments"]
            try:
                seq = json.loads(args).get("sequence", "")
            except json.JSONDecodeError:
                seq = args
            copied = seq.count("a")
            return {"n": n, "copied": copied, "exact": copied == n,
                    "arg_chars": len(args)}
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return {"n": n, "copied": None, "exact": False,
                        "detail": f"error: {str(exc)[:120]}"}
            time.sleep(4 * (attempt + 1))
    return {"n": n, "copied": None, "exact": False, "detail": "unreachable"}


def main() -> int:
    key = os.environ["OPENROUTER_API_KEY"]
    OUT.mkdir(parents=True, exist_ok=True)
    for model, slug, extra, lengths in MODELS:
        path = OUT / f"{slug}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["n"] for l in path.open()}
        todo = [n for n in lengths if n not in done]
        print(f"{model}: {len(todo)} probes", flush=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            for rec in pool.map(lambda n: probe(model, extra, n, key), todo):
                with path.open("a") as sink:
                    sink.write(json.dumps(rec) + "\n")
        exact = sum(1 for l in path.open() if json.loads(l)["exact"])
        total = sum(1 for _ in path.open())
        print(f"{model}: {exact}/{total} exact copies", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
