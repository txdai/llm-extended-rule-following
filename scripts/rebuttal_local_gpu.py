#!/usr/bin/env python3
"""Rebuttal Local Open-Weight Runner (Exp 1 Format Controls + Exp 3B Dual-Task Controls).

Loads each local checkpoint ONCE on the M2 Ultra, runs all requested conditions
with greedy decoding (do_sample=False), writes resumable JSONL records and fitted
two-regime summaries, and frees Metal memory before the next checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from counting.analysis.capacity import fit_transition, sufficiency
from counting.metrics import INTEGER_PATTERN
from counting.probes.common import ProbeSpec, load_model
from counting.probes.suite.scan import generate_one
from counting.tasks import build_task

REPO = Path(__file__).resolve().parents[1]
OUT_EXP1 = REPO / "results" / "rebuttal-exp1-local"
OUT_EXP3 = REPO / "results" / "rebuttal-interference-local"

LOCAL_MODELS = [
    {
        "name": "gpt-oss-20b",
        "config": "configs/probes/gpt-oss-20b.yaml",
        "prefill": "<|channel|>final<|message|>",
        "chat_kwargs": {},
        "cc": 36,
        "exp1_stop": 70,
        "run_exp3": True,
        "grid_u": 9,
        "row3_u": 11,
    },
    {
        "name": "gemma-4-26b-a4b-it",
        "config": "configs/probes/gemma-4-26b-a4b-it.yaml",
        "prefill": "",
        "chat_kwargs": {},
        "cc": 75,
        "exp1_stop": 120,
        "run_exp3": True,
        "grid_u": 18,
        "row3_u": 22,
    },
    {
        "name": "qwen3.6-35b-a3b",
        "config": "configs/probes/qwen3.6-35b-a3b.yaml",
        "prefill": "",
        "chat_kwargs": {"enable_thinking": False},
        "cc": 42,
        "exp1_stop": 75,
        "run_exp3": True,
        "grid_u": 10,
        "row3_u": 12,
    },
    {
        "name": "llama-3.3-70b-instruct",
        "config": "configs/probes/llama-3.3-70b-instruct.yaml",
        "prefill": "",
        "chat_kwargs": {},
        "cc": 32,
        "exp1_stop": 60,
        "run_exp3": False,
        "grid_u": 8,
        "row3_u": 9,
    },
]


def render_chat(tokenizer: Any, system: str, user: str, chat_kwargs: dict[str, Any], prefill: str) -> str:
    block_msgs = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    plain_msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            block_msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
    except (TypeError, ValueError):
        prompt = tokenizer.apply_chat_template(
            plain_msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
    return prompt + prefill


def fit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda r: r["length"])
    lengths = [r["length"] for r in ordered]
    correct = [bool(r["correct"]) for r in ordered]
    errors = [r["parsed"] is None for r in ordered]
    t = fit_transition(lengths, correct, errors)
    above = [r for r in ordered if r["length"] > t.scc]
    st = sufficiency(t, len(above), sum(1 for r in above if r["correct"]))
    sub_vals = [r["subtask_correct"] for r in ordered if r.get("subtask_correct") is not None]
    return {
        "n_probes": len(ordered),
        "sufficiency": st,
        "transition": asdict(t),
        "subtask_accuracy": round(sum(sub_vals) / len(sub_vals), 4) if sub_vals else None,
    }


def run_exp1_for_model(mspec: dict[str, Any], model: Any, tokenizer: Any, device: str) -> None:
    mname = mspec["name"]
    stop_len = mspec["exp1_stop"]
    for fmt in ("int_bare", "english_words", "zero_padded_spaced", "mod_10"):
        out_dir = OUT_EXP1 / mname / fmt
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_path = out_dir / "records.jsonl"
        records = [json.loads(l) for l in rec_path.read_text().splitlines() if l] if rec_path.exists() else []
        done = {r["length"] for r in records}
        task = build_task("formatted_count", output_format=fmt, item="a", delimiter=", ")
        max_new = 32 if fmt in ("english_words", "zero_padded_spaced") else 12
        t0 = time.time()
        with rec_path.open("a") as fh:
            for length in range(1, stop_len + 1):
                if length in done:
                    continue
                stim = task.build(length, random.Random(f"exp1:{mname}:{fmt}:{length}"))
                prompt = render_chat(tokenizer, stim.system, stim.user, mspec["chat_kwargs"], mspec["prefill"])
                reply = generate_one(model, tokenizer, prompt, device, max_new)
                parsed = task.parse(reply)
                rec = {
                    "length": length,
                    "expected": stim.target,
                    "parsed": parsed,
                    "correct": parsed == stim.target,
                    "signed_error": (parsed - stim.target) if parsed is not None else None,
                    "reply": reply[:500],
                }
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        summary = {"model": mname, "format": fmt, "elapsed_s": round(time.time() - t0, 1), **fit_records(records)}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[Local Exp1] {mname} {fmt}: CC={summary["transition"]["scc"]} "
              f"[{summary["transition"]["ci_low"]}, {summary["transition"]["ci_high"]}] "
              f"({summary["elapsed_s"]}s)", flush=True)


def run_exp3_for_model(mspec: dict[str, Any], model: Any, tokenizer: Any, device: str, seeds: int = 5) -> None:
    mname = mspec["name"]
    cc = mspec["cc"]
    grid_u = mspec["grid_u"]
    row3_u = mspec["row3_u"]
    conds = [
        ("solo",     0,                    "solo"),
        ("inert",    max(3, round(0.90 * grid_u)), "inert-90"),
        ("semantic", max(3, round(0.90 * grid_u)), "semantic-90"),
        ("pointer",  max(3, round(0.90 * grid_u)), "pointer-90"),
        ("row3",     max(3, round(0.60 * row3_u)), "row3-60"),
        ("grid",     max(3, round(0.60 * grid_u)), "grid-60"),
    ]
    start_l = max(4, round(0.20 * cc))
    stop_l = round(1.45 * cc) + 16
    step_l = 2

    for kind, size, tag in conds:
        n_seeds = 1 if kind == "solo" else seeds
        for rep in range(1, n_seeds + 1):
            out_dir = OUT_EXP3 / mname / tag / f"seed-{rep}"
            out_dir.mkdir(parents=True, exist_ok=True)
            rec_path = out_dir / "records.jsonl"
            records = [json.loads(l) for l in rec_path.read_text().splitlines() if l] if rec_path.exists() else []
            done = {r["length"] for r in records}
            task = (
                build_task("homogeneous_count", item="a", delimiter=", ")
                if kind == "solo"
                else build_task("count_with_load", load_kind=kind, load_size=size)
            )
            t0 = time.time()
            with rec_path.open("a") as fh:
                for length in range(start_l, stop_l + 1, step_l):
                    if length in done:
                        continue
                    stim = task.build(length, random.Random(f"exp3:{mname}:{tag}:{rep}:{length}"))
                    prompt = render_chat(tokenizer, stim.system, stim.user, mspec["chat_kwargs"], mspec["prefill"])
                    reply = generate_one(model, tokenizer, prompt, device, 24)
                    parsed = task.parse(reply)
                    ints = [int(x) for x in INTEGER_PATTERN.findall(reply)] if reply else []
                    sub_expected = stim.metadata.get("subtask_answer")
                    sub_parsed = ints[0] if len(ints) >= 2 else None
                    sub_ok = (sub_parsed == sub_expected) if sub_expected is not None else None
                    rec = {
                        "length": length,
                        "expected": stim.target,
                        "parsed": parsed,
                        "correct": parsed == stim.target,
                        "subtask_expected": sub_expected,
                        "subtask_parsed": sub_parsed,
                        "subtask_correct": sub_ok,
                        "reply": reply[:500],
                    }
                    records.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
            summary = {
                "model": mname, "condition": tag, "seed": rep,
                "load_size": size, "elapsed_s": round(time.time() - t0, 1),
                **fit_records(records),
            }
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"[Local Exp3] {mname} {tag} seed-{rep}: CC={summary["transition"]["scc"]} "
                  f"sub_acc={summary["subtask_accuracy"]} ({summary["elapsed_s"]}s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="subset of local model names")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selected = [m for m in LOCAL_MODELS if not args.models or m["name"] in args.models]
    print(f"Local GPU queue: {[m["name"] for m in selected]}", flush=True)
    if args.dry_run:
        return 0

    import torch
    for mspec in selected:
        print(f"=== Loading {mspec["name"]} ===", flush=True)
        spec = ProbeSpec.load(REPO / mspec["config"])
        model, tokenizer, device = load_model(spec)
        print(f"Loaded {mspec["name"]} on {device}", flush=True)
        try:
            run_exp1_for_model(mspec, model, tokenizer, device)
            if mspec["run_exp3"]:
                run_exp3_for_model(mspec, model, tokenizer, device, seeds=args.seeds)
        finally:
            del model, tokenizer
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    print("=== All Local GPU Experiments Complete ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
