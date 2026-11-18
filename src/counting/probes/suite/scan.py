"""Local behavioral scan: the validation gate for every mechanistic claim.

Reproduces the API harness's measurement locally on an open-weight checkpoint:
one greedy generation per length, ascending, scored exact/inexact, fitted with
the paper's two-regime estimator (`counting.analysis.capacity.fit_transition`)
and stopped by the same criterion the OpenRouter extension sweep used -- scan
until the transition is declared and its tail resolved.

The local CC must reproduce the model's catalogued CC before any probe result
on that checkpoint is trusted; that is what ties the mechanistic story to the
paper's behavioral numbers.

    python -m counting.probes.suite.scan configs/probes/gpt-oss-20b.yaml \
        --output-dir results/suite/gpt-oss-20b/scan-final \
        --prefill '<|channel|>final<|message|>' --max-new-tokens 16

`--prefill` appends text after the chat template's generation prompt. For
harmony-format models (gpt-oss) this forces the answer channel and skips the
analysis channel, making the model answer immediately like the rest of the
slate; without it the scan measures the model's default (reasoning) behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from ...analysis.capacity import fit_transition, sufficiency
from ..common import ProbeSpec, load_model
from ..common.prompts import parse_integer, render_prompts, tokenize_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, help="probe config yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--item", default="a")
    parser.add_argument("--delimiter", default=", ")
    parser.add_argument("--prefill", default="", help="text appended after the generation prompt")
    parser.add_argument(
        "--chat-kwargs", default="",
        help="JSON dict forwarded to apply_chat_template, e.g. "
        '\'{"enable_thinking": true}\' to toggle a reasoning mode',
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--ceiling", type=int, default=512)
    parser.add_argument(
        "--fit-every", type=int, default=8,
        help="re-evaluate the stop criterion every N probes once a mistake exists",
    )
    parser.add_argument(
        "--min-tail", type=int, default=16,
        help="minimum probes above the fitted transition before the scan may stop; "
        "guards against declaring on a thin tail when the stable regime is noisy",
    )
    return parser.parse_args()


def generate_one(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    if device == "mlx":
        from ..common.mlx_backend import generate_text

        return generate_text(model, tokenizer, prompt, max_new_tokens)

    import torch

    encoded = tokenize_prompts(tokenizer, [prompt], device)
    prompt_length = int(encoded["attention_mask"][0].sum().item())
    with torch.inference_mode():
        sequences = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False
        )
    continuation = sequences[0, prompt_length:]
    decode = getattr(tokenizer, "decode", None) or tokenizer.tokenizer.decode
    raw = decode(continuation, skip_special_tokens=False)
    # Replace special-token markers with newlines instead of stripping them:
    # silently dropping e.g. Gemma's end-of-thinking marker concatenates the
    # last number of the reasoning trace with the final answer ("11"+"11" ->
    # "1111") and corrupts the parse.
    return re.sub(r"<[^<>\s]{1,60}>", "\n", raw).strip()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"

    # Resume: keep completed probes from an interrupted scan.
    records: list[dict] = []
    if records_path.exists():
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
        print(f"resuming with {len(records)} existing probes")

    spec = ProbeSpec.load(args.config)
    model, tokenizer, device = load_model(spec)
    print(f"{spec.name} loaded on {device}")

    done_lengths = {record["length"] for record in records}
    handle = records_path.open("a")
    started = time.time()

    def current_fit():
        ordered = sorted(records, key=lambda r: r["length"])
        lengths = [r["length"] for r in ordered]
        correct = [bool(r["correct"]) for r in ordered]
        error = [r["parsed"] is None for r in ordered]
        transition = fit_transition(lengths, correct, error)
        n_above = sum(1 for r in ordered if r["length"] > transition.scc)
        ok_above = sum(1 for r in ordered if r["length"] > transition.scc and r["correct"])
        return transition, sufficiency(transition, n_above, ok_above), n_above

    template_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    any_mistake = any(not r["correct"] for r in records)
    for length in range(args.start, args.ceiling + 1):
        if length in done_lengths:
            continue
        prompt = render_prompts(
            tokenizer, [length], args.item, args.delimiter, template_kwargs
        )[0] + args.prefill
        reply = generate_one(model, tokenizer, prompt, device, args.max_new_tokens)
        parsed = parse_integer(reply)
        record = {
            "length": length,
            "parsed": parsed,
            "correct": parsed == length,
            "reply": reply[:2000],
        }
        records.append(record)
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        if not record["correct"]:
            any_mistake = True
            print(f"  L={length}: replied {parsed!r}  [{len(records)} probes, "
                  f"{time.time() - started:.0f}s]")

        # The criterion is its own stop rule: once the record can declare a
        # transition with a resolved tail, more probing cannot move the answer.
        if any_mistake and (length % args.fit_every == 0):
            transition, status, n_above = current_fit()
            if status == "ok" and n_above >= args.min_tail:
                print(f"resolved at probe {len(records)}: scc={transition.scc} "
                      f"[{transition.ci_low}, {transition.ci_high}]")
                break

    handle.close()
    transition, status, _ = current_fit()
    summary = {
        "name": spec.name,
        "checkpoint": spec.checkpoint,
        "item": args.item,
        "delimiter": args.delimiter,
        "prefill": args.prefill,
        "chat_kwargs": args.chat_kwargs,
        "max_new_tokens": args.max_new_tokens,
        "n_probes": len(records),
        "sufficiency": status,
        "transition": asdict(transition),
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
