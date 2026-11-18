"""Cache decode-point residual states for several prompts per count.

The single-prompt cache (`collect.py`) has one state per count, so any readout
richer than a single direction (digit classes, periodic features, count
subspaces) is fitted on 27-71 samples. This script varies the counted item and
the delimiter to give 8-16 states per count, keeping the paper's prompt
template, the chat template, the prefill and the decode-point definition.

Per prompt it records the decode-point state at every hooked site and the
teacher-forced correctness of the model's own answer (`greedy_exact`), which is
the stable label used downstream: a prompt is stable when its count is at or
below the scanned SCC and the teacher-forced answer is exact.

    python -m counting.probes.suite.collect_multi configs/probes/gpt-oss-20b.yaml \
        --output-dir results/suite/gpt-oss-20b/collect-multi \
        --scan-dir results/suite/gpt-oss-20b/scan-final \
        --prefill '<|channel|>final<|message|>'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..common import ProbeSpec, load_model
from ..common.prompts import render_prompts
from .util import encode_text, load_summary, scanned_scc

DEFAULT_ITEMS = "a,b,x,z,cat,dog,red,sun"
DEFAULT_DELIMITERS = ", | "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--prefill", default="")
    parser.add_argument("--chat-kwargs", default="", help="JSON dict for apply_chat_template")
    parser.add_argument("--items", default=DEFAULT_ITEMS, help="comma list of counted items")
    parser.add_argument(
        "--delimiters", default=DEFAULT_DELIMITERS,
        help="'|'-separated list of delimiters (default: ', ' and ' ')",
    )
    parser.add_argument(
        "--max-frac", type=float, default=1.25,
        help="largest count as a fraction of the scanned SCC",
    )
    parser.add_argument("--max-count", type=int, default=0, help="override the count ceiling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scc = scanned_scc(args.scan_dir)
    max_count = args.max_count or max(int(round(args.max_frac * scc)), 8)
    items = [s for s in args.items.split(",") if s]
    delimiters = args.delimiters.split("|")

    spec = ProbeSpec.load(args.config)
    model, tokenizer, device = load_model(spec)
    prompts = [(c, it, d) for c in range(1, max_count + 1) for it in items for d in delimiters]
    print(
        f"{spec.name} on {device}: {len(prompts)} prompts "
        f"({len(items)} items x {len(delimiters)} delimiters x counts 1..{max_count}; scc={scc})"
    )

    template_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    states: dict[str, list] = {h.label: [] for h in spec.hooks}
    rows: list[dict] = []
    started = time.time()

    for i, (count, item, delimiter) in enumerate(prompts):
        prompt = render_prompts(tokenizer, [count], item, delimiter, template_kwargs)[0]
        full_ids = encode_text(tokenizer, prompt + args.prefill)
        answer_ids = encode_text(tokenizer, str(count))
        token_ids = full_ids + list(answer_ids)
        q_idx = len(full_ids) - 1

        if device == "mlx":
            from ..common.mlx_backend import MlxActivationRecorder, forward_logits

            with MlxActivationRecorder(model, spec.hooks) as recorder:
                logits_np = forward_logits(model, token_ids)
                for hook in spec.hooks:
                    states[hook.label].append(
                        recorder.cache[hook.label][0][q_idx].astype(np.float16)
                    )
        else:
            import torch

            from ..common.model import ActivationRecorder

            ids = torch.tensor([token_ids], device=device)
            with ActivationRecorder(model, spec.hooks) as recorder, torch.inference_mode():
                output = model(ids)
                for hook in spec.hooks:
                    seq = recorder.cache[hook.label][0]
                    states[hook.label].append(seq[q_idx].float().cpu().numpy().astype(np.float16))
                logits_np = output.logits[0, q_idx : q_idx + len(answer_ids)].float().cpu().numpy()
            del output

        gaps: list[float] = []
        greedy_exact = True
        for j, token_id in enumerate(answer_ids):
            row = logits_np[q_idx + j] if device == "mlx" else logits_np[j]
            order = np.argsort(row)[-2:]
            top, second = float(row[order[-1]]), float(row[order[-2]])
            pred = int(order[-1])
            other = second if pred == token_id else top
            gaps.append(float(row[token_id]) - other)
            greedy_exact = greedy_exact and pred == token_id
        rows.append(
            {
                "index": i, "count": count, "item": item, "delimiter": delimiter,
                "min_gap": round(min(gaps), 4), "greedy_exact": greedy_exact,
                "first_answer_token": int(answer_ids[0]),
            }
        )
        del logits_np
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(prompts)}  [{time.time() - started:.0f}s]", flush=True)

    arrays: dict[str, np.ndarray] = {
        "counts": np.array([r["count"] for r in rows], dtype=np.int32),
        "greedy_exact": np.array([r["greedy_exact"] for r in rows], dtype=bool),
        "prompt_index": np.array([r["index"] for r in rows], dtype=np.int32),
    }
    for hook in spec.hooks:
        arrays[f"{hook.label}|decode_point"] = np.stack(states[hook.label])
    np.savez_compressed(args.output_dir / "activations.npz", **arrays)
    (args.output_dir / "prompts.json").write_text(json.dumps(rows, indent=1))

    stable = arrays["greedy_exact"] & (arrays["counts"] <= scc)
    meta = {
        "name": spec.name, "checkpoint": spec.checkpoint, "scan_dir": str(args.scan_dir),
        "scc": scc, "max_count": max_count, "items": items, "delimiters": delimiters,
        "prefill": args.prefill, "chat_kwargs": args.chat_kwargs,
        "hooks": [h.label for h in spec.hooks], "n_prompts": len(prompts),
        "n_stable": int(stable.sum()),
        "stable_by_item": {
            it: int((stable & (np.array([r["item"] for r in rows]) == it)).sum()) for it in items
        },
        "scan_summary": load_summary(args.scan_dir),
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"collected {len(prompts)} prompts ({int(stable.sum())} stable) in "
        f"{meta['elapsed_s']}s -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
