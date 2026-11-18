"""Cache residual states and teacher-forced decoder margins across counts.

One forward pass per count over `prompt + prefill + correct_answer` captures,
at every hooked layer:

  final_prompt   the residual state at the last token of the user prompt --
                 the state every geometry claim is about
  decode_point   the state at the token the decoder actually reads to emit
                 the answer (differs from final_prompt only when a prefill,
                 e.g. gpt-oss's forced final channel, is appended)

and, from the same pass, the teacher-forced logit margin on each digit of the
correct answer. Margins collapsing while the projection stays readable is the
"readable but not usable" signature the geometry analysis overlays.

    python -m counting.probes.suite.collect configs/probes/gpt-oss-20b.yaml \
        --output-dir results/suite/gpt-oss-20b/collect \
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scan-dir", type=Path, required=True,
        help="behavioral scan output; provides the SCC and per-count correctness labels",
    )
    parser.add_argument("--prefill", default="")
    parser.add_argument("--chat-kwargs", default="", help="JSON dict for apply_chat_template")
    parser.add_argument("--item", default="a")
    parser.add_argument("--delimiter", default=", ")
    parser.add_argument(
        "--max-count", type=int, default=0,
        help="largest count to collect; 0 means 2.2x the scanned SCC (min 48, cap 320)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scc = scanned_scc(args.scan_dir)
    max_count = args.max_count or min(max(int(2.2 * scc), 48), 320)

    spec = ProbeSpec.load(args.config)
    model, tokenizer, device = load_model(spec)
    print(f"{spec.name} on {device}: collecting counts 1..{max_count} (scc={scc})")

    counts = list(range(1, max_count + 1))
    final_states: dict[str, list] = {h.label: [] for h in spec.hooks}
    decode_states: dict[str, list] = {h.label: [] for h in spec.hooks}
    margins: list[dict] = []
    started = time.time()

    template_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    for count in counts:
        prompt = render_prompts(tokenizer, [count], args.item, args.delimiter, template_kwargs)[0]
        base_ids = encode_text(tokenizer, prompt)
        full_ids = encode_text(tokenizer, prompt + args.prefill) if args.prefill else list(base_ids)
        prefix_ok = full_ids[: len(base_ids)] == base_ids
        answer_ids = encode_text(tokenizer, str(count))
        token_ids = full_ids + list(answer_ids)
        p_idx, q_idx = len(base_ids) - 1, len(full_ids) - 1

        if device == "mlx":
            from ..common.mlx_backend import MlxActivationRecorder, forward_logits

            with MlxActivationRecorder(model, spec.hooks) as recorder:
                logits_np = forward_logits(model, token_ids)
                for hook in spec.hooks:
                    seq = recorder.cache[hook.label][0]
                    final_states[hook.label].append(seq[p_idx].astype(np.float16))
                    decode_states[hook.label].append(seq[q_idx].astype(np.float16))
        else:
            import torch

            from ..common.model import ActivationRecorder

            ids = torch.tensor([token_ids], device=device)
            with ActivationRecorder(model, spec.hooks) as recorder, torch.inference_mode():
                output = model(ids)
                for hook in spec.hooks:
                    seq = recorder.cache[hook.label][0].float().cpu().numpy()
                    final_states[hook.label].append(seq[p_idx].astype(np.float16))
                    decode_states[hook.label].append(seq[q_idx].astype(np.float16))
                logits_np = output.logits[0].float().cpu().numpy()

        gaps: list[float] = []
        greedy_exact = True
        for i, token_id in enumerate(answer_ids):
            row = logits_np[q_idx + i]
            order = np.argsort(row)[-2:]
            top, second = float(row[order[-1]]), float(row[order[-2]])
            pred = int(order[-1])
            other = second if pred == token_id else top
            gaps.append(float(row[token_id]) - other)
            greedy_exact = greedy_exact and pred == token_id
        margins.append(
            {
                "count": count,
                "min_gap": min(gaps),
                "gaps": [round(g, 4) for g in gaps],
                "greedy_exact": greedy_exact,
                "prefix_ok": prefix_ok,
            }
        )
        del logits_np
        if count % 32 == 0:
            print(f"  count {count}/{max_count}  [{time.time() - started:.0f}s]")

    arrays: dict[str, np.ndarray] = {"counts": np.array(counts, dtype=np.int32)}
    for hook in spec.hooks:
        arrays[f"{hook.label}|final_prompt"] = np.stack(final_states[hook.label])
        arrays[f"{hook.label}|decode_point"] = np.stack(decode_states[hook.label])
    np.savez_compressed(args.output_dir / "activations.npz", **arrays)
    (args.output_dir / "margins.json").write_text(json.dumps(margins, indent=1))

    meta = {
        "name": spec.name,
        "checkpoint": spec.checkpoint,
        "scan_dir": str(args.scan_dir),
        "scc": scc,
        "max_count": max_count,
        "prefill": args.prefill,
        "chat_kwargs": args.chat_kwargs,
        "item": args.item,
        "delimiter": args.delimiter,
        "hooks": [h.label for h in spec.hooks],
        "scan_summary": load_summary(args.scan_dir),
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"collected {len(counts)} counts in {meta['elapsed_s']}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
