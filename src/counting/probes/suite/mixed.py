"""Count-vs-length deconfound: is the linear coordinate a counter or a ruler?

On homogeneous sequences the count and the sequence length are the same
variable, so a "count direction" could equally be a position/length direction.
This probe breaks the confound: sequences of k target items shuffled among
L-k distractors, over a crossed (k, L) grid. For each hooked site it fits
independent ridge readouts for k and for L on the decode-point states and
reports held-out R^2 for both, the angle between the two directions, and the
model's behavioral accuracy on the mixed task itself.

    python -m counting.probes.suite.mixed configs/probes/gpt-oss-20b.yaml \
        --scan-dir results/suite/gpt-oss-20b/scan-final \
        --output-dir results/suite/gpt-oss-20b/mixed \
        --prefill '<|channel|>final<|message|>'
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from ..common import ProbeSpec, load_model
from ..common.prompts import SYSTEM_TEXT, _plain_messages, parse_integer
from .util import decode_tokens, encode_text, scanned_scc


def make_mixed_sequence(
    k: int, length: int, item: str, distractor: str, delimiter: str, seed: int = 0
) -> str:
    rng = random.Random(seed * 100003 + length * 1009 + k)
    tokens = [item] * k + [distractor] * (length - k)
    rng.shuffle(tokens)
    return delimiter.join(tokens)


def render_user_prompt(
    tokenizer: object, user_text: str, template_kwargs: dict | None = None
) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    extra = template_kwargs or {}
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            _plain_messages(messages), tokenize=False, add_generation_prompt=True, **extra
        )


def build_grid(scc: int) -> list[tuple[int, int]]:
    lengths = sorted({max(12, round(f * scc)) for f in (0.5, 0.8, 1.1)})
    pairs = []
    for length in lengths:
        ks = np.linspace(1, length - 1, min(14, length - 1)).round().astype(int)
        pairs.extend((int(k), length) for k in sorted(set(ks.tolist())))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill", default="")
    parser.add_argument("--chat-kwargs", default="", help="JSON dict for apply_chat_template")
    parser.add_argument("--item", default="a")
    parser.add_argument("--distractor", default="x")
    parser.add_argument("--delimiter", default=", ")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    from .geometry import choose_lambda, cv_predictions, ridge_fit

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scc = scanned_scc(args.scan_dir)
    grid = build_grid(scc)

    spec = ProbeSpec.load(args.config)
    model, tokenizer, device = load_model(spec)
    print(f"{spec.name} on {device}: mixed grid of {len(grid)} prompts (scc={scc})")

    states: dict[str, list] = {h.label: [] for h in spec.hooks}
    trials: list[dict] = []
    started = time.time()

    template_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    for k, length in grid:
        sequence = make_mixed_sequence(k, length, args.item, args.distractor, args.delimiter)
        user_text = f'How many "{args.item}" are in this sequence?\n\n{sequence}'
        prompt = render_user_prompt(tokenizer, user_text, template_kwargs) + args.prefill
        ids = encode_text(tokenizer, prompt)
        if device == "mlx":
            from ..common.mlx_backend import (
                MlxActivationRecorder,
                forward_logits,
                generate_tokens,
            )

            with MlxActivationRecorder(model, spec.hooks) as recorder:
                forward_logits(model, ids)
                for hook in spec.hooks:
                    seq = recorder.cache[hook.label][0]
                    states[hook.label].append(seq[len(ids) - 1].astype(np.float16))
            new_ids = generate_tokens(model, tokenizer, ids, args.max_new_tokens)
            decoded = decode_tokens(tokenizer, new_ids).strip()
        else:
            import torch

            from ..common.model import ActivationRecorder

            input_ids = torch.tensor([ids], device=device)
            attention = torch.ones_like(input_ids)
            # States come from a dedicated forward pass: recording during generate
            # would leave only the last one-token decode step in the hook cache.
            with ActivationRecorder(model, spec.hooks) as recorder, torch.inference_mode():
                model(input_ids)
                for hook in spec.hooks:
                    seq = recorder.cache[hook.label][0].float().cpu().numpy()
                    states[hook.label].append(seq[len(ids) - 1].astype(np.float16))
            with torch.inference_mode():
                sequences = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            decoded = decode_tokens(tokenizer, sequences[0, len(ids):]).strip()
        parsed = parse_integer(decoded)
        trials.append(
            {"k": k, "length": length, "decoded": decoded[:200], "parsed": parsed,
             "correct": parsed == k}
        )
        if len(trials) % 16 == 0:
            print(f"  {len(trials)}/{len(grid)}  [{time.time() - started:.0f}s]")

    k_values = np.array([t["k"] for t in trials], dtype=np.float64)
    l_values = np.array([t["length"] for t in trials], dtype=np.float64)
    accuracy = float(np.mean([t["correct"] for t in trials]))

    sites = {}
    arrays = {"k": k_values, "length": l_values}
    for hook in spec.hooks:
        x = np.stack(states[hook.label]).astype(np.float32)
        arrays[hook.label] = x.astype(np.float16)

        lam_k, r2_k = choose_lambda(x, k_values)
        lam_l, r2_l = choose_lambda(x, l_values)
        d_k, _, _ = ridge_fit(x, k_values, lam_k)
        d_l, _, _ = ridge_fit(x, l_values, lam_l)
        cosine = float(
            (d_k @ d_l) / max(np.linalg.norm(d_k) * np.linalg.norm(d_l), 1e-12)
        )
        # Partial checks: does the k-readout track k with L held out, and vice
        # versa? Residualize the CV predictions against the other variable.
        pred_k = cv_predictions(x, k_values, lam_k)
        pred_l = cv_predictions(x, l_values, lam_l)

        def partial_r(pred, target, control):
            design = np.column_stack([control, np.ones_like(control)])
            beta_t, *_ = np.linalg.lstsq(design, target, rcond=None)
            beta_p, *_ = np.linalg.lstsq(design, pred, rcond=None)
            rt, rp = target - design @ beta_t, pred - design @ beta_p
            denom = np.linalg.norm(rt) * np.linalg.norm(rp)
            return float(rt @ rp / denom) if denom > 1e-12 else 0.0

        sites[hook.label] = {
            "cv_r2_count": round(r2_k, 4),
            "cv_r2_length": round(r2_l, 4),
            "cosine_count_length": round(cosine, 4),
            "partial_r_count_given_length": round(partial_r(pred_k, k_values, l_values), 4),
            "partial_r_length_given_count": round(partial_r(pred_l, l_values, k_values), 4),
        }

    np.savez_compressed(args.output_dir / "states.npz", **arrays)
    result = {
        "name": spec.name,
        "scc_homogeneous": scc,
        "grid_size": len(grid),
        "behavioral_accuracy": round(accuracy, 4),
        "sites": sites,
        "trials": trials,
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.output_dir / "mixed.json").write_text(json.dumps(result, indent=1))
    print(f"accuracy={accuracy:.2%}; wrote {args.output_dir / 'mixed.json'}")


if __name__ == "__main__":
    main()
