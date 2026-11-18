"""Causal tests of the two-channel hypothesis: clamp transfer and chimera patches.

Hypothesis under test: inside the stable regime the residual state carries a
coarse *magnitude* channel that is linear in the count, plus a distributed
nonlinear component that carries *exactness*. If that is right:

  clamp      moving only the linear coordinate of a mid-regime state should
             move the decoded answer coarsely (staircase, systematic bias)
             but not land exactly on arbitrary targets;
  chimera    a donor state split as  h_d = center + z_d*u + r_d  lets us swap
             channels between a base and a donor prompt --
               linear_only    base residual, donor linear coordinate
               residual_only  donor residual, base linear coordinate
             the channel that determines the decoded integer is the one that
             carries exactness;
  full       donor state transplanted whole (positive control);
  random     matched-norm shift along a random direction (negative control).

Edits are applied at the decode-point token during the prefill forward pass of
a greedy generation, at each hooked layer in turn.

    python -m counting.probes.suite.intervene configs/probes/gpt-oss-20b.yaml \
        --collect-dir results/suite/gpt-oss-20b/collect \
        --scan-dir results/suite/gpt-oss-20b/scan-final \
        --output-dir results/suite/gpt-oss-20b/interventions \
        --prefill '<|channel|>final<|message|>'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..common import ProbeSpec, load_model
from ..common.directions import fit_counter_direction
from ..common.prompts import parse_integer, render_prompts
from .util import correctness_by_count, decode_tokens, encode_text, scanned_scc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--collect-dir", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill", default="")
    parser.add_argument("--chat-kwargs", default="", help="JSON dict for apply_chat_template")
    parser.add_argument("--item", default="a")
    parser.add_argument("--delimiter", default=", ")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-donors", type=int, default=20)
    parser.add_argument(
        "--base-frac", type=float, default=0.5,
        help="base count for all interventions, as a fraction of the scanned SCC",
    )
    parser.add_argument(
        "--experiments", default="baseline,clamp,chimera,step,random",
        help="comma list of experiment groups to run",
    )
    return parser.parse_args()


def subsample(values: list[int], k: int) -> list[int]:
    if len(values) <= k:
        return values
    idx = np.linspace(0, len(values) - 1, k).round().astype(int)
    return [values[i] for i in sorted(set(idx.tolist()))]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scc = scanned_scc(args.scan_dir)
    correct = correctness_by_count(args.scan_dir)

    cache = np.load(args.collect_dir / "activations.npz")
    counts = cache["counts"].astype(np.int64)
    stable = np.array([c <= scc and correct.get(int(c), False) for c in counts])

    spec = ProbeSpec.load(args.config)
    model, tokenizer, device = load_model(spec)
    print(f"{spec.name} on {device}: interventions at {len(spec.hooks)} sites (scc={scc})")

    base_count = max(4, round(args.base_frac * scc))
    while base_count not in counts[stable] and base_count > 2:
        base_count -= 1
    base_index = int(np.where(counts == base_count)[0][0])

    template_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    prompt = render_prompts(
        tokenizer, [base_count], args.item, args.delimiter, template_kwargs
    )[0] + args.prefill
    ids = encode_text(tokenizer, prompt)
    q_idx = len(ids) - 1

    if device == "mlx":
        from ..common.mlx_backend import MlxActivationEditor, generate_tokens

        def generate_with_edit(hook, edit_fn) -> str:
            with MlxActivationEditor(model) as editor:
                editor.edit(hook, edit_fn)
                new_ids = generate_tokens(model, tokenizer, ids, args.max_new_tokens)
            return decode_tokens(tokenizer, new_ids).strip()

        def prefill_edit(apply_hidden):
            """Apply `apply_hidden(h) -> h` at the decode-point token.

            mlx_lm's generate_step forwards the prompt as all-but-last tokens,
            then a one-token step for the last prompt token. The Torch
            `shape[1] > q_idx` test never matches that pattern, so we write at
            `q_idx` on a full sequence and at the last axis on the first
            one-token forward.
            """
            fired = {"once": False}

            def fn(arr):
                out = np.array(arr, copy=True)
                t = out.shape[1]
                if t > q_idx:
                    out[0, q_idx] = apply_hidden(out[0, q_idx])
                elif t == 1 and not fired["once"]:
                    fired["once"] = True
                    out[0, 0] = apply_hidden(out[0, 0])
                return out

            return fn

        def add_apply(vec):
            v = np.asarray(vec, dtype=np.float32)
            return prefill_edit(lambda h, _v=v: h + _v.astype(h.dtype))

        def set_apply(vec):
            v = np.asarray(vec, dtype=np.float32)
            return prefill_edit(lambda h, _v=v: _v.astype(h.dtype))
    else:
        import torch

        from ..common.model import ActivationEditor

        input_ids = torch.tensor([ids], device=device)
        attention = torch.ones_like(input_ids)

        def generate_with_edit(hook, edit_fn) -> str:
            with ActivationEditor(model) as editor, torch.inference_mode():
                editor.edit(hook, edit_fn)
                sequences = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            return decode_tokens(tokenizer, sequences[0, len(ids):]).strip()

        def prefill_edit(apply):
            def fn(tensor):
                if tensor.shape[1] > q_idx:
                    apply(tensor)
                return tensor
            return fn

        def add_apply(vec):
            def apply(t):
                t[0, q_idx] += torch.as_tensor(vec, device=t.device, dtype=t.dtype)
            return prefill_edit(apply)

        def set_apply(vec):
            def apply(t):
                t[0, q_idx] = torch.as_tensor(vec, device=t.device, dtype=t.dtype)
            return prefill_edit(apply)

    clamp_step = 1 if scc <= 48 else 2
    clamp_targets = list(range(1, int(1.3 * scc) + 1, clamp_step))
    donor_counts = subsample(
        [int(c) for c in counts[stable] if c != base_count], args.max_donors
    )
    rng = np.random.default_rng(7)

    experiments = set(args.experiments.split(","))
    rows_path = args.output_dir / "interventions.jsonl"
    handle = rows_path.open("a" if rows_path.exists() else "w")
    started = time.time()
    n_rows = 0

    def emit(row: dict) -> None:
        nonlocal n_rows
        row["parsed"] = parse_integer(row["decoded"])
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        n_rows += 1

    for hook in spec.hooks:
        states = cache[f"{hook.label}|decode_point"].astype(np.float32)
        fit = fit_counter_direction(states, counts.astype(np.float32), stable)
        u = fit.direction
        u_norm_sq = float(u @ u)
        z = fit.projected
        z_base = float(z[base_index])
        base_state = states[base_index]

        # Baseline: unedited generation (hook installed but identity edit).
        if "baseline" in experiments:
            emit({
                "experiment": "baseline", "site": hook.label, "base": base_count,
                "target": None,
                "decoded": generate_with_edit(hook, add_apply(np.zeros_like(base_state))),
            })

        # 1. Clamp transfer function along the fitted direction.
        for target in clamp_targets if "clamp" in experiments else []:
            z_target = float(fit.target_projection(target))
            shift = ((z_target - z_base) / u_norm_sq) * u
            emit({
                "experiment": "clamp", "site": hook.label, "base": base_count,
                "target": target,
                "decoded": generate_with_edit(hook, add_apply(shift)),
            })

        # 2. Chimera patches: swap linear coordinate vs residual channel.
        for donor in donor_counts if "chimera" in experiments else []:
            d_index = int(np.where(counts == donor)[0][0])
            donor_state = states[d_index]
            z_donor = float(z[d_index])
            delta = ((z_donor - z_base) / u_norm_sq) * u
            variants = {
                "chimera_full": donor_state,
                "chimera_linear_only": base_state + delta,
                "chimera_residual_only": donor_state - delta,
            }
            for name, new_state in variants.items():
                emit({
                    "experiment": name, "site": hook.label, "base": base_count,
                    "target": donor,
                    "decoded": generate_with_edit(hook, set_apply(new_state)),
                })

        # 3. Translate along the mean per-item trajectory step: the ambient-space
        # vector by which one additional item moves the state, averaged over the
        # stable regime. This is the strongest "linear channel" intervention --
        # it moves the state exactly as k more items would, on average.
        if "step" in experiments:
            mean_step = np.diff(states[stable], axis=0).mean(axis=0)
            for k in (-8, -4, -2, -1, 1, 2, 4, 8, 16):
                emit({
                    "experiment": "step_translate", "site": hook.label,
                    "base": base_count, "target": base_count + k,
                    "decoded": generate_with_edit(hook, add_apply(mean_step * float(k))),
                })

        # 4. Random-direction control with the norm of a mid-size clamp shift.
        reference = float(np.linalg.norm(
            ((float(fit.target_projection(int(0.9 * scc))) - z_base) / u_norm_sq) * u
        ))
        for scale in (1.0, 2.0, 4.0) if "random" in experiments else []:
            noise = rng.standard_normal(len(u)).astype(np.float32)
            noise *= reference * scale / float(np.linalg.norm(noise))
            emit({
                "experiment": "random_control", "site": hook.label, "base": base_count,
                "target": scale,
                "decoded": generate_with_edit(hook, add_apply(noise)),
            })

        print(f"  {hook.label} done  [{n_rows} rows, {time.time() - started:.0f}s]")

    handle.close()
    meta = {
        "name": spec.name, "scc": scc, "base_count": base_count,
        "clamp_targets": clamp_targets, "donor_counts": donor_counts,
        "prefill": args.prefill, "experiments": sorted(experiments), "n_rows": n_rows,
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n_rows} rows in {meta['elapsed_s']}s -> {rows_path}")


if __name__ == "__main__":
    main()
