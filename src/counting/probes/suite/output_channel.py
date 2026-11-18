"""Generic numerical-output control for the U transplant (round-2 E1).

If the decoder's answer-token subspace U is just the numeric output channel,
it should transfer and ablate the same way on tasks that only move a number
from the context to the output. Two such tasks, with answers drawn from the
same integer set as the counting donors:

  copy        "The number is N. Return only that number."
  retrieval   five "key: value" lines, "Return only the value of key K."

and, for the same code path, `counting` (the paper's prompt, states from the
existing cache). Groups:

  collect     decode-point states at every hooked site for N = 1..1.3 SCC,
              with the model's teacher-forced correctness
  localize    full donor-state transplant at every layer (five donors whose
              leading digit differs from the base's): the consolidation layer
  transplant  U projection and complement of the donor-base chord, 20 donors
  ablate      U, L (linear readout of N on the task's states), F (period-10
              pair), random in span of the task trajectory (k = 8, 16) and
              ambient random (8, 16) ablated to the task's stable mean
  interp      h_base + alpha (h_donor - h_base) at the last site
  layers      U projection / complement transplant at extra (unhooked) layers
              given by --layers, e.g. 20-27 for gpt-oss-120b

    python -m counting.probes.suite.output_channel configs/probes/gpt-oss-20b.yaml \
        --task copy --scan-dir results/suite/gpt-oss-20b/scan-final \
        --collect-dir results/suite/gpt-oss-20b/collect \
        --interventions-dir results/suite/gpt-oss-20b/interventions \
        --subspace-dir results/suite/gpt-oss-20b/subspace \
        --output-dir results/suite/gpt-oss-20b/round2 --prefill '<|channel|>final<|message|>'
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from ..common import ProbeSpec, load_model
from ..common.prompts import _plain_messages, parse_integer, render_prompts
from ..common.spec import HookSpec
from .editing import DecodeEditor, ablate_to_mean, project_delta, set_state
from .subspace import (
    LAMBDA_GRID,
    ambient_random_basis,
    choose_lambda_grouped,
    orthonormal_rows,
    random_bases,
    ridge_multi,
)
from .util import correctness_by_count, encode_text, scanned_scc

GROUPS = ("collect", "localize", "transplant", "ablate", "interp")
NEUTRAL_SYSTEM = "Return only one integer with no words, punctuation, or explanation."
KEYS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]


def task_user_text(task: str, n: int, scc: int, seed: int) -> str:
    if task == "copy":
        return f"The number is {n}. Return only that number."
    if task == "retrieval":
        rng = random.Random(seed * 7919 + n)
        keys = rng.sample(KEYS, 5)
        target = rng.randrange(5)
        pool = [v for v in range(1, int(1.3 * scc) + 2) if v != n]
        values = rng.sample(pool, 4)
        values.insert(target, n)
        lines = "\n".join(f"{k}: {v}" for k, v in zip(keys, values))
        return f"{lines}\n\nReturn only the value of key {keys[target]}."
    raise ValueError(task)


def render_task_prompt(tokenizer, task: str, n: int, scc: int, seed: int, template_kwargs) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": NEUTRAL_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": task_user_text(task, n, scc, seed)}]},
    ]
    extra = template_kwargs or {}
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(_plain_messages(messages), tokenize=False, add_generation_prompt=True, **extra)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--task", choices=("copy", "retrieval", "counting"), required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--collect-dir", type=Path, required=True, help="counting single-prompt cache")
    parser.add_argument("--interventions-dir", type=Path, required=True)
    parser.add_argument("--subspace-dir", type=Path, required=True, help="for unembed.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill", default="")
    parser.add_argument("--chat-kwargs", default="")
    parser.add_argument("--item", default="a")
    parser.add_argument("--delimiter", default=", ")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--layers", default="", help="extra layers for the `layers` group, e.g. 20-27")
    parser.add_argument("--n-localize-donors", type=int, default=5)
    parser.add_argument("--max-donors", type=int, default=0, help="debug: truncate the donor list")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = set(args.groups.split(","))
    scc = scanned_scc(args.scan_dir)
    iv_meta = json.loads((args.interventions_dir / "meta.json").read_text())
    base_n, donors = int(iv_meta["base_count"]), [int(d) for d in iv_meta["donor_counts"]]
    if args.max_donors:
        donors = donors[: args.max_donors]
    unembed = np.load(args.subspace_dir / "unembed.npz")
    u_basis = orthonormal_rows(unembed["rows"])
    max_n = int(1.3 * scc) + 1

    spec = ProbeSpec.load(args.config)
    model, tokenizer, device = load_model(spec)
    editor = DecodeEditor(model, tokenizer, device, args.max_new_tokens)
    template_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    sites = list(spec.hooks)
    hidden = int(unembed["rows"].shape[1])
    tag = args.task
    started = time.time()

    def prompt_ids(n: int) -> list[int]:
        if args.task == "counting":
            prompt = render_prompts(tokenizer, [n], args.item, args.delimiter, template_kwargs)[0]
        else:
            prompt = render_task_prompt(tokenizer, args.task, n, scc, args.seed, template_kwargs)
        return encode_text(tokenizer, prompt + args.prefill)

    def forward_states(ids: list[int], hooks: list[HookSpec]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Decode-point state at each hook, and the logits at the decode point."""
        q = len(ids) - 1
        if device == "mlx":
            from ..common.mlx_backend import MlxActivationRecorder, forward_logits

            with MlxActivationRecorder(model, hooks) as rec:
                logits = forward_logits(model, ids)[q]
                return {h.label: rec.cache[h.label][0][q].astype(np.float32) for h in hooks}, logits
        import torch

        from ..common.model import ActivationRecorder

        with ActivationRecorder(model, hooks) as rec, torch.inference_mode():
            out = model(torch.tensor([ids], device=device))
            states = {h.label: rec.cache[h.label][0][q].float().cpu().numpy() for h in hooks}
            logits = out.logits[0, q].float().cpu().numpy()
        return states, logits

    # -- collect ---------------------------------------------------------------
    cache_path = args.output_dir / f"output_{tag}_states.npz"
    if args.task == "counting":
        c = np.load(args.collect_dir / "activations.npz")
        counts = c["counts"].astype(int)
        correct = correctness_by_count(args.scan_dir)
        states = {h.label: c[f"{h.label}|decode_point"].astype(np.float32) for h in sites}
        exact = np.array([correct.get(int(n), False) for n in counts])
    elif "collect" in groups or not cache_path.exists():
        counts = np.arange(1, max_n + 1)
        states = {h.label: [] for h in sites}
        exact = []
        for n in counts:
            ids = prompt_ids(int(n))
            st, logits = forward_states(ids, sites)
            for h in sites:
                states[h.label].append(st[h.label])
            first_token = encode_text(tokenizer, str(int(n)))[0]
            exact.append(int(np.argmax(logits)) == int(first_token))
        states = {k: np.stack(v) for k, v in states.items()}
        exact = np.array(exact)
        np.savez_compressed(cache_path, counts=counts, exact=exact, **states)
        print(f"  {tag}: collected {len(counts)} prompts, first-token exact {exact.mean():.2f}  [{time.time() - started:.0f}s]")
    else:
        c = np.load(cache_path)
        counts, exact = c["counts"].astype(int), c["exact"].astype(bool)
        states = {h.label: c[h.label].astype(np.float32) for h in sites}
    index_of = {int(n): i for i, n in enumerate(counts)}
    stable = exact & (counts <= scc)
    print(f"{spec.name} {tag}: base {base_n}, {len(donors)} donors, stable {int(stable.sum())}/{len(counts)}")

    # Task bases: L (ridge readout of N), F (period-10 pair), trajectory span.
    bases: dict[str, dict] = {}
    ns = counts[stable].astype(np.float64)
    folds = np.arange(int(stable.sum())) % 5
    for h in sites:
        x = states[h.label][stable].astype(np.float64)
        lam, r2, _ = choose_lambda_grouped(x, ns, folds, grid=LAMBDA_GRID)
        w, _, _ = ridge_multi(x, ns, lam)
        l_basis = orthonormal_rows(w.T)
        targets = np.column_stack([np.cos(2 * np.pi * ns / 10), np.sin(2 * np.pi * ns / 10)])
        wf, _, _ = ridge_multi(x, targets, lam)
        f_basis = orthonormal_rows(wf.T)
        centered = x - x.mean(axis=0)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        span = vt[: int((s > 1e-6 * s[0]).sum())].astype(np.float32)
        bases[h.label] = {"L": l_basis, "F": f_basis, "span": span, "mu": x.mean(axis=0).astype(np.float32),
                          "linear_cv_r2": round(float(r2), 4)}
    (args.output_dir / f"output_{tag}_meta.json").write_text(json.dumps({
        "task": tag, "base": base_n, "donors": donors, "scc": scc, "n_stable": int(stable.sum()),
        "first_token_exact_rate": round(float(exact.mean()), 4),
        "linear_cv_r2": {h.label: bases[h.label]["linear_cv_r2"] for h in sites},
        "example_prompt": task_user_text(args.task, base_n, scc, args.seed) if args.task != "counting" else "paper prompt",
    }, indent=1))

    rows_path = args.output_dir / f"output_{tag}.jsonl"
    handle = rows_path.open("a" if rows_path.exists() else "w")
    n_rows = 0

    def emit(row: dict) -> None:
        nonlocal n_rows
        row["task"] = tag
        row["parsed"] = parse_integer(row["decoded"])
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        n_rows += 1

    base_ids = prompt_ids(base_n)
    informative = [d for d in donors if str(d)[0] != str(base_n)[0]]

    # -- localize --------------------------------------------------------------
    def all_layer_hooks() -> list[HookSpec]:
        prefix = sites[-1].module.rsplit(".", 1)[0]
        if device == "mlx":
            n_layers = len(model.model.layers)
        else:
            names = set(dict(model.named_modules()))
            n_layers = sum(1 for i in range(400) if f"{prefix}.{i}" in names)
        return [HookSpec(label=f"resid_post.layer{i}", module=f"{prefix}.{i}") for i in range(n_layers)]

    if "localize" in groups:
        hooks = all_layer_hooks()
        pick = np.linspace(0, len(informative) - 1, min(args.n_localize_donors, len(informative))).round().astype(int)
        loc_donors = [informative[i] for i in sorted(set(pick.tolist()))]
        donor_states = {d: forward_states(prompt_ids(d), hooks)[0] for d in loc_donors}
        emit({"group": "localize", "layer": None, "donor": None, "base": base_n,
              "decoded": editor.generate(base_ids, None, None)})
        for layer, hook in enumerate(hooks):
            for d in loc_donors:
                emit({"group": "localize", "layer": layer, "donor": d, "base": base_n,
                      "decoded": editor.generate(base_ids, hook, set_state(donor_states[d][hook.label]))})
        print(f"  localize done ({len(hooks)} layers)  [{n_rows} rows, {time.time() - started:.0f}s]", flush=True)

    # -- per-site groups -------------------------------------------------------
    for hook in sites:
        site = hook.label
        if not groups & {"transplant", "ablate", "interp"}:
            break
        x_site = states[site]
        base_state = x_site[index_of[base_n]]
        baseline = editor.generate(base_ids, hook, set_state(base_state))
        if "transplant" in groups:
            for d in donors:
                delta = x_site[index_of[d]] - base_state
                emit({"group": "transplant", "site": site, "family": "full", "donor": d, "complement": False,
                      "base": base_n, "baseline": baseline,
                      "decoded": editor.generate(base_ids, hook, set_state(x_site[index_of[d]]))})
                for comp in (False, True):
                    emit({"group": "transplant", "site": site, "family": "U", "donor": d, "complement": comp,
                          "base": base_n, "baseline": baseline,
                          "decoded": editor.generate(base_ids, hook, project_delta(base_state, delta, u_basis, comp))})
        if "ablate" in groups:
            b = bases[site]
            configs = [("U", u_basis), ("L", b["L"]), ("F", b["F"])]
            r_full = random_bases(b["span"], 1, args.seed + 3)[0]
            for k in (8, 16):
                if k <= b["span"].shape[0]:
                    configs.append((f"R{k}", r_full[:k]))
                configs.append((f"A{k}", ambient_random_basis(hidden, k, args.seed + 5 * k)))
            for n in donors:
                ids = prompt_ids(n)
                own = editor.generate(ids, hook, None)
                emit({"group": "ablate", "site": site, "family": "none", "count": n, "own": own, "decoded": own})
                for fam, basis in configs:
                    emit({"group": "ablate", "site": site, "family": fam, "count": n, "own": own,
                          "decoded": editor.generate(ids, hook, ablate_to_mean(basis, b["mu"]))})
        if "interp" in groups and hook is sites[-1]:
            for d in donors:
                delta = x_site[index_of[d]] - base_state
                for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                    emit({"group": "interp", "site": site, "donor": d, "alpha": alpha, "base": base_n,
                          "baseline": baseline,
                          "decoded": editor.generate(base_ids, hook, set_state(base_state + alpha * delta))})
        print(f"  {site} done  [{n_rows} rows, {time.time() - started:.0f}s]", flush=True)

    # -- extra layers (U vs complement at unhooked depths) ----------------------
    if "layers" in groups and args.layers:
        lo, hi = (int(v) for v in args.layers.split("-"))
        prefix = sites[-1].module.rsplit(".", 1)[0]
        hooks = [HookSpec(label=f"resid_post.layer{i}", module=f"{prefix}.{i}") for i in range(lo, hi + 1)]
        base_states = forward_states(base_ids, hooks)[0]
        for d in donors:
            d_states = forward_states(prompt_ids(d), hooks)[0]
            for hook in hooks:
                delta = d_states[hook.label] - base_states[hook.label]
                emit({"group": "layers", "site": hook.label, "family": "full", "donor": d, "complement": False,
                      "base": base_n, "decoded": editor.generate(base_ids, hook, set_state(d_states[hook.label]))})
                for comp in (False, True):
                    emit({"group": "layers", "site": hook.label, "family": "U", "donor": d, "complement": comp,
                          "base": base_n,
                          "decoded": editor.generate(base_ids, hook, project_delta(base_states[hook.label], delta, u_basis, comp))})
        print(f"  layers {args.layers} done  [{n_rows} rows, {time.time() - started:.0f}s]", flush=True)

    handle.close()
    print(f"wrote {n_rows} rows in {time.time() - started:.0f}s -> {rows_path}")


if __name__ == "__main__":
    main()
