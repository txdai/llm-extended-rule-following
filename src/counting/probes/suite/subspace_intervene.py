"""Causal tests of the subspace hypotheses at the decode point.

Same base prompt (0.5 x SCC), the same 20 donors and the same greedy decoding
as `intervene.py`; edits at the decode-point token. Groups:

  unembed     dump the gamma-scaled unembedding rows of the answer tokens
              (the decoder's own subspace U) and the first answer token per
              count, for the offline principal-angle and logit-lens analysis
  localize    full donor-state transplant at *every* layer for a few donors:
              the first layer at which the transplant carries the answer is
              the consolidation layer
  transplant  h_base + P_k (h_donor - h_base) and its complement for every
              hypothesis family (L, F, G, U, U_k, T_k, R_k in span(T), A_k
              ambient random), scored on the donor's answer
  interp      h_base + alpha (h_donor - h_base): number line or switch
  ablate      h - P_k h + P_k mu on the model's own prompts: necessity
  periodic    rotate the fitted (cos, sin) readout pair for a period by the
              phase of +d counts, against a random 2-plane in span(T)

    python -m counting.probes.suite.subspace_intervene configs/probes/gpt-oss-20b.yaml \
        --collect-dir results/suite/gpt-oss-20b/collect \
        --scan-dir results/suite/gpt-oss-20b/scan-final \
        --interventions-dir results/suite/gpt-oss-20b/interventions \
        --subspace-dir results/suite/gpt-oss-20b/subspace \
        --prefill '<|channel|>final<|message|>'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..common import ProbeSpec, load_model
from ..common.prompts import parse_integer, render_prompts
from ..common.spec import HookSpec
from .editing import DecodeEditor, ablate_to_mean, project_delta, rotate_pair, set_state
from .subspace import K_GRID, ambient_random_basis, random_bases
from .util import correctness_by_count, encode_text, scanned_scc

GROUPS = ("unembed", "localize", "transplant", "interp", "ablate", "periodic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--collect-dir", type=Path, required=True, help="single-prompt cache")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--interventions-dir", type=Path, required=True)
    parser.add_argument("--subspace-dir", type=Path, required=True)
    parser.add_argument("--prefill", default="")
    parser.add_argument("--chat-kwargs", default="")
    parser.add_argument("--item", default="a")
    parser.add_argument("--delimiter", default=", ")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--sites", default="last2", help="'last2', 'all', or comma list of hook labels")
    parser.add_argument("--k-grid", default=",".join(str(k) for k in K_GRID))
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--n-localize-donors", type=int, default=5)
    parser.add_argument("--max-donors", type=int, default=0, help="debug: truncate the donor list")
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument(
        "--complement-families", default="L,F,G,U,T",
        help="families whose complement (I - P) transplant is also run; random families are skipped",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def leading_digit(n: int) -> str:
    return str(n)[0]


def main() -> None:
    args = parse_args()
    args.subspace_dir.mkdir(parents=True, exist_ok=True)
    groups = [g for g in args.groups.split(",") if g]
    k_grid = [int(k) for k in args.k_grid.split(",")]
    complement_families = set(args.complement_families.split(","))
    scc = scanned_scc(args.scan_dir)
    correct = correctness_by_count(args.scan_dir)
    cache = np.load(args.collect_dir / "activations.npz")
    counts = cache["counts"].astype(np.int64)
    stable = np.array([c <= scc and correct.get(int(c), False) for c in counts])
    iv_meta = json.loads((args.interventions_dir / "meta.json").read_text())
    base_count = int(iv_meta["base_count"])
    donors = [int(d) for d in iv_meta["donor_counts"]]
    if args.max_donors:
        donors = donors[: args.max_donors]
    index_of = {int(c): i for i, c in enumerate(counts)}

    spec = ProbeSpec.load(args.config)
    model, tokenizer, device = load_model(spec)
    editor = DecodeEditor(model, tokenizer, device, args.max_new_tokens)
    template_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None

    def prompt_ids(count: int) -> list[int]:
        prompt = render_prompts(tokenizer, [count], args.item, args.delimiter, template_kwargs)[0]
        return encode_text(tokenizer, prompt + args.prefill)

    base_ids = prompt_ids(base_count)
    hidden = int(cache[f"{spec.hooks[0].label}|decode_point"].shape[1])
    if args.sites == "last2":
        sites = spec.hooks[-2:]
    elif args.sites == "all":
        sites = list(spec.hooks)
    else:
        sites = [spec.hook(label) for label in args.sites.split(",")]
    print(f"{spec.name} on {device}: groups {groups}, sites {[h.label for h in sites]}, base {base_count}, {len(donors)} donors")

    bases = dict(np.load(args.subspace_dir / "bases.npz")) if (args.subspace_dir / "bases.npz").exists() else {}
    started = time.time()
    handles: dict[str, object] = {}
    n_rows = 0

    def emit(group: str, row: dict) -> None:
        nonlocal n_rows
        if group not in handles:
            path = args.subspace_dir / f"{group}.jsonl"
            handles[group] = path.open("a" if path.exists() else "w")
        row["parsed"] = parse_integer(row["decoded"])
        handles[group].write(json.dumps(row) + "\n")
        handles[group].flush()
        n_rows += 1

    # -- unembed ---------------------------------------------------------------
    if "unembed" in groups:
        max_answer = int(1.3 * scc) + 1
        first_token = np.full(max(max_answer, int(counts.max())) + 1, -1, dtype=np.int64)
        for c in range(1, len(first_token)):
            first_token[c] = encode_text(tokenizer, str(c))[0]
        ids = sorted(set(int(t) for t in first_token[1:max_answer + 1]))
        last_hook = spec.hooks[-1]
        if device == "mlx":
            import mlx.core as mx

            gamma = np.array(model.model.norm(mx.ones((1, hidden))).astype(mx.float32))[0]
            w_t = model.lm_head(mx.eye(hidden))  # (hidden, vocab): column t is W_U[t]
            rows = np.array(w_t[:, ids].astype(mx.float32)).T
            del w_t
        else:
            import torch

            modules = dict(model.named_modules())
            prefix = last_hook.module.rsplit(".layers.", 1)[0]
            norm = modules[f"{prefix}.norm"]
            with torch.inference_mode():
                ones = torch.ones(hidden, device=device, dtype=next(model.parameters()).dtype)
                gamma = norm(ones).float().cpu().numpy()
                head = model.get_output_embeddings()
                if head is None:
                    head = modules["lm_head"]
                rows = head.weight[torch.tensor(ids, device=head.weight.device)].float().cpu().numpy()
        rows = rows * gamma[None, :]
        tok = getattr(tokenizer, "tokenizer", tokenizer)
        np.savez_compressed(
            args.subspace_dir / "unembed.npz",
            token_ids=np.array(ids, dtype=np.int64), rows=rows.astype(np.float32),
            gamma=gamma.astype(np.float32), first_token_by_count=first_token,
        )
        (args.subspace_dir / "unembed.json").write_text(json.dumps({
            "token_strings": {int(t): tok.decode([int(t)]) for t in ids},
            "answer_counts": f"1..{max_answer}", "hidden": hidden,
            "digit_split": bool(len(ids) <= 10),
        }, indent=1))
        print(f"  unembed: {len(ids)} answer tokens, gamma mean {gamma.mean():.3f}  [{time.time() - started:.0f}s]")

    # -- localize --------------------------------------------------------------
    if "localize" in groups:
        last_hook = spec.hooks[-1]
        prefix = last_hook.module.rsplit(".", 1)[0]
        if device == "mlx":
            n_layers = len(model.model.layers)
        else:
            names = set(dict(model.named_modules()))
            n_layers = sum(1 for i in range(400) if f"{prefix}.{i}" in names)
        all_hooks = [HookSpec(label=f"resid_post.layer{i}", module=f"{prefix}.{i}") for i in range(n_layers)]
        candidates = [d for d in donors if leading_digit(d) != leading_digit(base_count)] or donors
        pick = np.linspace(0, len(candidates) - 1, min(args.n_localize_donors, len(candidates))).round().astype(int)
        loc_donors = [candidates[i] for i in sorted(set(pick.tolist()))]

        donor_states: dict[int, np.ndarray] = {}
        for d in loc_donors:
            ids = prompt_ids(d)
            q_idx = len(ids) - 1
            if device == "mlx":
                from ..common.mlx_backend import MlxActivationRecorder, forward_logits

                with MlxActivationRecorder(model, all_hooks) as rec:
                    forward_logits(model, ids)
                    donor_states[d] = np.stack([rec.cache[h.label][0][q_idx] for h in all_hooks]).astype(np.float32)
            else:
                import torch

                from ..common.model import ActivationRecorder

                with ActivationRecorder(model, all_hooks) as rec, torch.inference_mode():
                    model(torch.tensor([ids], device=device))
                    donor_states[d] = np.stack(
                        [rec.cache[h.label][0][q_idx].float().cpu().numpy() for h in all_hooks]
                    )
        emit("localize", {"layer": None, "donor": None, "base": base_count,
                          "decoded": editor.generate(base_ids, None, None)})
        for layer, hook in enumerate(all_hooks):
            for d in loc_donors:
                emit("localize", {
                    "layer": layer, "donor": d, "base": base_count,
                    "decoded": editor.generate(base_ids, hook, set_state(donor_states[d][layer])),
                })
            if layer % 8 == 7:
                print(f"  localize layer {layer}/{n_layers}  [{n_rows} rows, {time.time() - started:.0f}s]", flush=True)
        (args.subspace_dir / "localize_meta.json").write_text(json.dumps({
            "n_layers": n_layers, "donors": loc_donors, "base": base_count}, indent=1))

    # -- per-site families -----------------------------------------------------
    def family_configs(site: str, with_complement: bool) -> list[dict]:
        configs: list[dict] = []
        t_basis = bases[f"{site}|T"]
        rank = int(t_basis.shape[0])

        def add(family: str, k: int, basis: np.ndarray, draw: int | None = None) -> None:
            if basis.shape[0] == 0:
                return
            configs.append({"family": family, "k": int(k), "draw": draw, "basis": basis})

        add("L", 1, bases[f"{site}|L"])
        if f"{site}|F" in bases:
            add("F", bases[f"{site}|F"].shape[0], bases[f"{site}|F"])
        if f"{site}|G" in bases:
            add("G", bases[f"{site}|G"].shape[0], bases[f"{site}|G"])
        if f"{site}|U" in bases:
            u_basis = bases[f"{site}|U"]
            add("U", u_basis.shape[0], u_basis)
            if u_basis.shape[0] > 2:
                rows = bases_unembed_rows if bases_unembed_rows is not None else u_basis
                _, _, vt = np.linalg.svd(rows.astype(np.float64), full_matrices=False)
                for k in k_grid:
                    if k < u_basis.shape[0]:
                        add("U_k", k, vt[:k].astype(np.float32))
        for k in k_grid:
            if k < rank:
                add("T", k, t_basis[:k])
        add("T", rank, t_basis)
        for draw, r_basis in enumerate(random_bases(t_basis, args.n_random, args.seed)):
            for k in k_grid:
                if k <= rank:
                    add("R", k, r_basis[:k], draw)
        for k in k_grid:
            add("A", k, ambient_random_basis(hidden, k, args.seed + 7 * k))
        return configs

    bases_unembed_rows = None
    if (args.subspace_dir / "unembed.npz").exists():
        bases_unembed_rows = np.load(args.subspace_dir / "unembed.npz")["rows"]
        if any(f"{h.label}|U" not in bases for h in sites):
            from .subspace import orthonormal_rows

            for h in sites:
                bases[f"{h.label}|U"] = orthonormal_rows(bases_unembed_rows)

    for hook in sites:
        site = hook.label
        if not any(g in groups for g in ("transplant", "interp", "ablate", "periodic")):
            break
        states = cache[f"{site}|decode_point"].astype(np.float32)
        base_state = states[index_of[base_count]]
        baseline = editor.generate(base_ids, hook, set_state(base_state))
        print(f"  {site}: baseline '{baseline}'  [{time.time() - started:.0f}s]", flush=True)

        if "transplant" in groups:
            configs = family_configs(site, True)
            for d in donors:
                donor_state = states[index_of[d]]
                delta = donor_state - base_state
                emit("transplant", {"site": site, "family": "full", "k": None, "draw": None, "donor": d,
                                    "complement": False, "base": base_count, "baseline": baseline,
                                    "decoded": editor.generate(base_ids, hook, set_state(donor_state))})
                for cfg in configs:
                    variants = (False, True) if cfg["family"] in complement_families else (False,)
                    for complement in variants:
                        emit("transplant", {
                            "site": site, "family": cfg["family"], "k": cfg["k"], "draw": cfg["draw"],
                            "donor": d, "complement": complement, "base": base_count, "baseline": baseline,
                            "decoded": editor.generate(
                                base_ids, hook, project_delta(base_state, delta, cfg["basis"], complement)
                            ),
                        })
                print(f"    transplant donor {d} done  [{n_rows} rows, {time.time() - started:.0f}s]", flush=True)

        if "interp" in groups:
            for d in donors:
                delta = states[index_of[d]] - base_state
                for alpha in [float(a) for a in args.alphas.split(",")]:
                    emit("interp", {"site": site, "donor": d, "alpha": alpha, "base": base_count,
                                    "baseline": baseline,
                                    "decoded": editor.generate(base_ids, hook, set_state(base_state + alpha * delta))})
            print(f"    interp done  [{n_rows} rows, {time.time() - started:.0f}s]", flush=True)

        if "ablate" in groups:
            mu = bases.get(f"{site}|mu_single", states[stable].mean(axis=0))
            configs = family_configs(site, False)
            for c in donors:
                ids = prompt_ids(c)
                own = editor.generate(ids, hook, None)
                emit("ablate", {"site": site, "family": "none", "k": None, "draw": None, "count": c,
                                "own": own, "decoded": own})
                for cfg in configs:
                    emit("ablate", {
                        "site": site, "family": cfg["family"], "k": cfg["k"], "draw": cfg["draw"],
                        "count": c, "own": own,
                        "decoded": editor.generate(ids, hook, ablate_to_mean(cfg["basis"], mu)),
                    })
                print(f"    ablate count {c} done  [{n_rows} rows, {time.time() - started:.0f}s]", flush=True)

        if "periodic" in groups:
            t_basis = bases[f"{site}|T"]
            control = random_bases(t_basis, 1, args.seed + 99)[0][:2]
            mu = bases.get(f"{site}|mu_single", states[stable].mean(axis=0))
            readouts_path = args.subspace_dir / "readouts.json"
            readable = set()
            if readouts_path.exists():
                readable = set(json.loads(readouts_path.read_text())["sites"].get(site, {})
                               .get("periodic", {}).get("readable_periods", []))
            for period in (10, 5, 2):
                key = f"{site}|P{period}_w"
                # P = 10 is the primary test and always runs; a secondary pair
                # that is not readable has near-zero weights, and rotating its
                # readout is a huge off-manifold edit rather than a phase change.
                if key not in bases or (period != 10 and period not in readable):
                    continue
                w, xc, yc = bases[key], bases[f"{site}|P{period}_xc"], bases[f"{site}|P{period}_yc"]
                for c in donors:
                    ids = prompt_ids(c)
                    own = editor.generate(ids, hook, None)
                    h_cached = states[index_of[c]]
                    for d in range(1, period):
                        phase = 2 * np.pi * d / period
                        fn = rotate_pair(w, xc, yc, phase)
                        ratio = float(np.linalg.norm(fn(h_cached) - h_cached) / np.linalg.norm(h_cached))
                        emit("periodic", {"site": site, "period": period, "family": "F", "count": c, "d": d,
                                          "own": own, "edit_norm_ratio": round(ratio, 4),
                                          "decoded": editor.generate(ids, hook, fn)})
                        if period == 10:
                            fn_r = rotate_pair(control, mu, np.zeros(2, np.float32), phase)
                            ratio_r = float(np.linalg.norm(fn_r(h_cached) - h_cached) / np.linalg.norm(h_cached))
                            emit("periodic", {"site": site, "period": period, "family": "R2", "count": c, "d": d,
                                              "own": own, "edit_norm_ratio": round(ratio_r, 4),
                                              "decoded": editor.generate(ids, hook, fn_r)})
                print(f"    periodic P={period} done  [{n_rows} rows, {time.time() - started:.0f}s]", flush=True)

    for handle in handles.values():
        handle.close()
    meta_path = args.subspace_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "name": spec.name, "scc": scc, "base_count": base_count, "donor_counts": donors,
        "sites": [h.label for h in sites], "k_grid": k_grid, "n_random": args.n_random,
        "prefill": args.prefill, "seed": args.seed,
        "runs": meta.get("runs", []) + [{"groups": groups, "n_rows": n_rows,
                                          "elapsed_s": round(time.time() - started, 1)}],
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {n_rows} rows in {time.time() - started:.0f}s -> {args.subspace_dir}")


if __name__ == "__main__":
    main()
