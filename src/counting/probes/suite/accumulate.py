"""Multi-position intervention on the accumulation (round-2 E4).

The decode-point nulls of the suite say nothing about whether the count
information at the *item* positions is used. This script edits the running
count at every item token from a chosen position onward and reads the answer.

Steps (`--groups`):

  collect   one forward pass at the longest prompt: because attention is
            causal and the prompts share their prefix, the state at the i-th
            item token is the same in every prompt with at least i items, so
            the item states of the longest prompt are the item states of all
            prompts. Saved as `positions.npz` (regenerable, not committed).
  fit       per site: ridge readout of the running count at item tokens
            (stable regime), the running-count subspace S_k (top-k principal
            components of the item states, k = the trajectory dimension that
            carried 90% of transfer in the subspace controls), and a random
            k-subspace inside the same span as control.
  shift     for prompts of length L inside the stable regime, at every item
            position from j onward, replace the S_k component of the state at
            running count i by that of running count i + delta (the
            trajectory's own increment, `variant = local`) or add delta times
            the mean per-item increment in S_k (`variant = mean`). Controls:
            the same edit through the random subspace, and the same
            (last-position) shift at the decode point only.
  beyond    the same shifts at L from 1.0 to 1.25 x SCC.
  rescale   at L beyond SCC, compress the running count at every item position
            so that the last item reads as `target` (the largest in-regime
            value): if consolidation fails because the accumulator leaves the
            decoder's range, the answer should follow the rescaled count.

    python -m counting.probes.suite.accumulate configs/probes/gpt-oss-20b.yaml \
        --scan-dir results/suite/gpt-oss-20b/scan-final \
        --output-dir results/suite/gpt-oss-20b/round2 --k 16 \
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
from .editing import DecodeEditor
from .subspace import orthonormal_rows, random_bases
from .util import correctness_by_count, encode_text, scanned_scc, underlying_tokenizer

GROUPS = ("collect", "fit", "shift", "beyond", "rescale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True, help="running-count subspace dimension")
    parser.add_argument("--prefill", default="")
    parser.add_argument("--chat-kwargs", default="")
    parser.add_argument("--item", default="a")
    parser.add_argument("--delimiter", default=", ")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--sites", default="all")
    parser.add_argument("--deltas", default="-8,-4,-2,-1,1,2,4,8")
    parser.add_argument("--js", default="0.25,0.5,0.75", help="fractions of L from which the edit applies")
    parser.add_argument("--lengths", default="0.35,0.5,0.7,0.9", help="in-regime prompt lengths, as fractions of SCC")
    parser.add_argument("--beyond", default="1.0,1.1,1.25", help="beyond-regime lengths, as fractions of SCC")
    parser.add_argument("--variants", default="local,mean")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def item_positions(tokenizer, ids: list[int], item: str, count: int) -> list[int]:
    """Positions of the last `count` tokens that decode to the item string."""
    tok = underlying_tokenizer(tokenizer)
    hits = [i for i, t in enumerate(ids) if tok.decode([int(t)]).strip() == item]
    if len(hits) < count:
        raise RuntimeError(f"found {len(hits)} item tokens, expected at least {count}")
    return hits[-count:]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = set(args.groups.split(","))
    scc = scanned_scc(args.scan_dir)
    correct = correctness_by_count(args.scan_dir)
    deltas = [int(d) for d in args.deltas.split(",")]
    js = [float(v) for v in args.js.split(",")]
    variants = args.variants.split(",")
    max_count = int(round(1.25 * scc)) + max(deltas)

    spec = ProbeSpec.load(args.config)
    model, tokenizer, device = load_model(spec)
    editor = DecodeEditor(model, tokenizer, device, args.max_new_tokens)
    template_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    sites = list(spec.hooks) if args.sites == "all" else [spec.hook(s) for s in args.sites.split(",")]
    rng = np.random.default_rng(args.seed)
    started = time.time()

    def prompt_ids(count: int) -> list[int]:
        prompt = render_prompts(tokenizer, [count], args.item, args.delimiter, template_kwargs)[0]
        return encode_text(tokenizer, prompt + args.prefill)

    # -- collect ---------------------------------------------------------------
    cache_path = args.output_dir / "positions.npz"
    if "collect" in groups or not cache_path.exists():
        ids = prompt_ids(max_count)
        pos = item_positions(tokenizer, ids, args.item, max_count)
        arrays = {"positions": np.array(pos, dtype=np.int64), "running_count": np.arange(1, max_count + 1)}
        if device == "mlx":
            from ..common.mlx_backend import MlxActivationRecorder, forward_logits

            with MlxActivationRecorder(model, sites) as rec:
                forward_logits(model, ids)
                for hook in sites:
                    arrays[hook.label] = rec.cache[hook.label][0][pos].astype(np.float32)
        else:
            import torch

            from ..common.model import ActivationRecorder

            with ActivationRecorder(model, sites) as rec, torch.inference_mode():
                model(torch.tensor([ids], device=device))
                for hook in sites:
                    arrays[hook.label] = rec.cache[hook.label][0][pos].float().cpu().numpy()
            # Sanity check of the shared-prefix argument on a shorter prompt.
            short = int(round(0.5 * scc))
            ids_s = prompt_ids(short)
            pos_s = item_positions(tokenizer, ids_s, args.item, short)
            with ActivationRecorder(model, sites[-1:]) as rec, torch.inference_mode():
                model(torch.tensor([ids_s], device=device))
                x_s = rec.cache[sites[-1].label][0][pos_s].float().cpu().numpy()
            rel = float(np.linalg.norm(x_s - arrays[sites[-1].label][:short]) / np.linalg.norm(x_s))
            arrays["prefix_check_rel_err"] = np.array(rel)
            print(f"  shared-prefix check at {sites[-1].label}: relative error {rel:.2e}")
        np.savez_compressed(cache_path, **arrays)
        print(f"  collected item states for {max_count} positions  [{time.time() - started:.0f}s]")
    cache = dict(np.load(cache_path))
    running = cache["running_count"]
    stable = np.array([c <= scc and correct.get(int(c), False) for c in running])

    # -- fit -------------------------------------------------------------------
    fits: dict[str, dict] = {}
    fit_summary: dict[str, dict] = {}
    for hook in sites:
        x = cache[hook.label].astype(np.float64)
        xs, ys = x[stable], running[stable].astype(np.float64)
        centered = xs - xs.mean(axis=0)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        rank = int((s > 1e-6 * s[0]).sum())
        k = min(args.k, rank)
        basis = vt[:k].astype(np.float32)
        span = vt[:rank].astype(np.float32)
        control = random_bases(span, 1, args.seed + 11)[0][:k]
        # Ridge readout of the running count (dual form, lambda by leave-one-out R^2 on a small grid).
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0):
            preds = np.empty(len(ys))
            for i in range(len(ys)):
                tr = np.arange(len(ys)) != i
                xc = xs[tr] - xs[tr].mean(axis=0)
                kern = xc @ xc.T
                scale = np.trace(kern) / len(xc)
                alpha = np.linalg.solve(kern + lam * scale * np.eye(len(xc)), ys[tr] - ys[tr].mean())
                preds[i] = (xs[i] - xs[tr].mean(axis=0)) @ (xc.T @ alpha) + ys[tr].mean()
            r2 = 1 - np.mean((preds - ys) ** 2) / np.var(ys)
            if best is None or r2 > best[1]:
                best = (lam, r2, float(np.sqrt(np.mean((preds - ys) ** 2))))
        steps = np.diff(x[: int(scc)], axis=0)
        mean_step = (basis.astype(np.float64) @ steps.mean(axis=0)) @ basis.astype(np.float64)
        fits[hook.label] = {"basis": basis, "control": control, "mean_step": mean_step.astype(np.float32),
                            "explained": [round(float(v), 4) for v in (s[:k] ** 2 / (s ** 2).sum())]}
        fit_summary[hook.label] = {
            "k": k, "rank": rank, "loo_r2": round(best[1], 4), "loo_rmse_items": round(best[2], 3),
            "lambda": best[0], "cum_explained_k": round(float((s[:k] ** 2).sum() / (s ** 2).sum()), 4),
            "mean_step_norm_over_state_norm": round(float(np.linalg.norm(mean_step) / np.linalg.norm(x[int(scc // 2)])), 4),
        }
        print(f"  {hook.label}: running-count LOO R2 {best[1]:.3f} (rmse {best[2]:.2f}); S_{k} explains {fit_summary[hook.label]['cum_explained_k']:.3f}")
    (args.output_dir / "fit.json").write_text(json.dumps({"scc": scc, "n_stable": int(stable.sum()), "sites": fit_summary}, indent=1))
    if not groups & {"shift", "beyond", "rescale"}:
        return

    # -- edits -----------------------------------------------------------------
    rows_path = args.output_dir / "accumulate.jsonl"
    handle = rows_path.open("a" if rows_path.exists() else "w")
    n_rows = 0

    def emit(row: dict) -> None:
        nonlocal n_rows
        row["parsed"] = parse_integer(row["decoded"])
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        n_rows += 1

    def shift_vectors(site: str, family: str, variant: str, counts_at_positions: np.ndarray, delta: int) -> np.ndarray:
        """Per-position shift vectors for a block of item states with the given running counts.

        The random control (`R`) is a fixed random direction inside the span of
        the item states (drawn once per site), applied with the sign of delta
        and, per position, the norm of the `S` shift. It is therefore matched
        in magnitude and coherent across positions but unrelated to the count
        code; projecting the true increment onto a random subspace would keep
        a correlated component and is not a null.
        """
        fit = fits[site]
        x_site = cache[site].astype(np.float32)
        bs = fit["basis"].astype(np.float32)
        if variant == "local":
            targets = np.clip(counts_at_positions + delta, 1, len(x_site))
            raw = np.stack([x_site[t - 1] - x_site[c - 1] for c, t in zip(counts_at_positions, targets)])
        else:
            raw = np.tile(delta * fit["mean_step"], (len(counts_at_positions), 1))
        s_shift = raw @ bs.T @ bs
        if family == "S":
            return s_shift
        direction = fit["control"][0].astype(np.float32)
        norms = np.linalg.norm(s_shift, axis=1, keepdims=True)
        return np.sign(delta) * norms * direction[None, :]

    def shift_fn(site: str, family: str, variant: str, counts_at_positions: np.ndarray, delta: int):
        shifts = shift_vectors(site, family, variant, counts_at_positions, delta)
        return lambda block: block + shifts

    def run_shifts(length: int, band: str) -> None:
        ids = prompt_ids(length)
        pos = item_positions(tokenizer, ids, args.item, length)
        baseline = editor.generate(ids, None, None)
        emit({"group": "baseline", "band": band, "length": length, "site": None, "family": None,
              "variant": None, "j": None, "delta": 0, "decoded": baseline})
        for hook in sites:
            site = hook.label
            for jf in js:
                j = max(1, int(round(jf * length)))
                edited = list(range(j - 1, length))  # item indices (0-based) from j onward
                positions = [pos[i] for i in edited]
                counts = np.array([i + 1 for i in edited])
                for variant in variants:
                    for family in ("S", "R"):
                        for delta in deltas:
                            emit({"group": "shift", "band": band, "length": length, "site": site,
                                  "family": family, "variant": variant, "j": j, "delta": delta,
                                  "baseline": baseline,
                                  "decoded": editor.generate(ids, hook, shift_fn(site, family, variant, counts, delta), positions)})
            # Decode-point-only control: the last position's shift applied at the decode point.
            for variant in variants:
                for delta in deltas:
                    fn_block = shift_fn(site, "S", variant, np.array([length]), delta)
                    fn = lambda h, _f=fn_block: _f(h[None, :])[0]
                    emit({"group": "shift", "band": band, "length": length, "site": site,
                          "family": "S_decode_point", "variant": variant, "j": None, "delta": delta,
                          "baseline": baseline, "decoded": editor.generate(ids, hook, fn)})
            print(f"    {band} L={length} {site} done  [{n_rows} rows, {time.time() - started:.0f}s]", flush=True)

    if "shift" in groups:
        for frac in [float(v) for v in args.lengths.split(",")]:
            length = max(4, int(round(frac * scc)))
            run_shifts(length, "stable")
    if "beyond" in groups:
        for frac in [float(v) for v in args.beyond.split(",")]:
            run_shifts(int(round(frac * scc)), "beyond")
    if "rescale" in groups:
        for frac in [float(v) for v in args.beyond.split(",")]:
            length = int(round(frac * scc))
            if length <= scc:
                continue
            ids = prompt_ids(length)
            pos = item_positions(tokenizer, ids, args.item, length)
            baseline = editor.generate(ids, None, None)
            for target in (scc, int(round(0.8 * scc))):
                for hook in sites:
                    site = hook.label
                    b = fits[site]["basis"]
                    x_site = cache[site].astype(np.float32)
                    counts = np.arange(1, length + 1)
                    mapped = np.clip(np.round(counts * target / length).astype(int), 1, len(x_site))
                    shifts = np.stack([b.T @ (b @ (x_site[m - 1] - x_site[c - 1])) for c, m in zip(counts, mapped)])
                    emit({"group": "rescale", "band": "beyond", "length": length, "site": site, "family": "S",
                          "variant": "compress", "j": 1, "delta": int(target - length), "target": int(target),
                          "baseline": baseline,
                          "decoded": editor.generate(ids, hook, lambda block, _s=shifts: block + _s, pos)})
            print(f"    rescale L={length} done  [{n_rows} rows, {time.time() - started:.0f}s]", flush=True)

    handle.close()
    meta = {"name": spec.name, "scc": scc, "k": args.k, "deltas": deltas, "js": js, "variants": variants,
            "lengths": args.lengths, "beyond": args.beyond, "sites": [h.label for h in sites],
            "n_rows": n_rows, "elapsed_s": round(time.time() - started, 1), "prefill": args.prefill}
    (args.output_dir / "accumulate_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n_rows} rows in {meta['elapsed_s']}s -> {rows_path}")


if __name__ == "__main__":
    main()
