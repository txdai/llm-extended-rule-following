"""Counting-only discovery and held-out gate for natural event-message patches.

    python -m counting.probes.suite.causal_crossing --model gpt-oss-20b

See docs/plans/causal-crossing-stage1-protocol.md. No API inference is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import platform
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..common.model import load_model
from ..common.prompts import render_prompts
from ..common.spec import ProbeSpec

SCC = {"gpt-oss-20b": 36, "gemma-4-26b-a4b-it": 75}
DATE = "2026-09-15"


def exact_integer(text):
    return int(text.strip()) if re.fullmatch(r"[+-]?\d+", text.strip()) else None


@dataclass(frozen=True)
class Route:
    family: str
    layers: tuple[int, ...]

    @property
    def name(self):
        return self.family + ":" + ",".join(map(str, self.layers))


def candidates(depth, sites):
    routes = [Route("residual", tuple(range(i, depth))) for i in range(depth)]
    routes += [Route("message", (i,)) for i in range(depth)]
    routes += [Route("message", tuple(range(i, depth))) for i in [0, *sites]]
    routes += [Route("message", tuple(range(i, min(i + 3, depth)))) for i in sites]
    return list(dict.fromkeys(routes))


def cases(model):
    counts = np.random.default_rng(7).permutation(np.arange(8, SCC[model] + 1))
    rows = []
    for split, ns, fractions in [("calibration", counts[:6], (.2, .5, .8)),
                                  ("heldout", counts[6:14], (.3, .6, .9))]:
        for n in sorted(map(int, ns)):
            for frac in fractions:
                j = max(1, min(n - 1, int(np.floor(n * frac + .5))))
                rows.append(dict(id=f"{split}-n{n}-j{j}", split=split, n=n, j=j,
                                 fraction=frac))
    return rows


def make_prompt(tokenizer, model, case):
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    text = render_prompts(tokenizer, [case["n"]])[0]
    if model.startswith("gpt-oss"):
        # This template calls strftime_now directly and ignores date_string.
        text = re.sub(r"Current date: \d{4}-\d{2}-\d{2}", f"Current date: {DATE}", text)
        text += "<|channel|>final<|message|>"
    ids = tok.encode(text, add_special_tokens=False)
    positions = [i for i, t in enumerate(ids) if tok.decode([t]).strip() == "a"][-case["n"]:]
    assert len(positions) == case["n"]
    end, source = positions[-1] + 1, positions[case["j"] - 1]
    prefix, suffix = ids[:end], ids[end:]
    target_piece = tok.decode([prefix[source]])
    replacement = tok.encode(target_piece.replace("a", "x"), add_special_tokens=False)
    assert len(replacement) == 1 and replacement[0] != prefix[source]
    donor = prefix.copy()
    donor[source] = replacement[0]
    # Encode just the added continuation: the tokenized history is never changed.
    factual_text = (f"\n\nInstead of counting, identify the item at position {case['j']} "
                    "in the sequence above (positions start at 1). "
                    "Return only 1 if that item is a, or 2 if that item is x.")
    factual = tok.encode(factual_text, add_special_tokens=False) + suffix
    alternatives = [p for p in positions if p != source]
    other = int(np.random.default_rng(7 + case["n"] * 100 + case["j"]).choice(alternatives))
    assert sum(a != b for a, b in zip(prefix, donor)) == 1
    return dict(**case, prefix=prefix, donor=donor, suffix=suffix, factual=factual,
                source=source, random_source=other, positions=positions,
                text=tok.decode(prefix), donor_text=tok.decode(donor),
                suffix_text=tok.decode(suffix), factual_text=tok.decode(factual))


def replace_source(output, weights, values, source, donor_message):
    """Replace a single source's weighted message; output is B,Q,H,D.

    Values must already be expanded from KV heads to attention heads. Earlier
    queries and the source query remain bit-identical. Donor has B,Q,H,D shape.
    """
    recipient = (weights[..., source:source + 1] * values[:, :, source:source + 1, :]).transpose(1, 2)
    edited = output.clone()
    # Add the difference in fp32, then cast once to limit bf16 cancellation.
    edited[:, source + 1:] = (output[:, source + 1:].float()
                              + (donor_message[:, source + 1:].float()
                                 - recipient[:, source + 1:].float())).to(output.dtype)
    return edited


class PrefixEngine:
    """Patch a history once; branch questions from independent copies of its cache."""

    def __init__(self, model, tokenizer, spec, device):
        import torch
        self.torch, self.model, self.device = torch, model, device
        self.tok = getattr(tokenizer, "tokenizer", tokenizer)
        path = spec.residual_hooks[0].module.rsplit(".", 1)[0]
        self.layers = model.get_submodule(path)
        self.depth = len(self.layers)
        self.module = importlib.import_module(type(self.layers[0].self_attn).__module__)
        self.original_attention = self.module.eager_attention_forward
        self.active = False
        self.pairs = {}
        self.handles = []
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._residual_hook(i)))
        self.module.eager_attention_forward = self._attention

    def close(self):
        self.module.eager_attention_forward = self.original_attention
        for handle in self.handles:
            handle.remove()

    def _residual_hook(self, layer):
        def hook(module, inputs, output):
            torch = self.torch
            if not self.active:
                return output
            h = output[0] if isinstance(output, tuple) else output
            self.protected[layer] = h.detach().clone()
            if self.route and self.route.family == "message":
                # MoE grouping changes bf16 GEMM rounding even for causally earlier
                # tokens. Enforce the exact-arithmetic invariant with recipient
                # states, never donor states, on positions that cannot be affected.
                original = self.recipient["protected"][layer][:, :self.source + 1]
                difference = h[:, :self.source + 1].float() - original.float()
                self.guard_drift[layer] = dict(max_abs=float(difference.abs().max()),
                    relative_l2=float(torch.linalg.vector_norm(difference)
                                      / torch.linalg.vector_norm(original.float()).clamp_min(1e-8)))
                h = h.clone()
                h[:, :self.source + 1] = original
            for source in self.sources:
                self.states[(layer, source)] = h[:, source].detach().clone()
            if self.route and self.route.family == "residual" and layer in self.route.layers:
                h = h.clone()
                h[:, self.source] = self.donor["states"][(layer, self.source)]
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return hook

    def _attention(self, module, query, key, value, attention_mask, **kwargs):
        output, weights = self.original_attention(module, query, key, value, attention_mask, **kwargs)
        if not self.active:
            return output, weights
        layer = module.layer_idx
        values = self.module.repeat_kv(value, module.num_key_value_groups)
        for source in self.sources:
            message = (weights[..., source:source + 1] * values[:, :, source:source + 1, :]).transpose(1, 2)
            self.messages[(layer, source)] = message.detach().clone()
            self.attention[(layer, source)] = weights[0, :, source + 1:, source].float().mean(dim=-1).cpu().tolist()
        if self.route and self.route.family == "message" and layer in self.route.layers:
            output = replace_source(output, weights, values, self.source,
                                    self.donor["messages"][(layer, self.source)])
        return output, weights

    def _forward(self, ids, cache=None):
        return self.model(input_ids=self.torch.tensor([ids], device=self.device),
                          past_key_values=cache, use_cache=True, logits_to_keep=1)

    def prefix(self, ids, sources, route=None, donor=None, source=None, recipient=None):
        self.sources, self.route, self.donor, self.source = sources, route, donor, source
        self.recipient = recipient
        self.states, self.messages, self.attention = {}, {}, {}
        self.protected, self.guard_drift = {}, {}
        self.active = True
        try:
            with self.torch.inference_mode():
                out = self._forward(ids)
        finally:
            self.active = False
        return dict(cache=out.past_key_values, states=self.states, messages=self.messages,
                    attention=self.attention, protected=self.protected, guard_drift=self.guard_drift)

    def decode(self, prefix, suffix):
        cache = copy.deepcopy(prefix["cache"])
        generated = []
        with self.torch.inference_mode():
            out = self._forward(suffix, cache)
            for step in range(8):
                token = int(out.logits[0, -1].argmax())
                if token in self.eos:
                    break
                generated.append(token)
                if step < 7:
                    out = self._forward([token], out.past_key_values)
        text = self.tok.decode(generated, skip_special_tokens=True)
        return dict(decoded=text, parsed=exact_integer(text), generated=generated)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def evaluate(engine, prompt, route=None, control="edit", factual=True):
    """Return all raw responses and diagnostics for one paired history."""
    import torch
    sources = [prompt["source"], prompt["random_source"]]
    if prompt["id"] not in engine.pairs:
        engine.pairs[prompt["id"]] = (engine.prefix(prompt["prefix"], sources),
                                       engine.prefix(prompt["donor"], sources))
    target, donor = engine.pairs[prompt["id"]]
    if route is None:
        result = dict(target_count=engine.decode(target, prompt["suffix"]),
                      donor_count=engine.decode(donor, prompt["suffix"]),
                      target_fact=engine.decode(target, prompt["factual"]),
                      donor_fact=engine.decode(donor, prompt["factual"]))
        result["ordinary_route"] = [dict(layer=i,
            attention_heads=target["attention"][(i, sources[0])],
            residual_difference=float(torch.linalg.vector_norm(
                target["states"][(i, sources[0])].float() - donor["states"][(i, sources[0])].float())))
            for i in range(engine.depth)]
        return result
    actual = route
    if control == "random_layer":
        ls = np.random.default_rng(7 + route.layers[0]).choice(engine.depth, len(route.layers), replace=False)
        actual = Route(route.family, tuple(sorted(map(int, ls))))
    source = prompt["random_source"] if control == "random_source" else prompt["source"]
    edited = engine.prefix(prompt["prefix"], sources, actual,
                           target if control == "sham" else donor, source, recipient=target)
    result = dict(count=engine.decode(edited, prompt["suffix"]),
                  applied_layers=list(actual.layers), applied_source=source,
                  layer_control_degenerate=control == "random_layer" and actual.layers == route.layers)
    if factual:
        result["fact"] = engine.decode(edited, prompt["factual"])
    if route.family == "message":
        diffs = [float((edited["states"][(i, source)] - target["states"][(i, source)]).abs().max())
                 for i in range(engine.depth)]
        result["max_source_residual_change"] = max(diffs)
        result["source_residual_change_by_layer"] = diffs
        result["max_source_relative_l2_change"] = max(
            float(torch.linalg.vector_norm((edited["states"][(i, source)]
                                             - target["states"][(i, source)]).float())
                  / torch.linalg.vector_norm(target["states"][(i, source)].float()).clamp_min(1e-8))
            for i in range(engine.depth))
        result["protected_prefix_numerical_drift"] = edited["guard_drift"]
        assert max(diffs) == 0, "protected source changed despite recipient identity guard"
    return result


def run(args):
    import torch
    import transformers
    spec = ProbeSpec.load(f"configs/probes/{args.model}.yaml")
    spec.attn_implementation = "eager"
    outdir = Path(args.out or f"results/suite/{args.model}/causal-crossing")
    outdir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model(spec)
    engine = PrefixEngine(model, tokenizer, spec, device)
    routes = candidates(engine.depth, [h.layer for h in spec.residual_hooks])
    prompts = [make_prompt(tokenizer, args.model, case) for case in cases(args.model)]
    manifest = dict(model=args.model, scc=SCC[args.model], seed=7, date=DATE,
                    protocol="docs/plans/causal-crossing-stage1-protocol.md", device=device,
                    python=platform.python_version(), torch=torch.__version__,
                    transformers=transformers.__version__, dtype=str(next(model.parameters()).dtype),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    routes=[dict(name=r.name, **asdict(r)) for r in routes], prompts=prompts)
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        assert previous == json.loads(json.dumps(manifest)), "run definition changed; use a new output directory"
    else:
        write_json(manifest_path, manifest)
    rows_path = outdir / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    keys = {(r["phase"], r["case"], r["route"], r["control"]) for r in rows}
    start = time.monotonic()

    def emit(phase, prompt, route=None, control="edit", factual=True):
        key = (phase, prompt["id"], route.name if route else "baseline", control)
        if key in keys:
            return
        result = evaluate(engine, prompt, route, control, factual)
        row = dict(phase=phase, case=prompt["id"], n=prompt["n"], j=prompt["j"],
                   route=key[2], control=control, **result)
        with rows_path.open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
        rows.append(row)
        keys.add(key)

    try:
        cal = [p for p in prompts if p["split"] == "calibration"]
        test = [p for p in prompts if p["split"] == "heldout"]
        if args.smoke:
            p = cal[0]
            emit("smoke", p)
            for route in [routes[0], Route("message", tuple(range(engine.depth)))]:
                for control in ["edit", "sham"]:
                    emit("smoke", p, route, control)
            # Compare identical question as a fresh pass and as a cache branch.
            fresh = engine.decode(dict(cache=None), p["prefix"] + p["suffix"])
            default_impl = "sdpa" if model._supports_sdpa else "eager"
            model.set_attn_implementation(default_impl)
            native = engine.decode(dict(cache=None), p["prefix"] + p["suffix"])
            model.set_attn_implementation("eager")
            write_json(outdir / "smoke_validation.json", dict(fresh=fresh, default=native,
                       comparison_implementation=default_impl))
            print(json.dumps(rows[-5:], indent=2), flush=True)
            return
        for p in cal:
            emit("calibration_baseline", p)
        for ri, route in enumerate(routes):
            for p in cal:
                emit("screen", p, route, factual=False)
            rr = [r for r in rows if r["phase"] == "screen" and r["route"] == route.name]
            score = np.mean([r["count"]["parsed"] == r["n"] - 1 for r in rr])
            print(f"screen {ri+1}/{len(routes)} {route.name} N-1={score:.3f} elapsed={time.monotonic()-start:.0f}s", flush=True)
        by_name = {r.name: r for r in routes}

        def screen_score(route):
            rr = [r for r in rows if r["phase"] == "screen" and r["route"] == route.name]
            return np.mean([r["count"]["parsed"] == r["n"] - 1 for r in rr])

        shortlist = []
        for family in ("residual", "message"):
            ranked = sorted([r for r in routes if r.family == family],
                            key=lambda r: (-screen_score(r), len(r.layers), r.layers))
            shortlist.extend(ranked[:3])
        for route in shortlist:
            for p in cal:
                emit("factual_screen", p, route)
        base_fact = np.mean([r["target_fact"]["parsed"] == 1 for r in rows if r["phase"] == "calibration_baseline"])

        def selection_key(route):
            rr = [r for r in rows if r["phase"] == "factual_screen" and r["route"] == route.name]
            count = np.mean([r["count"]["parsed"] == r["n"] - 1 for r in rr])
            fact = np.mean([r["fact"]["parsed"] == 1 for r in rr])
            joint = np.mean([r["fact"]["parsed"] == 1 and r["count"]["parsed"] == r["n"] - 1 for r in rr])
            return (-int(count >= .8 and fact >= base_fact), -joint, -count, len(route.layers), route.layers)

        selected = [min([r for r in shortlist if r.family == f], key=selection_key)
                    for f in ("residual", "message")]
        frozen = dict(selected=[asdict(r) for r in selected],
                      selection_scores={r.name: list(selection_key(r)[:-1]) for r in shortlist})
        # Explicit boundary: no held-out response has been computed before this file.
        write_json(outdir / "frozen.json", frozen)
        print("frozen " + json.dumps(frozen), flush=True)
        engine.pairs.clear()
        for p in test:
            emit("heldout_baseline", p)
            for route in selected:
                for control in ("edit", "sham", "random_source", "random_layer"):
                    emit("heldout", p, route, control)
            print(f"heldout {p['id']} elapsed={time.monotonic()-start:.0f}s", flush=True)
        write_json(outdir / "complete.json", dict(rows=len(rows), elapsed_seconds=time.monotonic()-start,
                                                  selected=[r.name for r in selected]))
        print(f"done {args.model}: {len(rows)} rows", flush=True)
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SCC, required=True)
    parser.add_argument("--out")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
