"""Logit-level interpolation along the donor chord (round-2 E2).

An abrupt switch of the decoded answer under interpolation is what argmax
does to any continuously varying logits. The number-line question is whether
the answer-token logits of the integers strictly between base and donor ever
lead at intermediate alpha. For every donor pair of the suite at the last
hooked site, the answer-token logits at alpha in {0, 0.05, ..., 1}:

  base_logit, donor_logit, max_intermediate_logit, intermediate softmax mass
  (over the answer tokens) per alpha.

For digit-splitting tokenizers the first answer token is a digit, so
"intermediate" means the first digits strictly between the base's and the
donor's, and pairs whose leading digits are adjacent have no intermediate
token.

    python -m counting.probes.suite.logit_interp configs/probes/gpt-oss-20b.yaml \
        --scan-dir ... --collect-dir ... --interventions-dir ... --subspace-dir ... \
        --output-dir results/suite/gpt-oss-20b/round2 --prefill '<|channel|>final<|message|>'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..common import ProbeSpec, load_model
from ..common.prompts import render_prompts
from .editing import DecodeEditor, set_state
from .util import encode_text, scanned_scc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--collect-dir", type=Path, required=True)
    parser.add_argument("--interventions-dir", type=Path, required=True)
    parser.add_argument("--subspace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill", default="")
    parser.add_argument("--chat-kwargs", default="")
    parser.add_argument("--item", default="a")
    parser.add_argument("--delimiter", default=", ")
    parser.add_argument("--alphas", default=",".join(f"{a / 20:.2f}" for a in range(21)))
    parser.add_argument("--site", default="", help="hook label; default the last hooked site")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scc = scanned_scc(args.scan_dir)
    iv_meta = json.loads((args.interventions_dir / "meta.json").read_text())
    base_n, donors = int(iv_meta["base_count"]), [int(d) for d in iv_meta["donor_counts"]]
    unembed = np.load(args.subspace_dir / "unembed.npz")
    token_ids = unembed["token_ids"].astype(int)
    first_token = unembed["first_token_by_count"].astype(int)
    digit_split = bool(json.loads((args.subspace_dir / "unembed.json").read_text()).get("digit_split"))
    cache = np.load(args.collect_dir / "activations.npz")
    counts = cache["counts"].astype(int)
    index_of = {int(n): i for i, n in enumerate(counts)}

    spec = ProbeSpec.load(args.config)
    model, tokenizer, device = load_model(spec)
    editor = DecodeEditor(model, tokenizer, device)
    template_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    hook = spec.hook(args.site) if args.site else spec.hooks[-1]
    states = cache[f"{hook.label}|decode_point"].astype(np.float32)
    alphas = [float(a) for a in args.alphas.split(",")]

    prompt = render_prompts(tokenizer, [base_n], args.item, args.delimiter, template_kwargs)[0]
    base_ids = encode_text(tokenizer, prompt + args.prefill)
    base_state = states[index_of[base_n]]
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    tok_strings = {int(t): tok.decode([int(t)]) for t in token_ids}
    started = time.time()

    def token_of(n: int) -> int:
        return int(first_token[n])

    rows = []
    for d in donors:
        lo, hi = sorted((base_n, d))
        if digit_split:
            lo_d, hi_d = sorted((int(str(base_n)[0]), int(str(d)[0])))
            inter_tokens = sorted({int(encode_text(tokenizer, str(x))[0]) for x in range(lo_d + 1, hi_d)})
        else:
            inter_tokens = sorted({token_of(x) for x in range(lo + 1, hi)})
        delta = states[index_of[d]] - base_state
        for alpha in alphas:
            logits = editor.logits(base_ids, hook, set_state(base_state + alpha * delta))
            answer_logits = logits[token_ids]
            probs = np.exp(answer_logits - answer_logits.max())
            probs /= probs.sum()
            pos = {int(t): i for i, t in enumerate(token_ids)}
            base_l = float(logits[token_of(base_n)])
            donor_l = float(logits[token_of(d)])
            inter_l = float(max(logits[t] for t in inter_tokens)) if inter_tokens else None
            inter_mass = float(sum(probs[pos[t]] for t in inter_tokens if t in pos)) if inter_tokens else 0.0
            top = int(np.argmax(logits))
            rows.append({
                "site": hook.label, "base": base_n, "donor": d, "alpha": alpha,
                "base_logit": round(base_l, 4), "donor_logit": round(donor_l, 4),
                "max_intermediate_logit": round(inter_l, 4) if inter_l is not None else None,
                "n_intermediate_tokens": len(inter_tokens),
                "intermediate_mass": round(inter_mass, 6),
                "top_token": top, "top_is_base": top == token_of(base_n), "top_is_donor": top == token_of(d),
                "top_is_intermediate": top in set(inter_tokens),
                "top_string": tok_strings.get(top),
                "answer_logits": [round(float(v), 3) for v in answer_logits],
            })
        print(f"  donor {d} done  [{time.time() - started:.0f}s]", flush=True)

    out = args.output_dir / "logit_interp.jsonl"
    with out.open("w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    (args.output_dir / "logit_interp_meta.json").write_text(json.dumps({
        "site": hook.label, "base": base_n, "donors": donors, "alphas": alphas, "digit_split": digit_split,
        "scc": scc, "answer_token_ids": [int(t) for t in token_ids],
        "answer_token_strings": [tok_strings[int(t)] for t in token_ids],
        "elapsed_s": round(time.time() - started, 1)}, indent=1))
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
