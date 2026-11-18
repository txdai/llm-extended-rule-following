#!/usr/bin/env python3
"""Run all 11 character/delimiter motifs locally across the open-weight suite models ($0 API cost)."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from counting.analysis.capacity import fit_transition, sufficiency
from counting.probes.common import ProbeSpec, load_model
from counting.probes.common.prompts import parse_integer, render_prompts
from counting.probes.suite.scan import generate_one

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "rebuttal-motif-local"

MOTIFS = [
    ("baseline-a-comma",    "a",  ", "),
    ("letter-b-comma",      "b",  ", "),
    ("letter-x-comma",      "x",  ", "),
    ("double-aa-comma",     "aa", ", "),
    ("greek-alpha-comma",   "α",  ", "),
    ("greek-beta-comma",    "β",  ", "),
    ("chinese-han-comma",   "汉", ", "),
    ("chinese-zhong-comma", "中", ", "),
    ("letter-a-space",      "a",  " "),
    ("letter-a-pipe",       "a",  " | "),
    ("letter-a-section",    "a",  " § "),
]

MODELS = [
    ("gpt-oss-20b",            "configs/probes/gpt-oss-20b.yaml",            "<|channel|>final<|message|>", {}, 160),
    ("gemma-4-26b-a4b-it",     "configs/probes/gemma-4-26b-a4b-it.yaml",     "", {}, 180),
    ("qwen3.6-35b-a3b",        "configs/probes/qwen3.6-35b-a3b.yaml",        "", {"enable_thinking": False}, 140),
    ("muse-glimmer-30b",       "configs/probes/muse-glimmer-30b.yaml",       " to=user<|message|>", {}, 150),
    ("llama-3.3-70b-instruct", "configs/probes/llama-3.3-70b-instruct.yaml", "", {}, 110),
    ("gpt-oss-120b",           "configs/probes/gpt-oss-120b.yaml",           "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>", {}, 130),
]


def main() -> int:
    import torch
    tok_unwrap = lambda tok: getattr(tok, "tokenizer", tok)

    for mname, cfg_rel, prefill, kw, ceiling in MODELS:
        print(f"=== Loading {mname} for 11-motif sweep ===", flush=True)
        spec = ProbeSpec.load(REPO / cfg_rel)
        model, tokenizer, device = load_model(spec)
        inner_tok = tok_unwrap(tokenizer)
        try:
            for motif_id, item, delim in MOTIFS:
                out_dir = OUT / mname / motif_id
                out_dir.mkdir(parents=True, exist_ok=True)
                rec_path = out_dir / "records.jsonl"
                records = [json.loads(l) for l in rec_path.read_text().splitlines() if l] if rec_path.exists() else []
                done = {r["length"] for r in records}
                t0 = time.time()

                def cur_fit():
                    ordered = sorted(records, key=lambda r: r["length"])
                    ls = [r["length"] for r in ordered]
                    ok = [bool(r["correct"]) for r in ordered]
                    er = [r["parsed"] is None for r in ordered]
                    tr = fit_transition(ls, ok, er, calibrate=False)
                    above = [r for r in ordered if r["length"] > tr.scc]
                    return tr, sufficiency(tr, len(above), sum(1 for r in above if r["correct"])), len(above)

                any_mistake = any(not r["correct"] for r in records)
                with rec_path.open("a") as fh:
                    for length in range(1, ceiling + 1):
                        if length in done:
                            continue
                        prompt = render_prompts(tokenizer, [length], item, delim, kw or None)[0] + prefill
                        n_in_tok = len(inner_tok.encode(prompt, add_special_tokens=False))
                        reply = generate_one(model, tokenizer, prompt, device, 12)
                        parsed = parse_integer(reply)
                        rec = {
                            "length": length,
                            "parsed": parsed,
                            "correct": parsed == length,
                            "input_tokens": n_in_tok,
                            "reply": reply[:200],
                        }
                        records.append(rec)
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        if not rec["correct"]:
                            any_mistake = True
                        if any_mistake and (length % 4 == 0) and len(records) >= 20:
                            tr, st, n_above = cur_fit()
                            if st == "ok" and n_above >= 14:
                                break
                tr, st, _ = cur_fit()
                # Fit full calibrated interval at end
                ordered = sorted(records, key=lambda r: r["length"])
                tr_cal = fit_transition(
                    [r["length"] for r in ordered],
                    [bool(r["correct"]) for r in ordered],
                    [r["parsed"] is None for r in ordered],
                )
                summary = {
                    "model": mname, "motif": motif_id, "item": item, "delimiter": delim,
                    "n_probes": len(ordered), "sufficiency": st,
                    "transition": asdict(tr_cal), "elapsed_s": round(time.time() - t0, 1),
                }
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
                print(f"[MotifLocal] {mname:24s} | {motif_id:20s} | CC={tr_cal.scc:3d} [{tr_cal.ci_low:3d},{tr_cal.ci_high:3d}] ({summary["elapsed_s"]}s)", flush=True)
        finally:
            del model, tokenizer
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    print("=== All Local 11-Motif Sweeps Complete ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
