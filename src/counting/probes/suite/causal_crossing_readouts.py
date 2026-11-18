"""Readout change at the natural target/neutral item, using existing E4 fits.

This is a descriptive route check, never an intervention-selection objective.
The historical all-a running-count probe is confounded with item position.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..common.model import ActivationRecorder, load_model
from ..common.spec import ProbeSpec
from .causal_crossing import SCC, PrefixEngine, write_json
from .util import correctness_by_count


def fit_historical(model):
    directory = Path("results/suite") / model
    fits = json.loads((directory / "round2/fit.json").read_text())
    scan = directory / ("scan-final" if model == "gpt-oss-20b" else "scan-nothink")
    correct = correctness_by_count(scan)
    cache_path = directory / "round2/positions.npz"
    cache = np.load(cache_path)
    running = cache["running_count"]
    mask = np.array([n <= SCC[model] and correct.get(int(n), False) for n in running])
    assert int(mask.sum()) == fits["n_stable"], "historical training mask does not match E4"
    weights = {}
    for site, fit in fits["sites"].items():
        x, y = cache[site][mask].astype(float), running[mask].astype(float)
        center = x.mean(axis=0)
        xc = x - center
        kernel = xc @ xc.T
        scale = np.trace(kernel) / len(x)
        w = xc.T @ np.linalg.solve(kernel + fit["lambda"] * scale * np.eye(len(x)), y - y.mean())
        weights[site] = dict(weights=w.tolist(), intercept=float(y.mean() - center @ w),
                             historical_lambda=fit["lambda"], historical_loo_rmse=fit["loo_rmse_items"])
    return dict(model=model, source=str(cache_path), source_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                n_train=int(mask.sum()), scan_dir=str(scan), sites=weights)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=SCC)
    args = parser.parse_args()
    directory = Path("results/suite") / args.model / "causal-crossing"
    manifest = json.loads((directory / "manifest.json").read_text())
    weights_path = directory / "historical_readout_weights.json"
    # Saving the small weights makes evaluation reproducible without the old cache.
    weights = json.loads(weights_path.read_text()) if weights_path.exists() else fit_historical(args.model)
    write_json(weights_path, weights)
    spec = ProbeSpec.load(f"configs/probes/{args.model}.yaml")
    spec.attn_implementation = "eager"
    model, tokenizer, device = load_model(spec)
    engine = PrefixEngine(model, tokenizer, spec, device)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    rows, decoys = [], []
    for prompt in manifest["prompts"]:
        predictions = {}
        for kind in ("prefix", "donor"):
            with ActivationRecorder(model, spec.residual_hooks) as recorder, torch.inference_mode():
                output = model(input_ids=torch.tensor([prompt[kind]], device=device), use_cache=True, logits_to_keep=1)
                predictions[kind] = {}
                for site, fit in weights["sites"].items():
                    state = recorder.cache[site][0, prompt["source"]].float().cpu().numpy()
                    predictions[kind][site] = float(state @ np.array(fit["weights"]) + fit["intercept"])
        # A neutral prefix contains x at j but a at this other item: detecting any
        # x in the sequence is not enough to answer this supplementary query.
        other = prompt["positions"].index(prompt["random_source"]) + 1
        fact = (f"\n\nInstead of counting, identify the item at position {other} "
                "in the sequence above (positions start at 1). "
                "Return only 1 if that item is a, or 2 if that item is x.")
        suffix = tok.encode(fact, add_special_tokens=False) + prompt["suffix"]
        response = engine.decode(dict(cache=output.past_key_values), suffix)
        decoys.append(dict(case=prompt["id"], split=prompt["split"], n=prompt["n"], j=prompt["j"],
                           query_j=other, expected=1, **response))
        for site in weights["sites"]:
            rows.append(dict(case=prompt["id"], split=prompt["split"], n=prompt["n"], j=prompt["j"], site=site,
                             target_readout=predictions["prefix"][site], neutral_readout=predictions["donor"][site],
                             neutral_minus_target=predictions["donor"][site] - predictions["prefix"][site]))
        print(prompt["id"], flush=True)
    write_json(directory / "ordinary_readout.json", rows)
    write_json(directory / "factual_decoy.json", decoys)
    engine.close()


if __name__ == "__main__":
    main()
