#!/usr/bin/env python3
"""Enrich harvested traces with fitted CC and correctness; emit agent batches.

Joins each harvested record to its model's fitted capacity (preferring the
reasoning-enabled variant when a hybrid model has several rows in scc.csv),
parses the returned integer, and writes one enriched JSONL per model plus a
batches.json that splits models into agent-sized groups balanced by trace
volume.
"""
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "harvested-traces"
OUT = REPO / "results" / "harvested-traces-enriched"
OUT.mkdir(exist_ok=True)

# --- CC per model slug, preferring reasoning-enabled variants -------------
rows = list(csv.DictReader((REPO / "results/counting-data/scc.csv").open()))
by_model = defaultdict(list)
for r in rows:
    by_model[r["model"]].append(r)

def pick(rs):
    def rank(r):
        v = (r.get("variant") or "").lower()
        return 0 if ("think" in v and "nothink" not in v) or "reason" in v else 1
    rs = sorted(rs, key=rank)
    return rs[0]

cc_of = {}
for m, rs in by_model.items():
    r = pick(rs)
    try:
        cc_of[m] = (float(r["scc"]), r.get("variant") or "")
    except (ValueError, KeyError):
        pass

INT = re.compile(r"-?\d[\d,]*")

def last_int(text):
    if not text:
        return None
    hits = INT.findall(text)
    if not hits:
        return None
    try:
        return int(hits[-1].replace(",", ""))
    except ValueError:
        return None

stats = defaultdict(int)
sizes = {}
for f in sorted(glob.glob(str(SRC / "*.jsonl"))):
    model = Path(f).stem.replace("__", "/")
    cc, variant = cc_of.get(model, (None, ""))
    out_records = []
    for line in open(f):
        r = json.loads(line)
        L = r.get("sequence_length")
        if not r.get("counting_task") or not (r.get("reasoning") or "").strip() or not L:
            stats["skipped"] += 1
            continue
        ans = last_int(r.get("completion") or "")
        rel = None
        if cc:
            ratio = L / cc
            rel = "below" if ratio < 0.8 else ("near" if ratio <= 1.2 else "above")
        out_records.append({
            "model": model,
            "generation_id": r["generation_id"],
            "sequence_length": L,
            "model_cc": cc,
            "relative_to_cc": rel,
            "answer_returned": ans,
            "correct": (ans == L),
            "tokens_reasoning": r.get("tokens_reasoning"),
            "finish": r.get("finish"),
            "reasoning": r.get("reasoning"),
            "completion": (r.get("completion") or "")[:400],
        })
        stats["kept"] += 1
    if out_records:
        path = OUT / (Path(f).stem + ".jsonl")
        with path.open("w") as sink:
            for rec in out_records:
                sink.write(json.dumps(rec) + "\n")
        sizes[str(path)] = len(out_records)

# --- batches balanced by trace count --------------------------------------
items = sorted(sizes.items(), key=lambda kv: -kv[1])
n_batches = min(13, max(1, math.ceil(len(items) / 10)))
batches = [{"files": [], "n": 0} for _ in range(n_batches)]
for path, n in items:
    b = min(batches, key=lambda b: b["n"])
    b["files"].append(path)
    b["n"] += n
Path(OUT / "batches.json").write_text(json.dumps(batches, indent=1))
print(f"kept {stats['kept']} traces, skipped {stats['skipped']}, "
      f"{len(sizes)} models, {n_batches} batches "
      f"(sizes {[b['n'] for b in batches]})")
no_cc = [m for m in (Path(f).stem.replace('__','/') for f in sizes) if m not in cc_of]
print("models without CC join:", no_cc[:10] if no_cc else "none")
