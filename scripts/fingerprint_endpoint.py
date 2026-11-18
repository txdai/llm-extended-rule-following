"""CC fingerprint of an OpenAI-compatible endpoint. Endpoint from env FINGERPRINT_BASE_URL, key from env FINGERPRINT_API_KEY; nothing is written except <model>.json in the working directory.
usage: python fingerprint.py smoke <model>            # three probes, prints replies
       python fingerprint.py scan <model> [<model>...] # full schedule, writes <model>.json + summary
"""
import os, sys, re, json, time, statistics
from concurrent.futures import ThreadPoolExecutor
import requests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from counting.analysis.capacity import fit_transition, sufficiency
BASE = os.environ.get("FINGERPRINT_BASE_URL", "https://api.openai.com/v1"); KEY = os.environ["FINGERPRINT_API_KEY"]
OUT = os.path.dirname(os.path.abspath(__file__))
SYSTEM = "You count items exactly. Return only one integer with no words,\npunctuation, or explanation."
USER = 'How many occurrences of "a" are in this sequence?\n\n{seq}'
SCHEDULE = list(range(1, 61)) + list(range(62, 201, 2)) + list(range(205, 401, 5)) + list(range(410, 801, 10))

def probe(model, n, tries=3):
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(seq=", ".join(["a"] * n))}], "temperature": 0, "max_tokens": 512}
    last = None
    for t in range(tries):
        t0 = time.time()
        try:
            r = requests.post(f"{BASE}/chat/completions", headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=180)
            lat = time.time() - t0
            if r.status_code == 400 and "temperature" in r.text.lower() and "temperature" in body:
                body.pop("temperature"); continue
            if r.status_code != 200:
                last = f"HTTP {r.status_code}: {r.text[:160]}"; time.sleep(2 + 3 * t); continue
            d = r.json(); c = d["choices"][0]["message"].get("content") or ""
            ints = re.findall(r"-?\d+", c); parsed = int(ints[-1]) if ints else None
            return dict(n=n, parsed=parsed, exact=(parsed == n), error=False, reply=c[:120], served_model=d.get("model"), usage=d.get("usage"), latency=round(lat, 2), finish=d["choices"][0].get("finish_reason"))
        except Exception as e:
            last = repr(e)[:160]; time.sleep(2 + 3 * t)
    return dict(n=n, parsed=None, exact=False, error=True, reply=last, served_model=None, usage=None, latency=None, finish=None)

def scan(model):
    with ThreadPoolExecutor(8) as ex: rows = list(ex.map(lambda n: probe(model, n), SCHEDULE))
    rows.sort(key=lambda r: r["n"])
    ok = [r for r in rows if not r["error"]]
    lengths = [r["n"] for r in rows]; correct = [bool(r["exact"]) for r in rows]; error = [bool(r["error"]) for r in rows]
    t = fit_transition(lengths, correct, error=error)
    above = [c for L, c in zip(lengths, correct) if L > t.scc]
    suff = sufficiency(t, len(above), sum(above)) if not t.censored else "censored"
    served = sorted({r["served_model"] for r in ok if r["served_model"]})
    wrong = [(r["n"], r["parsed"]) for r in ok if not r["exact"]][:12]
    lat = statistics.median([r["latency"] for r in ok]) if ok else None
    summary = dict(model=model, probes=len(rows), errors=sum(error), cc=t.scc, ci=[t.ci_low, t.ci_high], censored=t.censored, p_stable=round(t.p_stable, 2), p_after=(round(t.p_after, 2) if t.p_after is not None else None), sufficiency=suff, served_model_field=served, median_latency_s=lat, first_wrong=wrong)
    json.dump(dict(summary=summary, rows=rows), open(f"{OUT}/{model}.json", "w"), indent=1)
    print(json.dumps(summary)); return summary

if __name__ == "__main__":
    cmd, models = sys.argv[1], sys.argv[2:]
    if cmd == "smoke":
        for n in (5, 50, 200): print(json.dumps(probe(models[0], n)))
    else:
        for m in models: scan(m)
