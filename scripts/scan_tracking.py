"""State-tracking capacity scan (docs/plans/state-tracking-scan.md).

Object tracking as a CC-style scan: four named objects in four named locations,
a history of T swaps, then the location of an object the last swap moved, answered
as one integer label among four (labels randomized per probe). The schedule is
coarse to fine: one probe at T = 100, the step doubling while the reply is exact
and halving when it is not, until the bracket between the last exact and the
first inexact length is tight; then a short dense scan across the bracket, one
probe per length, ascending, with the catalogue's stop rule (the two-regime fit
declares a transition whose broken regime is established: Wilson upper bound of
the accuracy above it below one half). Every probe enters the record; TC is the
fitted changepoint with the calibrated interval.

Side questions at 0.5x, 1.0x, and 1.5x the fitted boundary, five probes each:
the last swap's text (event access) and the location question with the correct
current state written into the prompt (supplied state).

Every model sees the same histories (seeds depend on length and probe only).
Decoding follows the catalogue: temperature 0 unless the model locks it, the
roster's reasoning_effort ("none" switches thinking off), 8,192 output tokens.

    OPENROUTER_API_KEY=... python scripts/scan_tracking.py --subset results/tracking_subset.json
        [--workers 12] [--only gpt-oss] [--smoke] [--dry-run]

Records: results/tracking/<model>#<variant>.jsonl (one line per probe) and
<...>.summary.json; a run resumes from its jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from counting.analysis.capacity import fit_transition, sufficiency  # noqa: E402
from counting.metrics import parse_integer  # noqa: E402
from counting.providers.openrouter_api import OpenRouterProvider  # noqa: E402
from counting.types import Stimulus  # noqa: E402

LOCATIONS = ("A", "B", "C", "D")
OBJECTS = ("red", "blue", "green", "yellow")
SYSTEM_TEXT = ("You track objects exactly. Return only one integer with no words, punctuation, "
               "or explanation.")
START, MAX_LENGTH = 100, 1600
DENSE_POINTS = 24
SIDE_PROBES = 5
BATCH = 4


# -- task generation ------------------------------------------------------------

def rng_for(length: int, kind: str, index: int) -> np.random.Generator:
    return np.random.default_rng([7, length, {"scan": 0, "event": 1, "supplied": 2}[kind], index])


def execute(initial: dict, operations: list) -> dict:
    state = dict(initial)
    for a, b in operations:
        state[a], state[b] = state[b], state[a]
    return state


def where(state: dict, obj: str) -> str:
    return next(loc for loc, o in state.items() if o == obj)


def make_history(rng: np.random.Generator, length: int) -> dict:
    perm = rng.permutation(len(OBJECTS))
    initial = {loc: OBJECTS[int(perm[i])] for i, loc in enumerate(LOCATIONS)}
    operations = [tuple(map(str, rng.choice(LOCATIONS, 2, replace=False))) for _ in range(length)]
    return dict(initial=initial, operations=operations, final=execute(initial, operations))


def render_history(task: dict) -> str:
    initial = ", ".join(f"{loc}: {task['initial'][loc]}" for loc in LOCATIONS)
    ops = "\n".join(f"{i + 1}. Swap the objects in {a} and {b}." for i, (a, b) in enumerate(task["operations"]))
    return (f"Locations {', '.join(LOCATIONS)} each hold one object. Initially: {initial}.\n"
            f"Operations, in order:\n{ops}")


def labels(rng: np.random.Generator, n: int) -> list[int]:
    return [int(v) for v in rng.choice(np.arange(10, 90), n, replace=False)]


def moved_object(rng: np.random.Generator, task: dict) -> str:
    a, b = task["operations"][-1]
    candidates = [task["final"][a], task["final"][b]]
    moved = [o for o in candidates if where(task["initial"], o) != where(task["final"], o)]
    pool = moved or candidates
    return str(pool[int(rng.integers(len(pool)))])


def location_question(rng: np.random.Generator, obj: str) -> tuple[str, dict]:
    lab = labels(rng, len(LOCATIONS))
    order = rng.permutation(len(LOCATIONS))
    options = {lab[k]: LOCATIONS[int(i)] for k, i in enumerate(order)}
    text = (f"\n\nWhich location holds the {obj} object now?\n"
            + "\n".join(f"{label}: {loc}" for label, loc in options.items())
            + "\nReturn only the integer label of that location.")
    return text, options


def score(text: str | None, probe: dict) -> tuple[int | None, bool, str]:
    """Parsed answer, correctness, and how it was read: an integer label, or a bare
    location letter (the last one in the reply), which some models return despite
    the instruction; a letter counts when it names the expected location."""
    parsed = parse_integer(text)
    if parsed is not None:
        return parsed, parsed == probe["expected"], "label"
    # a bare letter reply only: the whole reply, or its last line, is one location letter
    lines = [ln.strip().rstrip(".") for ln in (text or "").strip().splitlines() if ln.strip()]
    tail = lines[-1] if lines else ""
    if tail in LOCATIONS and probe.get("options") and (len(lines) == 1 or len(tail) == 1):
        label = next((k for k, v in probe["options"].items() if v == tail), None)
        return label, label == probe["expected"], "letter"
    return None, False, "none"


def build_probe(length: int, kind: str, index: int) -> dict:
    rng = rng_for(length, kind, index)
    task = make_history(rng, length)
    history = render_history(task)
    if kind in ("scan", "supplied"):
        obj = moved_object(rng, task)
        text, options = location_question(rng, obj)
        expected = next(label for label, loc in options.items() if loc == where(task["final"], obj))
        if kind == "supplied":
            # the final state alone, at the same length of history, so that the
            # decision is tested without any tracking and re-tracing is impossible
            state = ", ".join(f"{loc}: {task['final'][loc]}" for loc in LOCATIONS)
            user = f"Locations {', '.join(LOCATIONS)} each hold one object. The current state is: {state}." + text
        else:
            user = history + text
        return dict(length=length, kind=kind, index=index, expected=int(expected), user=user,
                    options={int(k): v for k, v in options.items()})
    # event: which two locations were named in the last operation
    a, b = task["operations"][-1]
    while True:
        c, d = map(str, rng.choice(LOCATIONS, 2, replace=False))
        if {c, d} != {a, b}:
            break
    lab = labels(rng, 2)
    pairs = [(a, b), (c, d)]
    if rng.random() < 0.5:
        pairs = pairs[::-1]
    options = {lab[i]: p for i, p in enumerate(pairs)}
    expected = next(label for label, pair in options.items() if set(pair) == {a, b})
    text = (f"\n\nWhich two locations were named in operation {length}?\n"
            + "\n".join(f"{label}: {x} and {y}" for label, (x, y) in options.items())
            + "\nReturn only the integer label of that pair.")
    return dict(length=length, kind="event", index=index, expected=int(expected), user=history + text)


# -- one model ---------------------------------------------------------------------

class Scan:
    def __init__(self, entry: dict, provider: OpenRouterProvider, out: Path, max_output_tokens: int, smoke: bool):
        self.entry = entry
        self.model = entry["id"]
        self.variant = entry.get("variant", "")
        self.key = self.model + (f"#{self.variant}" if self.variant else "")
        self.effort = entry.get("reasoning_effort")
        self.locked = bool(entry.get("temperature_locked", False))
        self.provider = provider
        self.max_output_tokens = max_output_tokens
        self.smoke = smoke
        safe = self.key.replace("/", "__")
        self.jsonl = out / f"{safe}.jsonl"
        self.summary_path = out / f"{safe}.summary.json"
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        if self.jsonl.exists():
            self.rows = [json.loads(line) for line in self.jsonl.read_text().splitlines() if line.strip()]
        self.tokens = {"input": 0, "output": 0, "reasoning": 0}

    def ask(self, probe: dict) -> dict:
        stim = Stimulus(target=probe["expected"], system=SYSTEM_TEXT, user=probe["user"])
        t0 = time.monotonic()
        resp = self.provider.complete(stim, self.model, self.max_output_tokens,
                                      reasoning_effort=self.effort, temperature_locked=self.locked)
        parsed, ok, how = score(resp.text, probe) if resp.status != "error" else (None, False, "none")
        row = dict(key=self.key, length=probe["length"], kind=probe["kind"], index=probe["index"],
                   expected=probe["expected"], reply=(resp.text or "")[-400:], parsed=parsed, read_as=how,
                   correct=ok, status=resp.status, detail=resp.detail,
                   input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                   reasoning_tokens=resp.usage.reasoning_tokens, seconds=round(time.monotonic() - t0, 1))
        with self.lock:
            self.rows.append(row)
            with self.jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.tokens["input"] += resp.usage.input_tokens or 0
            self.tokens["output"] += resp.usage.output_tokens or 0
            self.tokens["reasoning"] += resp.usage.reasoning_tokens or 0
        return row

    def scan_rows(self) -> list[dict]:
        return sorted((r for r in self.rows if r["kind"] == "scan"), key=lambda r: r["length"])

    def fit(self, calibrate: bool):
        rows = self.scan_rows()
        lengths = [r["length"] for r in rows]
        correct = [bool(r["correct"]) for r in rows]
        error = [(not r["correct"]) and (r["parsed"] is None) for r in rows]
        t = fit_transition(lengths, correct, error, calibrate=calibrate)
        above = [r for r in rows if r["length"] > t.scc]
        s = sufficiency(t, len(above), sum(1 for r in above if r["correct"]))
        return t, s

    def probe_scan(self, length: int) -> dict:
        prior = [r for r in self.rows if r["kind"] == "scan" and r["length"] == length]
        return prior[0] if prior else self.ask(build_probe(length, "scan", 0))

    def coarse_point(self, length: int) -> dict:
        """Two probes at a coarse length; exact only if both are exact."""
        rows = [r for r in self.rows if r["kind"] == "scan" and r["length"] == length]
        need = [i for i in (0, 1) if not any(r["index"] == i for r in rows)]
        if need:
            with ThreadPoolExecutor(max_workers=len(need)) as pool:
                rows += list(pool.map(lambda i: self.ask(build_probe(length, "scan", i)), need))
        return dict(correct=all(r["correct"] for r in rows), status="error" if all(r["status"] == "error" for r in rows) else "ok")

    def bracket(self) -> tuple[int, int | None, int]:
        """Coarse phase: last exact length, first inexact length, error count."""
        lo, hi, step, errors = 0, None, START, 0
        length = START
        while True:
            row = self.coarse_point(length)
            errors += int(row["status"] == "error")
            if errors >= 4:
                break
            if row["correct"]:
                lo = length
                if hi is not None and hi - lo <= max(2, lo // 10):
                    break
                step = step * 2 if hi is None else max(1, step // 2)
                length = min(lo + step, MAX_LENGTH) if hi is None else lo + max(1, (hi - lo) // 2)
                if hi is None and lo >= MAX_LENGTH:
                    break
            else:
                hi = length
                if hi - lo <= max(2, lo // 10):
                    break
                step = max(1, step // 2)
                length = lo + max(1, (hi - lo) // 2)
            if length == lo or (hi is not None and length >= hi):
                break
        return lo, hi, errors

    def run(self) -> dict:
        if self.summary_path.exists():
            return json.loads(self.summary_path.read_text())
        lo, hi, errors = self.bracket()
        # dense phase across the bracket, widened so the fit sees both regimes
        start = max(1, round(0.75 * lo)) if lo else 1
        end = min(MAX_LENGTH, round(1.5 * hi) if hi is not None else max(2 * lo, lo + 8))
        if self.smoke:
            end = min(end, start + 8)
        step = max(1, round((end - start) / DENSE_POINTS))
        done = {r["length"] for r in self.rows if r["kind"] == "scan"}
        pending = [L for L in range(start, end + 1, step) if L not in done]
        last, first = end, start
        while errors < 6:
            if not pending:
                t, s = self.fit(calibrate=False)
                if s in ("censored", "tail_unresolved") and last < MAX_LENGTH:
                    # still exact at the top of the range, or too few probes above the
                    # transition to establish the broken regime: keep climbing, skipping
                    # lengths a previous pass already probed
                    pending = []
                    while not pending and last < MAX_LENGTH:
                        top = min(MAX_LENGTH, last + step * (DENSE_POINTS // 2))
                        pending = [L for L in range(last + step, top + 1, step) if L not in done]
                        last = top
                elif s == "never_stable" and first > 1:
                    # no stable regime seen: the coarse bracket sat above the boundary; go down
                    new_first = max(1, first // 2)
                    dstep = max(1, round((first - new_first) / (DENSE_POINTS // 2)))
                    pending = [L for L in range(new_first, first, dstep) if L not in done]
                    first = new_first
                else:
                    break
                if not pending:
                    break
            batch, pending = pending[:BATCH], pending[BATCH:]
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                results = list(pool.map(lambda L: self.ask(build_probe(L, "scan", 0)), batch))
            done.update(batch)
            errors += sum(1 for r in results if r["status"] == "error")
            t, s = self.fit(calibrate=False)
            if s == "ok":
                break
        t, s = self.fit(calibrate=True)
        tc = int(t.scc)
        # side questions around the boundary
        sides = sorted({max(1, round(0.5 * tc)), max(1, tc), min(MAX_LENGTH, max(1, round(1.5 * tc)))})
        if t.censored:
            sides = sorted({max(1, round(0.5 * tc)), max(1, tc)})
        side_done = {(r["kind"], r["length"], r["index"]) for r in self.rows if r["kind"] != "scan"}
        probes = [build_probe(L, kind, i) for L in sides for kind in ("event", "supplied") for i in range(SIDE_PROBES)
                  if (kind, L, i) not in side_done]
        if probes:
            with ThreadPoolExecutor(max_workers=min(10, len(probes))) as pool:
                list(pool.map(self.ask, probes))
        side = {}
        for kind in ("event", "supplied"):
            for L in sides:
                rows = [r for r in self.rows if r["kind"] == kind and r["length"] == L]
                side[f"{kind}@{L}"] = dict(n=len(rows), correct=sum(1 for r in rows if r["correct"]))
        summary = dict(key=self.key, model=self.model, variant=self.variant, reasoning_effort=self.effort,
                       tc=tc, ci=[int(t.ci_low), int(t.ci_high)], p_stable=t.p_stable, p_after=t.p_after,
                       censored=t.censored, bound_by=t.bound_by, sufficiency=s, n_probes=t.n_probes,
                       errors=errors, bracket=[lo, hi], side=side, side_lengths=sides, tokens=self.tokens,
                       scan=[(r["length"], bool(r["correct"]), r["parsed"]) for r in self.scan_rows()])
        self.summary_path.write_text(json.dumps(summary, indent=1))
        return summary


FLOOR = list(range(1, 9))


def floor_sweep(scan: "Scan") -> dict | None:
    """Post-pass: two probes at every length 1..8 for a run that has a summary, then refit.

    The coarse phase starts at 100 and can leave a record with too few short
    histories to anchor the stable regime; this guarantees them for every run.
    """
    if not scan.summary_path.exists():
        return None
    s = json.loads(scan.summary_path.read_text())
    if "failed" in s:
        return s
    have = {(r["length"], r["index"]) for r in scan.rows if r["kind"] == "scan"}
    probes = [build_probe(L, "scan", i) for L in FLOOR for i in (0, 1) if (L, i) not in have]
    if s.get("sufficiency") == "never_stable":
        # no stable regime declared: give the fit four more probes at each of the
        # shortest six histories before concluding that none exists
        probes += [build_probe(L, "scan", i) for L in FLOOR[:6] for i in (2, 3, 4, 5) if (L, i) not in have]
    if probes:
        with ThreadPoolExecutor(max_workers=min(8, len(probes))) as pool:
            list(pool.map(scan.ask, probes))
    t, suff = scan.fit(calibrate=True)
    scan_rows = scan.scan_rows()
    s.update(tc=int(t.scc), ci=[int(t.ci_low), int(t.ci_high)], p_stable=t.p_stable, p_after=t.p_after, censored=t.censored,
             bound_by=t.bound_by, sufficiency=suff, n_probes=t.n_probes, floor_sweep=True,
             scan=[(r["length"], bool(r["correct"]), r["parsed"]) for r in scan_rows])
    scan.summary_path.write_text(json.dumps(s, indent=1))
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", action="store_true", help="post-pass: floor sweep and refit for every run with a summary")
    ap.add_argument("--subset", default=str(ROOT / "results" / "tracking_subset.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tracking"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default=None, help="substring filter on model id")
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--smoke", action="store_true", help="short dense phase")
    ap.add_argument("--dry-run", action="store_true", help="print one prompt and exit")
    args = ap.parse_args()
    if args.dry_run:
        for kind in ("scan", "event", "supplied"):
            p = build_probe(6, kind, 0)
            print(f"--- {kind} (expected {p['expected']})\n{SYSTEM_TEXT}\n\n{p['user']}\n")
        return 0
    entries = json.loads(Path(args.subset).read_text())
    if args.only:
        entries = [e for e in entries if args.only in e["id"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provider = OpenRouterProvider(temperature=0.0)
    scans = [Scan(e, provider, out, args.max_output_tokens, args.smoke) for e in entries]
    print(f"{len(scans)} runs, {args.workers} in parallel", flush=True)
    if args.floor:
        def sweep(scan: Scan):
            s = floor_sweep(scan)
            if s and "failed" not in s:
                print(f"{scan.key:48s} TC={s['tc']:>4} [{s['ci'][0]},{s['ci'][1]}] {s['sufficiency']:15s} probes={s['n_probes']}", flush=True)
            return s
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(sweep, scans))
        print("floor sweep done", flush=True)
        return 0

    def one(scan: Scan) -> dict:
        t0 = time.monotonic()
        try:
            s = scan.run()
            print(f"{scan.key:48s} TC={s['tc']:>3} [{s['ci'][0]},{s['ci'][1]}] {s['sufficiency']:15s} "
                  f"probes={s['n_probes']} side={ {k: f'{v['correct']}/{v['n']}' for k, v in s['side'].items()} } "
                  f"tokens={s['tokens']} {round(time.monotonic() - t0)}s", flush=True)
            return s
        except Exception as exc:  # noqa: BLE001
            print(f"{scan.key:48s} FAILED: {exc}", flush=True)
            return dict(key=scan.key, failed=str(exc))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        summaries = list(pool.map(one, scans))
    tot = {k: sum(s.get("tokens", {}).get(k, 0) for s in summaries) for k in ("input", "output", "reasoning")}
    print("total tokens:", tot)
    (out / "_all_summaries.json").write_text(json.dumps(summaries, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
