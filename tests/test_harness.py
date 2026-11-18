"""End-to-end checks that need no API access.

A FakeProvider stands in for a real backend: it counts perfectly up to a chosen
capacity and then emits a fixed "attractor" value, which is what real models do
past their boundary. That is enough to exercise the whole loop -- task ->
provider -> metrics -> strategy -> storage -- and to assert that both search
strategies recover the capacity they were given.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from counting.config import deep_merge, load_experiment, load_roster
from counting.metrics import parse_integer, summarize_slice, tier_passed
from counting.providers.base import Provider
from counting.runner import Runner, RunnerSettings
from counting.search import build_grid, build_strategy
from counting.storage import RunPaths, completed_models, write_model_output, write_run_indexes
from counting.tasks import build_task, ladder
from counting.types import RawResponse, TierVerdict, Usage


class FakeProvider(Provider):
    """Counts exactly up to `capacity`, then returns `attractor`."""

    name = "fake"

    def __init__(self, capacity: int = 200, attractor: int = 100, **options):
        super().__init__(**options)
        self.capacity = capacity
        self.attractor = attractor
        self.calls = 0

    def complete(self, stimulus, model, max_output_tokens, reasoning_effort=None, **kw):
        self.calls += 1
        answer = stimulus.target if stimulus.target <= self.capacity else self.attractor
        return RawResponse(
            text=str(answer),
            status="completed",
            usage=Usage(input_tokens=stimulus.target * 2, output_tokens=1, total_tokens=1),
        )


class TruncatingProvider(Provider):
    """Truncates until the budget passes `needs_tokens`, then answers."""

    name = "truncating"

    def __init__(self, needs_tokens: int = 8192, **options):
        super().__init__(**options)
        self.needs_tokens = needs_tokens
        self.budgets: list[int] = []

    def complete(self, stimulus, model, max_output_tokens, reasoning_effort=None, **kw):
        self.budgets.append(max_output_tokens)
        if max_output_tokens < self.needs_tokens:
            return RawResponse(text="", status="incomplete", detail="max_output_tokens")
        return RawResponse(text=str(stimulus.target), status="completed")


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        raise AssertionError(label)


def test_parsing() -> None:
    print("parsing")
    check("last integer wins", parse_integer("let me count: 1, 2, 3 ... answer is 47") == 47)
    check("bare integer", parse_integer("  512 ") == 512)
    check("negative", parse_integer("-12") == -12)
    check("no integer -> None", parse_integer("I cannot count this") is None)
    check("empty -> None", parse_integer("") is None)


def test_tasks() -> None:
    print("tasks")
    rng = random.Random(0)

    homogeneous = build_task("homogeneous_count")
    stim = homogeneous.build(5, rng)
    sequence = stim.user.split("\n\n", 1)[1]
    check("homogeneous sequence length", sequence.count("a") == 5, sequence)
    check("homogeneous sequence form", sequence == "a, a, a, a, a", sequence)
    check("homogeneous target", stim.target == 5)

    motif = build_task("motif_count", item="x", delimiter=" | ")
    stim = motif.build(4, rng)
    check("motif substitution", "x | x | x | x" in stim.user)
    check("motif level", motif.level == 2)

    nested = build_task("nested_key_path")
    stim = nested.build(8, rng)
    records = stim.user.split(" ; ")
    matches = sum(1 for r in records if _is_match(r))
    check("nested match count == target", matches == 8, f"{matches} of {len(records)} records")
    check("nested pads with distractors", len(records) > 8, f"{len(records)} records")
    check("nested is deterministic", nested.build(8, rng).user == stim.user)

    dual = build_task("dual_task_count", condition="plain")
    stim = dual.build(6, rng)
    check("dual-task asks for JSON", '"count"' in stim.user)
    check("dual-task parses JSON", dual.parse('{"count": 6}') == 6)
    check("dual-task parses fenced JSON", dual.parse('```json\n{"count": 41}\n```') == 41)
    check(
        "dual-task ignores secondary count",
        dual.parse('{"count": 6, "secondary_count": 999}') == 6,
    )
    check("dual-task rejects prose", dual.parse("the count is 6") is None)

    check("ladder is ordered by level", [t.level for t in ladder()] == [1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9])


def _is_match(record: str) -> bool:
    """Independent re-implementation of the match rule, for cross-checking.

    Walks the PATH string tracking bracket depth and takes the token sitting at
    the greatest depth, rather than assuming any particular bracket geometry.
    """
    key = record.split("KEY=")[1].split(";")[0].strip()
    path = record.split("PATH=")[1].split(";")[0].strip()

    depth = 0
    best_depth = -1
    best_token = ""
    token = ""
    for char in path:
        if char in "[({<":
            depth += 1
            token = ""
        elif char in "])}>":
            if token and depth > best_depth:
                best_depth, best_token = depth, token
            depth -= 1
            token = ""
        elif char.isalnum():
            token += char
        else:
            token = ""
    return key == best_token


def test_grids() -> None:
    print("search grids")
    check("linear grid", build_grid(start=32, stop=128, step=32) == [32, 64, 96, 128])
    check("explicit points win", build_grid(points=[5, 1, 5, 3]) == [1, 3, 5])
    log_grid = build_grid(start=32, stop=1024, spacing="log", num=6)
    check("log grid endpoints", log_grid[0] == 32 and log_grid[-1] == 1024, str(log_grid))
    check("log grid is ascending", log_grid == sorted(log_grid))


def test_ladder_finds_capacity() -> None:
    print("ladder search")
    provider = FakeProvider(capacity=300, attractor=100)
    task = build_task("homogeneous_count")
    runner = Runner(provider, task, RunnerSettings(samples_per_length=8, parallel_trials_per_batch=8))
    strategy = build_strategy("ladder", initial_length=32, max_length=4096, step_fraction=0.1)
    trials, summary = runner.run_model("fake-model", strategy)

    scc = summary["stable_count"]
    upper = summary["unstable_upper_bound"]
    # Jitter means a tier at L samples up to 1.2L, so the recovered boundary sits
    # below the true capacity. It must still bracket it sensibly.
    check("ladder is bounded", summary["bounded"], f"scc={scc} upper={upper}")
    check("ladder scc below capacity", 0 < scc <= 300, f"scc={scc}")
    check("ladder bracket contains transition", upper is not None and upper > scc)
    check("ladder is cheap", summary["evaluation_count"] <= 20, f"{summary['evaluation_count']} tiers")
    check("trials recorded", len(trials) == summary["trial_count"])
    check("stable token usage computed", summary["stable_token_usage"]["trials"] > 0)


def test_dense_reports_curve() -> None:
    print("dense sweep")
    provider = FakeProvider(capacity=300, attractor=100)
    task = build_task("homogeneous_count")
    runner = Runner(provider, task, RunnerSettings(samples_per_length=8, parallel_trials_per_batch=8))
    strategy = build_strategy("dense", start=32, stop=512, step=32)
    _, summary = runner.run_model("fake-model", strategy)

    check("dense evaluated whole grid", summary["tiers_evaluated"] == summary["grid_size"])
    check("dense reports a curve", len(summary["curve"]) == summary["grid_size"])
    check("dense scc is bounded", 0 < summary["stable_count"] <= 300, str(summary["stable_count"]))
    check("dense found a failure", summary["unstable_upper_bound"] is not None)
    check("curve has both outcomes", {p["passed"] for p in summary["curve"]} == {True, False})


def test_dense_early_stop() -> None:
    print("dense early stop")
    provider = FakeProvider(capacity=100, attractor=50)
    runner = Runner(provider, build_task("homogeneous_count"), RunnerSettings(samples_per_length=4))
    strategy = build_strategy(
        "dense", start=32, stop=2000, step=32, stop_after_consecutive_failures=3
    )
    _, summary = runner.run_model("fake-model", strategy)
    check("stopped early", summary["stopped_early"] == "consecutive_failures")
    check("did not run whole grid", summary["tiers_evaluated"] < summary["grid_size"],
          f"{summary['tiers_evaluated']}/{summary['grid_size']}")


def test_budget_escalation() -> None:
    print("budget escalation")
    provider = TruncatingProvider(needs_tokens=8192)
    runner = Runner(
        provider,
        build_task("homogeneous_count"),
        RunnerSettings(samples_per_length=1, max_output_tokens=1024, retry_max_output_tokens=32768),
    )
    strategy = build_strategy("dense", points=[32])
    trials, _ = runner.run_model("fake-model", strategy)
    check("doubled to the needed budget", provider.budgets[:4] == [1024, 2048, 4096, 8192],
          str(provider.budgets[:4]))
    check("retry count recorded", trials[0].retry_count == 3, str(trials[0].retry_count))
    check("answer eventually parsed", trials[0].parsed == trials[0].expected)

    capped = TruncatingProvider(needs_tokens=10**9)
    runner = Runner(
        capped,
        build_task("homogeneous_count"),
        RunnerSettings(samples_per_length=1, max_output_tokens=4096, retry_max_output_tokens=16384),
    )
    trials, _ = runner.run_model("fake-model", build_strategy("dense", points=[32]))
    check("stops at the ceiling", max(capped.budgets) == 16384, str(capped.budgets))
    check("records retry_limit_reached", "retry_limit_reached" in (trials[0].response_detail or ""))


def test_preflight() -> None:
    print("preflight")
    from counting.providers.base import Provider
    from counting.runner import ProviderUnavailable

    class DeadProvider(Provider):
        name = "dead"

        def __init__(self, **options):
            super().__init__(**options)
            self.calls = 0

        def complete(self, stimulus, model, max_output_tokens, reasoning_effort=None, **kw):
            self.calls += 1
            return RawResponse(status="error", detail="model_not_found")

    dead = DeadProvider()
    runner = Runner(dead, build_task("homogeneous_count"), RunnerSettings(samples_per_length=16))
    try:
        runner.run_model("bogus-model", build_strategy("dense", start=32, stop=2000, step=32))
        check("dead model raises ProviderUnavailable", False)
    except ProviderUnavailable as exc:
        check("dead model raises ProviderUnavailable", "preflight" in str(exc))
    # The whole point: one request, not a 62-tier sweep.
    check("dead model costs exactly one request", dead.calls == 1, f"{dead.calls} calls")

    live = FakeProvider(capacity=300)
    runner = Runner(live, build_task("homogeneous_count"), RunnerSettings(samples_per_length=4))
    trials, summary = runner.run_model("fake-model", build_strategy("dense", points=[32]))
    check("preflight trial is recorded", any(t.phase == "preflight" for t in trials))
    check("preflight is not a search tier",
          all(e["phase"] != "preflight" for e in summary["evaluations"]))
    check("preflight does not shift the SCC", summary["stable_count"] == 32)

    off = FakeProvider(capacity=300)
    runner = Runner(off, build_task("homogeneous_count"),
                    RunnerSettings(samples_per_length=2, preflight_length=0))
    trials, _ = runner.run_model("fake-model", build_strategy("dense", points=[32]))
    check("preflight_length=0 skips it", not any(t.phase == "preflight" for t in trials))


def test_tier_rule() -> None:
    print("tier rule")
    check("unparsed reply fails the tier", not tier_passed({"parse_rate": 0.9, "mape": 0.0}, 0.05))
    check("nMAE above tolerance fails", not tier_passed({"parse_rate": 1.0, "mape": 0.06}, 0.05))
    check("nMAE at tolerance passes", tier_passed({"parse_rate": 1.0, "mape": 0.05}, 0.05))


def _verdict(center: int, passed: bool, index: int = 0) -> TierVerdict:
    return TierVerdict(center, "p", index, center, center, passed, 16, 1.0, 1.0, 0.0, 0.0)


def _probe(center: int, outcome: str, index: int = 0) -> TierVerdict:
    """One scan probe. The scan classifies on parse_rate and exact_match_rate,
    so those must reflect the outcome rather than being fixed at 1.0."""
    parse_rate = 0.0 if outcome == "error" else 1.0
    exact = 1.0 if outcome == "correct" else 0.0
    return TierVerdict(center, "scan", index, center, center, outcome == "correct",
                       1, parse_rate, exact, 0.0, 0.0)


def test_search_edge_cases() -> None:
    """Degenerate searches must terminate and report honestly."""
    print("search edge cases")

    # Fails the very first tier: no lower bound exists to bisect from, so the
    # refinement is skipped and SCC is 0 (reported as "<initial_length").
    s = build_strategy("ladder", initial_length=32)
    s.record(_verdict(s.next_center(), False))
    check("first-tier failure skips refinement", s.next_center() is None)
    r = s.result()
    check("first-tier failure -> scc 0", r["stable_count"] == 0, str(r["stable_count"]))
    check("first-tier failure is bounded at 32", r["unstable_upper_bound"] == 32)

    # Never fails: must stop at max_length and report itself unbounded.
    s = build_strategy("ladder", initial_length=32, max_length=256)
    seen, guard = [], 0
    while (c := s.next_center()) is not None and guard < 50:
        seen.append(c); s.record(_verdict(c, True)); guard += 1
    check("never-fails terminates", guard < 50, f"{seen}")
    check("never-fails is unbounded", not s.result()["bounded"])
    check("never-fails never exceeds max_length", max(seen) <= 256, str(max(seen)))

    # A growth factor of 1 would make the expansion stand still.
    s = build_strategy("ladder", initial_length=32, growth_factor=1.0, max_length=1000)
    guard = 0
    while (c := s.next_center()) is not None and guard < 50:
        s.record(_verdict(c, True)); guard += 1
    check("growth_factor=1.0 terminates", guard < 50, f"{guard} tiers")

    # A pass above the first failure is real data but must not extend the stable
    # regime -- and must not be silently dropped either.
    s = build_strategy("dense", points=[32, 64, 96, 128])
    for i, (c, p) in enumerate([(32, True), (64, False), (96, True), (128, False)]):
        s.next_center(); s.record(_verdict(c, p, i))
    r = s.result()
    check("pass above first failure does not extend scc", r["stable_count"] == 32, str(r["stable_count"]))
    check("first failure is the lowest failure", r["unstable_upper_bound"] == 64)
    check("late passes are reported", r["passes_above_first_failure"] == [96],
          str(r["passes_above_first_failure"]))

    s = build_strategy("dense", points=[32, 64])
    for i, c in enumerate([32, 64]):
        s.next_center(); s.record(_verdict(c, False, i))
    check("all-fail -> scc 0", s.result()["stable_count"] == 0)


def test_task_edge_cases() -> None:
    print("task edge cases")
    rng = random.Random(0)
    check("target=1 builds", build_task("homogeneous_count").build(1, rng).user.endswith("a"))
    nested = build_task("nested_key_path").build(1, rng)
    check("nested target=1 still pads", len(nested.user.split(" ; ")) >= 3,
          str(len(nested.user.split(" ; "))))

    dual = build_task("dual_task_count", condition="plain")
    check("dual-task rejects a bool count", dual.parse('{"count": true}') is None)
    check("dual-task reads a float count", dual.parse('{"count": 12.0}') == 12)
    check("dual-task reads a string count", dual.parse('{"count": "12"}') == 12)
    check("dual-task on malformed json -> None", dual.parse('{"count":') is None)
    try:
        build_task("dual_task_count", condition="benchmark").build(5, rng)
        check("benchmark condition demands a manifest", False)
    except ValueError as exc:
        check("benchmark condition demands a manifest", "manifest" in str(exc))
    try:
        build_task("dual_task_count", condition="nonsense")
        check("unknown condition is rejected", False)
    except ValueError:
        check("unknown condition is rejected", True)


def test_single_provider() -> None:
    print("single provider")
    from counting.providers import PROVIDERS, build_provider, resolve_provider_name
    check("only openrouter is registered", set(PROVIDERS) == {"openrouter"}, str(set(PROVIDERS)))
    check("legacy names alias to openrouter", resolve_provider_name("responses") == "openrouter")
    check("builds", build_provider("openrouter").name == "openrouter")
    try:
        build_provider("nonesuch")
        check("unknown provider rejected", False)
    except KeyError as exc:
        check("unknown provider rejected", "OpenRouter" in str(exc))


def test_fixed_output_budget() -> None:
    """Escalation off: one call at the cap, truncation scored as a failure."""
    print("fixed output budget")
    from counting.providers.base import Provider

    class Truncating(Provider):
        name = "truncating"

        def __init__(self, **options):
            super().__init__(**options)
            self.budgets = []

        def complete(self, stimulus, model, max_output_tokens, reasoning_effort=None, **kw):
            self.budgets.append(max_output_tokens)
            return RawResponse(text="", status="incomplete", detail="max_output_tokens")

    provider = Truncating()
    runner = Runner(
        provider,
        build_task("homogeneous_count"),
        RunnerSettings(samples_per_length=1, max_output_tokens=8192,
                       escalate_output_budget=False, preflight_length=0),
    )
    trials, summary = runner.run_model("m", build_strategy("dense", points=[10]))
    check("exactly one call at the cap", provider.budgets == [8192], str(provider.budgets))
    check("no retries", trials[0].retry_count == 0)
    check("empty reply does not parse", trials[0].parsed is None)
    check("recorded as budget-exhausted",
          "fixed_budget_exhausted" in (trials[0].response_detail or ""))
    check("tier fails", not summary["evaluations"][0]["passed"])

    # Escalation on is still the default and still doubles.
    other = Truncating()
    runner = Runner(
        other,
        build_task("homogeneous_count"),
        RunnerSettings(samples_per_length=1, max_output_tokens=1024,
                       retry_max_output_tokens=4096, preflight_length=0),
    )
    runner.run_model("m", build_strategy("dense", points=[10]))
    check("escalation still works when enabled", other.budgets == [1024, 2048, 4096],
          str(other.budgets))


def test_stop_beyond_first_failure() -> None:
    print("stop beyond first failure")
    s = build_strategy("dense", start=1, stop=500, step=1,
                       stop_beyond_first_failure_ratio=1.0)
    seen, i = [], 0
    while (c := s.next_center()) is not None and i < 600:
        seen.append(c); s.record(_verdict(c, c < 20, i)); i += 1
    r = s.result()
    check("stops at twice the first failure", seen[-1] == 40, f"last={seen[-1]}")
    check("reports why it stopped", r["stopped_early"] == "beyond_first_failure")
    check("first failure recorded", r["unstable_upper_bound"] == 20)
    check("scc is the last pass below it", r["stable_count"] == 19)

    # Consecutive-failure guard fires first when it is the tighter bound.
    s = build_strategy("dense", start=1, stop=500, step=1,
                       stop_after_consecutive_failures=10,
                       stop_beyond_first_failure_ratio=1.0)
    seen, i = [], 0
    while (c := s.next_center()) is not None and i < 600:
        seen.append(c); s.record(_verdict(c, c < 100, i)); i += 1
    check("consecutive guard wins when tighter",
          s.result()["stopped_early"] == "consecutive_failures", str(s.result()["stopped_early"]))
    check("stopped 10 failures after the first", seen[-1] == 109, f"last={seen[-1]}")


def test_stop_floor() -> None:
    """An early isolated slip must not truncate the scan.

    Without a floor, a model that first errs at length 2 stops at 4 and records
    a capacity of 1 -- which the 1..50 probe shows is wrong for 71 of 295
    measurements.
    """
    print("stop floor")

    def run(first_bad: int, **kw) -> list[int]:
        s = build_strategy("scan", start=1, stop=20000, **kw)
        seen, i = [], 0
        while (c := s.next_center()) is not None and i < 3000:
            seen.append(c)
            s.record(_probe(c, "correct" if c != first_bad else "mistake", i))
            i += 1
        return seen, s

    seen, _ = run(2, min_stop_length=100, stop_at_first_mistake_multiple=2.0)
    check("early slip scans to the floor", seen[-1] == 100, str(seen[-1]))
    seen, _ = run(60, min_stop_length=100, stop_at_first_mistake_multiple=2.0)
    check("a later mistake uses the multiple", seen[-1] == 120, str(seen[-1]))
    seen, _ = run(500, min_stop_length=100, stop_at_first_mistake_multiple=2.0)
    check("the multiple governs once above the floor", seen[-1] == 1000, str(seen[-1]))
    seen, _ = run(2, stop_at_first_mistake_multiple=2.0)
    check("without a floor the early slip truncates", seen[-1] == 4, str(seen[-1]))

    # The floor must not delay the error stop: a model returning nothing should
    # still cost ten probes, not a hundred.
    s = build_strategy("scan", start=1, stop=20000, min_stop_length=100,
                       stop_after_consecutive_errors=10)
    seen, i = [], 0
    while (c := s.next_center()) is not None and i < 3000:
        seen.append(c)
        s.record(_probe(c, "error", i))
        i += 1
    check("errors are not floored", seen[-1] == 10, str(seen[-1]))
    check("and report why", s.result()["stopped_early"] == "consecutive_errors")


def test_step_one_grid_is_exact() -> None:
    """With jitter off, tier L must test exactly L -- that is the whole point."""
    print("step-1 exactness")
    seen_targets = []

    class Recording(FakeProvider):
        def complete(self, stimulus, model, max_output_tokens, reasoning_effort=None, **kw):
            seen_targets.append(stimulus.target)
            return super().complete(stimulus, model, max_output_tokens, reasoning_effort, **kw)

    runner = Runner(
        Recording(capacity=10**9),
        build_task("homogeneous_count"),
        RunnerSettings(samples_per_length=1, jitter_fraction=0.0, preflight_length=0),
    )
    runner.run_model("m", build_strategy("dense", start=1, stop=25, step=1))
    check("tests every integer 1..25", seen_targets == list(range(1, 26)), str(seen_targets[:8]))

    grid = build_grid(start=1, stop=10, step=1)
    check("step-1 grid has no gaps", grid == list(range(1, 11)), str(grid))


def test_formal_roster() -> None:
    print("formal roster")
    config = load_experiment("configs/experiments/openrouter_formal.yaml")
    roster = load_roster(config["models"])
    check("roster is large", len(roster) > 200, f"{len(roster)} entries")
    check("every entry is namespaced", all("/" in e["id"] for e in roster))
    efforts = {e.get("reasoning_effort") for e in roster}
    check("efforts are medium/high/none/unset", efforts <= {"medium", "high", "none", None},
          str(efforts))
    check("most reasoning models request medium",
          sum(1 for e in roster if e.get("reasoning_effort") == "medium") > 80)

    # A toggleable model contributes two measurements; they must be distinguishable.
    from counting.storage import run_key
    keys = [run_key(e) for e in roster]
    check("run keys are unique", len(keys) == len(set(keys)),
          f"{len(keys) - len(set(keys))} collisions")
    think = {e["id"] for e in roster if e.get("variant") == "think"}
    nothink = {e["id"] for e in roster if e.get("variant") == "nothink"}
    check("think and nothink come in pairs", think == nothink,
          f"{len(think ^ nothink)} unpaired")
    check("nothink entries disable reasoning",
          all(e.get("reasoning_effort") == "none"
              for e in roster if e.get("variant") == "nothink"))
    # effort "none" is rejected by mandatory-reasoning models, so it must never
    # be paired with one.
    check("no variant on single-entry models",
          all(run_key(e) == e["id"] for e in roster if e.get("variant") is None))
    locked = [e for e in roster if e.get("temperature_locked")]
    check("temperature-locked models are flagged", len(locked) > 20, f"{len(locked)} entries")
    check("the flag is a bool", all(e["temperature_locked"] is True for e in locked))
    settings = RunnerSettings.from_config(config)
    check("budget fixed at 8192", settings.max_output_tokens == 8192)
    check("escalation disabled", settings.escalate_output_budget is False)
    check("strategy is the linear scan", config["search"]["strategy"] == "scan")
    check("starts at 1", config["search"]["start"] == 1)
    check("step of 1", config["search"]["step"] == 1)
    check("10 consecutive errors abort", config["search"]["stop_after_consecutive_errors"] == 10)
    check("2x first mistake abort", config["search"]["stop_at_first_mistake_multiple"] == 2.0)


class ScanModel(Provider):
    """Exact up to `capacity`, then a fixed wrong answer; empty above `dies_at`."""

    name = "scan-mock"

    def __init__(self, capacity: int = 25, dies_at: int | None = None, **options):
        super().__init__(**options)
        self.capacity = capacity
        self.dies_at = dies_at
        self.lengths: list[int] = []

    def complete(self, stimulus, model, max_output_tokens, reasoning_effort=None, **kw):
        n = stimulus.target
        self.lengths.append(n)
        if self.dies_at is not None and n >= self.dies_at:
            return RawResponse(text="", status="incomplete", detail="max_output_tokens")
        return RawResponse(text=str(n if n <= self.capacity else 100), status="completed")


SCAN_SETTINGS = RunnerSettings(
    samples_per_length=1, jitter_fraction=0.0, max_output_tokens=8192,
    escalate_output_budget=False, preflight_length=0,
)


def test_scan_is_dumb_and_exact() -> None:
    """No tiers, no randomness: length n is asked once, in order, from 1."""
    print("linear scan")
    provider = ScanModel(capacity=25)
    runner = Runner(provider, build_task("homogeneous_count"), SCAN_SETTINGS)
    trials, summary = runner.run_scan("m", build_strategy("scan", start=1, stop=100000))

    check("probes 1,2,3,... in order", provider.lengths == list(range(1, 53)),
          f"{provider.lengths[:6]}..{provider.lengths[-2:]}")
    check("one request per length", len(provider.lengths) == len(set(provider.lengths)))
    check("counting capacity is first_mistake - 1", summary["stable_count"] == 25)
    check("first mistake recorded", summary["first_mistake"] == 26)
    check("stops at twice the first mistake",
          summary["highest_length_probed"] == 52 and
          summary["stopped_early"] == "past_first_mistake_multiple")
    check("no errors on a healthy model", summary["error_count"] == 0)
    check("every response kept", len(trials) == summary["lengths_probed"])
    check("curve records each outcome", len(summary["curve"]) == summary["lengths_probed"])


def test_scan_can_issue_bounded_parallel_batches() -> None:
    """Extension scans may batch fixed grid points without changing outcomes."""
    print("linear scan: bounded parallel batches")
    provider = ScanModel(capacity=5)
    settings = RunnerSettings(
        samples_per_length=1, jitter_fraction=0.0, max_output_tokens=8192,
        escalate_output_budget=False, preflight_length=0,
        parallel_scan_probes=4,
    )
    seen = []
    runner = Runner(
        provider, build_task("homogeneous_count"), settings,
        on_trial=lambda trial: seen.append(trial.center_length),
    )
    _, summary = runner.run_scan("m", build_strategy("scan", start=1, stop=100))
    check("batched scan covers each load once", sorted(provider.lengths) == list(range(1, 13)))
    check("parallel checkpoint keeps each response", sorted(seen) == list(range(1, 13)))
    check("batched scan preserves fitted capacity", summary["stable_count"] == 5)
    check("batch respects the same multiple ceiling", summary["highest_length_probed"] == 12)


def test_scan_error_vs_mistake() -> None:
    """An unparsed reply is an error; a wrong number is a mistake. Different stops."""
    print("scan: errors vs mistakes")
    provider = ScanModel(capacity=10**9, dies_at=40)
    runner = Runner(provider, build_task("homogeneous_count"), SCAN_SETTINGS)
    _, summary = runner.run_scan("m", build_strategy("scan", start=1, stop=100000))
    check("ten consecutive errors abort", summary["stopped_early"] == "consecutive_errors")
    check("stopped ten past the first error", summary["highest_length_probed"] == 49)
    check("first error recorded", summary["first_error"] == 40)
    check("errors counted, not mistakes",
          summary["error_count"] == 10 and summary["mistake_count"] == 0)
    check("capacity is the last correct length", summary["stable_count"] == 39)

    # An isolated error must not abort: the counter is consecutive.
    class Flaky(ScanModel):
        def complete(self, stimulus, model, max_output_tokens, reasoning_effort=None, **kw):
            if stimulus.target in (5, 11, 17):
                self.lengths.append(stimulus.target)
                return RawResponse(text="", status="completed")
            return super().complete(stimulus, model, max_output_tokens, reasoning_effort, **kw)

    runner = Runner(Flaky(capacity=30), build_task("homogeneous_count"), SCAN_SETTINGS)
    _, summary = runner.run_scan("m", build_strategy("scan", start=1, stop=100000))
    check("scattered errors do not abort", summary["stopped_early"] == "past_first_mistake_multiple")
    check("they are still counted", summary["error_count"] == 3)


def test_scan_checkpoint_and_resume(tmp_path: Path) -> None:
    """Every response is written as it arrives; an interrupted scan resumes."""
    print("scan: checkpoint and resume")
    from counting.storage import TrialLog

    log_path = tmp_path / "resume.jsonl"

    class Dies(ScanModel):
        def complete(self, stimulus, model, max_output_tokens, reasoning_effort=None, **kw):
            if stimulus.target >= 15:
                raise KeyboardInterrupt("killed")
            return super().complete(stimulus, model, max_output_tokens, reasoning_effort, **kw)

    log = TrialLog(log_path)
    runner = Runner(Dies(capacity=25), build_task("homogeneous_count"), SCAN_SETTINGS,
                    on_trial=log.append)
    try:
        runner.run_scan("m", build_strategy("scan", start=1, stop=1000))
    except KeyboardInterrupt:
        pass
    log.close()
    check("responses written as they arrive",
          len(log_path.read_text().splitlines()) == 14,
          f"{len(log_path.read_text().splitlines())} lines")

    log = TrialLog(log_path)
    done = log.completed()
    check("checkpoint reads back", len(done) == 14 and max(done) == 14)

    provider = ScanModel(capacity=25)
    runner = Runner(provider, build_task("homogeneous_count"), SCAN_SETTINGS, on_trial=log.append)
    trials, summary = runner.run_scan(
        "m", build_strategy("scan", start=1, stop=1000), already_done=done
    )
    log.close()
    check("resumes at the next length", provider.lengths[0] == 15, str(provider.lengths[0]))
    check("does not re-pay for 1..14", min(provider.lengths) == 15)
    check("reaches the same answer", summary["stable_count"] == 25)
    check("stop condition survives the resume",
          summary["stopped_early"] == "past_first_mistake_multiple")
    check("no duplicate lines",
          len(log_path.read_text().splitlines()) == summary["lengths_probed"])

    # A truncated final line is what an interrupted write leaves behind.
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write('{"provider": "x", "model": "m", "trunca')
    check("truncated tail is ignored, not fatal", len(TrialLog(log_path).completed()) == 52)


def test_body_level_errors() -> None:
    """OpenRouter returns upstream failures as HTTP 200 with an error body.

    Left unhandled such a response has no `choices`, reads as empty text, and
    gets labelled `completed` -- so a provider outage is scored as the model
    failing to count. It must surface as an error instead.
    """
    print("body-level API errors")
    from counting.providers.http import APIError
    from counting.providers.openrouter_api import _status_and_detail, raise_for_body_error

    try:
        raise_for_body_error({"error": {"message": "Internal Server Error", "code": 500}})
        check("error body raises", False)
    except APIError as exc:
        check("error body raises", exc.status_code == 500, f"status {exc.status_code}")

    try:
        raise_for_body_error({"error": "something odd"})
        check("non-dict error body raises", False)
    except APIError as exc:
        check("non-dict error body raises", exc.status_code == 502, f"status {exc.status_code}")

    ok = {"choices": [{"message": {"content": "42"}, "finish_reason": "stop"}]}
    check("a good response passes through", raise_for_body_error(ok) is ok)

    # 500 is in TRANSIENT_STATUS, so the retry policy sees it.
    from counting.providers.base import status_from_codes
    from counting.providers.openrouter_api import TRANSIENT_STATUS
    check("500 is retryable", status_from_codes(APIError(500, "x"), TRANSIENT_STATUS))

    status, detail = _status_and_detail({})
    check("no choices is an error, not a completed empty reply",
          status == "error", f"got {status!r}")
    check("and says why", "no choices" in (detail or ""))


def test_datastore(tmp_path: Path) -> None:
    """The two flat tables, and that later runs stack onto earlier ones."""
    print("datastore")
    from counting.datastore import build_from_run, load_catalogue, write_tables
    from counting.storage import RunPaths, write_model_output

    provider = ScanModel(capacity=8)
    runner = Runner(provider, build_task("homogeneous_count"), SCAN_SETTINGS)
    trials, summary = runner.run_scan(
        "lab/model-x",
        build_strategy("scan", start=1, stop=20, stop_at_first_mistake_multiple=None),
        reasoning_effort="medium",
    )
    summary["run_key"] = "lab/model-x#think"
    summary["variant"] = "think"
    summary["temperature_locked"] = False

    paths = RunPaths.create(tmp_path / "ds", "run-a")
    write_model_output(paths, {"search": {"start": 1, "stop": 20, "step": 1}},
                       "lab/model-x#think", summary, trials, {"variant": "think"})

    payload = json.loads(paths.model_path("lab/model-x#think").read_text())
    check("model id stored without the variant", payload["model"] == "lab/model-x",
          payload["model"])
    check("run key stored separately", payload["run_key"] == "lab/model-x#think")

    catalogue = {"lab/model-x": {"id": "lab/model-x", "release_date": "2026-01-01",
                                 "total_params_b": "7", "architecture_class": "Dense",
                                 "prompt_usd_per_mtok": "0.5", "completion_usd_per_mtok": "1.5",
                                 "reasoning_supported": "True", "reasoning_mandatory": "False",
                                 "knowledge_cutoff": "", "active_params_b": "7",
                                 "context_length": "128000"}}
    registry, responses = build_from_run(paths.run_dir, catalogue)
    check("one registry row per measurement", len(registry) == 1, str(len(registry)))
    row = registry[0]
    check("registry keys on the run key", row["run_key"] == "lab/model-x#think")
    check("registry carries the variant", row["variant"] == "think")
    check("registry carries catalogue metadata", row["release_date"] == "2026-01-01")
    check("registry records the measured capacity", row["cc"] == 8, str(row["cc"]))
    check("registry records the scan window", (row["scan_start"], row["scan_stop"]) == (1, 20))
    check("a wrong answer bounds the capacity", row["cc_bound_by"] == "mistake",
          row["cc_bound_by"])

    # A capacity ended by the output budget is not a counting failure and must
    # be separable from one that is.
    from counting.datastore import _bound_by
    budget = {"stopped_early": "consecutive_errors", "first_mistake": None}
    exhausted_trials = [{"response_detail": "max_output_tokens|fixed_budget_exhausted"}] * 10
    check("budget exhaustion is labelled",
          _bound_by(budget, {"trials": exhausted_trials}) == ("budget", 10))
    other = {"stopped_early": "consecutive_errors", "first_mistake": None}
    check("other errors are labelled separately",
          _bound_by(other, {"trials": [{"response_detail": "no choices"}]}) == ("error", 0))
    check("a mistake wins over any errors",
          _bound_by({"first_mistake": 12, "stopped_early": "consecutive_errors"},
                    {"trials": exhausted_trials})[0] == "mistake")
    check("no stop means the range ended",
          _bound_by({"first_mistake": None, "stopped_early": None}, {"trials": []})[0]
          == "range_end")

    check("one response row per probe", len(responses) == 20, str(len(responses)))
    check("responses are ordered by input count",
          [r["input_count"] for r in responses] == list(range(1, 21)))
    check("responses record input against output",
          all(r["output_count"] == r["input_count"] for r in responses if r["exact"]))
    wrong = [r for r in responses if r["outcome"] == "mistake"]
    check("a wrong answer keeps its value and signed error",
          all(r["output_count"] != r["input_count"] and r["signed_error"] != "" for r in wrong))
    check("preflight is excluded from the curve",
          all(r["input_count"] >= 1 for r in responses) and len(responses) == 20)

    # A second run of the same model must stack, not overwrite.
    paths_b = RunPaths.create(tmp_path / "ds2", "run-b")
    write_model_output(paths_b, {"search": {"start": 1, "stop": 20, "step": 1}},
                       "lab/model-x#think", summary, trials, {"variant": "think"})
    reg_a, resp_a = build_from_run(paths.run_dir, catalogue)
    reg_b, resp_b = build_from_run(paths_b.run_dir, catalogue)
    merged = reg_a + reg_b
    check("runs stack as separate rows", len(merged) == 2)
    check("and are distinguished by run", {r["run"] for r in merged} == {"run-a", "run-b"})

    reg_path, resp_path = write_tables(tmp_path / "out", merged, resp_a + resp_b)
    with reg_path.open() as h:
        check("registry round-trips through CSV", len(list(csv.DictReader(h))) == 2)
    with resp_path.open() as h:
        check("responses round-trip through CSV", len(list(csv.DictReader(h))) == 40)


def test_config_stack() -> None:
    print("config stack")
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": [1]}, {"a": {"y": 3}, "b": [2]})
    check("mappings merge", merged["a"] == {"x": 1, "y": 3})
    check("lists replace", merged["b"] == [2])

    config = load_experiment("configs/experiments/dense_sweep.yaml")
    check("inherits from base", config["seed"] == 7)
    check("inherits provider", config["provider"]["name"] == "openrouter")
    check("inherits task", config["task"]["name"] == "homogeneous_count")
    check("experiment overrides strategy grid", config["search"]["step"] == 16)
    check("strategy layer survives merge", config["search"]["strategy"] == "dense")
    check("base search keys survive", config["search"]["samples_per_length"] == 16)

    roster = load_roster(load_experiment("configs/experiments/openrouter_formal.yaml")["models"])
    check("roster resolves", len(roster) > 200, f"{len(roster)} models")
    check("roster entries are mappings", all("id" in entry for entry in roster))

    for path in sorted(Path("configs/experiments").glob("*.yaml")):
        cfg = load_experiment(path)
        build_strategy(
            {k: v for k, v in cfg["search"].items()}.pop("strategy", "ladder"),
            **{k: v for k, v in cfg["search"].items() if k != "strategy"},
        )
        opts = dict(cfg.get("task") or {})
        build_task(opts.pop("name", "homogeneous_count"), **opts)
    check("every experiment config builds", True, f"{len(list(Path('configs/experiments').glob('*.yaml')))} configs")


def test_storage_roundtrip(tmp_path: Path) -> None:
    print("storage")
    provider = FakeProvider(capacity=200)
    runner = Runner(provider, build_task("homogeneous_count"), RunnerSettings(samples_per_length=4))
    trials, summary = runner.run_model("fake-model", build_strategy("ladder", initial_length=32))

    paths = RunPaths.create(tmp_path, "unit-test-run")
    write_model_output(paths, {"seed": 7}, "fake-model", summary, trials)
    write_run_indexes(paths, {"seed": 7}, ["fake-model"], {"fake-model": summary}, status="complete")

    check("model file written", paths.model_path("fake-model").exists())
    check("manifest written", paths.manifest_path.exists())

    payload = json.loads(paths.model_path("fake-model").read_text())
    check("trials round-trip", len(payload["trials"]) == len(trials))
    check("summary round-trips", payload["summary"]["stable_count"] == summary["stable_count"])
    check("resume detects the model", "fake-model" in completed_models(paths))

    # Two configurations of one model must not be confused for each other.
    variant_summary = dict(summary)
    variant_summary["run_key"] = "fake-model#nothink"
    write_model_output(paths, {"seed": 7}, "fake-model#nothink", variant_summary, trials)
    done = completed_models(paths)
    check("variants are tracked separately",
          {"fake-model", "fake-model#nothink"} <= set(done), str(sorted(done)))

    # The flat tables are derived, not written by the run loop; see test_datastore.
    from counting.datastore import build_from_run
    registry, responses = build_from_run(paths.run_dir, {})
    # Both measurements written above appear, the plain one and the variant.
    check("registry derives from the model files", len(registry) == 2, str(len(registry)))
    check("registry keys both measurements",
          {r["run_key"] for r in registry} == {"fake-model", "fake-model#nothink"})
    check("a ladder run is tagged method=ladder",
          all(r["method"] == "ladder" for r in registry), str(registry[0]["method"]))
    check("ladder counts are derived from trials",
          registry[0]["n_probed"] > 0 and registry[0]["n_correct"] > 0)
    scanned = [x for x in trials if x.phase != "preflight"]
    check("responses derive from the trials", len(responses) == 2 * len(scanned),
          f"{len(responses)} vs {2 * len(scanned)}")
    check("preflight probes stay out of the curve", len(scanned) < len(trials))
    check("ladder responses keep their phase",
          {r["phase"] for r in responses} <= {"expansion", "refinement", "refine"},
          str({r["phase"] for r in responses}))


def main() -> int:
    import tempfile

    test_parsing()
    test_tasks()
    test_grids()
    test_tier_rule()
    test_preflight()
    test_ladder_finds_capacity()
    test_dense_reports_curve()
    test_dense_early_stop()
    test_budget_escalation()
    test_search_edge_cases()
    test_task_edge_cases()
    test_config_stack()
    test_body_level_errors()
    test_single_provider()
    test_fixed_output_budget()
    test_stop_beyond_first_failure()
    test_stop_floor()
    test_step_one_grid_is_exact()
    test_formal_roster()
    test_scan_is_dumb_and_exact()
    test_scan_error_vs_mistake()
    with tempfile.TemporaryDirectory() as tmp:
        test_storage_roundtrip(Path(tmp))
        test_scan_checkpoint_and_resume(Path(tmp))
        test_datastore(Path(tmp))
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
