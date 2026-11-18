"""The evaluation loop.

Everything specific lives behind one of the four interfaces; this module only
composes them:

    strategy picks a center length L
      task    builds K stimuli around L
      provider answers each one
      metrics scores them into a verdict
    strategy takes the verdict and picks the next L

Two policies live here rather than in a provider, because they must be identical
across providers for results to be comparable:

  * **Jittered targets.** Trials at tier L draw targets uniformly from
    [0.8L, 1.2L] rather than all using L. This is what makes the assay resistant
    to magnitude guessing -- see the bound in the paper's Appendix B.
  * **Output budget.** With `escalate_output_budget` on (the default), a reply
    truncated by the token cap with no parseable answer gets the cap doubled and
    is retried, up to `retry_max_output_tokens`. A model that needs a bigger
    scratchpad gets one; a model in a generation loop hits the ceiling and is
    recorded as a failure rather than costing unbounded tokens.

    With it off, every trial gets exactly `max_output_tokens` and a truncated or
    empty reply is scored as a failure on the spot. That is the right setting
    when the budget itself is the experimental control: escalation would give
    verbose models more compute than terse ones at the same nominal cap, so the
    thing being measured would no longer be the same across models.
"""

from __future__ import annotations

import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .metrics import score_trial, summarize_slice, tier_passed
from .providers.base import Provider
from .search.base import SearchStrategy
from .tasks.base import Task
from .types import TierVerdict, TrialResult


class ProviderUnavailable(RuntimeError):
    """Preflight could not get an answer out of the model at all."""


@dataclass
class RunnerSettings:
    samples_per_length: int = 16
    jitter_fraction: float = 0.2
    tolerance_fraction: float = 0.05
    max_output_tokens: int = 4096
    retry_max_output_tokens: int = 32768
    parallel_trials_per_batch: int = 15
    parallel_scan_probes: int = 1
    """Number of consecutive linear-scan loads issued as one bounded batch.

    The default remains strictly sequential. Extension sweeps may raise this
    to reduce wall time for high-latency models; the stopping rule is evaluated
    after each returned batch, so resolution is overshot by at most one short
    batch without changing any scored response.
    """
    seed: int = 7
    escalate_output_budget: bool = True
    """Whether to double the cap and retry when a reply is truncated unparsed.

    Turn this off to hold the output budget fixed across every model and every
    trial; see the module docstring for why that changes what is measured.
    """

    preflight_length: int = 8
    """Length of the single connectivity trial run before the search.

    A model that is misnamed, unavailable, or rejecting the request fails here
    for one request instead of after a full ladder -- or, for a dense sweep,
    after hundreds. Set to 0 to skip it.
    """

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RunnerSettings":
        search = config.get("search") or {}
        known = {field for field in cls.__dataclass_fields__}
        values: dict[str, Any] = {}
        for source in (config, search):
            values.update({k: v for k, v in source.items() if k in known})
        return cls(**values)


class Runner:
    """Evaluates one model against one task with one search strategy."""

    def __init__(
        self,
        provider: Provider,
        task: Task,
        settings: RunnerSettings,
        on_tier: Callable[[TierVerdict], None] | None = None,
        on_trial: Callable[[TrialResult], None] | None = None,
    ) -> None:
        self.provider = provider
        self.task = task
        self.settings = settings
        self.on_tier = on_tier
        # Called with every completed trial, before anything is aggregated. The CLI
        # uses it to append to the checkpoint so an interrupted run keeps every
        # response it already paid for.
        self.on_trial = on_trial
        self._trial_callback_lock = threading.Lock()

    def _emit_trial(self, trial: TrialResult) -> None:
        """Checkpoint safely when several scan probes return together."""
        if self.on_trial is not None:
            with self._trial_callback_lock:
                self.on_trial(trial)

    # -- one trial ---------------------------------------------------------

    def run_trial(
        self,
        model: str,
        target: int,
        *,
        phase: str,
        evaluation_index: int,
        center_length: int,
        trial_index: int,
        sample_low: int,
        sample_high: int,
        rng: random.Random,
        reasoning_effort: str | None,
        max_output_tokens: int,
        temperature_locked: bool = False,
    ) -> TrialResult:
        stimulus = self.task.build(target, rng)
        started = time.perf_counter()

        budget = max_output_tokens
        retry_count = 0
        while True:
            response = self.provider.complete(
                stimulus, model, budget, reasoning_effort=reasoning_effort,
                temperature_locked=temperature_locked,
            )
            parsed = self.task.parse(response.text)
            # Escalate only when the cap is demonstrably the problem: generation
            # was cut off *and* nothing usable came back.
            if not self.settings.escalate_output_budget:
                if response.hit_output_cap and parsed is None:
                    response.detail = f"{response.detail}|fixed_budget_exhausted"
                break
            if not response.hit_output_cap or parsed is not None:
                break
            if budget >= self.settings.retry_max_output_tokens:
                response.detail = f"{response.detail}|retry_limit_reached"
                break
            budget *= 2
            retry_count += 1

        latency = time.perf_counter() - started
        # Ground truth is the stimulus's own target. For every counting task it
        # equals the probed length, but a task may probe length as load while
        # the correct answer is derived from it (the solo state machines), and
        # scoring must follow the stimulus rather than the scan variable.
        scores = score_trial(parsed, stimulus.target)
        return TrialResult(
            provider=self.provider.name,
            model=model,
            task=self.task.name,
            phase=phase,
            evaluation_index=evaluation_index,
            center_length=center_length,
            trial_index=trial_index,
            sample_low=sample_low,
            sample_high=sample_high,
            expected=stimulus.target,
            raw_response=response.text,
            response_status=response.status,
            response_detail=response.detail,
            parsed=parsed,
            latency_seconds=latency,
            requested_max_output_tokens=max_output_tokens,
            final_max_output_tokens=budget,
            retry_count=retry_count,
            metadata={
                **stimulus.metadata,
                **({"reasoning_text": response.reasoning_text}
                   if response.reasoning_text is not None else {}),
                **({"generation_id": response.generation_id}
                   if response.generation_id is not None else {}),
            },
            **scores,
            **response.usage.as_dict(),
        )

    # -- one tier ----------------------------------------------------------

    def evaluate_tier(
        self,
        model: str,
        center_length: int,
        *,
        phase: str,
        evaluation_index: int,
        rng: random.Random,
        reasoning_effort: str | None,
    ) -> tuple[list[TrialResult], TierVerdict]:
        s = self.settings
        low = max(1, math.floor(center_length * (1 - s.jitter_fraction)))
        high = max(low, math.ceil(center_length * (1 + s.jitter_fraction)))
        targets = [rng.randint(low, high) for _ in range(s.samples_per_length)]

        def execute(index: int) -> TrialResult:
            # Each trial gets its own generator, derived from the shared seed, so
            # results do not depend on the order threads happen to finish in.
            trial_rng = random.Random(f"{s.seed}:{model}:{center_length}:{index}")
            return self.run_trial(
                model,
                targets[index],
                phase=phase,
                evaluation_index=evaluation_index,
                center_length=center_length,
                trial_index=index,
                sample_low=low,
                sample_high=high,
                rng=trial_rng,
                reasoning_effort=reasoning_effort,
                max_output_tokens=s.max_output_tokens,
            )

        workers = max(1, s.parallel_trials_per_batch)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            trials = list(pool.map(execute, range(len(targets))))

        summary = summarize_slice(trials)
        verdict = TierVerdict(
            center_length=center_length,
            phase=phase,
            evaluation_index=evaluation_index,
            sample_low=low,
            sample_high=high,
            passed=tier_passed(summary, s.tolerance_fraction),
            trial_count=len(trials),
            parse_rate=summary["parse_rate"],
            exact_match_rate=summary["success_rate"],
            mape=summary["mape"],
            mae=summary["mae"],
            summary=summary,
        )
        return trials, verdict

    def probe_length(
        self,
        model: str,
        length: int,
        *,
        index: int,
        reasoning_effort: str | None,
        temperature_locked: bool = False,
    ) -> tuple[TrialResult, TierVerdict]:
        """One request at exactly `length`. The unit of a linear scan.

        Deliberately not evaluate_tier with samples=1: there is no tier here, no
        sampling window, and no tolerance band. The length asked is the length
        recorded, and the verdict is just correct / mistake / error.
        """
        trial = self.run_trial(
            model,
            length,
            phase="scan",
            evaluation_index=index,
            center_length=length,
            trial_index=0,
            sample_low=length,
            sample_high=length,
            rng=random.Random(f"{self.settings.seed}:{model}:{length}"),
            reasoning_effort=reasoning_effort,
            max_output_tokens=self.settings.max_output_tokens,
            temperature_locked=temperature_locked,
        )
        self._emit_trial(trial)

        summary = summarize_slice([trial])
        summary["predicted"] = trial.parsed
        verdict = TierVerdict(
            center_length=length,
            phase="scan",
            evaluation_index=index,
            sample_low=length,
            sample_high=length,
            # A probe "passes" only on an exact match. No tolerance band: the
            # scan is measuring exact rule execution, not approximate magnitude.
            passed=bool(trial.exact_match),
            trial_count=1,
            parse_rate=summary["parse_rate"],
            exact_match_rate=summary["success_rate"],
            mape=summary["mape"],
            mae=summary["mae"],
            summary=summary,
        )
        return trial, verdict

    def run_scan(
        self,
        model: str,
        strategy: SearchStrategy,
        reasoning_effort: str | None = None,
        already_done: dict[int, TrialResult] | None = None,
        temperature_locked: bool = False,
    ) -> tuple[list[TrialResult], dict[str, Any]]:
        """Drive a linear scan, resuming from any lengths already on disk."""
        trials: list[TrialResult] = []
        index = 0

        if already_done:
            replay = []
            for length in sorted(already_done):
                trial = already_done[length]
                summary = summarize_slice([trial])
                summary["predicted"] = trial.parsed
                replay.append(
                    TierVerdict(
                        center_length=length, phase="scan", evaluation_index=index,
                        sample_low=length, sample_high=length,
                        passed=bool(trial.exact_match), trial_count=1,
                        parse_rate=summary["parse_rate"],
                        exact_match_rate=summary["success_rate"],
                        mape=summary["mape"], mae=summary["mae"], summary=summary,
                    )
                )
                trials.append(trial)
                index += 1
            resume = getattr(strategy, "resume_from", None)
            if resume is not None:
                resume(replay)

        check = self.preflight(model, reasoning_effort) if not already_done else None
        if check is not None:
            trials.append(check)
            self._emit_trial(check)
            if check.response_status == "error":
                raise ProviderUnavailable(
                    f"preflight failed at L={check.center_length}: {check.response_detail}"
                )

        batch_size = max(1, int(self.settings.parallel_scan_probes))
        while (length := strategy.next_center()) is not None:
            if batch_size == 1:
                batch_lengths = [length]
            else:
                # LinearScan exposes its fixed grid. Batching is deliberately
                # confined to that strategy; adaptive searches must choose each
                # next point from the preceding verdict.
                step = int(getattr(strategy, "step", 1))
                stop = int(getattr(strategy, "stop", length))
                batch_lengths = [
                    candidate
                    for candidate in (
                        length + offset * step for offset in range(batch_size)
                    )
                    if candidate <= stop
                ]

            def execute(item: tuple[int, int]):
                offset, candidate = item
                return self.probe_length(
                    model, candidate, index=index + offset,
                    reasoning_effort=reasoning_effort,
                    temperature_locked=temperature_locked,
                )

            if len(batch_lengths) == 1:
                results = [execute((0, batch_lengths[0]))]
            else:
                with ThreadPoolExecutor(max_workers=len(batch_lengths)) as pool:
                    results = list(pool.map(execute, enumerate(batch_lengths)))

            # ``map`` preserves grid order even when requests return out of
            # order, so strategy state and the fitted stopping criterion remain
            # deterministic. At most one short batch lies beyond the first
            # point at which the criterion would have stopped sequentially.
            for trial, verdict in results:
                trials.append(trial)
                strategy.record(verdict)
                if self.on_tier is not None:
                    self.on_tier(verdict)
                index += 1

        summary = strategy.result()
        summary.update(
            {
                "model": model,
                "provider": self.provider.name,
                "task": self.task.name,
                "task_level": self.task.level,
                "task_config": self.task.config_record(),
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": self.settings.max_output_tokens,
                # False means the run asked for temperature 0 and got it. True
                # means the model fixes decoding internally, so its answers are
                # a single draw rather than a deterministic reading.
                "temperature_locked": temperature_locked,
                "trial_count": len(trials),
                "overall": summarize_slice(trials),
            }
        )
        return trials, summary

    # -- preflight ---------------------------------------------------------

    def preflight(self, model: str, reasoning_effort: str | None = None) -> TrialResult | None:
        """One cheap trial to confirm the model answers at all.

        Returns None when preflight is disabled. The caller decides what to do
        with a failure; the trial is returned rather than raised so it can be
        recorded alongside the search trials, as the pre-refactor harness did.
        """
        length = self.settings.preflight_length
        if length <= 0:
            return None
        return self.run_trial(
            model,
            length,
            phase="preflight",
            evaluation_index=-1,
            center_length=length,
            trial_index=0,
            sample_low=length,
            sample_high=length,
            rng=random.Random(f"{self.settings.seed}:{model}:preflight"),
            reasoning_effort=reasoning_effort,
            max_output_tokens=self.settings.max_output_tokens,
        )

    # -- one model ---------------------------------------------------------

    def run_model(
        self,
        model: str,
        strategy: SearchStrategy,
        reasoning_effort: str | None = None,
    ) -> tuple[list[TrialResult], dict[str, Any]]:
        rng = random.Random(f"{self.settings.seed}:{model}")
        all_trials: list[TrialResult] = []
        evaluation_index = 0

        check = self.preflight(model, reasoning_effort)
        if check is not None:
            all_trials.append(check)
            if check.response_status == "error":
                raise ProviderUnavailable(
                    f"preflight failed at L={check.center_length}: {check.response_detail}"
                )

        while (center := strategy.next_center()) is not None:
            phase = getattr(strategy, "phase_label", lambda: strategy.name)()
            trials, verdict = self.evaluate_tier(
                model,
                center,
                phase=phase,
                evaluation_index=evaluation_index,
                rng=rng,
                reasoning_effort=reasoning_effort,
            )
            all_trials.extend(trials)
            strategy.record(verdict)
            if self.on_tier is not None:
                self.on_tier(verdict)
            evaluation_index += 1

        summary = strategy.result()
        summary.update(
            {
                "model": model,
                "provider": self.provider.name,
                "task": self.task.name,
                "task_level": self.task.level,
                "task_config": self.task.config_record(),
                "reasoning_effort": reasoning_effort,
                "samples_per_length": self.settings.samples_per_length,
                "jitter_fraction": self.settings.jitter_fraction,
                "tolerance_fraction": self.settings.tolerance_fraction,
                "trial_count": len(all_trials),
                "overall": summarize_slice(all_trials),
                "stable_token_usage": stable_token_usage(summary, all_trials),
            }
        )
        return all_trials, summary


def stable_token_usage(
    summary: dict[str, Any], trials: Sequence[TrialResult]
) -> dict[str, float | None]:
    """Average token spend on the trials inside the stable regime.

    Restricted to tiers that passed: token cost measured over failing tiers is
    dominated by runaway generation and says nothing about the cost of tracking.
    """
    passed = {
        (str(item["phase"]), int(item["evaluation_index"]))
        for item in summary.get("evaluations", [])
        if item.get("passed")
    }
    stable = [t for t in trials if (t.phase, t.evaluation_index) in passed]
    if not stable:
        return {"avg_output_tokens": None, "avg_total_tokens": None, "trials": 0}

    def average(attr: str) -> float | None:
        values = [getattr(t, attr) for t in stable if getattr(t, attr) is not None]
        return sum(values) / len(values) if values else None

    return {
        "avg_output_tokens": average("output_tokens"),
        "avg_total_tokens": average("total_tokens"),
        "trials": len(stable),
    }
