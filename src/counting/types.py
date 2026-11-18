"""Data carried between the four layers of the harness.

The harness is a product of four independent axes, and these types are the only
thing they agree on:

    task      -> Stimulus            what to ask
    provider  -> RawResponse         how to ask it
    runner    -> TrialResult         what came back
    search    -> TierVerdict         where to ask next

Adding a provider, a task, or a search strategy means implementing one interface
and touching nothing else. Keep this module free of provider- and task-specific
fields so that stays true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Stimulus:
    """One prompt with known ground truth, produced by a Task."""

    target: int
    """The exact integer the model is supposed to return."""

    system: str
    """System / developer instruction."""

    user: str
    """User message body."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Task-specific provenance (motif, delimiter, benchmark item id, ...).

    Written verbatim into the trial record, so anything here is available to the
    analysis layer without the analysis layer knowing which task produced it.
    """


@dataclass
class Usage:
    """Token accounting, normalised across providers. None means not reported."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_input_tokens": self.cached_input_tokens,
        }


@dataclass
class RawResponse:
    """One completed provider call, before parsing or scoring.

    `status` is normalised to a small vocabulary so the runner can act on it
    without provider-specific knowledge:

        "completed"  -> the model finished on its own
        "incomplete" -> generation was cut short; see `detail`
        "error"      -> the call failed; see `detail`

    `detail` is "max_output_tokens" when the cut-off was the token budget. That
    exact pairing is what triggers the runner's budget escalation, so providers
    must map their own finish reasons onto it.
    """

    text: str = ""
    status: str | None = None
    detail: str | None = None
    usage: Usage = field(default_factory=Usage)
    reasoning_text: str | None = None
    """The provider-exposed reasoning stream. None when the provider returned
    none (Anthropic exposes no trace through OpenRouter; Gemini returns
    summaries; DeepSeek and Qwen return full traces) or when capture is
    disabled for the run."""
    generation_id: str | None = None
    """The provider's id for this generation, kept so a trial can be joined to
    billing and generation metadata later."""

    @property
    def hit_output_cap(self) -> bool:
        return self.status == "incomplete" and self.detail == "max_output_tokens"

    @property
    def is_error(self) -> bool:
        return self.status == "error"


@dataclass
class TrialResult:
    """One scored trial. This is the unit written to disk and read by analysis.

    Field names are kept compatible with the pre-refactor `benchmark.py` records
    so that existing runs under results/ still load.
    """

    provider: str
    model: str
    task: str
    phase: str
    evaluation_index: int
    center_length: int
    trial_index: int
    sample_low: int
    sample_high: int
    expected: int
    raw_response: str
    response_status: str | None
    response_detail: str | None
    parsed: int | None
    exact_match: bool
    absolute_error: float | None
    squared_error: float | None
    relative_error: float | None
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    requested_max_output_tokens: int = 0
    final_max_output_tokens: int = 0
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TierVerdict:
    """The outcome of evaluating one tier, handed back to the search strategy.

    A tier is a center length L plus the K trials drawn around it. `passed` is
    the only field a strategy has to look at; the rest is carried for the run
    record and for dense sweeps, which report curves rather than a boundary.
    """

    center_length: int
    phase: str
    evaluation_index: int
    sample_low: int
    sample_high: int
    passed: bool
    trial_count: int
    parse_rate: float | None
    exact_match_rate: float | None
    mape: float | None
    mae: float | None
    summary: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        record = {
            "center_length": self.center_length,
            "phase": self.phase,
            "evaluation_index": self.evaluation_index,
            "sample_low": self.sample_low,
            "sample_high": self.sample_high,
            "passed": self.passed,
        }
        record.update(self.summary)
        return record
