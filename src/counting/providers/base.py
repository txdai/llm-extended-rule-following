"""Provider interface.

A provider does exactly one thing: turn a `Stimulus` plus a token budget into a
`RawResponse`, retrying its own transient failures. It does not parse, score, or
decide whether to raise the budget -- the runner owns all three, so that policy
is identical across providers.

To add a provider:

    1. subclass Provider, implement `complete`
    2. map the provider's finish reason onto ("incomplete", "max_output_tokens")
       when generation was cut off by the token cap -- the runner keys its budget
       escalation on exactly that pair
    3. register the class in providers/__init__.py
    4. add configs/providers/<name>.yaml

Nothing else in the codebase needs to change.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from ..types import RawResponse, Stimulus, Usage


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff for transient (rate-limit / 5xx) failures."""

    attempts: int = 8
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0

    def delay_for(self, attempt: int, suggested: float | None = None) -> float:
        delay = self.base_delay_seconds * (2 ** attempt)
        if suggested is not None:
            delay = max(delay, suggested)
        return min(self.max_delay_seconds, delay)


class Provider(ABC):
    """One model-serving backend."""

    name: str = "provider"

    def __init__(self, retry: RetryPolicy | None = None, **options: Any) -> None:
        self.retry = retry or RetryPolicy()
        self.options = options

    @abstractmethod
    def complete(
        self,
        stimulus: Stimulus,
        model: str,
        max_output_tokens: int,
        reasoning_effort: str | None = None,
        temperature_locked: bool = False,
    ) -> RawResponse:
        """Issue one request and return the normalised response.

        `temperature_locked` marks a model that fixes decoding internally and
        rejects a temperature; such a model samples at the provider default and
        its answers are not deterministic.
        """

    # -- helpers shared by concrete providers -------------------------------

    def _with_transient_retries(
        self,
        call: Callable[[], Any],
        is_transient: Callable[[Exception], bool],
        suggested_delay: Callable[[Exception], float | None],
    ) -> Any:
        """Run `call`, backing off while `is_transient` says the error is worth retrying.

        Re-raises the last exception once the attempt budget is exhausted, so the
        caller still sees the real failure rather than a synthesised one.
        """
        attempt = 0
        while True:
            try:
                return call()
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise freely
                if not is_transient(exc) or attempt >= self.retry.attempts:
                    raise
                time.sleep(self.retry.delay_for(attempt, suggested_delay(exc)))
                attempt += 1

    @staticmethod
    def _error(exc: Exception) -> RawResponse:
        return RawResponse(status="error", detail=f"{type(exc).__name__}: {exc}")


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mapping_or_attr(value: Any, key: str) -> Any:
    """Read `key` from a dict or an object, whichever the SDK handed back."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


_DELAY_PATTERNS = (
    re.compile(r"retry[- ]after[: ]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
    re.compile(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
    re.compile(r"'retryDelay':\s*'([0-9]+(?:\.[0-9]+)?)s'", re.IGNORECASE),
)


def suggested_retry_delay(exc: Exception) -> float | None:
    """Honour a server-supplied backoff hint, from a header or the message text."""
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    message = str(exc)
    for pattern in _DELAY_PATTERNS:
        match = pattern.search(message)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def status_from_codes(exc: Exception, codes: set[int]) -> bool:
    """True when the exception carries, or merely mentions, a retryable status."""
    status_code = getattr(exc, "status_code", None)
    if status_code in codes:
        return True
    message = str(exc).upper()
    return any(str(code) in message for code in codes)


__all__ = [
    "Provider",
    "RetryPolicy",
    "RawResponse",
    "Stimulus",
    "Usage",
    "mapping_or_attr",
    "safe_int",
    "status_from_codes",
    "suggested_retry_delay",
]
