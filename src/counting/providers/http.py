"""Minimal JSON-over-HTTP client.

Anthropic and OpenRouter are called directly rather than through their SDKs, so
that a run does not depend on SDK versions drifting under it. Both need the same
three things: bearer-ish auth headers, JSON in / JSON out, and the HTTP status
preserved on failure so the retry policy can see whether it was a 429.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

# urlopen defaults to no timeout, which lets a half-open socket block a worker
# forever: a full run was observed with two models holding slots for five hours
# having returned nothing, so the sweep could never finish. The cap has to clear
# the slowest legitimate request rather than the typical one -- across 42k
# probes the median is 2.4s but the maximum is 3131s, a thinking model emitting
# a four-figure count. An hour sits above everything real and still bounds a hang.
DEFAULT_TIMEOUT_SECONDS = 3600.0


class APIError(RuntimeError):
    """An HTTP error with the status code and Retry-After preserved."""

    def __init__(self, status_code: int, body: str, retry_after: str | None = None):
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        super().__init__(f"HTTP {status_code}: {body}")


def request_json(
    url: str,
    *,
    headers: dict[str, str],
    method: str = "POST",
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    service: str = "API",
    timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Issue one request and return the decoded JSON object."""
    if query:
        encoded = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        if encoded:
            url = f"{url}?{encoded}"

    data: bytes | None = None
    request_headers = dict(headers)
    request_headers.setdefault("accept", "application/json")
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["content-type"] = "application/json"

    request = urllib.request.Request(
        url, data=data, headers=request_headers, method=method.upper()
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise APIError(exc.code, body, exc.headers.get("retry-after")) from exc
    except urllib.error.URLError as exc:
        # A timeout arrives here wrapped when the connection stalls, and bare
        # from read() when the body does. Both are transient, so they are given
        # a retryable status instead of being reported as the model's failure.
        if isinstance(getattr(exc, "reason", None), TimeoutError):
            raise APIError(504, f"request exceeded {timeout}s") from exc
        raise RuntimeError(str(exc)) from exc
    except TimeoutError as exc:
        raise APIError(504, f"request exceeded {timeout}s") from exc
    except http.client.IncompleteRead as exc:
        # Seen in the wild mid-run: the response body was truncated in transit.
        # The model answered; the transport lost it. Retry rather than score it.
        raise APIError(502, f"truncated response body: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {service}: {body[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected {service} response shape: {type(parsed).__name__}")
    return parsed
