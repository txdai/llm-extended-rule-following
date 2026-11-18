"""OpenRouter chat completions, called directly over HTTP.

This is the backend that covers most open-weight models in the paper, so it sees
the widest spread of upstream behaviour: some providers behind OpenRouter return
content as a string, others as a list of blocks, and both are handled here.
"""

from __future__ import annotations

import os
from typing import Any

from ..types import RawResponse, Stimulus, Usage
from .base import Provider, mapping_or_attr, safe_int, status_from_codes, suggested_retry_delay
from .http import APIError, request_json

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
TRANSIENT_STATUS = {429, 500, 502, 503, 504, 524, 529}


def raise_for_body_error(response: dict[str, Any]) -> dict[str, Any]:
    """Surface an error that arrived with a 200 status.

    OpenRouter returns upstream failures as HTTP 200 with an error object in the
    body rather than an error status. Left alone, such a response has no
    `choices`, so it reads as a completed reply with empty text -- which the scan
    scores as the model failing to count, when in fact the request never reached
    a model. Converting it to an APIError puts it back on the retry path and,
    if it persists, records it honestly as an error rather than a wrong answer.
    """
    error = response.get("error")
    if not error:
        return response
    if isinstance(error, dict):
        code = error.get("code")
        message = str(error.get("message") or error)
    else:
        code, message = None, str(error)
    try:
        status = int(code)
    except (TypeError, ValueError):
        status = 502  # unknown upstream failure; treat as transient
    raise APIError(status, message)


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "OPENROUTER_API_KEY",
        temperature: float = 0.0,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self._api_key: str | None = None

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            key = os.getenv(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self.api_key_env} is required to run the openrouter provider"
                )
            self._api_key = key
        return self._api_key

    def complete(
        self,
        stimulus: Stimulus,
        model: str,
        max_output_tokens: int,
        reasoning_effort: str | None = None,
        temperature_locked: bool = False,
    ) -> RawResponse:
        capture_reasoning = bool(self.options.get("capture_reasoning", True))
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": stimulus.system},
                {"role": "user", "content": stimulus.user},
            ],
            "max_tokens": max_output_tokens,
        }
        # Models that fix decoding internally (the OpenAI o-series and GPT-5
        # family, some Anthropic models) do not accept these. Sending them is
        # noise at best and a 400 at worst, and it would imply the run pinned
        # decoding when it did not.
        if not temperature_locked:
            payload["temperature"] = self.temperature
            # Mistral rejects temperature 0 unless top_p is pinned to 1:
            # "top_p must be 1 when using greedy sampling".
            if self.temperature == 0:
                payload["top_p"] = 1

        effort = (reasoning_effort or "").strip().lower()
        if effort:
            # "none" disables reasoning outright, which is how a thinking model
            # is measured with thinking off. Mandatory-reasoning models reject
            # it, so the roster never pairs "none" with one.
            payload["reasoning"] = {"effort": effort}

        try:
            response = self._with_transient_retries(
                lambda: raise_for_body_error(
                    request_json(
                        f"{self.base_url}/chat/completions",
                        headers={"authorization": f"Bearer {self.api_key}"},
                        payload=payload,
                        service="OpenRouter",
                    )
                ),
                is_transient=lambda exc: status_from_codes(exc, TRANSIENT_STATUS),
                suggested_delay=suggested_retry_delay,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc)

        status, detail = _status_and_detail(response)
        return RawResponse(
            text=_response_text(response),
            status=status,
            detail=detail,
            usage=_usage(response),
            reasoning_text=_reasoning_text(response) if capture_reasoning else None,
            generation_id=response.get("id"),
        )


def _reasoning_text(response: dict[str, Any]) -> str | None:
    """The reasoning stream, wherever the upstream provider put it."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    details = message.get("reasoning_details")
    if isinstance(details, list):
        parts = [d.get("text") for d in details
                 if isinstance(d, dict) and isinstance(d.get("text"), str)]
        if parts:
            return "\n".join(parts)
    return None


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _status_and_detail(response: dict[str, Any]) -> tuple[str | None, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        # No choices and no error object: nothing came back, but the request did
        # not fail either. Recording it as an error is honest; recording it as a
        # completed empty reply would blame the model.
        return "error", "no choices in response"
    finish_reason = choices[0].get("finish_reason")
    if finish_reason == "length":
        return "incomplete", "max_output_tokens"
    if isinstance(finish_reason, str) and finish_reason:
        return "completed", finish_reason
    return "completed", None


def _usage(response: dict[str, Any]) -> Usage:
    usage = mapping_or_attr(response, "usage")
    if usage is None:
        return Usage()
    details = mapping_or_attr(usage, "completion_tokens_details")
    return Usage(
        input_tokens=safe_int(mapping_or_attr(usage, "prompt_tokens")),
        output_tokens=safe_int(mapping_or_attr(usage, "completion_tokens")),
        total_tokens=safe_int(mapping_or_attr(usage, "total_tokens")),
        reasoning_tokens=safe_int(mapping_or_attr(details, "reasoning_tokens"))
        if details is not None
        else None,
    )
