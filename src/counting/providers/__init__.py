"""Provider registry.

    from counting.providers import build_provider
    provider = build_provider("openrouter", retry={"attempts": 8})

OpenRouter is the only backend. Every model in the sweep is reached through one
provider, one billing path, and one retry policy, so a difference between two
models cannot be a difference between two serving stacks. The earlier
multi-backend harness (OpenAI Responses and Chat, Anthropic Messages, Google
GenAI) carried exactly that confound and was removed; it is recoverable from
git history if a direct-API comparison is ever needed.

Adding a backend back means one module implementing `Provider.complete` plus one
line here -- the interface did not go away with the implementations.
"""

from __future__ import annotations

from typing import Any

from .base import Provider, RetryPolicy
from .openrouter_api import OpenRouterProvider

PROVIDERS: dict[str, type[Provider]] = {
    "openrouter": OpenRouterProvider,
}

# Names used by configs that predate the single-provider decision. They resolve
# to OpenRouter rather than failing, because the model ids in those configs are
# no longer valid either -- the error should come from the roster, not here.
ALIASES = {
    "openai": "openrouter",
    "openai_responses": "openrouter",
    "openai_chat": "openrouter",
    "responses": "openrouter",
    "chat": "openrouter",
    "anthropic": "openrouter",
    "google": "openrouter",
    "google_genai": "openrouter",
    "gemini": "openrouter",
}


def resolve_provider_name(name: str) -> str:
    key = str(name).strip().lower()
    return ALIASES.get(key, key)


def build_provider(name: str, **options: Any) -> Provider:
    """Instantiate a provider by name, translating a `retry:` mapping."""
    key = resolve_provider_name(name)
    if key not in PROVIDERS:
        raise KeyError(
            f"unknown provider {name!r}; this harness serves every model through "
            f"OpenRouter (known: {', '.join(sorted(PROVIDERS))})"
        )

    options = dict(options)
    retry = options.pop("retry", None)
    if isinstance(retry, dict):
        options["retry"] = RetryPolicy(**retry)
    elif isinstance(retry, RetryPolicy):
        options["retry"] = retry

    return PROVIDERS[key](**options)


__all__ = [
    "PROVIDERS",
    "Provider",
    "RetryPolicy",
    "build_provider",
    "resolve_provider_name",
]
