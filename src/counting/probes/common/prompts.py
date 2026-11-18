"""Prompt rendering and decoder-margin measurement for open-weight probes.

The probe prompt is deliberately the same counting prompt the API harness uses,
rendered through the checkpoint's own chat template. Keeping them the same is
what lets a mechanistic result be read against a behavioural SCC number.

The margin functions measure how strongly the decoder prefers the correct integer
under teacher forcing. That is the quantity that collapses at the boundary while
the residual projection is still visibly present -- the gap between "the state is
readable" and "the state is usable".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import torch

SYSTEM_TEXT = (
    "You count items exactly. Return only one integer with no words, punctuation, "
    "or explanation."
)


def make_sequence(count: int, item: str = "a", delimiter: str = ", ") -> str:
    return delimiter.join(item for _ in range(count))


def build_messages(count: int, item: str = "a", delimiter: str = ", ") -> list[dict[str, Any]]:
    """Chat messages in the multimodal content-block form Gemma 3 expects.

    Text-only checkpoints accept this shape too when rendered through
    `apply_chat_template`, so one builder covers both loaders.
    """
    sequence = make_sequence(count, item, delimiter)
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f'How many "{item}" are in this sequence?\n\n{sequence}'}
            ],
        },
    ]


def _plain_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten content blocks to plain strings for text-only chat templates."""
    flat = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = "".join(block["text"] for block in content if block.get("type") == "text")
        flat.append({"role": message["role"], "content": content})
    return flat


def render_prompts(
    tokenizer: Any,
    counts: Sequence[int],
    item: str = "a",
    delimiter: str = ", ",
    template_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Apply the checkpoint's chat template, leaving the assistant turn open.

    Messages are built in the multimodal content-block form first (what VLM
    processors such as Gemma's expect); templates that only accept plain string
    content (e.g. gpt-oss's harmony template) get the flattened form instead.
    `template_kwargs` passes through to the template (e.g. Qwen's
    `enable_thinking`).
    """
    prompts: list[str] = []
    extra = template_kwargs or {}
    for count in counts:
        messages = build_messages(count, item, delimiter)
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **extra
            )
        except (TypeError, ValueError):
            prompt = tokenizer.apply_chat_template(
                _plain_messages(messages), tokenize=False, add_generation_prompt=True, **extra
            )
        if not isinstance(prompt, str):
            raise TypeError(f"expected a string prompt from the chat template, got {type(prompt)!r}")
        prompts.append(prompt)
    return prompts


def tokenize_prompts(tokenizer: Any, prompts: Sequence[str], device: str) -> dict[str, "torch.Tensor"]:
    """Batch-tokenize rendered prompts.

    `add_special_tokens=False` because the chat template already emitted them;
    letting the tokenizer add another BOS shifts every position by one and
    quietly invalidates position-indexed analyses.

    Processors are unwrapped to their inner tokenizer: a VLM processor's
    `__call__` routes its first positional argument to `images`.
    """
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    encoded = tok(
        list(prompts), padding=True, return_tensors="pt", add_special_tokens=False
    )
    return {key: value.to(device) for key, value in encoded.items()}


def batched(items: Sequence[Any], batch_size: int) -> list[list[Any]]:
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def continuation_token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def top_other_logit(logits: "torch.Tensor", token_id: int) -> float:
    """Highest logit among tokens other than `token_id`."""
    import torch

    values, indices = torch.topk(logits, k=2)
    return float(values[1]) if int(indices[0]) == token_id else float(values[0])


def first_divergence_index(correct_ids: Sequence[int], predicted_ids: Sequence[int]) -> int | None:
    """First position where two token sequences differ, or None if one prefixes the other."""
    limit = min(len(correct_ids), len(predicted_ids))
    for index in range(limit):
        if correct_ids[index] != predicted_ids[index]:
            return index
    return limit if len(correct_ids) != len(predicted_ids) else None


def parse_integer(text: str | None) -> int | None:
    """Same rule as the API harness: the last integer in the reply."""
    from ...metrics import parse_integer as _parse

    return _parse(text)
