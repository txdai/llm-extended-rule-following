"""MLX backend for gpt-oss-120b (and any mlx-lm checkpoint).

MLX has no PyTorch hook API. This module wraps residual-stream layers by
monkeypatching `__call__`, converts activations to numpy for the rest of the
suite, and exposes greedy generation through `mlx_lm.generate.generate_step`.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from .spec import HookSpec


def load_mlx(checkpoint: str) -> tuple[Any, Any, str]:
    from mlx_lm import load

    model, tokenizer = load(checkpoint)
    model.eval()
    return model, tokenizer, "mlx"


def resolve_layer_index(module: str) -> int:
    """`model.layers.8` -> 8."""
    parts = module.split(".")
    if len(parts) >= 3 and parts[-2] == "layers":
        return int(parts[-1])
    raise KeyError(f"cannot resolve MLX module path {module!r}")


class _LayerTap:
    """Instance-level `__call__` is ignored by `layer(...)` (looks up the class).

    Replacing the list entry with this proxy is what actually intercepts the
    residual stream: `GptOssMoeModel` iterates `self.layers` and calls each item.
    """

    def __init__(self, inner: Any, on_out: Callable[[Any], Any]) -> None:
        self._inner = inner
        self._on_out = on_out

    def __call__(self, x, mask, cache=None):
        return self._on_out(self._inner(x, mask, cache))


def _to_numpy(value: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.array(value.astype(mx.float32))


def generate_tokens(model: Any, tokenizer: Any, token_ids: list[int], max_new_tokens: int) -> list[int]:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    eos = set(getattr(tokenizer, "eos_token_ids", None) or [])
    if not eos:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            eos = {int(eos_id)}
    new_ids: list[int] = []
    for token, _logprobs in generate_step(mx.array(token_ids), model, max_tokens=max_new_tokens):
        tid = int(token)
        if tid in eos:
            break
        new_ids.append(tid)
        if len(new_ids) >= max_new_tokens:
            break
    return new_ids


def generate_text(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    import re

    tok = getattr(tokenizer, "tokenizer", tokenizer)
    ids = list(tok.encode(prompt, add_special_tokens=False))
    new_ids = generate_tokens(model, tokenizer, ids, max_new_tokens)
    try:
        raw = tok.decode(new_ids, skip_special_tokens=False)
    except TypeError:
        raw = tok.decode(new_ids)
    return re.sub(r"<[^<>\s]{1,60}>", "\n", raw).strip()


def _layer_list(model: Any) -> list:
    return model.model.layers


class MlxActivationRecorder:
    """Capture residual-stream outputs at named TransformerBlocks."""

    def __init__(self, model: Any, hooks: Iterable[HookSpec]) -> None:
        self.model = model
        self.hooks = list(hooks)
        self.cache: dict[str, np.ndarray] = {}
        self._swaps: list[tuple[int, Any]] = []

    def __enter__(self) -> "MlxActivationRecorder":
        layers = _layer_list(self.model)
        for spec in self.hooks:
            idx = resolve_layer_index(spec.module)
            inner = layers[idx]

            def on_out(out, *, _label=spec.label):
                self.cache[_label] = _to_numpy(out)
                return out

            layers[idx] = _LayerTap(inner, on_out)
            self._swaps.append((idx, inner))
        return self

    def __exit__(self, *exc_info: object) -> None:
        layers = _layer_list(self.model)
        for idx, inner in self._swaps:
            layers[idx] = inner
        self._swaps.clear()


class MlxActivationEditor:
    """Rewrite one residual-stream site during the next forward / generate."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._swaps: list[tuple[int, Any]] = []

    def __enter__(self) -> "MlxActivationEditor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        layers = _layer_list(self.model)
        for idx, inner in self._swaps:
            layers[idx] = inner
        self._swaps.clear()

    def edit(self, spec: HookSpec, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        import mlx.core as mx

        layers = _layer_list(self.model)
        idx = resolve_layer_index(spec.module)
        inner = layers[idx]

        def on_out(out, *, _fn=fn):
            return mx.array(_fn(_to_numpy(out)))

        layers[idx] = _LayerTap(inner, on_out)
        self._swaps.append((idx, inner))


def forward_logits(model: Any, token_ids: list[int]) -> np.ndarray:
    """Teacher-forced logits, shape (seq, vocab)."""
    import mlx.core as mx

    logits = model(mx.array([token_ids]))
    mx.eval(logits)
    return _to_numpy(logits[0])
