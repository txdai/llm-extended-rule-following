"""Per-model probe specification.

Everything that differed between the Gemma and Qwen probe scripts is here: the
checkpoint, the loader class, the module paths to hook, the hidden size at each
hook, and (when the model has one) the SAE dictionary to attach.

That means adding a new open-weight model is a YAML file in configs/probes/,
not a fork of a 2500-line script. Finding the module paths for a new model is
mechanical -- `counting-probe inspect <checkpoint>` prints the named modules and
guesses the residual-stream path for each layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...config import load_config

# How a hook reads its tensor:
#   forward      the module's output      (residual stream, MLP output)
#   forward_pre  the module's input       (attention output, read before o_proj)
HOOK_MODES = ("forward", "forward_pre")


@dataclass(frozen=True)
class HookSpec:
    """One activation site."""

    label: str
    module: str
    mode: str = "forward"
    hidden_size: int | None = None
    sae: str | None = None
    """Path to the SAE dictionary for this site, relative to the spec's sae_root."""

    def __post_init__(self) -> None:
        if self.mode not in HOOK_MODES:
            raise ValueError(f"hook {self.label}: mode must be one of {HOOK_MODES}")

    @property
    def family(self) -> str:
        """"resid_post.layer31" -> "resid_post"."""
        return self.label.split(".", 1)[0]

    @property
    def layer(self) -> int | None:
        tail = self.label.rsplit("layer", 1)
        if len(tail) != 2:
            return None
        try:
            return int(tail[1])
        except ValueError:
            return None


@dataclass
class ProbeSpec:
    """Everything needed to run a probe against one checkpoint."""

    name: str
    checkpoint: str
    loader: str = "causal_lm"
    """`causal_lm`, `conditional_generation` (VLMs), or `mlx` (mlx-lm)."""

    tokenizer: str = "auto"
    """`auto` (AutoTokenizer) or `processor` (AutoProcessor, for Gemma 3)."""

    dtype: str = "bfloat16"
    device: str = "auto"
    attn_implementation: str | None = None
    hooks: list[HookSpec] = field(default_factory=list)
    sae_root: Path | None = None
    full_attention_layers: list[int] = field(default_factory=list)
    """For hybrid/sliding-window models: the layers with full attention.

    Attention analyses are only meaningful at these layers; leaving it empty
    means every layer is treated as full attention.
    """

    label: str = ""
    notes: str = ""

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ProbeSpec":
        probe = config.get("probe", config)
        hooks = [
            HookSpec(
                label=label,
                module=spec["module"],
                mode=spec.get("mode", "forward"),
                hidden_size=spec.get("hidden_size"),
                sae=spec.get("sae"),
            )
            for label, spec in (probe.get("hooks") or {}).items()
        ]
        sae_root = probe.get("sae_root")
        return cls(
            name=probe["name"],
            checkpoint=probe["checkpoint"],
            loader=probe.get("loader", "causal_lm"),
            tokenizer=probe.get("tokenizer", "auto"),
            dtype=probe.get("dtype", "bfloat16"),
            device=probe.get("device", "auto"),
            attn_implementation=probe.get("attn_implementation"),
            hooks=hooks,
            sae_root=Path(sae_root) if sae_root else None,
            full_attention_layers=list(probe.get("full_attention_layers") or []),
            label=probe.get("label", probe["name"]),
            notes=probe.get("notes", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProbeSpec":
        return cls.from_config(load_config(path))

    # -- lookups -----------------------------------------------------------

    def hook(self, label: str) -> HookSpec:
        for spec in self.hooks:
            if spec.label == label:
                return spec
        known = ", ".join(spec.label for spec in self.hooks)
        raise KeyError(f"{self.name} has no hook {label!r}; available: {known}")

    def hooks_in_family(self, family: str) -> list[HookSpec]:
        return [spec for spec in self.hooks if spec.family == family]

    @property
    def residual_hooks(self) -> list[HookSpec]:
        return self.hooks_in_family("resid_post")

    @property
    def sae_hooks(self) -> list[HookSpec]:
        return [spec for spec in self.hooks if spec.sae]

    def is_full_attention(self, layer: int) -> bool:
        return not self.full_attention_layers or layer in self.full_attention_layers
