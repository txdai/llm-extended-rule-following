"""Shared machinery for mechanistic probes on open-weight checkpoints.

The Gemma and Qwen probe scripts were near-duplicates that differed in a hook
table and a loader class. Both are now data:

    configs/probes/<model>.yaml   checkpoint, loader, hook module paths, SAE root
    ProbeSpec                     that file, parsed
    ActivationRecorder            read activations at the spec's hook sites
    ActivationEditor              rewrite them mid-forward-pass
    SparseAutoencoder             Gemma Scope-style JumpReLU dictionaries
    fit_counter_direction         the 1-D progress coordinate
    interventions                 steer / patch / translate / clamp

Adding a model is a YAML file. Start from configs/probes/gemma3-27b-it.yaml and
use `counting-probe inspect <checkpoint>` to find the module paths.
"""

from __future__ import annotations

from .directions import (
    CounterDirection,
    fit_counter_direction,
    fit_principal_components,
    monotonicity_score,
    pearson,
    pearson_by_feature,
    project_position_window,
)
from .interventions import (
    clamp_projection,
    patch_positions,
    resample_positions,
    steer_final_token,
    translate_positions,
)
from .model import ActivationEditor, ActivationRecorder, load_model, resolve_device, resolve_dtype
from .prompts import (
    SYSTEM_TEXT,
    batched,
    build_messages,
    continuation_token_ids,
    first_divergence_index,
    make_sequence,
    parse_integer,
    render_prompts,
    tokenize_prompts,
    top_other_logit,
)
from .sae import SparseAutoencoder, load_saes
from .spec import HookSpec, ProbeSpec

__all__ = [
    "ActivationEditor",
    "ActivationRecorder",
    "CounterDirection",
    "HookSpec",
    "ProbeSpec",
    "SYSTEM_TEXT",
    "SparseAutoencoder",
    "batched",
    "build_messages",
    "clamp_projection",
    "continuation_token_ids",
    "first_divergence_index",
    "fit_counter_direction",
    "fit_principal_components",
    "load_model",
    "load_saes",
    "make_sequence",
    "monotonicity_score",
    "parse_integer",
    "patch_positions",
    "pearson",
    "pearson_by_feature",
    "project_position_window",
    "render_prompts",
    "resample_positions",
    "resolve_device",
    "resolve_dtype",
    "steer_final_token",
    "tokenize_prompts",
    "top_other_logit",
    "translate_positions",
]
