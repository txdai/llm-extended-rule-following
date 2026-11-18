"""Layered YAML configuration.

One experiment file names the pieces it wants and overrides what it needs:

    # configs/experiments/openrouter_formal.yaml
    extends:
      - base.yaml
      - providers/openrouter.yaml
      - tasks/homogeneous_count.yaml
      - strategies/dense.yaml
    models: {roster: models/openrouter_frontier.yaml}
    search: {step: 1}

`extends` entries are resolved relative to the configs/ root, merged left to
right, and the including file wins over everything it extends. Merging is
recursive for mappings and replace-wholesale for lists and scalars -- a list in
an experiment file replaces the inherited list rather than appending to it,
which is what you want for model rosters and stimulus sets.

Nothing here knows what any particular key means. Validation lives with the
component that consumes the key, so adding a provider or a task never requires
editing this file.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(RuntimeError):
    pass


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} and ${VAR:-default} inside string leaves."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved is None:
                if default is None:
                    raise ConfigError(
                        f"config references ${{{name}}} but that variable is not set"
                    )
                return default
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge; mappings merge, everything else is replaced."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve(path: str | Path, relative_to: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for root in (relative_to, CONFIG_ROOT, REPO_ROOT):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    raise ConfigError(f"config file not found: {path} (looked under {CONFIG_ROOT})")


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load one config file with its `extends` chain resolved and merged."""
    resolved = _resolve(path, CONFIG_ROOT).resolve()
    if resolved in _seen:
        raise ConfigError(f"circular extends detected at {resolved}")

    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved} must contain a mapping at the top level")

    parents = raw.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = _resolve(parent, resolved.parent)
        merged = deep_merge(merged, load_config(parent_path, _seen | {resolved}))

    merged = deep_merge(merged, raw)
    # Assign rather than setdefault: the outermost file in the chain is the one
    # worth recording, and it is merged last.
    try:
        merged["_source"] = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        merged["_source"] = str(resolved)
    return merged


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and expand environment references.

    Env expansion happens once, at the end, so that an `${OPENROUTER_API_KEY}` in a
    provider file is not expanded while merely being inherited by something that
    overrides it.
    """
    return _expand_env(load_config(path))


def load_roster(spec: Any) -> list[dict[str, Any]]:
    """Normalise a `models:` block into a list of model entries.

    Accepts, in increasing order of explicitness:

        models: [openai/gpt-5.4]                              # bare ids
        models: {roster: models/openrouter_frontier.yaml}      # a roster file
        models: [{id: openai/gpt-5.4, reasoning_effort: medium}, ...]

    A roster file is a YAML mapping with a `models:` list using any of the same
    forms, so rosters can be shared between experiments.
    """
    if spec is None:
        return []
    if isinstance(spec, dict):
        if "roster" in spec:
            roster = load_config(spec["roster"])
            entries = load_roster(roster.get("models"))
            overrides = {k: v for k, v in spec.items() if k != "roster"}
            if overrides:
                entries = [deep_merge(entry, overrides) for entry in entries]
            return entries
        spec = [spec]
    if not isinstance(spec, list):
        raise ConfigError(f"models: expects a list or a mapping, got {type(spec).__name__}")

    entries: list[dict[str, Any]] = []
    for item in spec:
        if isinstance(item, str):
            entries.append({"id": item})
        elif isinstance(item, dict):
            if "id" not in item:
                raise ConfigError(f"model entry is missing an 'id': {item}")
            entries.append(dict(item))
        else:
            raise ConfigError(f"unsupported model entry: {item!r}")
    return entries


def require(config: dict[str, Any], key: str) -> Any:
    """Fetch a dotted key, raising a config error that names the missing path."""
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"config is missing required key '{key}'")
        node = node[part]
    return node


def get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Fetch a dotted key, falling back to `default`."""
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
