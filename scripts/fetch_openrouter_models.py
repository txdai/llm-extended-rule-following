#!/usr/bin/env python3
"""Build the OpenRouter model catalogue for a formal run.

    python scripts/fetch_openrouter_models.py
    python scripts/fetch_openrouter_models.py --no-enrich      # skip HF/Epoch lookups
    python scripts/fetch_openrouter_models.py --labs openai anthropic
    python scripts/fetch_openrouter_models.py --all-labs

Writes three things:

    configs/models/openrouter_frontier.yaml   the roster a run consumes
    results/model_catalogue/openrouter_models.csv    full table, one row per model
    results/model_catalogue/openrouter_models.json   same, with nested fields intact

The catalogue is not just a list of entry points. Per model it records the
release date, context window, output cap, prompt and completion price, reasoning
support, modality, tokenizer, and -- where it can be established -- total and
active parameter counts.

Parameter counts are the one thing OpenRouter does not serve, so they are
resolved from three sources in decreasing order of authority:

    1. the published slug convention, e.g. qwen3-235b-a22b -> 235B total /
       22B active, or mixtral-8x7b -> 8 experts of 7B. This is the number the
       lab itself publishes and is the only reliable source for *active* params.
    2. the HuggingFace API's safetensors.total, which is an exact count of
       tensor elements, used for total params when a hugging_face_id is known.
    3. Epoch AI's notable_ai_models table, which covers proprietary models that
       have no HF entry but whose size has been reported.

Every row records which source produced its numbers in `parameter_source`, so a
downstream analysis can drop inferred values if it needs only published ones.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counting.config import load_config  # noqa: E402
from counting.datastore import TEMPERATURE_IGNORED  # noqa: E402

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
HF_API = "https://huggingface.co/api/models/"
EPOCH_MODELS_URL = "https://epoch.ai/data/notable_ai_models.csv"

OUT_ROSTER = REPO_ROOT / "configs" / "models" / "openrouter_frontier.yaml"
OUT_DIR = REPO_ROOT / "results" / "model_catalogue"

# Variant suffixes that are a different way of serving an existing model rather
# than a different model. Keeping them would double-count and, worse, measure the
# serving path instead of the model.
#   :batch  asynchronous batch API -- same weights, no latency semantics
#   :free   rate-limited community mirror, frequently a different provider
#   :nitro  throughput-optimised routing
#   :floor  price-optimised routing
# ":thinking" is deliberately NOT here: it is a genuine reasoning variant and is
# exactly what the base-versus-reasoning comparison needs.
SERVING_VARIANTS = {"batch", "free", "nitro", "floor", "extended", "online"}

# Same idea as SERVING_VARIANTS but the suffix uses "-" rather than ":".
# Anthropic's "-fast" models are the same weights on a priority queue at 2-6x
# the price -- claude-opus-4.7-fast is $30/Mtok against $5 for claude-opus-4.7.
# Identical weights, so they add no capability data point.
SERVING_SUFFIXES = ("-fast",)

# Default ceiling on prompt price, in USD per million tokens.
#
# Set to keep gpt-4 and o1 at $15-30 -- gpt-4 is a weak non-thinking model whose
# replies are ~3 tokens, so its bill is input-only and modest, and both are
# historically interesting -- while still cutting the tier above them. o1-pro at
# $150/Mtok is 300x the roster median and would alone be a quarter of a run;
# o3-pro and the gpt-5.x -pro variants are the same base model at higher
# test-time compute, which this project's own result says does not recover
# counting.
#
# Price, not name, is the filter. Google's gemini-3.1-pro ($2) and DeepSeek's
# v4-pro ($0.50) are flagship tiers, and the gpt-5.6 *-pro models are reasoning
# modes priced identically to their base -- a "-pro" pattern would wrongly cut
# all of them.
DEFAULT_MAX_PROMPT_USD = 31.0

# Models that cannot be measured on this assay, with the reason. Each was
# confirmed by probing, not inferred from its name -- gpt-oss-safeguard-20b has
# "safeguard" in its id and scores CC=42, so the name is not a usable signal.
UNUSABLE_MODELS = {
    "meta-llama/llama-guard-4-12b":
        "safety classifier: replies 'safe' to every prompt, including 'say hello'",
    "mistralai/mistral-small-3.1-24b-instruct":
        "sole provider (Cloudflare) returns HTTP 500 for any system prompt that "
        "constrains output to a bare integer; answers normally without one",
    "qwen/qwen-2.5-coder-32b-instruct":
        "sole provider (Cloudflare) returns HTTP 500 for any system prompt that "
        "constrains output to a bare integer; answers normally without one",
    "mistralai/mistral-7b-instruct":
        "listed by the models API but has no live endpoint (HTTP 404)",
}

# Endpoints that are not a single model. The assay measures one model counting;
# an orchestration of several behind one id measures the orchestration, and its
# capacity is not a property of any model in the study. Excluded on what the
# endpoint *is*, not on how it scored.
NOT_SINGLE_MODEL = {
    "x-ai/grok-4.20-multi-agent":
        "multi-agent orchestration over grok-4.20, not a single model",
}

# Models excluded regardless of price: same base model, more test-time compute.
EXPENSIVE_COMPUTE_VARIANTS = {
    "openai/o1-pro",
    "openai/o3-pro",
    "openai/gpt-5-pro",
    "openai/gpt-5.2-pro",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.5-pro",
}

# Reasoning effort the roster requests, and the ladder used to approximate it on
# models that expose a different set. Ranked low to high; "medium" is rank 3.
EFFORT_RANK = {"minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5, "max": 6}
TARGET_EFFORT = "medium"


def is_text_model(model: dict[str, Any]) -> bool:
    """True when the model outputs text.

    Image and audio generators are carried on OpenRouter alongside text models
    and pass every other filter, but asking one to count a sequence and return
    an integer is off-task. Vision *input* is fine -- qwen3-vl reads images and
    writes text, and is a normal text model for this assay.
    """
    out = (model.get("architecture") or {}).get("output_modalities") or ["text"]
    return "text" in out and not ({"image", "audio"} & set(out))


def choose_effort(row: dict[str, Any]) -> str | None:
    """Which reasoning effort to request when reasoning is on.

    `medium` wherever it is offered. A model that advertises reasoning without
    enumerating efforts also gets `medium`. A model that enumerates a set without
    `medium` gets the nearest rung, preferring the higher one on a tie, because a
    reasoning model's lowest rung is often effectively off.

    Returns None for a model with no reasoning at all, and for one that always
    reasons but exposes no depth control -- there is nothing to set, and sending
    an effort it never advertised would be noise in the run record rather than a
    setting.
    """
    if not row["reasoning_supported"]:
        return None
    efforts = [e for e in (row["reasoning_efforts"] or "").split(",") if e]
    if row["reasoning_mandatory"] and not efforts:
        return None
    if not efforts or TARGET_EFFORT in efforts:
        return TARGET_EFFORT
    ranked = [(EFFORT_RANK[e], e) for e in efforts if e in EFFORT_RANK]
    if not ranked:
        return TARGET_EFFORT
    target = EFFORT_RANK[TARGET_EFFORT]
    ranked.sort(key=lambda pair: (abs(pair[0] - target), -pair[0]))
    return ranked[0][1]


def roster_entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The measurement(s) this model contributes.

    A model whose reasoning can be switched off is two measurements, not one:
    the same weights with thinking on and with thinking off. That pairing is
    what a base-versus-reasoning comparison needs, and taking it from one model
    id controls for everything except the setting.

    Mandatory-reasoning models reject `effort: "none"` outright, so they stay a
    single entry, as do models with no reasoning at all.

    Entries also carry `temperature_locked` when the model does not accept a
    temperature. The assay asks for temperature 0 because the task has exactly
    one correct answer, so sampling only adds variance -- but 31 models (the
    OpenAI o-series and GPT-5 family, and the Anthropic releases from Opus 4.7
    onward, which accept the parameter and ignore it) fix decoding internally. Their answers are a single draw from
    whatever the provider defaults to, not a deterministic reading, and a scan of
    one probe per length cannot tell the difference. Flagging it keeps that
    visible in the results rather than assuming determinism the run never had.
    """
    effort = row["configured_effort"]
    accepts = row["supports_temperature"] and row["id"] not in TEMPERATURE_IGNORED
    locked = {} if accepts else {"temperature_locked": True}
    if not row["reasoning_supported"]:
        return [{"id": row["id"], **locked}]
    if row["reasoning_mandatory"]:
        entry = {"id": row["id"]}
        if effort:
            entry["reasoning_effort"] = effort
        return [{**entry, **locked}]
    on = {"id": row["id"], "variant": "think"}
    if effort:
        on["reasoning_effort"] = effort
    off = {"id": row["id"], "variant": "nothink", "reasoning_effort": "none"}
    return [{**on, **locked}, {**off, **locked}]


# -- fetching ---------------------------------------------------------------

def get_json(url: str, timeout: float = 60.0) -> Any:
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(url: str, timeout: float = 120.0) -> str:
    request = urllib.request.Request(url, headers={"accept": "text/csv"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


# -- parameter inference ----------------------------------------------------

BILLION = 1e9

# qwen3-235b-a22b, glm-4.5-air-106b-a12b, ...  total-B / active-B
MOE_SLUG = re.compile(r"(\d+(?:\.\d+)?)\s*b[-_]?a(\d+(?:\.\d+)?)\s*b", re.I)
# mixtral-8x7b, ...  experts x expert-size
EXPERT_SLUG = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b", re.I)
# llama-3.3-70b-instruct, gemma-3-27b-it, ...  a single size token
DENSE_SLUG = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", re.I)

DESC_TOTAL = re.compile(r"(\d+(?:\.\d+)?)\s*b(?:illion)?\s+(?:total\s+)?(?:parameter|params?)", re.I)
DESC_ACTIVE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?:illion)?\s+(?:active|activated)", re.I)


def params_from_slug(*candidates: str) -> tuple[float | None, float | None, str | None]:
    """Total and active params in billions, read off the published naming."""
    blob = " ".join(c for c in candidates if c).lower()

    if (m := MOE_SLUG.search(blob)):
        return float(m.group(1)), float(m.group(2)), "slug_moe"
    if (m := EXPERT_SLUG.search(blob)):
        experts, size = float(m.group(1)), float(m.group(2))
        # Two experts are routed per token in the Mixtral-style configs this
        # pattern matches; the total is not experts*size exactly (attention is
        # shared) but it is the figure these models are published under.
        return experts * size, 2 * size, "slug_experts"

    sizes = [float(x) for x in DENSE_SLUG.findall(blob)]
    # Filter out version numbers that look like sizes (e.g. "4b" in a date).
    sizes = [s for s in sizes if 0.1 <= s <= 3000]
    if sizes:
        total = max(sizes)
        return total, total, "slug_dense"
    return None, None, None


def params_from_description(description: str) -> tuple[float | None, float | None, str | None]:
    total = active = None
    if (m := DESC_TOTAL.search(description or "")):
        total = float(m.group(1))
    if (m := DESC_ACTIVE.search(description or "")):
        active = float(m.group(1))
    return total, active, "description" if (total or active) else None


def params_from_huggingface(hf_id: str, cache: dict[str, Any]) -> tuple[float | None, str | None]:
    """Exact total parameter count from the HF safetensors index."""
    if hf_id in cache:
        payload = cache[hf_id]
    else:
        try:
            payload = get_json(HF_API + hf_id, timeout=25.0)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            payload = None
        cache[hf_id] = payload
        time.sleep(0.1)  # be polite to the HF API
    if not payload:
        return None, None
    total = ((payload.get("safetensors") or {}).get("total"))
    if isinstance(total, (int, float)) and total > 0:
        return float(total) / BILLION, "huggingface"
    return None, None


def load_epoch_params() -> dict[str, float]:
    """Model display name (lowercased) -> total parameters in billions."""
    try:
        text = get_text(EPOCH_MODELS_URL)
    except Exception as exc:  # noqa: BLE001
        print(f"  epoch lookup unavailable ({type(exc).__name__}); continuing", file=sys.stderr)
        return {}
    table: dict[str, float] = {}
    for row in csv.DictReader(text.splitlines()):
        name = (row.get("Model") or "").strip().lower()
        raw = (row.get("Parameters") or "").strip()
        if not name or not raw:
            continue
        try:
            table[name] = float(raw) / BILLION
        except ValueError:
            continue
    return table


KNOWN_ACTIVE_PARAMS_B: dict[str, float] = {
    "openai/gpt-oss-120b": 5.1,          # 117B total, 5.1B active (OpenAI model card)
    "openai/gpt-oss-20b": 3.6,           # 21B total, 3.6B active
    "openai/gpt-oss-safeguard-20b": 3.6,
    "mistralai/mixtral-8x22b-instruct": 39.0,  # 141B total, 39B active (Mistral)
}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def resolve_parameters(
    model: dict[str, Any], epoch: dict[str, float], hf_cache: dict[str, Any], enrich: bool
) -> dict[str, Any]:
    slug, name = model["id"], model.get("name", "")
    hf_id = model.get("hugging_face_id") or ""
    description = model.get("description") or ""

    total, active, source = params_from_slug(slug, model.get("canonical_slug", ""), name, hf_id)
    if total is None:
        total, active, source = params_from_description(description)

    # HF gives an exact total; prefer it over a slug-rounded figure, but keep the
    # slug's active count, which HF cannot give.
    if enrich and hf_id:
        hf_total, hf_source = params_from_huggingface(hf_id, hf_cache)
        if hf_total is not None:
            if total is None or abs(hf_total - total) / max(total, 1) > 0.02:
                source = hf_source if total is None else f"{source}+huggingface"
            total = hf_total

    if total is None and epoch:
        key = normalise(name.split(":")[-1])
        for epoch_name, value in epoch.items():
            if normalise(epoch_name) == key:
                total, source = value, "epoch"
                break

    # Mixture models whose published names carry no active-parameter marker.
    # Values from the vendors' model cards; without this table the slug rule
    # below would record them as dense (gpt-oss-120b was listed as 120B active).
    base_slug = slug.split(":")[0]
    if base_slug in KNOWN_ACTIVE_PARAMS_B:
        active = KNOWN_ACTIVE_PARAMS_B[base_slug]
        source = f"{source}+vendor_card"

    if total is not None and active is None:
        # A model with no MoE marker in its published name is dense: every
        # parameter is active. Recorded explicitly so the column is not empty.
        active = total

    return {
        "total_params_b": round(total, 3) if total is not None else None,
        "active_params_b": round(active, 3) if active is not None else None,
        "parameter_source": source,
    }


# -- assembly ---------------------------------------------------------------

def price_per_million(value: Any) -> float | None:
    """OpenRouter quotes USD per token; per-million tokens is the readable unit."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return round(price * 1_000_000, 6)


def build_row(model: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    pricing = model.get("pricing") or {}
    provider = model.get("top_provider") or {}
    architecture = model.get("architecture") or {}
    reasoning = model.get("reasoning") or {}
    supported = model.get("supported_parameters") or []
    created = model.get("created")

    return {
        "id": model["id"],
        "lab": model["id"].split("/")[0],
        "name": model.get("name"),
        "canonical_slug": model.get("canonical_slug"),
        "release_date": (
            datetime.fromtimestamp(created, timezone.utc).date().isoformat() if created else None
        ),
        "knowledge_cutoff": model.get("knowledge_cutoff"),
        "total_params_b": params["total_params_b"],
        "active_params_b": params["active_params_b"],
        "parameter_source": params["parameter_source"],
        "architecture_class": (
            "MoE"
            if params["total_params_b"] and params["active_params_b"]
            and params["active_params_b"] < params["total_params_b"] * 0.95
            else ("Dense" if params["total_params_b"] else None)
        ),
        "context_length": model.get("context_length"),
        "max_output_tokens": provider.get("max_completion_tokens"),
        "prompt_usd_per_mtok": price_per_million(pricing.get("prompt")),
        "completion_usd_per_mtok": price_per_million(pricing.get("completion")),
        "cache_read_usd_per_mtok": price_per_million(pricing.get("input_cache_read")),
        "is_free": price_per_million(pricing.get("prompt")) == 0,
        "reasoning_supported": bool(reasoning) or "reasoning" in supported,
        "reasoning_mandatory": bool(reasoning.get("mandatory")),
        "reasoning_efforts": ",".join(reasoning.get("supported_efforts") or []),
        "default_effort": reasoning.get("default_effort"),
        "supports_seed": "seed" in supported,
        "supports_temperature": "temperature" in supported,
        "modality": architecture.get("modality"),
        "tokenizer": architecture.get("tokenizer"),
        "instruct_type": architecture.get("instruct_type"),
        "hugging_face_id": model.get("hugging_face_id"),
        "is_moderated": provider.get("is_moderated"),
    }


def with_effort(row: dict[str, Any]) -> dict[str, Any]:
    row["configured_effort"] = choose_effort(row)
    return row


def allowed_labs(
    config_path: Path,
    override: list[str] | None,
    all_labs: bool,
    tiers: list[str] | None,
) -> set[str] | None:
    """Resolve the namespace allowlist from labs.yaml, or from an override."""
    if all_labs:
        return None
    if override:
        return set(override)
    config = load_config(config_path)
    groups = config["labs"]
    chosen = tiers or config.get("select") or list(groups)
    unknown = [t for t in chosen if t not in groups]
    if unknown:
        raise SystemExit(
            f"unknown lab tier(s) {unknown}; labs.yaml defines {sorted(groups)}"
        )
    return {lab for tier in chosen for lab in groups[tier]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labs", nargs="+", help="namespaces to keep, overriding labs.yaml")
    parser.add_argument("--all-labs", action="store_true", help="keep every namespace")
    parser.add_argument(
        "--tiers", nargs="+", help="lab tiers from labs.yaml, overriding its `select`"
    )
    parser.add_argument(
        "--no-enrich", action="store_true", help="skip the HuggingFace and Epoch lookups"
    )
    parser.add_argument(
        "--keep-serving-variants",
        action="store_true",
        help="keep :batch/:free/:nitro and -fast variants (dropped by default as duplicates)",
    )
    parser.add_argument(
        "--max-prompt-usd",
        type=float,
        default=DEFAULT_MAX_PROMPT_USD,
        help=f"drop models above this prompt price per Mtok (default {DEFAULT_MAX_PROMPT_USD}; "
             "0 disables)",
    )
    parser.add_argument(
        "--keep-non-text",
        action="store_true",
        help="keep image/audio-output models (dropped by default as off-task)",
    )
    parser.add_argument(
        "--keep-unusable",
        action="store_true",
        help="keep models known to be unmeasurable on this assay (see UNUSABLE_MODELS)",
    )
    parser.add_argument(
        "--keep-compute-variants",
        action="store_true",
        help="keep o1-pro/o3-pro/gpt-5.x-pro (dropped by default: same base, more compute)",
    )
    parser.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="drop models released before this date",
    )
    args = parser.parse_args()

    print(f"fetching {OPENROUTER_MODELS_URL}", file=sys.stderr)
    models = get_json(OPENROUTER_MODELS_URL)["data"]
    print(f"  {len(models)} models offered", file=sys.stderr)

    # Aliases retarget between runs, so they can never be part of a formal sweep.
    concrete = [m for m in models if not m["id"].startswith("~") and not m.get("alias_target")]
    print(f"  {len(models) - len(concrete)} moving aliases dropped", file=sys.stderr)

    if not args.keep_serving_variants:
        before = len(concrete)
        concrete = [
            m for m in concrete
            if (":" not in m["id"] or m["id"].split(":", 1)[1] not in SERVING_VARIANTS)
            and not m["id"].endswith(SERVING_SUFFIXES)
        ]
        print(f"  {before - len(concrete)} serving variants dropped "
              f"({'/'.join(sorted(SERVING_VARIANTS))}, {'/'.join(SERVING_SUFFIXES)})",
              file=sys.stderr)

    if not args.keep_non_text:
        before = len(concrete)
        concrete = [m for m in concrete if is_text_model(m)]
        print(f"  {before - len(concrete)} image/audio-output models dropped (off-task)",
              file=sys.stderr)

    keep = allowed_labs(
        REPO_ROOT / "configs" / "models" / "labs.yaml", args.labs, args.all_labs, args.tiers
    )
    selected = [m for m in concrete if keep is None or m["id"].split("/")[0] in keep]
    print(f"  {len(selected)} models from {len(keep) if keep else 'all'} allowlisted labs",
          file=sys.stderr)

    if args.since:
        before = len(selected)
        selected = [
            m for m in selected
            if m.get("created")
            and datetime.fromtimestamp(m["created"], timezone.utc).date().isoformat() >= args.since
        ]
        print(f"  {before - len(selected)} models released before {args.since} dropped",
              file=sys.stderr)

    epoch = {} if args.no_enrich else load_epoch_params()
    if epoch:
        print(f"  epoch table: {len(epoch)} models with parameter counts", file=sys.stderr)

    hf_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for index, model in enumerate(sorted(selected, key=lambda m: m["id"]), 1):
        params = resolve_parameters(model, epoch, hf_cache, enrich=not args.no_enrich)
        rows.append(with_effort(build_row(model, params)))
        if index % 25 == 0:
            print(f"    {index}/{len(selected)}", file=sys.stderr)

    if not args.keep_unusable:
        before = len(rows)
        hit = [r for r in rows if r["id"] in UNUSABLE_MODELS]
        rows = [r for r in rows if r["id"] not in UNUSABLE_MODELS]
        if hit:
            print(f"\n  {before - len(rows)} unmeasurable models dropped:", file=sys.stderr)
            for r in hit:
                print(f"    {r['id']}\n      {UNUSABLE_MODELS[r['id']]}", file=sys.stderr)

    before = len(rows)
    hit = [r for r in rows if r["id"] in NOT_SINGLE_MODEL]
    rows = [r for r in rows if r["id"] not in NOT_SINGLE_MODEL]
    if hit:
        print(f"\n  {before - len(rows)} non-single-model endpoints dropped:", file=sys.stderr)
        for r in hit:
            print(f"    {r['id']}\n      {NOT_SINGLE_MODEL[r['id']]}", file=sys.stderr)

    if not args.keep_compute_variants:
        before = len(rows)
        rows = [r for r in rows if r["id"] not in EXPENSIVE_COMPUTE_VARIANTS]
        dropped = before - len(rows)
        if dropped:
            print(f"\n  {dropped} test-time-compute variants dropped "
                  f"(same base model, {'/'.join(sorted(i.split('/')[-1] for i in EXPENSIVE_COMPUTE_VARIANTS))})",
                  file=sys.stderr)

    if args.max_prompt_usd:
        priced_out = [
            r for r in rows
            if r["prompt_usd_per_mtok"] is not None
            and r["prompt_usd_per_mtok"] >= args.max_prompt_usd
        ]
        rows = [r for r in rows if r not in priced_out]
        if priced_out:
            print(f"\n  {len(priced_out)} models over ${args.max_prompt_usd:g}/Mtok dropped:",
                  file=sys.stderr)
            for r in sorted(priced_out, key=lambda r: -r["prompt_usd_per_mtok"]):
                print(f"    ${r['prompt_usd_per_mtok']:>8,.2f}/Mtok  {r['id']}", file=sys.stderr)

    rows.sort(key=lambda r: (r["lab"], r["release_date"] or "", r["id"]))
    write_outputs(rows, keep)
    return 0


def write_outputs(rows: list[dict[str, Any]], labs: set[str] | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    (OUT_DIR / "openrouter_models.json").write_text(
        json.dumps({"fetched_at_utc": stamp, "labs": sorted(labs) if labs else None,
                    "model_count": len(rows), "models": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = OUT_DIR / "openrouter_models.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # The roster carries only what a run needs; the full table stays in results/.
    lines = [
        "# OpenRouter roster for formal runs -- GENERATED, do not edit by hand.",
        f"#   python scripts/fetch_openrouter_models.py     (fetched {stamp})",
        "#",
        f"# {len(rows)} models from the labs allowlisted in configs/models/labs.yaml.",
        "# Moving aliases (~openai/gpt-latest and friends) are excluded: they would",
        "# retarget between runs.",
        "#",
        "# The full table -- release date, parameter counts, price, context window,",
        "# reasoning support -- is results/model_catalogue/openrouter_models.csv.",
        "",
        "models:",
    ]
    current = None
    n_entries = 0
    for row in rows:
        if row["lab"] != current:
            current = row["lab"]
            lines.append(f"\n  # --- {current} ---")
        note = []
        if row["release_date"]:
            note.append(row["release_date"])
        if row["total_params_b"]:
            note.append(
                f"{row['total_params_b']:g}B"
                + (f"/{row['active_params_b']:g}B active" if row["architecture_class"] == "MoE" else "")
            )
        if row["prompt_usd_per_mtok"] is not None:
            note.append(f"${row['prompt_usd_per_mtok']:g}/${row['completion_usd_per_mtok']:g} per Mtok")
        for entry in roster_entries(row):
            n_entries += 1
            body = ", ".join(
                f"{k}: {json.dumps(v) if k == 'id' else json.dumps(v) if isinstance(v, bool) else v}"
                for k, v in entry.items()
            )
            lines.append(
                f"  - {{{body}}}" + (f"   # {'  '.join(note)}" if note else "")
            )
    lines.insert(3, f"# {n_entries} measurements over {len(rows)} models: a model whose reasoning")
    lines.insert(4, "# can be switched off contributes two, think and nothink.")

    OUT_ROSTER.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {OUT_ROSTER.relative_to(REPO_ROOT)}  ({len(rows)} models)", file=sys.stderr)
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(f"wrote {(OUT_DIR / 'openrouter_models.json').relative_to(REPO_ROOT)}", file=sys.stderr)

    efforts = collections.Counter(r["configured_effort"] for r in rows if r["configured_effort"])
    print(f"\nreasoning effort requested: "
          + ", ".join(f"{v} x {k}" for k, v in efforts.most_common())
          + f", {sum(1 for r in rows if not r['configured_effort'])} x none",
          file=sys.stderr)

    have_params = sum(1 for r in rows if r["total_params_b"])
    have_price = sum(1 for r in rows if r["prompt_usd_per_mtok"] is not None)
    have_date = sum(1 for r in rows if r["release_date"])
    print(
        f"\ncoverage: {have_date}/{len(rows)} dates, {have_params}/{len(rows)} parameter counts, "
        f"{have_price}/{len(rows)} prices",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
