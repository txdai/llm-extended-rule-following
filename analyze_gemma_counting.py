#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoProcessor, Gemma3ForConditionalGeneration



INTEGER_PATTERN = re.compile(r"-?\d+")
SYSTEM_TEXT = (
    "You count items exactly. Return only one integer with no words, punctuation, "
    "or explanation."
)

# Explicit extended subset, matching the local counting-extended Gemma Scope 2 download.
HOOK_SPECS = {
    "resid_post.layer16": {
        "config": Path("resid_post/layer_16_width_65k_l0_medium"),
        "hook_name": "model.language_model.layers.16",
        "mode": "forward",
        "hidden_size": 5376,
    },
    "resid_post.layer31": {
        "config": Path("resid_post/layer_31_width_65k_l0_medium"),
        "hook_name": "model.language_model.layers.31",
        "mode": "forward",
        "hidden_size": 5376,
    },
    "resid_post.layer40": {
        "config": Path("resid_post/layer_40_width_65k_l0_medium"),
        "hook_name": "model.language_model.layers.40",
        "mode": "forward",
        "hidden_size": 5376,
    },
    "resid_post.layer53": {
        "config": Path("resid_post/layer_53_width_65k_l0_medium"),
        "hook_name": "model.language_model.layers.53",
        "mode": "forward",
        "hidden_size": 5376,
    },
    "attn_out.layer31": {
        "config": Path("attn_out/layer_31_width_65k_l0_medium"),
        "hook_name": "model.language_model.layers.31.self_attn.o_proj",
        "mode": "forward_pre",
        "hidden_size": 4096,
    },
    "attn_out.layer40": {
        "config": Path("attn_out/layer_40_width_65k_l0_medium"),
        "hook_name": "model.language_model.layers.40.self_attn.o_proj",
        "mode": "forward_pre",
        "hidden_size": 4096,
    },
    "mlp_out.layer31": {
        "config": Path("mlp_out/layer_31_width_65k_l0_medium"),
        "hook_name": "model.language_model.layers.31.post_feedforward_layernorm",
        "mode": "forward",
        "hidden_size": 5376,
    },
    "mlp_out.layer40": {
        "config": Path("mlp_out/layer_40_width_65k_l0_medium"),
        "hook_name": "model.language_model.layers.40.post_feedforward_layernorm",
        "mode": "forward",
        "hidden_size": 5376,
    },
}

HOOK_ORDER = [
    "resid_post.layer16",
    "resid_post.layer31",
    "resid_post.layer40",
    "resid_post.layer53",
    "attn_out.layer31",
    "attn_out.layer40",
    "mlp_out.layer31",
    "mlp_out.layer40",
]
RESIDUAL_HOOKS = [
    "resid_post.layer16",
    "resid_post.layer31",
    "resid_post.layer40",
    "resid_post.layer53",
]
ATTN_LAYERS = [31, 40]

ATTN_BINS = 48
ATTN_REGION_WINDOW = 16
TOP_FEATURE_SAVE_K = 64
FEATURE_STRUCTURE_K = 6
FEATURE_RANK_MAX = 26
POSITION_WINDOW = 8
POSITION_OFFSETS = list(range(-(POSITION_WINDOW - 1), 1))
ATTN_MAIN_LAYER = 40
STEERING_COUNTS = [24, 26, 27, 32, 60, 100]
STEERING_ALPHA_STDS = [-1.0, -0.5, 0.5, 1.0]


@dataclass
class SweepRecord:
    count: int
    prompt_tokens: int
    prediction_text: str
    parsed_prediction: int | None
    exact: bool
    absolute_error: float | None
    min_correct_logit_gap: float | None
    mean_correct_logit_gap: float | None
    first_divergence_logit_gap: float | None


class GemmaScopeSAE:
    def __init__(self, root: Path, device: str) -> None:
        self.root = root
        self.device = device
        config = json.loads((root / "config.json").read_text())
        self.width = int(config["width"])
        with safe_open(root / "params.safetensors", framework="pt", device="cpu") as handle:
            self.b_dec = handle.get_tensor("b_dec").to(device=device)
            self.b_enc = handle.get_tensor("b_enc").to(device=device)
            self.threshold = handle.get_tensor("threshold").to(device=device)
            self.w_dec = handle.get_tensor("w_dec").to(device=device)
            self.w_enc = handle.get_tensor("w_enc").to(device=device)
        self.hidden_size = int(self.w_enc.shape[0])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.w_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))


class ActivationRecorder:
    def __init__(self, model: torch.nn.Module):
        self.module_map = dict(model.named_modules())
        self.cache: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def _extract_tensor(self, value: object) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, tuple):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item
        raise TypeError(f"Could not extract tensor from hook value of type {type(value)!r}")

    def _forward_hook(self, label: str) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            self.cache[label] = self._extract_tensor(output)

        return hook

    def _forward_pre_hook(self, label: str) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
            self.cache[label] = self._extract_tensor(inputs)

        return hook

    def register(self) -> None:
        for label in HOOK_ORDER:
            spec = HOOK_SPECS[label]
            module = self.module_map[str(spec["hook_name"])]
            if spec["mode"] == "forward_pre":
                handle = module.register_forward_pre_hook(self._forward_pre_hook(label))
            else:
                handle = module.register_forward_hook(self._forward_hook(label))
            self.handles.append(handle)

    def clear(self) -> None:
        self.cache.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.cache.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extended Gemma Scope 2 counting analysis on Gemma 3 27B IT."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/gemma-3-27b-it"),
        help="Local Gemma 3 27B IT path.",
    )
    parser.add_argument(
        "--scope-dir",
        type=Path,
        default=Path("model/gemma-scope-2-27b-it"),
        help="Local Gemma Scope 2 subset path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gemma_counting_mechanistic_analysis"),
        help="Directory for analysis payloads.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=160,
        help="Dense count sweep upper bound.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation and activation collection.",
    )
    parser.add_argument(
        "--attn-counts",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit attention snapshot counts. Otherwise derived from the run summary.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use. Defaults to auto -> mps, cuda, then cpu.",
    )
    parser.add_argument(
        "--feature-rank-max",
        type=int,
        default=FEATURE_RANK_MAX,
        help="Rank sparse features on counts 1..N, then record their full trajectories over the entire sweep.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stream_and_layer(label: str) -> tuple[str, int]:
    stream, layer_str = label.split(".layer")
    return stream, int(layer_str)


def safe_label(label: str) -> str:
    return label.replace(".", "_")


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_integer(text: str) -> int | None:
    match = INTEGER_PATTERN.search(text.strip())
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def make_sequence(count: int) -> str:
    return ", ".join("a" for _ in range(count))


def build_messages(count: int) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_TEXT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f'How many "a" are in this sequence?\n\n{make_sequence(count)}',
                }
            ],
        },
    ]


def batched(items: list[int] | list[str], batch_size: int) -> list[list[object]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def render_prompts(processor: AutoProcessor, counts: list[int]) -> list[str]:
    prompts: list[str] = []
    for count in counts:
        prompt = processor.apply_chat_template(
            build_messages(count),
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt, str):
            raise TypeError(f"Expected string prompt, got {type(prompt)!r}")
        prompts.append(prompt)
    return prompts


def tokenize_prompts(tokenizer, prompts: list[str], device: str) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def continuation_token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def top_other_logit(logits: torch.Tensor, token_id: int) -> float:
    top_values, top_indices = torch.topk(logits, k=2)
    if int(top_indices[0]) == token_id:
        return float(top_values[1])
    return float(top_values[0])


def first_divergence_index(correct_ids: list[int], predicted_ids: list[int]) -> int | None:
    limit = min(len(correct_ids), len(predicted_ids))
    for index in range(limit):
        if correct_ids[index] != predicted_ids[index]:
            return index
    if len(correct_ids) != len(predicted_ids):
        return limit
    return None


def teacher_forced_margin_batch(
    model: Gemma3ForConditionalGeneration,
    tokenizer,
    prompts: list[str],
    correct_texts: list[str],
    predicted_texts: list[str],
    device: str,
) -> list[dict[str, float | None]]:
    prompt_id_rows = [continuation_token_ids(tokenizer, prompt) for prompt in prompts]
    correct_id_rows = [continuation_token_ids(tokenizer, text) for text in correct_texts]
    predicted_id_rows = [continuation_token_ids(tokenizer, text) for text in predicted_texts]

    combined_rows = [
        prompt_ids + correct_ids
        for prompt_ids, correct_ids in zip(prompt_id_rows, correct_id_rows)
    ]
    max_len = max(len(row) for row in combined_rows)
    input_ids = torch.full(
        (len(combined_rows), max_len),
        fill_value=int(tokenizer.pad_token_id or 0),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((len(combined_rows), max_len), dtype=torch.long, device=device)
    for index, row in enumerate(combined_rows):
        row_tensor = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[index, : len(row)] = row_tensor
        attention_mask[index, : len(row)] = 1

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits.detach().float().cpu()

    rows: list[dict[str, float | None]] = []
    for index, (prompt_ids, correct_ids, predicted_ids) in enumerate(
        zip(prompt_id_rows, correct_id_rows, predicted_id_rows)
    ):
        token_margins: list[float] = []
        for step, correct_token_id in enumerate(correct_ids):
            logit_row = logits[index, len(prompt_ids) - 1 + step]
            correct_logit = float(logit_row[correct_token_id])
            token_margins.append(correct_logit - top_other_logit(logit_row, correct_token_id))

        divergence_gap = None
        divergence_index = first_divergence_index(correct_ids, predicted_ids)
        if divergence_index is not None and divergence_index < len(correct_ids) and divergence_index < len(predicted_ids):
            logit_row = logits[index, len(prompt_ids) - 1 + divergence_index]
            divergence_gap = float(logit_row[correct_ids[divergence_index]] - logit_row[predicted_ids[divergence_index]])

        rows.append(
            {
                "min_correct_logit_gap": float(min(token_margins)) if token_margins else None,
                "mean_correct_logit_gap": float(np.mean(token_margins)) if token_margins else None,
                "first_divergence_logit_gap": divergence_gap,
            }
        )
    return rows


def run_generation_sweep(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer,
    counts: list[int],
    batch_size: int,
    device: str,
) -> list[SweepRecord]:
    prompts = render_prompts(processor, counts)
    records: list[SweepRecord] = []
    for count_batch, prompt_batch in zip(
        batched(counts, batch_size),
        batched(prompts, batch_size),
    ):
        prompt_list = [str(item) for item in prompt_batch]
        count_list = [int(item) for item in count_batch]
        encoded = tokenize_prompts(tokenizer, prompt_list, device)
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=12,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        sequences = generated.sequences
        predicted_texts: list[str] = []
        parsed_predictions: list[int | None] = []
        for index, count in enumerate(count_list):
            continuation = sequences[index, int(input_lengths[index]) :]
            text = processor.decode(continuation, skip_special_tokens=True).strip()
            parsed = parse_integer(text)
            predicted_texts.append(str(parsed) if parsed is not None else text)
            parsed_predictions.append(parsed)
        margin_rows = teacher_forced_margin_batch(
            model,
            tokenizer,
            prompt_list,
            [str(count) for count in count_list],
            predicted_texts,
            device,
        )
        for index, count in enumerate(count_list):
            text = predicted_texts[index] if parsed_predictions[index] is not None else predicted_texts[index]
            parsed = parsed_predictions[index]
            raw_text = processor.decode(sequences[index, int(input_lengths[index]) :], skip_special_tokens=True).strip()
            records.append(
                SweepRecord(
                    count=count,
                    prompt_tokens=int(input_lengths[index]),
                    prediction_text=raw_text,
                    parsed_prediction=parsed,
                    exact=(parsed == count),
                    absolute_error=None if parsed is None else float(abs(parsed - count)),
                    min_correct_logit_gap=margin_rows[index]["min_correct_logit_gap"],
                    mean_correct_logit_gap=margin_rows[index]["mean_correct_logit_gap"],
                    first_divergence_logit_gap=margin_rows[index]["first_divergence_logit_gap"],
                )
            )
    return records


def collect_raw_activations(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer,
    counts: list[int],
    batch_size: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    prompts = render_prompts(processor, counts)
    recorder = ActivationRecorder(model)
    recorder.register()
    final_outputs: dict[str, list[np.ndarray]] = {label: [] for label in HOOK_ORDER}
    position_windows: dict[str, list[np.ndarray]] = {label: [] for label in RESIDUAL_HOOKS}
    gather_back = torch.tensor(
        list(range(POSITION_WINDOW - 1, -1, -1)),
        device=device,
        dtype=torch.long,
    )
    try:
        for prompt_batch in batched(prompts, batch_size):
            prompt_list = [str(item) for item in prompt_batch]
            encoded = tokenize_prompts(tokenizer, prompt_list, device)
            last_positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(last_positions.shape[0], device=device)
            recorder.clear()
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            for label in HOOK_ORDER:
                tensor = recorder.cache[label]
                expected_hidden = int(HOOK_SPECS[label]["hidden_size"])
                if int(tensor.shape[-1]) != expected_hidden:
                    raise ValueError(
                        f"{label} produced hidden size {tensor.shape[-1]}, expected {expected_hidden}"
                    )
                final_tensor = tensor[batch_indices, last_positions]
                final_outputs[label].append(final_tensor.detach().float().cpu().numpy())
                if label in RESIDUAL_HOOKS:
                    positions = (last_positions.unsqueeze(1) - gather_back.unsqueeze(0)).clamp(min=0)
                    window_tensor = tensor[batch_indices.unsqueeze(1), positions]
                    position_windows[label].append(window_tensor.detach().float().cpu().numpy())
    finally:
        recorder.close()
    final_arrays = {
        label: np.concatenate(chunks, axis=0) for label, chunks in final_outputs.items()
    }
    position_arrays = {
        label: np.concatenate(chunks, axis=0) for label, chunks in position_windows.items()
    }
    return final_arrays, position_arrays


def encode_hook_features(
    sae: GemmaScopeSAE,
    activations: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, activations.shape[0], batch_size):
        stop = start + batch_size
        batch = torch.from_numpy(activations[start:stop]).to(device=device)
        with torch.inference_mode():
            encoded = sae.encode(batch)
        chunks.append(encoded.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_centered = x.astype(np.float32) - float(np.mean(x))
    y_centered = y.astype(np.float32) - float(np.mean(y))
    denom = math.sqrt(float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denom < 1e-8:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denom)


def pearson_by_feature(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = features - features.mean(axis=0, keepdims=True)
    y = target.astype(np.float32) - float(target.mean())
    numerator = (x * y[:, None]).sum(axis=0)
    denominator = np.sqrt((x * x).sum(axis=0) * float((y * y).sum()))
    denominator = np.where(denominator < 1e-8, np.inf, denominator)
    return numerator / denominator


def monotonicity_score(series: np.ndarray) -> float:
    if series.size < 2:
        return float("nan")
    diffs = np.diff(series.astype(np.float32))
    sign = 1.0 if float(series[-1] - series[0]) >= 0.0 else -1.0
    return float(np.mean(sign * diffs >= 0.0))


def top_feature_table(
    features: np.ndarray,
    counts: np.ndarray,
    top_k: int,
    rank_mask: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    if rank_mask is None:
        rank_mask = np.ones(features.shape[0], dtype=bool)
    rank_counts = counts[rank_mask].astype(np.float32)
    rank_features = features[rank_mask]
    rank_correlations = pearson_by_feature(rank_features, rank_counts)
    full_correlations = pearson_by_feature(features, counts.astype(np.float32))
    rank_var = max(float(np.var(rank_counts)), 1e-8)
    full_var = max(float(np.var(counts.astype(np.float32))), 1e-8)
    order = np.argsort(-np.abs(rank_correlations))[:top_k]
    rows: list[dict[str, float | int]] = []
    for feature_id in order.tolist():
        activations = features[:, feature_id]
        rank_activations = rank_features[:, feature_id]
        rank_slope = float(
            np.cov(rank_counts, rank_activations.astype(np.float32), bias=True)[0, 1] / rank_var
        )
        full_slope = float(
            np.cov(counts.astype(np.float32), activations.astype(np.float32), bias=True)[0, 1] / full_var
        )
        rows.append(
            {
                "feature_id": int(feature_id),
                "correlation": float(rank_correlations[feature_id]),
                "rank_window_correlation": float(rank_correlations[feature_id]),
                "full_correlation": float(full_correlations[feature_id]),
                "mean_activation": float(activations.mean()),
                "active_rate": float((activations > 0).mean()),
                "rank_window_mean_activation": float(rank_activations.mean()),
                "rank_window_active_rate": float((rank_activations > 0).mean()),
                "slope": rank_slope,
                "slope_sign": int(1 if rank_slope >= 0 else -1),
                "full_slope": full_slope,
                "monotonicity": monotonicity_score(rank_activations),
                "full_monotonicity": monotonicity_score(activations),
            }
        )
    return rows


def fit_principal_components(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    explained = (s * s) / max(float((s * s).sum()), 1e-8)
    return coords.astype(np.float32), explained[:n_components].astype(np.float32)


def fit_counter_direction(
    x: np.ndarray,
    counts: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    fit_x = x[fit_mask]
    fit_y = counts[fit_mask].astype(np.float32)
    center = fit_x.mean(axis=0, keepdims=True)
    centered_x = fit_x - center
    centered_y = fit_y - float(fit_y.mean())
    direction, *_ = np.linalg.lstsq(centered_x, centered_y, rcond=None)
    projected = (x - center) @ direction
    z0 = projected[:-1]
    z1 = projected[1:]
    design = np.column_stack([z0, np.ones_like(z0)])
    rho, delta = np.linalg.lstsq(design, z1, rcond=None)[0]
    return (
        center.astype(np.float32),
        direction.astype(np.float32),
        projected.astype(np.float32),
        float(rho),
        float(delta),
    )


def project_position_window(
    position_window: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    centered = position_window - center.reshape(1, 1, -1)
    return np.tensordot(centered, direction, axes=([2], [0])).astype(np.float32)


def bin_matrix(matrix: np.ndarray, bins: int = ATTN_BINS) -> np.ndarray:
    row_edges = np.linspace(0, matrix.shape[0], bins + 1, dtype=int)
    col_edges = np.linspace(0, matrix.shape[1], bins + 1, dtype=int)
    binned = np.zeros((bins, bins), dtype=np.float32)
    for i in range(bins):
        row_slice = slice(row_edges[i], row_edges[i + 1])
        for j in range(bins):
            col_slice = slice(col_edges[j], col_edges[j + 1])
            block = matrix[row_slice, col_slice]
            binned[i, j] = float(block.mean()) if block.size else 0.0
    return binned


def renormalize_attention_without_first_key(matrix: np.ndarray) -> np.ndarray:
    renorm = matrix.astype(np.float32, copy=True)
    if renorm.shape[1] == 0:
        return renorm
    renorm[:, 0] = 0.0
    row_sums = renorm.sum(axis=1, keepdims=True)
    return np.divide(
        renorm,
        np.clip(row_sums, 1e-8, None),
        out=np.zeros_like(renorm),
        where=row_sums > 1e-8,
    )


def summarize_records(records: list[SweepRecord]) -> dict[str, object]:
    failed = [record for record in records if not record.exact]
    parsed_failures = [record.parsed_prediction for record in failed if record.parsed_prediction is not None]
    attractors = Counter(parsed_failures).most_common(8)
    first_failure = next((record.count for record in records if not record.exact), None)
    initial_exact_prefix = 0
    for record in records:
        if not record.exact:
            break
        initial_exact_prefix = record.count
    exact_counts = [record.count for record in records if record.exact]
    late_exact_pockets = [count for count in exact_counts if first_failure is not None and count > first_failure]
    return {
        "initial_exact_prefix": initial_exact_prefix,
        "first_failure": first_failure,
        "max_exact_count": max(exact_counts, default=0),
        "late_exact_pockets": late_exact_pockets,
        "accuracy": float(sum(record.exact for record in records) / max(len(records), 1)),
        "top_failed_attractors": [[int(value), int(freq)] for value, freq in attractors],
    }


def derive_attention_counts(
    run_summary: dict[str, object],
    max_count: int,
    explicit_counts: list[int] | None,
) -> list[int]:
    if explicit_counts:
        return sorted({count for count in explicit_counts if 1 <= count <= max_count})
    candidates = [
        8,
        int(run_summary["initial_exact_prefix"]),
        int(run_summary["first_failure"] or 0),
        32,
        64,
        96,
        128,
        max_count,
    ]
    counts = sorted({count for count in candidates if 1 <= count <= max_count})
    return counts


def sequence_token_masks(prompt: str, sequence: str, tokenizer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = prompt.rfind(sequence)
    if start < 0:
        raise ValueError("Could not locate repeated sequence inside rendered prompt")
    end = start + len(sequence)
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    seq_mask = np.zeros(len(offsets), dtype=bool)
    item_mask = np.zeros(len(offsets), dtype=bool)
    sep_mask = np.zeros(len(offsets), dtype=bool)
    for index, (left, right) in enumerate(offsets):
        if right <= start or left >= end:
            continue
        seq_mask[index] = True
        snippet = prompt[max(left, start) : min(right, end)]
        item_mask[index] = "a" in snippet
        sep_mask[index] = "," in snippet
    return seq_mask, item_mask, sep_mask


def attention_region_indices(sequence_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seq_indices = np.flatnonzero(sequence_mask)
    if seq_indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    tail = seq_indices[-ATTN_REGION_WINDOW:]
    center = seq_indices[
        max(0, (seq_indices.size // 2) - (ATTN_REGION_WINDOW // 2)) :
        max(0, (seq_indices.size // 2) - (ATTN_REGION_WINDOW // 2)) + ATTN_REGION_WINDOW
    ]
    return center.astype(int), tail.astype(int)


def head_attention_metrics(
    attention: np.ndarray,
    sequence_mask: np.ndarray,
    item_mask: np.ndarray,
    separator_mask: np.ndarray,
    count: int,
    layer: int,
) -> list[dict[str, float | int]]:
    final_attention = attention[:, -1, :]
    middle_indices, tail_indices = attention_region_indices(sequence_mask)
    rows: list[dict[str, float | int]] = []
    for head_index in range(final_attention.shape[0]):
        values = final_attention[head_index].astype(np.float32)
        entropy = float(-np.sum(values * np.log(np.clip(values, 1e-8, None))) / math.log(max(len(values), 2)))
        rows.append(
            {
                "count": count,
                "layer": layer,
                "head": head_index,
                "seq_len": int(values.shape[0]),
                "final_to_first_key": float(values[0]),
                "final_to_sequence": float(values[sequence_mask].sum()) if sequence_mask.any() else 0.0,
                "final_to_item_tokens": float(values[item_mask].sum()) if item_mask.any() else 0.0,
                "final_to_separator_tokens": float(values[separator_mask].sum()) if separator_mask.any() else 0.0,
                "final_to_middle16": float(values[middle_indices].sum()) if middle_indices.size else 0.0,
                "final_to_tail16": float(values[tail_indices].sum()) if tail_indices.size else 0.0,
                "final_entropy": entropy,
                "final_top1_mass": float(values.max()),
            }
        )
    return rows


def capture_attention_head_metrics(
    model_path: Path,
    processor: AutoProcessor,
    tokenizer,
    counts: list[int],
    device: str,
) -> dict[int, list[dict[str, float | int]]]:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to(device)
    metrics_by_layer: dict[int, list[dict[str, float | int]]] = {layer: [] for layer in ATTN_LAYERS}
    try:
        prompts = render_prompts(processor, counts)
        for count, prompt in zip(counts, prompts):
            encoded = tokenize_prompts(tokenizer, [prompt], device)
            with torch.inference_mode():
                outputs = model(**encoded, use_cache=False, output_attentions=True)
            sequence = make_sequence(count)
            seq_mask, item_mask, sep_mask = sequence_token_masks(prompt, sequence, tokenizer)
            for layer in ATTN_LAYERS:
                attention = outputs.attentions[layer][0].detach().float().cpu().numpy()
                metrics_by_layer[layer].extend(
                    head_attention_metrics(attention, seq_mask, item_mask, sep_mask, count, layer)
                )
    finally:
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()
    return metrics_by_layer


def summarize_attention_metrics(
    rows: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    by_count: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_keys = [
        "final_entropy",
        "final_to_first_key",
        "final_to_sequence",
        "final_to_item_tokens",
        "final_to_separator_tokens",
        "final_to_middle16",
        "final_to_tail16",
        "final_top1_mass",
    ]
    for row in rows:
        count = int(row["count"])
        for key in metric_keys:
            by_count[count][key].append(float(row[key]))
    summary_rows: list[dict[str, float | int]] = []
    for count in sorted(by_count):
        payload = {"count": count}
        for key in metric_keys:
            values = by_count[count][key]
            payload[f"mean_{key}"] = float(np.mean(values))
            payload[f"std_{key}"] = float(np.std(values))
        summary_rows.append(payload)
    return summary_rows


def select_representative_attention_heads(
    metrics_by_layer: dict[int, list[dict[str, float | int]]],
) -> dict[int, dict[str, int]]:
    selection: dict[int, dict[str, int]] = {}
    for layer, rows in metrics_by_layer.items():
        by_head: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            head = int(row["head"])
            by_head[head]["final_to_first_key"].append(float(row["final_to_first_key"]))
            by_head[head]["final_to_tail16"].append(float(row["final_to_tail16"]))
            by_head[head]["final_entropy"].append(float(row["final_entropy"]))
        scored: list[dict[str, float | int]] = []
        for head, metrics in by_head.items():
            mean_first = float(np.mean(metrics["final_to_first_key"]))
            mean_tail = float(np.mean(metrics["final_to_tail16"]))
            scored.append(
                {
                    "head": head,
                    "mean_first_key": mean_first,
                    "mean_tail16": mean_tail,
                    "mean_entropy": float(np.mean(metrics["final_entropy"])),
                    "local_recency_score": mean_tail - mean_first,
                }
            )
        first_key_head = int(max(scored, key=lambda row: float(row["mean_first_key"]))["head"])
        local_candidates = [row for row in scored if int(row["head"]) != first_key_head] or scored
        local_recency_head = int(
            max(local_candidates, key=lambda row: float(row["local_recency_score"]))["head"]
        )
        selection[layer] = {
            "first_key_head": first_key_head,
            "local_recency_head": local_recency_head,
        }
    return selection


def capture_attention_snapshots(
    model_path: Path,
    processor: AutoProcessor,
    tokenizer,
    counts: list[int],
    device: str,
    selected_heads: dict[int, dict[str, int]],
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[str, int, int], np.ndarray]]:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to(device)
    average_snapshots: dict[tuple[int, int], np.ndarray] = {}
    selected_snapshots: dict[tuple[str, int, int], np.ndarray] = {}
    try:
        prompts = render_prompts(processor, counts)
        for count, prompt in zip(counts, prompts):
            encoded = tokenize_prompts(tokenizer, [prompt], device)
            with torch.inference_mode():
                outputs = model(**encoded, use_cache=False, output_attentions=True)
            for layer in ATTN_LAYERS:
                attention = outputs.attentions[layer][0].detach().float().cpu().numpy()
                average_snapshots[(layer, count)] = attention.mean(axis=0)
                for role, head in selected_heads[layer].items():
                    selected_snapshots[(role, layer, count)] = attention[int(head)]
    finally:
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()
    return average_snapshots, selected_snapshots


def build_binned_attention_payload(
    average_snapshots: dict[tuple[int, int], np.ndarray],
    selected_snapshots: dict[tuple[str, int, int], np.ndarray],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for (layer, count), matrix in average_snapshots.items():
        payload[f"avg_layer{layer}_count{count}"] = bin_matrix(matrix, bins=ATTN_BINS).astype(np.float32)
        payload[f"masked_avg_layer{layer}_count{count}"] = bin_matrix(
            renormalize_attention_without_first_key(matrix),
            bins=ATTN_BINS,
        ).astype(np.float32)
    for (role, layer, count), matrix in selected_snapshots.items():
        payload[f"head_{role}_layer{layer}_count{count}"] = bin_matrix(matrix, bins=ATTN_BINS).astype(np.float32)
        payload[f"masked_head_{role}_layer{layer}_count{count}"] = bin_matrix(
            renormalize_attention_without_first_key(matrix),
            bins=ATTN_BINS,
        ).astype(np.float32)
    return payload


def save_attention_payload(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def load_attention_payload(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def extract_average_attention_matrices(
    payload: dict[str, np.ndarray],
    masked: bool,
) -> dict[tuple[int, int], np.ndarray]:
    matrices: dict[tuple[int, int], np.ndarray] = {}
    prefix = "masked_avg_layer" if masked else "avg_layer"
    for key, matrix in payload.items():
        if not key.startswith(prefix):
            continue
        match = re.fullmatch(r"(?:masked_)?avg_layer(\d+)_count(\d+)", key)
        if not match:
            continue
        layer = int(match.group(1))
        count = int(match.group(2))
        matrices[(layer, count)] = matrix
    return matrices


def extract_selected_attention_matrices(
    payload: dict[str, np.ndarray],
    role: str,
    masked: bool,
) -> dict[tuple[int, int], np.ndarray]:
    matrices: dict[tuple[int, int], np.ndarray] = {}
    prefix = f"masked_head_{role}_layer" if masked else f"head_{role}_layer"
    for key, matrix in payload.items():
        if not key.startswith(prefix):
            continue
        match = re.fullmatch(rf"(?:masked_)?head_{role}_layer(\d+)_count(\d+)", key)
        if not match:
            continue
        layer = int(match.group(1))
        count = int(match.group(2))
        matrices[(layer, count)] = matrix
    return matrices


def run_steering_probe(
    model_path: Path,
    processor: AutoProcessor,
    tokenizer,
    device: str,
    residual_directions: dict[str, dict[str, np.ndarray | float]],
) -> list[dict[str, object]]:
    counts = [count for count in STEERING_COUNTS if count > 0]
    prompts = render_prompts(processor, counts)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    rows: list[dict[str, object]] = []
    try:
        baseline_predictions: dict[int, int | None] = {}
        for count, prompt in zip(counts, prompts):
            encoded = tokenize_prompts(tokenizer, [prompt], device)
            with torch.inference_mode():
                sequences = model.generate(**encoded, max_new_tokens=12, do_sample=False)
            continuation = sequences[0, int(encoded["attention_mask"][0].sum().item()) :]
            baseline_predictions[count] = parse_integer(processor.decode(continuation, skip_special_tokens=True).strip())

        for label in RESIDUAL_HOOKS:
            module_name = str(HOOK_SPECS[label]["hook_name"])
            unit_direction = np.asarray(residual_directions[label]["unit_direction"], dtype=np.float32)
            scale_std = float(residual_directions[label]["exact_prefix_std"])
            for alpha_std in STEERING_ALPHA_STDS:
                with ResidualSteerer(model, module_name, unit_direction, alpha_std * scale_std):
                    for count, prompt in zip(counts, prompts):
                        encoded = tokenize_prompts(tokenizer, [prompt], device)
                        with torch.inference_mode():
                            sequences = model.generate(**encoded, max_new_tokens=12, do_sample=False)
                        continuation = sequences[0, int(encoded["attention_mask"][0].sum().item()) :]
                        steered_prediction = parse_integer(
                            processor.decode(continuation, skip_special_tokens=True).strip()
                        )
                        baseline_prediction = baseline_predictions[count]
                        rows.append(
                            {
                                "label": label,
                                "count": count,
                                "alpha_std": float(alpha_std),
                                "baseline_prediction": baseline_prediction,
                                "steered_prediction": steered_prediction,
                                "prediction_shift": None
                                if baseline_prediction is None or steered_prediction is None
                                else int(steered_prediction - baseline_prediction),
                            }
                        )
    finally:
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def feature_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"feature_rankings_{safe_label(label)}.json"


def feature_trajectory_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"feature_activation_trajectories_{safe_label(label)}.npz"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    counts = list(range(1, args.max_count + 1))

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    print(f"Using device: {device}")
    print("Loading Gemma 3 27B IT for generation and activation sweeps...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    print("Running dense generation sweep...")
    records = run_generation_sweep(model, processor, tokenizer, counts, args.batch_size, device)
    run_summary = summarize_records(records)

    print("Collecting raw activations and last-window position payloads...")
    raw_activations, position_windows = collect_raw_activations(
        model,
        processor,
        tokenizer,
        counts,
        args.batch_size,
        device,
    )

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()

    counts_array = np.array(counts, dtype=np.int32)
    exact_mask = np.array([record.exact for record in records], dtype=bool)
    exact_prefix_mask = counts_array <= int(run_summary["initial_exact_prefix"])
    post_failure_mask = (
        counts_array >= int(run_summary["first_failure"])
        if run_summary["first_failure"] is not None
        else np.zeros_like(counts_array, dtype=bool)
    )
    late_exact_pockets = set(int(item) for item in run_summary["late_exact_pockets"])
    late_exact_mask = np.array([count in late_exact_pockets for count in counts_array], dtype=bool)
    feature_rank_mask = counts_array <= min(int(args.feature_rank_max), int(counts_array[-1]))
    save_json(output_dir / "count_sweep_records.json", [record.__dict__ for record in records])
    save_json(output_dir / "count_sweep_summary.json", run_summary)

    hook_leaderboard: list[dict[str, object]] = []
    stream_layer_summary: list[dict[str, object]] = []
    residual_counter_metrics: list[dict[str, object]] = []
    exact_failed_comparison: list[dict[str, object]] = []
    residual_direction_payload: dict[str, dict[str, np.ndarray | float]] = {}

    analysis_arrays: dict[str, np.ndarray] = {
        "counts": counts_array,
        "exact_mask": exact_mask,
        "exact_prefix_mask": exact_prefix_mask,
        "post_failure_mask": post_failure_mask,
    }
    position_arrays: dict[str, np.ndarray] = {
        "counts": counts_array,
        "position_offsets": np.array(POSITION_OFFSETS, dtype=np.int32),
    }

    for label in HOOK_ORDER:
        print(f"Encoding features for {label}...")
        safe = safe_label(label)
        stream, layer = stream_and_layer(label)

        sae = GemmaScopeSAE(args.scope_dir / Path(HOOK_SPECS[label]["config"]), device)
        encoded_features = encode_hook_features(
            sae,
            raw_activations[label],
            device,
            args.batch_size,
        )

        feature_rows = top_feature_table(
            encoded_features,
            counts_array,
            top_k=TOP_FEATURE_SAVE_K,
            rank_mask=feature_rank_mask,
        )
        save_json(feature_path(output_dir, label), feature_rows)

        best_row = feature_rows[0]
        top5_mean_abs = float(np.mean([abs(float(row["correlation"])) for row in feature_rows[:5]]))
        best_feature_values = encoded_features[:, int(best_row["feature_id"])]
        exact_vs_failed_sep = float(
            best_feature_values[exact_prefix_mask].mean() - best_feature_values[post_failure_mask].mean()
        ) if np.any(exact_prefix_mask) and np.any(post_failure_mask) else float("nan")

        hook_leaderboard.append(
            {
                "label": label,
                "stream": stream,
                "layer": layer,
                "best_feature_id": int(best_row["feature_id"]),
                "best_feature_pearson": float(best_row["correlation"]),
                "best_feature_mean_activation": float(best_row["mean_activation"]),
                "best_feature_active_rate": float(best_row["active_rate"]),
            }
        )

        exact_rows = (
            top_feature_table(encoded_features[exact_prefix_mask], counts_array[exact_prefix_mask], top_k=1)
            if np.sum(exact_prefix_mask) >= 2
            else []
        )
        post_rows = (
            top_feature_table(encoded_features[post_failure_mask], counts_array[post_failure_mask], top_k=1)
            if np.sum(post_failure_mask) >= 2
            else []
        )
        late_rows = (
            top_feature_table(encoded_features[late_exact_mask], counts_array[late_exact_mask], top_k=1)
            if np.sum(late_exact_mask) >= 2
            else []
        )
        exact_failed_comparison.append(
            {
                "label": label,
                "stream": stream,
                "layer": layer,
                "exact_prefix_best_feature_id": int(exact_rows[0]["feature_id"]) if exact_rows else None,
                "exact_prefix_best_feature_corr": float(exact_rows[0]["correlation"]) if exact_rows else None,
                "post_failure_best_feature_id": int(post_rows[0]["feature_id"]) if post_rows else None,
                "post_failure_best_feature_corr": float(post_rows[0]["correlation"]) if post_rows else None,
                "late_exact_best_feature_id": int(late_rows[0]["feature_id"]) if late_rows else None,
                "late_exact_best_feature_corr": float(late_rows[0]["correlation"]) if late_rows else None,
                "best_feature_exact_prefix_mean": float(best_feature_values[exact_prefix_mask].mean())
                if np.any(exact_prefix_mask)
                else None,
                "best_feature_post_failure_mean": float(best_feature_values[post_failure_mask].mean())
                if np.any(post_failure_mask)
                else None,
                "best_feature_separation": exact_vs_failed_sep,
            }
        )

        top_ids = [int(row["feature_id"]) for row in feature_rows[:TOP_FEATURE_SAVE_K]]
        top_matrix = encoded_features[:, top_ids].astype(np.float32)
        structure_ids = top_ids[:FEATURE_STRUCTURE_K]
        decoder = sae.w_dec[structure_ids].detach().float().cpu().numpy()
        decoder = decoder / np.clip(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-8, None)
        decoder_cosine = (decoder @ decoder.T).astype(np.float32)

        analysis_arrays[f"top_feature_matrix_{safe}"] = top_matrix
        analysis_arrays[f"top_feature_decoder_cosine_{safe}"] = decoder_cosine
        analysis_arrays[f"top_feature_ids_{safe}"] = np.array(top_ids, dtype=np.int32)
        analysis_arrays[f"top_feature_values64_{safe}"] = encoded_features[
            :, [int(row["feature_id"]) for row in feature_rows[:TOP_FEATURE_SAVE_K]]
        ].astype(np.float32)
        analysis_arrays[f"top_feature_ids64_{safe}"] = np.array(
            [int(row["feature_id"]) for row in feature_rows[:TOP_FEATURE_SAVE_K]],
            dtype=np.int32,
        )
        np.savez_compressed(
            feature_trajectory_path(output_dir, label),
            counts=counts_array.astype(np.int32),
            rank_mask=feature_rank_mask.astype(np.int8),
            feature_ids=np.array([int(row["feature_id"]) for row in feature_rows[:TOP_FEATURE_SAVE_K]], dtype=np.int32),
            rank_window_correlations=np.array(
                [float(row["rank_window_correlation"]) for row in feature_rows[:TOP_FEATURE_SAVE_K]],
                dtype=np.float32,
            ),
            full_correlations=np.array(
                [float(row["full_correlation"]) for row in feature_rows[:TOP_FEATURE_SAVE_K]],
                dtype=np.float32,
            ),
            values=encoded_features[:, [int(row["feature_id"]) for row in feature_rows[:TOP_FEATURE_SAVE_K]]].astype(np.float32),
        )
        counter_projection_corr = None
        if label in RESIDUAL_HOOKS:
            coords, explained = fit_principal_components(raw_activations[label])
            center, direction, projected, rho, delta = fit_counter_direction(
                raw_activations[label],
                counts_array,
                exact_prefix_mask,
            )
            unit_direction = direction / np.clip(np.linalg.norm(direction), 1e-8, None)
            exact_prefix_projection = ((raw_activations[label] - center) @ unit_direction).astype(np.float32)
            exact_prefix_std = float(np.std(exact_prefix_projection[exact_prefix_mask]))
            projection_count_corr = pearson(projected, counts_array.astype(np.float32))
            exact_prefix_projection_corr = pearson(
                projected[exact_prefix_mask],
                counts_array[exact_prefix_mask].astype(np.float32),
            )
            post_failure_projection_corr = pearson(
                projected[post_failure_mask],
                counts_array[post_failure_mask].astype(np.float32),
            ) if np.sum(post_failure_mask) >= 2 else float("nan")
            adjacent_exact = np.diff(projected[exact_prefix_mask])
            eta_projection_threshold = None
            if run_summary["first_failure"] is not None and int(run_summary["first_failure"]) >= 2:
                boundary_index = int(run_summary["first_failure"]) - 2
                if 0 <= boundary_index < projected.shape[0] - 1:
                    eta_projection_threshold = float(abs(np.diff(projected)[boundary_index]))
            counter_projection_corr = projection_count_corr
            residual_counter_metrics.append(
                {
                    "label": label,
                    "stream": stream,
                    "layer": layer,
                    "rho": float(rho),
                    "delta": float(delta),
                    "projection_count_corr": float(projection_count_corr),
                    "projection_monotonicity": float(np.mean(adjacent_exact > 0.0)) if adjacent_exact.size else None,
                    "adjacent_delta_mean": float(adjacent_exact.mean()) if adjacent_exact.size else None,
                    "adjacent_delta_std": float(adjacent_exact.std()) if adjacent_exact.size else None,
                    "exact_prefix_projection_corr": float(exact_prefix_projection_corr),
                    "post_failure_projection_corr": float(post_failure_projection_corr),
                    "eta_projection_threshold": eta_projection_threshold,
                    "exact_prefix_projection_std": exact_prefix_std,
                }
            )
            analysis_arrays[f"pca_coords_{safe}"] = coords
            analysis_arrays[f"pca_explained_{safe}"] = explained
            analysis_arrays[f"projected_counter_{safe}"] = projected.astype(np.float32)
            analysis_arrays[f"counter_center_{safe}"] = center.astype(np.float32)
            analysis_arrays[f"counter_direction_unit_{safe}"] = unit_direction.astype(np.float32)
            analysis_arrays[f"counter_exact_prefix_std_{safe}"] = np.array([exact_prefix_std], dtype=np.float32)
            projected_window = project_position_window(position_windows[label], center, direction)
            position_arrays[f"projected_position_window_{safe}"] = projected_window
            residual_direction_payload[label] = {
                "unit_direction": unit_direction.astype(np.float32),
                "exact_prefix_std": exact_prefix_std,
            }

        stream_layer_summary.append(
            {
                "label": label,
                "stream": stream,
                "layer": layer,
                "best_feature_corr": float(best_row["correlation"]),
                "top5_mean_abs_corr": top5_mean_abs,
                "counter_projection_corr": counter_projection_corr,
                "exact_vs_failed_separation": exact_vs_failed_sep,
            }
        )

        del encoded_features
        del sae
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()

    hook_leaderboard_sorted = sorted(
        hook_leaderboard,
        key=lambda row: abs(float(row["best_feature_pearson"])),
        reverse=True,
    )
    save_json(output_dir / "hook_feature_leaderboard.json", hook_leaderboard_sorted)
    write_csv(output_dir / "hook_feature_leaderboard.csv", hook_leaderboard_sorted)
    save_json(output_dir / "layer_stream_summary.json", stream_layer_summary)
    save_json(output_dir / "residual_count_fit_metrics.json", residual_counter_metrics)
    save_json(output_dir / "exact_vs_failed_feature_comparison.json", exact_failed_comparison)
    np.savez_compressed(output_dir / "residual_analysis_arrays.npz", **analysis_arrays)
    np.savez_compressed(output_dir / "position_window_payload.npz", **position_arrays)
    attention_counts = derive_attention_counts(run_summary, args.max_count, args.attn_counts)
    print(f"Capturing attention head metrics at counts {attention_counts}...")
    attention_head_metrics = capture_attention_head_metrics(
        args.model_path,
        processor,
        tokenizer,
        attention_counts,
        device,
    )
    attention_metric_summary = {
        layer: summarize_attention_metrics(rows) for layer, rows in attention_head_metrics.items()
    }
    attention_head_selection = select_representative_attention_heads(attention_head_metrics)
    save_json(output_dir / "attention_head_metric_summary.json", attention_metric_summary)
    save_json(output_dir / "selected_attention_heads.json", attention_head_selection)
    print(f"Capturing average and selected-head attention snapshots at counts {attention_counts}...")
    average_snapshots, selected_snapshots = capture_attention_snapshots(
        args.model_path,
        processor,
        tokenizer,
        attention_counts,
        device,
        attention_head_selection,
    )
    attention_payload = build_binned_attention_payload(average_snapshots, selected_snapshots)
    save_attention_payload(output_dir / "attention_snapshot_payload.npz", attention_payload)
    for layer, rows in attention_head_metrics.items():
        save_json(output_dir / f"attention_head_metrics_layer_{layer}.json", rows)

    print("Running causal steering probe on residual counter directions...")
    steering_probe = run_steering_probe(
        args.model_path,
        processor,
        tokenizer,
        device,
        residual_direction_payload,
    )
    save_json(output_dir / "steering_probe_results.json", steering_probe)

    save_json(
        output_dir / "analysis_metadata.json",
        {
            "hook_order": HOOK_ORDER,
            "residual_hooks": RESIDUAL_HOOKS,
            "attention_layers": ATTN_LAYERS,
            "attention_counts": attention_counts,
            "attention_bins": ATTN_BINS,
            "top_feature_save_k": TOP_FEATURE_SAVE_K,
            "feature_structure_k": FEATURE_STRUCTURE_K,
            "feature_rank_max": int(args.feature_rank_max),
            "position_offsets": POSITION_OFFSETS,
            "attention_main_layer": ATTN_MAIN_LAYER,
            "steering_counts": STEERING_COUNTS,
            "steering_alpha_stds": STEERING_ALPHA_STDS,
        },
    )

    print(f"Analysis complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
