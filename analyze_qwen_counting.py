#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import transformers.integrations.moe as moe
from transformers import AutoModelForCausalLM, AutoTokenizer

import analyze_gemma_counting as base

MODEL_LABEL = "Qwen3.5-35B-A3B"
FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
FULL_ATTENTION_INDEX = {layer: index for index, layer in enumerate(FULL_ATTENTION_LAYERS)}

HOOK_SPECS = {
    "resid_post.layer15": {
        "hook_name": "model.layers.15",
        "mode": "forward",
        "hidden_size": 2048,
    },
    "resid_post.layer23": {
        "hook_name": "model.layers.23",
        "mode": "forward",
        "hidden_size": 2048,
    },
    "resid_post.layer31": {
        "hook_name": "model.layers.31",
        "mode": "forward",
        "hidden_size": 2048,
    },
    "resid_post.layer39": {
        "hook_name": "model.layers.39",
        "mode": "forward",
        "hidden_size": 2048,
    },
    "attn_out.layer31": {
        "hook_name": "model.layers.31.self_attn.o_proj",
        "mode": "forward_pre",
        "hidden_size": 4096,
    },
    "attn_out.layer39": {
        "hook_name": "model.layers.39.self_attn.o_proj",
        "mode": "forward_pre",
        "hidden_size": 4096,
    },
    "mlp_out.layer31": {
        "hook_name": "model.layers.31.mlp",
        "mode": "forward",
        "hidden_size": 2048,
    },
    "mlp_out.layer39": {
        "hook_name": "model.layers.39.mlp",
        "mode": "forward",
        "hidden_size": 2048,
    },
}

HOOK_ORDER = [
    "resid_post.layer15",
    "resid_post.layer23",
    "resid_post.layer31",
    "resid_post.layer39",
    "attn_out.layer31",
    "attn_out.layer39",
    "mlp_out.layer31",
    "mlp_out.layer39",
]
RESIDUAL_HOOKS = [
    "resid_post.layer15",
    "resid_post.layer23",
    "resid_post.layer31",
    "resid_post.layer39",
]
ATTN_LAYERS = [31, 39]
ATTN_MAIN_LAYER = 39

ATTN_BINS = base.ATTN_BINS
TOP_FEATURE_SAVE_K = base.TOP_FEATURE_SAVE_K
FEATURE_RANK_MAX = base.FEATURE_RANK_MAX
POSITION_WINDOW = base.POSITION_WINDOW
POSITION_OFFSETS = base.POSITION_OFFSETS
STEERING_COUNTS = [32, 48, 64, 96, 128, 160]
STEERING_ALPHA_STDS = base.STEERING_ALPHA_STDS

_MPS_PATCHED = False


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


class ResidualSteerer:
    def __init__(self, model: torch.nn.Module, module_name: str, direction: np.ndarray, scale: float):
        self.module = dict(model.named_modules())[module_name]
        parameter = next(model.parameters())
        self.vector = torch.from_numpy(direction.astype(np.float32)).to(
            device=parameter.device,
            dtype=parameter.dtype,
        ) * float(scale)
        self.handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        if isinstance(output, torch.Tensor):
            steered = output.clone()
            steered[:, -1, :] = steered[:, -1, :] + self.vector
            return steered
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            first = output[0].clone()
            first[:, -1, :] = first[:, -1, :] + self.vector
            return (first, *output[1:])
        raise TypeError(f"Unsupported hooked output type for steering: {type(output)!r}")

    def __enter__(self) -> "ResidualSteerer":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-35B-A3B counting analysis without SAE features."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/Qwen3.5-35B-A3B"),
        help="Local Qwen3.5-35B-A3B path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/qwen_counting_mechanistic_analysis"),
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
        default=1,
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
        help="Device to use. Defaults to auto -> cuda, mps, then cpu.",
    )
    parser.add_argument(
        "--feature-rank-max",
        type=int,
        default=FEATURE_RANK_MAX,
        help="Unused for SAE ranking here, but preserved for schema compatibility.",
    )
    return parser.parse_args()


def patch_moe_for_mps() -> None:
    global _MPS_PATCHED
    if _MPS_PATCHED:
        return

    def grouped_mm_experts_forward_mps_safe(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        device = hidden_states.device
        num_top_k = top_k_index.size(-1)
        num_tokens = hidden_states.size(0)
        hidden_dim = hidden_states.size(-1)

        token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
        sample_weights = top_k_weights.reshape(-1)
        expert_ids = top_k_index.reshape(-1)

        perm = torch.argsort(expert_ids)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device)

        expert_ids_g = expert_ids[perm]
        sample_weights_g = sample_weights[perm]
        selected_hidden_states_g = hidden_states[token_idx[perm]]

        histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
        tokens_per_expert = torch.histc(
            histc_input,
            bins=self.num_experts,
            min=0,
            max=self.num_experts - 1,
        )
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

        if self.has_gate:
            selected_weights = self.gate_up_proj
            selected_biases = self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
        else:
            selected_weights = self.up_proj
            selected_biases = self.up_proj_bias[expert_ids_g] if self.has_bias else None

        proj_out = moe._grouped_linear(
            selected_hidden_states_g,
            selected_weights,
            offsets,
            bias=selected_biases,
            is_transposed=self.is_transposed,
        )
        proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)

        selected_weights = self.down_proj
        selected_biases = self.down_proj_bias[expert_ids_g] if self.has_bias else None
        proj_out = moe._grouped_linear(
            proj_out,
            selected_weights,
            offsets,
            bias=selected_biases,
            is_transposed=self.is_transposed,
        )
        weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
        weighted_out = weighted_out[inv_perm]
        final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        return final_hidden_states.to(hidden_states.dtype)

    moe.ALL_EXPERTS_FUNCTIONS._global_mapping["grouped_mm"] = grouped_mm_experts_forward_mps_safe
    moe.ExpertsInterface._global_mapping["grouped_mm"] = grouped_mm_experts_forward_mps_safe
    _MPS_PATCHED = True


def resolve_device(requested: str) -> str:
    return base.resolve_device(requested)


def stream_and_layer(label: str) -> tuple[str, int]:
    return base.stream_and_layer(label)


def safe_label(label: str) -> str:
    return base.safe_label(label)


def save_json(path: Path, payload: object) -> None:
    base.save_json(path, payload)


def load_json(path: Path) -> object:
    return base.load_json(path)


def build_messages(count: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": base.SYSTEM_TEXT},
        {"role": "user", "content": f'How many "a" are in this sequence?\n\n{base.make_sequence(count)}'},
    ]


def render_prompts(tokenizer, counts: list[int]) -> list[str]:
    prompts: list[str] = []
    for count in counts:
        prompt = tokenizer.apply_chat_template(
            build_messages(count),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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


def load_model(model_path: Path, device: str, attn_implementation: str | None = None) -> torch.nn.Module:
    if device == "mps":
        patch_moe_for_mps()
    kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval().to(device)


def teacher_forced_margin_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    correct_texts: list[str],
    predicted_texts: list[str],
    device: str,
) -> list[dict[str, float | None]]:
    prompt_id_rows = [base.continuation_token_ids(tokenizer, prompt) for prompt in prompts]
    correct_id_rows = [base.continuation_token_ids(tokenizer, text) for text in correct_texts]
    predicted_id_rows = [base.continuation_token_ids(tokenizer, text) for text in predicted_texts]

    combined_rows = [
        prompt_ids + correct_ids for prompt_ids, correct_ids in zip(prompt_id_rows, correct_id_rows)
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
            token_margins.append(correct_logit - base.top_other_logit(logit_row, correct_token_id))

        divergence_gap = None
        divergence_index = base.first_divergence_index(correct_ids, predicted_ids)
        if (
            divergence_index is not None
            and divergence_index < len(correct_ids)
            and divergence_index < len(predicted_ids)
        ):
            logit_row = logits[index, len(prompt_ids) - 1 + divergence_index]
            divergence_gap = float(
                logit_row[correct_ids[divergence_index]] - logit_row[predicted_ids[divergence_index]]
            )

        rows.append(
            {
                "min_correct_logit_gap": float(min(token_margins)) if token_margins else None,
                "mean_correct_logit_gap": float(np.mean(token_margins)) if token_margins else None,
                "first_divergence_logit_gap": divergence_gap,
            }
        )
    return rows


def run_generation_sweep(
    model: torch.nn.Module,
    tokenizer,
    counts: list[int],
    batch_size: int,
    device: str,
) -> list[base.SweepRecord]:
    prompts = render_prompts(tokenizer, counts)
    records: list[base.SweepRecord] = []
    for count_batch, prompt_batch in zip(base.batched(counts, batch_size), base.batched(prompts, batch_size)):
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
        raw_texts: list[str] = []
        for index in range(len(count_list)):
            continuation = sequences[index, int(input_lengths[index]) :]
            raw_text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            parsed = base.parse_integer(raw_text)
            raw_texts.append(raw_text)
            predicted_texts.append(str(parsed) if parsed is not None else raw_text)
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
            parsed = parsed_predictions[index]
            records.append(
                base.SweepRecord(
                    count=count,
                    prompt_tokens=int(input_lengths[index]),
                    prediction_text=raw_texts[index],
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
    model: torch.nn.Module,
    tokenizer,
    counts: list[int],
    batch_size: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    prompts = render_prompts(tokenizer, counts)
    recorder = ActivationRecorder(model)
    recorder.register()
    final_outputs: dict[str, list[np.ndarray]] = {label: [] for label in HOOK_ORDER}
    position_windows: dict[str, list[np.ndarray]] = {label: [] for label in RESIDUAL_HOOKS}
    gather_back = torch.tensor(list(range(POSITION_WINDOW - 1, -1, -1)), device=device, dtype=torch.long)
    try:
        for prompt_batch in base.batched(prompts, batch_size):
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
    final_arrays = {label: np.concatenate(chunks, axis=0) for label, chunks in final_outputs.items()}
    position_arrays = {label: np.concatenate(chunks, axis=0) for label, chunks in position_windows.items()}
    return final_arrays, position_arrays


def capture_attention_head_metrics(
    model_path: Path,
    tokenizer,
    counts: list[int],
    device: str,
) -> dict[int, list[dict[str, float | int]]]:
    model = load_model(model_path, device, attn_implementation="eager")
    metrics_by_layer: dict[int, list[dict[str, float | int]]] = {layer: [] for layer in ATTN_LAYERS}
    try:
        prompts = render_prompts(tokenizer, counts)
        for count, prompt in zip(counts, prompts):
            encoded = tokenize_prompts(tokenizer, [prompt], device)
            with torch.inference_mode():
                outputs = model(**encoded, use_cache=False, output_attentions=True)
            sequence = base.make_sequence(count)
            seq_mask, item_mask, sep_mask = base.sequence_token_masks(prompt, sequence, tokenizer)
            for layer in ATTN_LAYERS:
                attention = outputs.attentions[FULL_ATTENTION_INDEX[layer]][0].detach().float().cpu().numpy()
                metrics_by_layer[layer].extend(
                    base.head_attention_metrics(attention, seq_mask, item_mask, sep_mask, count, layer)
                )
    finally:
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()
    return metrics_by_layer


def capture_attention_snapshots(
    model_path: Path,
    tokenizer,
    counts: list[int],
    device: str,
    selected_heads: dict[int, dict[str, int]],
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[str, int, int], np.ndarray]]:
    model = load_model(model_path, device, attn_implementation="eager")
    average_snapshots: dict[tuple[int, int], np.ndarray] = {}
    selected_snapshots: dict[tuple[str, int, int], np.ndarray] = {}
    try:
        prompts = render_prompts(tokenizer, counts)
        for count, prompt in zip(counts, prompts):
            encoded = tokenize_prompts(tokenizer, [prompt], device)
            with torch.inference_mode():
                outputs = model(**encoded, use_cache=False, output_attentions=True)
            for layer in ATTN_LAYERS:
                attention = outputs.attentions[FULL_ATTENTION_INDEX[layer]][0].detach().float().cpu().numpy()
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


def feature_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"features_{safe_label(label)}.json"


def feature_trajectory_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"feature_trajectories_{safe_label(label)}.npz"
def run_steering_probe(
    model_path: Path,
    tokenizer,
    device: str,
    residual_directions: dict[str, dict[str, np.ndarray | float]],
) -> list[dict[str, object]]:
    counts = [count for count in STEERING_COUNTS if count > 0]
    prompts = render_prompts(tokenizer, counts)
    model = load_model(model_path, device)
    rows: list[dict[str, object]] = []
    try:
        baseline_predictions: dict[int, int | None] = {}
        for count, prompt in zip(counts, prompts):
            encoded = tokenize_prompts(tokenizer, [prompt], device)
            with torch.inference_mode():
                sequences = model.generate(**encoded, max_new_tokens=12, do_sample=False)
            continuation = sequences[0, int(encoded["attention_mask"][0].sum().item()) :]
            baseline_predictions[count] = base.parse_integer(
                tokenizer.decode(continuation, skip_special_tokens=True).strip()
            )

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
                        steered_prediction = base.parse_integer(
                            tokenizer.decode(continuation, skip_special_tokens=True).strip()
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

def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    counts = list(range(1, args.max_count + 1))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Using device: {device}")
    print(f"Loading {MODEL_LABEL} for generation and activation sweeps...")
    model = load_model(args.model_path, device)

    print("Running dense generation sweep...")
    records = run_generation_sweep(model, tokenizer, counts, args.batch_size, device)
    run_summary = base.summarize_records(records)

    print("Collecting raw activations and last-window position payloads...")
    raw_activations, position_windows = collect_raw_activations(model, tokenizer, counts, args.batch_size, device)

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
    save_json(output_dir / "run_summary.json", run_summary)

    hook_leaderboard: list[dict[str, object]] = []
    hook_counter_metrics: list[dict[str, object]] = []
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
        print(f"Fitting raw counter geometry for {label}...")
        safe = safe_label(label)
        stream, layer = stream_and_layer(label)
        activations = raw_activations[label]

        coords, explained = base.fit_principal_components(activations)
        center, direction, projected, rho, delta = base.fit_counter_direction(
            activations,
            counts_array,
            exact_prefix_mask,
        )
        unit_direction = direction / np.clip(np.linalg.norm(direction), 1e-8, None)
        unit_projection = ((activations - center) @ unit_direction).astype(np.float32)
        exact_prefix_std = float(np.std(unit_projection[exact_prefix_mask]))
        projection_count_corr = base.pearson(projected, counts_array.astype(np.float32))
        exact_prefix_projection_corr = base.pearson(
            projected[exact_prefix_mask],
            counts_array[exact_prefix_mask].astype(np.float32),
        )
        post_failure_projection_corr = (
            base.pearson(projected[post_failure_mask], counts_array[post_failure_mask].astype(np.float32))
            if np.sum(post_failure_mask) >= 2
            else float("nan")
        )
        adjacent_exact = np.diff(projected[exact_prefix_mask])
        eta_projection_threshold = None
        if run_summary["first_failure"] is not None and int(run_summary["first_failure"]) >= 2:
            boundary_index = int(run_summary["first_failure"]) - 2
            if 0 <= boundary_index < projected.shape[0] - 1:
                eta_projection_threshold = float(abs(np.diff(projected)[boundary_index]))
        exact_vs_failed_sep = (
            float(projected[exact_prefix_mask].mean() - projected[post_failure_mask].mean())
            if np.any(exact_prefix_mask) and np.any(post_failure_mask)
            else float("nan")
        )

        metric_row = {
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
        hook_counter_metrics.append(metric_row)
        if label in RESIDUAL_HOOKS:
            residual_counter_metrics.append(metric_row)

        hook_leaderboard.append(
            {
                "label": label,
                "stream": stream,
                "layer": layer,
                "best_feature_id": None,
                "best_feature_pearson": None,
                "best_feature_mean_activation": None,
                "best_feature_active_rate": None,
                "projection_count_corr": float(projection_count_corr),
                "exact_prefix_projection_corr": float(exact_prefix_projection_corr),
                "post_failure_projection_corr": float(post_failure_projection_corr),
            }
        )
        stream_layer_summary.append(
            {
                "label": label,
                "stream": stream,
                "layer": layer,
                "best_feature_corr": None,
                "top5_mean_abs_corr": None,
                "counter_projection_corr": float(projection_count_corr),
                "exact_vs_failed_separation": exact_vs_failed_sep,
            }
        )
        exact_failed_comparison.append(
            {
                "label": label,
                "stream": stream,
                "layer": layer,
                "exact_prefix_best_feature_id": None,
                "exact_prefix_best_feature_corr": None,
                "post_failure_best_feature_id": None,
                "post_failure_best_feature_corr": None,
                "late_exact_best_feature_id": None,
                "late_exact_best_feature_corr": None,
                "best_feature_exact_prefix_mean": None,
                "best_feature_post_failure_mean": None,
                "best_feature_separation": None,
                "projection_exact_prefix_mean": float(projected[exact_prefix_mask].mean()) if np.any(exact_prefix_mask) else None,
                "projection_post_failure_mean": float(projected[post_failure_mask].mean()) if np.any(post_failure_mask) else None,
                "projection_late_exact_mean": float(projected[late_exact_mask].mean()) if np.any(late_exact_mask) else None,
                "projection_separation": exact_vs_failed_sep,
                "exact_prefix_projection_corr": float(exact_prefix_projection_corr),
                "post_failure_projection_corr": float(post_failure_projection_corr),
            }
        )

        analysis_arrays[f"pca_coords_{safe}"] = coords.astype(np.float32)
        analysis_arrays[f"pca_explained_{safe}"] = explained.astype(np.float32)
        analysis_arrays[f"projected_counter_{safe}"] = projected.astype(np.float32)
        analysis_arrays[f"counter_center_{safe}"] = center.astype(np.float32)
        analysis_arrays[f"counter_direction_unit_{safe}"] = unit_direction.astype(np.float32)
        analysis_arrays[f"counter_exact_prefix_std_{safe}"] = np.array([exact_prefix_std], dtype=np.float32)
        analysis_arrays[f"top_feature_matrix_{safe}"] = np.zeros((counts_array.shape[0], 0), dtype=np.float32)
        analysis_arrays[f"top_feature_decoder_cosine_{safe}"] = np.zeros((0, 0), dtype=np.float32)
        analysis_arrays[f"top_feature_ids_{safe}"] = np.zeros((0,), dtype=np.int32)
        analysis_arrays[f"top_feature_ids64_{safe}"] = np.zeros((0,), dtype=np.int32)
        analysis_arrays[f"top_feature_values64_{safe}"] = np.zeros((counts_array.shape[0], 0), dtype=np.float32)

        save_json(feature_path(output_dir, label), [])
        np.savez_compressed(
            feature_trajectory_path(output_dir, label),
            counts=counts_array.astype(np.int32),
            rank_mask=feature_rank_mask.astype(np.int8),
            feature_ids=np.zeros((0,), dtype=np.int32),
            rank_window_correlations=np.zeros((0,), dtype=np.float32),
            full_correlations=np.zeros((0,), dtype=np.float32),
            values=np.zeros((counts_array.shape[0], 0), dtype=np.float32),
        )
        if label in RESIDUAL_HOOKS:
            projected_window = base.project_position_window(position_windows[label], center, direction)
            position_arrays[f"projected_position_window_{safe}"] = projected_window
            residual_direction_payload[label] = {
                "unit_direction": unit_direction.astype(np.float32),
                "exact_prefix_std": exact_prefix_std,
            }

    hook_leaderboard_sorted = sorted(
        hook_leaderboard,
        key=lambda row: abs(float(row["projection_count_corr"])),
        reverse=True,
    )
    save_json(output_dir / "hook_leaderboard.json", hook_leaderboard_sorted)
    base.write_csv(output_dir / "hook_leaderboard.csv", hook_leaderboard_sorted)
    save_json(output_dir / "hook_counter_metrics.json", hook_counter_metrics)
    save_json(output_dir / "stream_layer_summary.json", stream_layer_summary)
    save_json(output_dir / "residual_counter_metrics.json", residual_counter_metrics)
    save_json(output_dir / "exact_failed_comparison.json", exact_failed_comparison)
    np.savez_compressed(output_dir / "residual_analysis_arrays.npz", **analysis_arrays)
    np.savez_compressed(output_dir / "position_window_payload.npz", **position_arrays)

    attention_counts = base.derive_attention_counts(run_summary, args.max_count, args.attn_counts)
    print(f"Capturing attention head metrics at counts {attention_counts}...")
    attention_head_metrics = capture_attention_head_metrics(
        args.model_path,
        tokenizer,
        attention_counts,
        device,
    )
    attention_metric_summary = {
        layer: base.summarize_attention_metrics(rows) for layer, rows in attention_head_metrics.items()
    }
    attention_head_selection = base.select_representative_attention_heads(attention_head_metrics)
    save_json(output_dir / "attention_metric_summary.json", attention_metric_summary)
    save_json(output_dir / "selected_attention_heads.json", attention_head_selection)
    base.ATTN_LAYERS = ATTN_LAYERS

    print(f"Capturing average and selected-head attention snapshots at counts {attention_counts}...")
    average_snapshots, selected_snapshots = capture_attention_snapshots(
        args.model_path,
        tokenizer,
        attention_counts,
        device,
        attention_head_selection,
    )
    attention_payload = base.build_binned_attention_payload(average_snapshots, selected_snapshots)
    base.save_attention_payload(output_dir / "attention_snapshot_payload.npz", attention_payload)
    for layer, rows in attention_head_metrics.items():
        save_json(output_dir / f"attention_head_metrics_layer{layer}.json", rows)

    print("Running causal steering probe on residual counter directions...")
    steering_probe = run_steering_probe(args.model_path, tokenizer, device, residual_direction_payload)
    save_json(output_dir / "steering_probe_results.json", steering_probe)

    save_json(
        output_dir / "analysis_metadata.json",
        {
            "model_label": MODEL_LABEL,
            "hook_order": HOOK_ORDER,
            "residual_hooks": RESIDUAL_HOOKS,
            "attention_layers": ATTN_LAYERS,
            "full_attention_layers": FULL_ATTENTION_LAYERS,
            "attention_counts": attention_counts,
            "attention_bins": ATTN_BINS,
            "feature_rank_max": int(args.feature_rank_max),
            "position_offsets": POSITION_OFFSETS,
            "attention_main_layer": ATTN_MAIN_LAYER,
            "steering_counts": STEERING_COUNTS,
            "steering_alpha_stds": STEERING_ALPHA_STDS,
            "sae_available": False,
        },
    )

    print(f"Analysis complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
