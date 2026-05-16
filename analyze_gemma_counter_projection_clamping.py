#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

import analyze_gemma_counting as base

DEFAULT_TARGET_COUNTS = [27, 31, 34, 37, 48, 64, 80, 96, 128, 160]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Counter-projection clamping analysis for Gemma 3 counting failure."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/gemma-3-27b-it"),
        help="Local Gemma 3 27B IT path.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/gemma_counting_mechanistic_analysis"),
        help="Existing analysis directory with saved counter geometry.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gemma_counter_projection_clamping"),
        help="Directory for causal patch outputs.",
    )
    parser.add_argument(
        "--target-counts",
        type=int,
        nargs="*",
        default=DEFAULT_TARGET_COUNTS,
        help="Failed counts to test.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use. Defaults to auto -> mps, cuda, then cpu.",
    )
    return parser.parse_args()


def load_counter_geometry(analysis_dir: Path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray | float]]]:
    analysis_metadata = json.loads((analysis_dir / "analysis_metadata.json").read_text())
    feature_rank_max = int(analysis_metadata["feature_rank_max"])
    analysis_payload = np.load(analysis_dir / "residual_analysis_arrays.npz")
    counts = analysis_payload["counts"].astype(np.int32)
    fit_mask = counts <= feature_rank_max
    geometry: dict[str, dict[str, np.ndarray | float]] = {}
    for label in base.RESIDUAL_HOOKS:
        safe = base.safe_label(label)
        projected = analysis_payload[f"projected_counter_{safe}"].astype(np.float32)
        center = analysis_payload[f"counter_center_{safe}"].astype(np.float32)
        direction = analysis_payload[f"counter_direction_unit_{safe}"].astype(np.float32)
        slope, intercept = np.polyfit(counts[fit_mask].astype(np.float32), projected[fit_mask], deg=1)
        geometry[label] = {
            "center": center.reshape(-1).astype(np.float32),
            "direction": direction.astype(np.float32),
            "projected": projected.astype(np.float32),
            "fit_slope": float(slope),
            "fit_intercept": float(intercept),
        }
    return counts, geometry


def summarize_rows(rows: list[dict[str, object]], target_counts: list[int]) -> dict[str, object]:
    successful = [row for row in rows if bool(row["patched_exact"])]
    min_gap_deltas = [
        float(row["min_gap_delta"])
        for row in rows
        if row["min_gap_delta"] is not None
    ]
    layer53_rescues = [
        float(row["resid_post.layer53_error_reduction"])
        for row in rows
        if row["resid_post.layer53_error_reduction"] is not None
    ]
    best_generation = max(
        rows,
        key=lambda row: float("-inf") if row["min_gap_delta"] is None else float(row["min_gap_delta"]),
    )
    best_state_rescue = max(
        rows,
        key=lambda row: float("-inf")
        if row["resid_post.layer53_error_reduction"] is None
        else float(row["resid_post.layer53_error_reduction"]),
    )
    return {
        "target_counts": target_counts,
        "intervention_count": len(rows),
        "exact_generation_rescues": len(successful),
        "mean_min_gap_delta": None if not min_gap_deltas else float(np.mean(min_gap_deltas)),
        "mean_layer53_error_reduction": None if not layer53_rescues else float(np.mean(layer53_rescues)),
        "best_generation_margin_rescue": {
            "patch_layer": best_generation["patch_layer"],
            "count": best_generation["count"],
            "baseline_prediction": best_generation["baseline_prediction"],
            "patched_prediction": best_generation["patched_prediction"],
            "min_gap_delta": best_generation["min_gap_delta"],
        },
        "best_layer53_state_rescue": {
            "patch_layer": best_state_rescue["patch_layer"],
            "count": best_state_rescue["count"],
            "baseline_prediction": best_state_rescue["baseline_prediction"],
            "patched_prediction": best_state_rescue["patched_prediction"],
            "layer53_error_reduction": best_state_rescue["resid_post.layer53_error_reduction"],
        },
    }


def expected_projection(geometry: dict[str, np.ndarray | float], count: int) -> float:
    return float(geometry["fit_intercept"]) + float(geometry["fit_slope"]) * float(count)


def make_prompt(processor: AutoProcessor, count: int) -> str:
    prompt = processor.apply_chat_template(
        base.build_messages(count),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str):
        raise TypeError(f"Expected string prompt, got {type(prompt)!r}")
    return prompt


class CounterProjectionClamper:
    def __init__(
        self,
        model: Gemma3ForConditionalGeneration,
        module_name: str,
        center: np.ndarray,
        direction: np.ndarray,
        target_projection: float,
        patch_index: int,
    ) -> None:
        self.module = dict(model.named_modules())[module_name]
        parameter = next(model.parameters())
        self.center = torch.from_numpy(center.astype(np.float32)).to(parameter.device, dtype=torch.float32)
        self.direction = torch.from_numpy(direction.astype(np.float32)).to(parameter.device, dtype=torch.float32)
        self.target_projection = float(target_projection)
        self.patch_index = int(patch_index)
        self.handle: torch.utils.hooks.RemovableHandle | None = None
        self.applied = False
        self.current_projection: float | None = None
        self.delta_projection: float | None = None

    def _patch_tensor(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.applied or hidden.shape[1] <= self.patch_index:
            return hidden
        modified = hidden.clone()
        vector = modified[:, self.patch_index, :].to(torch.float32)
        current_projection = torch.sum((vector - self.center) * self.direction, dim=-1)
        delta_projection = self.target_projection - current_projection
        patch = delta_projection[:, None] * self.direction[None, :]
        modified[:, self.patch_index, :] = (vector + patch).to(modified.dtype)
        self.current_projection = float(current_projection[0].item())
        self.delta_projection = float(delta_projection[0].item())
        self.applied = True
        return modified

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        if isinstance(output, torch.Tensor):
            return self._patch_tensor(output)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            first = self._patch_tensor(output[0])
            return (first, *output[1:])
        raise TypeError(f"Unsupported hooked output type for clamping: {type(output)!r}")

    def __enter__(self) -> "CounterProjectionClamper":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def teacher_forced_margin_single(
    model: Gemma3ForConditionalGeneration,
    tokenizer,
    prompt: str,
    correct_text: str,
    predicted_text: str,
    device: str,
    patcher: CounterProjectionClamper | None = None,
) -> dict[str, float | None]:
    prompt_ids = base.continuation_token_ids(tokenizer, prompt)
    correct_ids = base.continuation_token_ids(tokenizer, correct_text)
    predicted_ids = base.continuation_token_ids(tokenizer, predicted_text)
    row = prompt_ids + correct_ids
    input_ids = torch.tensor([row], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    context = patcher if patcher is not None else nullcontext()
    with context:
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits.detach().float().cpu()[0]

    token_margins: list[float] = []
    for step, correct_token_id in enumerate(correct_ids):
        logit_row = logits[len(prompt_ids) - 1 + step]
        correct_logit = float(logit_row[correct_token_id])
        token_margins.append(correct_logit - base.top_other_logit(logit_row, correct_token_id))

    divergence_gap = None
    divergence_index = base.first_divergence_index(correct_ids, predicted_ids)
    if divergence_index is not None and divergence_index < len(correct_ids) and divergence_index < len(predicted_ids):
        logit_row = logits[len(prompt_ids) - 1 + divergence_index]
        divergence_gap = float(logit_row[correct_ids[divergence_index]] - logit_row[predicted_ids[divergence_index]])

    return {
        "min_correct_logit_gap": float(min(token_margins)) if token_margins else None,
        "mean_correct_logit_gap": float(np.mean(token_margins)) if token_margins else None,
        "first_divergence_logit_gap": divergence_gap,
    }


def generate_single(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer,
    prompt: str,
    device: str,
    patcher: CounterProjectionClamper | None = None,
) -> tuple[int | None, str]:
    encoded = base.tokenize_prompts(tokenizer, [prompt], device)
    prompt_length = int(encoded["attention_mask"][0].sum().item())
    context = patcher if patcher is not None else nullcontext()
    with context:
        with torch.inference_mode():
            sequences = model.generate(**encoded, max_new_tokens=12, do_sample=False)
    continuation = sequences[0, prompt_length:]
    text = processor.decode(continuation, skip_special_tokens=True).strip()
    return base.parse_integer(text), text


def collect_projected_states(
    model: Gemma3ForConditionalGeneration,
    tokenizer,
    prompt: str,
    device: str,
    geometry: dict[str, dict[str, np.ndarray | float]],
    patcher: CounterProjectionClamper | None = None,
) -> dict[str, float]:
    encoded = base.tokenize_prompts(tokenizer, [prompt], device)
    last_position = int(encoded["attention_mask"][0].sum().item()) - 1
    context = patcher if patcher is not None else nullcontext()
    with context:
        recorder = base.ActivationRecorder(model)
        recorder.register()
        try:
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            rows: dict[str, float] = {}
            for label in base.RESIDUAL_HOOKS:
                tensor = recorder.cache[label]
                hidden = tensor[0, last_position].detach().float().cpu().numpy()
                center = np.asarray(geometry[label]["center"], dtype=np.float32)
                direction = np.asarray(geometry[label]["direction"], dtype=np.float32)
                rows[label] = float(np.dot(hidden - center, direction))
            return rows
        finally:
            recorder.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    counts, geometry = load_counter_geometry(args.analysis_dir)
    count_sweep_records = json.loads((args.analysis_dir / "count_sweep_records.json").read_text())
    record_by_count = {int(row["count"]): row for row in count_sweep_records}
    target_counts = [count for count in args.target_counts if count in record_by_count and not bool(record_by_count[count]["exact"])]

    device = base.resolve_device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    try:
        baseline_projected = {
            label: np.asarray(geometry[label]["projected"], dtype=np.float32)
            for label in base.RESIDUAL_HOOKS
        }

        rows: list[dict[str, object]] = []
        for label in base.RESIDUAL_HOOKS:
            print(f"Running causal clamps at {label}...")
            module_name = str(base.HOOK_SPECS[label]["hook_name"])
            for count in target_counts:
                prompt = make_prompt(processor, count)
                baseline = record_by_count[count]
                prompt_ids = base.continuation_token_ids(tokenizer, prompt)
                target_z = expected_projection(geometry[label], count)

                patcher = CounterProjectionClamper(
                    model,
                    module_name,
                    np.asarray(geometry[label]["center"], dtype=np.float32),
                    np.asarray(geometry[label]["direction"], dtype=np.float32),
                    target_z,
                    patch_index=len(prompt_ids) - 1,
                )
                patched_prediction, patched_text = generate_single(
                    model,
                    processor,
                    tokenizer,
                    prompt,
                    device,
                    patcher=patcher,
                )

                margin_patcher = CounterProjectionClamper(
                    model,
                    module_name,
                    np.asarray(geometry[label]["center"], dtype=np.float32),
                    np.asarray(geometry[label]["direction"], dtype=np.float32),
                    target_z,
                    patch_index=len(prompt_ids) - 1,
                )
                patched_margins = teacher_forced_margin_single(
                    model,
                    tokenizer,
                    prompt,
                    str(count),
                    str(patched_prediction) if patched_prediction is not None else patched_text,
                    device,
                    patcher=margin_patcher,
                )

                state_patcher = CounterProjectionClamper(
                    model,
                    module_name,
                    np.asarray(geometry[label]["center"], dtype=np.float32),
                    np.asarray(geometry[label]["direction"], dtype=np.float32),
                    target_z,
                    patch_index=len(prompt_ids) - 1,
                )
                patched_states = collect_projected_states(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    geometry,
                    patcher=state_patcher,
                )

                row: dict[str, object] = {
                    "patch_layer": label,
                    "count": count,
                    "baseline_prediction": baseline["parsed_prediction"],
                    "patched_prediction": patched_prediction,
                    "patched_prediction_text": patched_text,
                    "patched_exact": patched_prediction == count,
                    "baseline_min_correct_logit_gap": baseline["min_correct_logit_gap"],
                    "patched_min_correct_logit_gap": patched_margins["min_correct_logit_gap"],
                    "min_gap_delta": None
                    if baseline["min_correct_logit_gap"] is None or patched_margins["min_correct_logit_gap"] is None
                    else float(patched_margins["min_correct_logit_gap"]) - float(baseline["min_correct_logit_gap"]),
                    "baseline_first_divergence_logit_gap": baseline["first_divergence_logit_gap"],
                    "patched_first_divergence_logit_gap": patched_margins["first_divergence_logit_gap"],
                    "applied_projection_before": state_patcher.current_projection,
                    "applied_projection_target": target_z,
                    "applied_projection_delta": state_patcher.delta_projection,
                }
                for downstream_label in base.RESIDUAL_HOOKS:
                    expected_z = expected_projection(geometry[downstream_label], count)
                    base_z = float(baseline_projected[downstream_label][count - 1])
                    patched_z = float(patched_states[downstream_label])
                    base_err = abs(base_z - expected_z)
                    patched_err = abs(patched_z - expected_z)
                    reduction = 0.0 if base_err < 1e-6 else 1.0 - (patched_err / base_err)
                    row[f"baseline_{downstream_label}_projection"] = base_z
                    row[f"patched_{downstream_label}_projection"] = patched_z
                    row[f"expected_{downstream_label}_projection"] = expected_z
                    row[f"{downstream_label}_error_reduction"] = float(reduction)
                rows.append(row)
    finally:
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()

    base.save_json(output_dir / "counter_projection_clamping_rows.json", rows)
    base.save_json(output_dir / "counter_projection_clamping_summary.json", summarize_rows(rows, target_counts))
    print(f"Counter-projection clamping analysis complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
