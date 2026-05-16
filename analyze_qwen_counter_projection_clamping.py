#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import analyze_qwen_counting as qbase

DEFAULT_TARGET_COUNTS = [40, 48, 60, 64, 80, 96, 128, 160]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Counter-projection clamping analysis for Qwen counting failure."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/Qwen3.5-35B-A3B"),
        help="Local Qwen3.5-35B-A3B path.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/qwen_counting_mechanistic_analysis"),
        help="Existing Qwen analysis directory with saved counter geometry.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/qwen_counter_projection_clamping"),
        help="Directory for clamping outputs.",
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
        help="Device to use. Defaults to auto -> cuda, mps, then cpu.",
    )
    return parser.parse_args()


def load_counter_geometry(analysis_dir: Path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray | float]]]:
    analysis_metadata = json.loads((analysis_dir / "analysis_metadata.json").read_text())
    feature_rank_max = int(analysis_metadata["feature_rank_max"])
    analysis_payload = np.load(analysis_dir / "residual_analysis_arrays.npz")
    counts = analysis_payload["counts"].astype(np.int32)
    fit_mask = counts <= feature_rank_max
    geometry: dict[str, dict[str, np.ndarray | float]] = {}
    for label in qbase.RESIDUAL_HOOKS:
        safe = qbase.safe_label(label)
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


def expected_projection(geometry: dict[str, np.ndarray | float], count: int) -> float:
    return float(geometry["fit_intercept"]) + float(geometry["fit_slope"]) * float(count)


class CounterProjectionClamper:
    def __init__(
        self,
        model: torch.nn.Module,
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
        self.applied = False
        self.current_projection = None
        self.delta_projection = None
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def teacher_forced_margin_single(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    correct_text: str,
    predicted_text: str,
    device: str,
    patcher: CounterProjectionClamper | None = None,
) -> dict[str, float | None]:
    prompt_ids = qbase.base.continuation_token_ids(tokenizer, prompt)
    correct_ids = qbase.base.continuation_token_ids(tokenizer, correct_text)
    predicted_ids = qbase.base.continuation_token_ids(tokenizer, predicted_text)
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
        token_margins.append(correct_logit - qbase.base.top_other_logit(logit_row, correct_token_id))

    divergence_gap = None
    divergence_index = qbase.base.first_divergence_index(correct_ids, predicted_ids)
    if divergence_index is not None and divergence_index < len(correct_ids) and divergence_index < len(predicted_ids):
        logit_row = logits[len(prompt_ids) - 1 + divergence_index]
        divergence_gap = float(logit_row[correct_ids[divergence_index]] - logit_row[predicted_ids[divergence_index]])

    return {
        "min_correct_logit_gap": float(min(token_margins)) if token_margins else None,
        "mean_correct_logit_gap": float(np.mean(token_margins)) if token_margins else None,
        "first_divergence_logit_gap": divergence_gap,
    }


def generate_single(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: str,
    patcher: CounterProjectionClamper | None = None,
) -> tuple[int | None, str]:
    encoded = qbase.tokenize_prompts(tokenizer, [prompt], device)
    prompt_length = int(encoded["attention_mask"][0].sum().item())
    context = patcher if patcher is not None else nullcontext()
    with context:
        with torch.inference_mode():
            sequences = model.generate(**encoded, max_new_tokens=12, do_sample=False)
    continuation = sequences[0, prompt_length:]
    text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    return qbase.base.parse_integer(text), text


def collect_projected_states(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: str,
    geometry: dict[str, dict[str, np.ndarray | float]],
    patcher: CounterProjectionClamper | None = None,
) -> dict[str, float]:
    encoded = qbase.tokenize_prompts(tokenizer, [prompt], device)
    last_position = int(encoded["attention_mask"][0].sum().item()) - 1
    context = patcher if patcher is not None else nullcontext()
    with context:
        recorder = qbase.ActivationRecorder(model)
        recorder.register()
        try:
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            rows: dict[str, float] = {}
            for label in qbase.RESIDUAL_HOOKS:
                tensor = recorder.cache[label]
                hidden = tensor[0, last_position].detach().float().cpu().numpy()
                center = np.asarray(geometry[label]["center"], dtype=np.float32)
                direction = np.asarray(geometry[label]["direction"], dtype=np.float32)
                rows[label] = float(np.dot(hidden - center, direction))
            return rows
        finally:
            recorder.close()


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    exact_rescues = sum(bool((not row["baseline_exact"]) and row["patched_exact"]) for row in rows)
    mean_gap_delta = float(np.mean([float(row["min_correct_logit_gap_delta"]) for row in rows]))
    mean_layer39_reduction = float(np.mean([float(row["layer39_projection_error_reduction"]) for row in rows]))
    by_label: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)

    layer_summaries: list[dict[str, object]] = []
    for label in qbase.RESIDUAL_HOOKS:
        subset = by_label.get(label, [])
        if not subset:
            continue
        mean_gap = float(np.mean([float(row["min_correct_logit_gap_delta"]) for row in subset]))
        mean_proj = float(np.mean([float(row["layer39_projection_error_reduction"]) for row in subset]))
        exact = sum(bool((not row["baseline_exact"]) and row["patched_exact"]) for row in subset)
        layer_summaries.append(
            {
                "label": label,
                "exact_generation_rescues": exact,
                "trial_count": len(subset),
                "mean_min_logit_gap_delta": mean_gap,
                "mean_layer39_projection_error_reduction": mean_proj,
            }
        )
    return {
        "target_counts": sorted({int(row["count"]) for row in rows}),
        "intervention_count": total,
        "exact_generation_rescues": exact_rescues,
        "mean_min_logit_gap_delta": mean_gap_delta,
        "mean_layer39_projection_error_reduction": mean_layer39_reduction,
        "layer_summaries": layer_summaries,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = qbase.resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    counts, geometry = load_counter_geometry(args.analysis_dir)
    summary = json.loads((args.analysis_dir / "run_summary.json").read_text())
    first_failure = int(summary["first_failure"]) if summary["first_failure"] is not None else None
    count_sweep_records = json.loads((args.analysis_dir / "count_sweep_records.json").read_text())
    failed_counts = {int(row["count"]) for row in count_sweep_records if not bool(row["exact"])}

    target_counts = sorted({count for count in args.target_counts if count in counts.tolist()})
    if first_failure is not None:
        target_counts = [count for count in target_counts if count >= first_failure]
    target_counts = [count for count in target_counts if count in failed_counts]

    model = qbase.load_model(args.model_path, device)
    rows: list[dict[str, object]] = []
    try:
        for count in target_counts:
            prompt = qbase.render_prompts(tokenizer, [count])[0]
            baseline_prediction, baseline_text = generate_single(model, tokenizer, prompt, device)
            baseline_margin = teacher_forced_margin_single(
                model,
                tokenizer,
                prompt,
                str(count),
                str(baseline_prediction) if baseline_prediction is not None else baseline_text,
                device,
            )
            baseline_states = collect_projected_states(model, tokenizer, prompt, device, geometry)

            for label in qbase.RESIDUAL_HOOKS:
                module_name = str(qbase.HOOK_SPECS[label]["hook_name"])
                encoded = qbase.tokenize_prompts(tokenizer, [prompt], device)
                patch_index = int(encoded["attention_mask"][0].sum().item()) - 1
                clamper = CounterProjectionClamper(
                    model,
                    module_name,
                    np.asarray(geometry[label]["center"], dtype=np.float32),
                    np.asarray(geometry[label]["direction"], dtype=np.float32),
                    expected_projection(geometry[label], count),
                    patch_index,
                )

                patched_prediction, patched_text = generate_single(model, tokenizer, prompt, device, clamper)
                patched_margin = teacher_forced_margin_single(
                    model,
                    tokenizer,
                    prompt,
                    str(count),
                    str(patched_prediction) if patched_prediction is not None else patched_text,
                    device,
                    clamper,
                )
                patched_states = collect_projected_states(model, tokenizer, prompt, device, geometry, clamper)

                rows.append(
                    {
                        "label": label,
                        "count": count,
                        "baseline_prediction": baseline_prediction,
                        "patched_prediction": patched_prediction,
                        "baseline_exact": baseline_prediction == count,
                        "patched_exact": patched_prediction == count,
                        "baseline_min_correct_logit_gap": baseline_margin["min_correct_logit_gap"],
                        "patched_min_correct_logit_gap": patched_margin["min_correct_logit_gap"],
                        "min_correct_logit_gap_delta": None
                        if baseline_margin["min_correct_logit_gap"] is None or patched_margin["min_correct_logit_gap"] is None
                        else float(patched_margin["min_correct_logit_gap"] - baseline_margin["min_correct_logit_gap"]),
                        "baseline_layer39_projection": float(baseline_states["resid_post.layer39"]),
                        "patched_layer39_projection": float(patched_states["resid_post.layer39"]),
                        "layer39_projection_error_reduction": float(
                            abs(expected_projection(geometry["resid_post.layer39"], count) - baseline_states["resid_post.layer39"])
                            - abs(expected_projection(geometry["resid_post.layer39"], count) - patched_states["resid_post.layer39"])
                        ),
                    }
                )
    finally:
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()

    qbase.save_json(output_dir / "counter_projection_clamping_rows.json", rows)
    qbase.save_json(output_dir / "counter_projection_clamping_summary.json", summarize_rows(rows))

    print(f"Counter-projection clamping analysis complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
