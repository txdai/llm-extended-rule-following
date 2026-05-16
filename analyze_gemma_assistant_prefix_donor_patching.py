#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from analyze_gemma_counting import (
    HOOK_SPECS,
    RESIDUAL_HOOKS,
    ActivationRecorder,
    parse_integer,
    render_prompts,
    resolve_device,
    save_json,
    stream_and_layer,
    tokenize_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch the full assistant-prefix hidden state with donor states from successful fake-count prompts."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/gemma-3-27b-it"),
        help="Local Gemma 3 27B IT path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gemma_assistant_prefix_donor_patching"),
        help="Directory for donor patch payloads.",
    )
    parser.add_argument(
        "--base-count",
        type=int,
        default=10,
        help="True count of the fixed prompt to patch.",
    )
    parser.add_argument(
        "--fake-counts",
        type=int,
        nargs="*",
        default=None,
        help="Donor fake counts to test. Defaults to the successful exact prefix 1..26.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use. Defaults to auto -> mps, cuda, then cpu.",
    )
    return parser.parse_args()


def collect_assistant_prefix_states(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer,
    counts: list[int],
    device: str,
) -> tuple[dict[int, str], dict[int, int], dict[str, dict[int, np.ndarray]]]:
    prompts = render_prompts(processor, counts)
    prompt_by_count = {int(count): prompt for count, prompt in zip(counts, prompts)}
    position_by_count: dict[int, int] = {}
    activations: dict[str, dict[int, np.ndarray]] = {label: {} for label in RESIDUAL_HOOKS}
    recorder = ActivationRecorder(model)
    recorder.register()
    try:
        for count in counts:
            prompt = prompt_by_count[int(count)]
            encoded = tokenize_prompts(tokenizer, [prompt], device)
            last_position = int(encoded["attention_mask"][0].sum().item()) - 1
            position_by_count[int(count)] = last_position
            recorder.clear()
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            for label in RESIDUAL_HOOKS:
                tensor = recorder.cache[label]
                activations[label][int(count)] = tensor[0, last_position].detach().float().cpu().numpy()
    finally:
        recorder.close()
    return prompt_by_count, position_by_count, activations


class PrefixDonorPatcher:
    def __init__(
        self,
        model: Gemma3ForConditionalGeneration,
        module_name: str,
        prompt_len: int,
        target_position: int,
        donor_state: np.ndarray,
    ) -> None:
        self.module = dict(model.named_modules())[module_name]
        self.prompt_len = int(prompt_len)
        self.target_position = int(target_position)
        self.donor_state = torch.from_numpy(donor_state.astype(np.float32)).to(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        self.handle: torch.utils.hooks.RemovableHandle | None = None
        self.applied = False

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        if self.applied:
            return output
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor output for residual patching, got {type(output)!r}")
        if int(output.shape[1]) != self.prompt_len:
            return output
        patched = output.clone()
        patched[:, self.target_position, :] = self.donor_state.unsqueeze(0)
        self.applied = True
        return patched

    def __enter__(self) -> "PrefixDonorPatcher":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def generate_prediction(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer,
    prompt: str,
    device: str,
) -> tuple[str, int | None]:
    encoded = tokenize_prompts(tokenizer, [prompt], device)
    prompt_len = int(encoded["attention_mask"][0].sum().item())
    with torch.inference_mode():
        sequences = model.generate(**encoded, max_new_tokens=12, do_sample=False)
    continuation = sequences[0, prompt_len:]
    text = processor.decode(continuation, skip_special_tokens=True).strip()
    return text, parse_integer(text)


def summarize_rows(
    rows: list[dict[str, object]],
    base_count: int,
    baseline_prediction: int | None,
    baseline_text: str,
    fake_counts: list[int],
) -> dict[str, object]:
    by_label = {label: [row for row in rows if str(row["label"]) == label] for label in RESIDUAL_HOOKS}
    layer_summaries: list[dict[str, object]] = []
    for label in RESIDUAL_HOOKS:
        subset = by_label[label]
        predictions = [row["patched_prediction"] for row in subset if row["patched_prediction"] is not None]
        unique_predictions = sorted({int(value) for value in predictions})
        hits = sum(
            int(row["patched_prediction"]) == int(row["fake_count"])
            for row in subset
            if row["patched_prediction"] is not None
        )
        corr = float("nan")
        if predictions:
            valid = [
                (int(row["fake_count"]), int(row["patched_prediction"]))
                for row in subset
                if row["patched_prediction"] is not None
            ]
            if len(valid) >= 2:
                fake = np.array([item[0] for item in valid], dtype=np.float32)
                pred = np.array([item[1] for item in valid], dtype=np.float32)
                corr = float(np.corrcoef(fake, pred)[0, 1])
        shifted = [
            row for row in subset
            if baseline_prediction is not None and row["patched_prediction"] is not None and int(row["patched_prediction"]) != int(baseline_prediction)
        ]
        if shifted:
            strongest = max(shifted, key=lambda row: abs(int(row["patched_prediction"]) - int(baseline_prediction)))
            shift_text = (
                f"largest shift at fake count {strongest['fake_count']}: "
                f"{baseline_prediction} -> {strongest['patched_prediction']}"
            )
        else:
            shift_text = "no non-baseline decoded shifts"
        mean_abs_patch_delta = float(np.mean([float(row["mean_abs_patch_delta"]) for row in subset]))
        layer_summaries.append(
            {
                "label": label,
                "target_hits": hits,
                "trial_count": len(fake_counts),
                "unique_predictions": unique_predictions,
                "fake_vs_prediction_correlation": corr,
                "mean_abs_patch_delta": mean_abs_patch_delta,
                "largest_nonbaseline_shift": shift_text,
            }
        )
    layer53 = by_label["resid_post.layer53"]
    exact_hits53 = [
        int(row["fake_count"])
        for row in layer53
        if row["patched_prediction"] is not None and int(row["patched_prediction"]) == int(row["fake_count"])
    ]
    nontrivial_hits53 = [count for count in exact_hits53 if count != base_count]
    late_plateau53 = [
        int(row["fake_count"])
        for row in layer53
        if row["patched_prediction"] is not None and int(row["patched_prediction"]) == 20
    ]
    return {
        "base_count": base_count,
        "baseline_prediction": baseline_prediction,
        "baseline_text": baseline_text,
        "fake_counts": fake_counts,
        "layer_summaries": layer_summaries,
        "layer53_exact_fake_counts": exact_hits53,
        "layer53_nontrivial_exact_fake_counts": nontrivial_hits53,
        "layer53_plateau_20_fake_counts": late_plateau53,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    fake_counts = list(range(1, 27)) if args.fake_counts is None else sorted(set(args.fake_counts))

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    print(f"Using device: {device}")
    print("Loading Gemma 3 27B IT...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    counts_to_collect = sorted({int(args.base_count), *fake_counts})
    print(f"Collecting assistant-prefix donor states for counts {counts_to_collect}...")
    prompt_by_count, prefix_positions, activations = collect_assistant_prefix_states(
        model,
        processor,
        tokenizer,
        counts_to_collect,
        device,
    )

    base_prompt = prompt_by_count[int(args.base_count)]
    base_position = prefix_positions[int(args.base_count)]
    encoded_base = tokenize_prompts(tokenizer, [base_prompt], device)
    base_prompt_len = int(encoded_base["attention_mask"][0].sum().item())
    baseline_text, baseline_prediction = generate_prediction(model, processor, tokenizer, base_prompt, device)

    rows: list[dict[str, object]] = []
    for label in RESIDUAL_HOOKS:
        print(f"Testing assistant-prefix donor patches at {label}...")
        module_name = str(HOOK_SPECS[label]["hook_name"])
        base_state = activations[label][int(args.base_count)]
        for fake_count in fake_counts:
            donor_state = activations[label][int(fake_count)]
            with PrefixDonorPatcher(
                model,
                module_name,
                base_prompt_len,
                base_position,
                donor_state,
            ):
                patched_text, patched_prediction = generate_prediction(
                    model,
                    processor,
                    tokenizer,
                    base_prompt,
                    device,
                )
            stream, layer = stream_and_layer(label)
            rows.append(
                {
                    "label": label,
                    "stream": stream,
                    "layer": layer,
                    "base_count": int(args.base_count),
                    "fake_count": int(fake_count),
                    "base_prediction": baseline_prediction,
                    "patched_prediction": patched_prediction,
                    "patched_text": patched_text,
                    "assistant_prefix_position": int(base_position),
                    "mean_abs_patch_delta": float(np.mean(np.abs(donor_state - base_state))),
                }
            )

    save_json(output_dir / "assistant_prefix_donor_patching_rows.json", rows)
    save_json(
        output_dir / "patch_metadata.json",
        {
            "base_count": int(args.base_count),
            "baseline_prediction": baseline_prediction,
            "baseline_text": baseline_text,
            "fake_counts": fake_counts,
            "residual_hooks": RESIDUAL_HOOKS,
            "assistant_prefix_position": int(base_position),
        },
    )
    save_json(
        output_dir / "assistant_prefix_donor_patching_summary.json",
        summarize_rows(rows, int(args.base_count), baseline_prediction, baseline_text, fake_counts),
    )

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Assistant-prefix donor patch analysis complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
