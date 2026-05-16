#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

import analyze_gemma_counting as base




BASELINE_VARIANT = "a_comma"
FEATURE_HOOKS = list(base.RESIDUAL_HOOKS)
TOP_FEATURE_SAVE_K = 64
FEATURE_RANK_MAX = 26


@dataclass(frozen=True)
class VariantSpec:
    key: str
    item: str
    delimiter: str
    label: str


VARIANTS = [
    VariantSpec("a_comma", "a", ", ", "a + comma"),
    VariantSpec("b_comma", "b", ", ", "b + comma"),
    VariantSpec("x_comma", "x", ", ", "x + comma"),
    VariantSpec("aa_comma", "aa", ", ", "aa + comma"),
    VariantSpec("alpha_comma", "α", ", ", "alpha + comma"),
    VariantSpec("beta_comma", "β", ", ", "beta + comma"),
    VariantSpec("han_comma", "汉", ", ", "han + comma"),
    VariantSpec("zhong_comma", "中", ", ", "zhong + comma"),
    VariantSpec("a_space", "a", " ", "a + space"),
    VariantSpec("a_pipe", "a", " | ", "a + pipe"),
    VariantSpec("a_section", "a", " § ", "a + section"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gemma 3 counting directions and SAE features across item / delimiter motifs."
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
        default=Path("data/gemma_motif_invariance_analysis"),
        help="Directory for analysis tables and payloads.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=80,
        help="Dense count sweep upper bound for each motif variant.",
    )
    parser.add_argument(
        "--common-fit-max",
        type=int,
        default=24,
        help="Target shared early-count window for fitting counter directions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation and activation collection.",
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
        help="Use counts 1..N as the canonical ranking window for sparse features across all motifs.",
    )
    return parser.parse_args()


def make_sequence(item: str, delimiter: str, count: int) -> str:
    return delimiter.join(item for _ in range(count))


def build_messages(spec: VariantSpec, count: int) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": base.SYSTEM_TEXT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f'How many "{spec.item}" are in this sequence?\n\n{make_sequence(spec.item, spec.delimiter, count)}',
                }
            ],
        },
    ]


def render_prompts(processor: AutoProcessor, spec: VariantSpec, counts: list[int]) -> list[str]:
    prompts: list[str] = []
    for count in counts:
        prompt = processor.apply_chat_template(
            build_messages(spec, count),
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt, str):
            raise TypeError(f"Expected string prompt, got {type(prompt)!r}")
        prompts.append(prompt)
    return prompts


def run_generation_sweep_for_prompts(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer,
    counts: list[int],
    prompts: list[str],
    batch_size: int,
    device: str,
) -> list[base.SweepRecord]:
    records: list[base.SweepRecord] = []
    for count_batch, prompt_batch in zip(
        base.batched(counts, batch_size),
        base.batched(prompts, batch_size),
    ):
        count_list = [int(item) for item in count_batch]
        prompt_list = [str(item) for item in prompt_batch]
        encoded = base.tokenize_prompts(tokenizer, prompt_list, device)
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
        for index, count in enumerate(count_list):
            continuation = sequences[index, int(input_lengths[index]) :]
            text = processor.decode(continuation, skip_special_tokens=True).strip()
            raw_texts.append(text)
            parsed = base.parse_integer(text)
            predicted_texts.append(str(parsed) if parsed is not None else text)
            parsed_predictions.append(parsed)
        margin_rows = base.teacher_forced_margin_batch(
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


def collect_raw_final_activations(
    model: Gemma3ForConditionalGeneration,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    recorder = base.ActivationRecorder(model)
    recorder.register()
    final_outputs: dict[str, list[np.ndarray]] = {label: [] for label in base.HOOK_ORDER}
    try:
        for prompt_batch in base.batched(prompts, batch_size):
            prompt_list = [str(item) for item in prompt_batch]
            encoded = base.tokenize_prompts(tokenizer, prompt_list, device)
            last_positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(last_positions.shape[0], device=device)
            recorder.clear()
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            for label in base.HOOK_ORDER:
                tensor = recorder.cache[label]
                final_tensor = tensor[batch_indices, last_positions]
                final_outputs[label].append(final_tensor.detach().float().cpu().numpy())
    finally:
        recorder.close()
    return {label: np.concatenate(chunks, axis=0) for label, chunks in final_outputs.items()}


def top_feature_rank(correlations: np.ndarray, feature_id: int) -> int:
    order = np.argsort(-np.abs(correlations))
    rank_index = int(np.flatnonzero(order == feature_id)[0])
    return rank_index + 1


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def feature_trajectory_path(output_dir: Path, variant: str, label: str) -> Path:
    return output_dir / "feature_activation_trajectories" / f"{variant}__{base.safe_label(label)}.npz"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    feature_dir = output_dir / "feature_rankings"
    feature_trajectory_dir = output_dir / "feature_activation_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_trajectory_dir.mkdir(parents=True, exist_ok=True)

    device = base.resolve_device(args.device)
    counts = list(range(1, args.max_count + 1))
    counts_array = np.array(counts, dtype=np.int32)
    feature_rank_mask = counts_array <= min(int(args.feature_rank_max), int(counts_array[-1]))

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    print(f"Using device: {device}")
    print("Loading Gemma 3 27B IT for motif-invariance sweeps...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    records_by_variant: dict[str, list[base.SweepRecord]] = {}
    raw_by_variant: dict[str, dict[str, np.ndarray]] = {}
    behavior_rows: list[dict[str, object]] = []

    for spec in VARIANTS:
        print(f"Running motif variant {spec.key}...")
        prompts = render_prompts(processor, spec, counts)
        records = run_generation_sweep_for_prompts(
            model,
            processor,
            tokenizer,
            counts,
            prompts,
            args.batch_size,
            device,
        )
        raw_activations = collect_raw_final_activations(
            model,
            tokenizer,
            prompts,
            args.batch_size,
            device,
        )
        run_summary = base.summarize_records(records)
        prompt_tokens = np.array([record.prompt_tokens for record in records], dtype=np.float32)
        behavior_row = {
            "variant": spec.key,
            "variant_label": spec.label,
            "item": spec.item,
            "delimiter": spec.delimiter,
            "initial_exact_prefix": int(run_summary["initial_exact_prefix"]),
            "first_failure": run_summary["first_failure"],
            "max_exact_count": int(run_summary["max_exact_count"]),
            "accuracy": float(run_summary["accuracy"]),
            "mean_prompt_tokens": float(prompt_tokens.mean()),
            "top_failed_attractors": run_summary["top_failed_attractors"],
        }
        behavior_rows.append(behavior_row)
        records_by_variant[spec.key] = records
        raw_by_variant[spec.key] = raw_activations
        base.save_json(output_dir / f"sweep_records_{spec.key}.json", [record.__dict__ for record in records])

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()

    direction_payload: dict[str, dict[str, dict[str, object]]] = {spec.key: {} for spec in VARIANTS}
    direction_rows: list[dict[str, object]] = []

    for spec in VARIANTS:
        variant_key = spec.key
        initial_exact_prefix = next(
            int(row["initial_exact_prefix"]) for row in behavior_rows if row["variant"] == variant_key
        )
        fit_max = min(args.common_fit_max, initial_exact_prefix)
        fit_mask = counts_array <= fit_max
        if int(fit_mask.sum()) < 2:
            raise ValueError(f"{variant_key} has too few exact counts for fitting: fit_max={fit_max}")
        for label in base.RESIDUAL_HOOKS:
            x = raw_by_variant[variant_key][label]
            center, direction, projected, rho, delta = base.fit_counter_direction(x, counts_array, fit_mask)
            unit_direction = direction / np.clip(np.linalg.norm(direction), 1e-8, None)
            direction_payload[variant_key][label] = {
                "fit_max": int(fit_max),
                "fit_mask": fit_mask.copy(),
                "center": center.astype(np.float32),
                "direction": direction.astype(np.float32),
                "unit_direction": unit_direction.astype(np.float32),
                "projected": projected.astype(np.float32),
                "rho": float(rho),
                "delta": float(delta),
            }

    baseline_payload = direction_payload[BASELINE_VARIANT]
    for spec in VARIANTS:
        variant_key = spec.key
        for label in base.RESIDUAL_HOOKS:
            payload = direction_payload[variant_key][label]
            fit_mask = np.asarray(payload["fit_mask"], dtype=bool)
            x = raw_by_variant[variant_key][label]
            self_projected = np.asarray(payload["projected"], dtype=np.float32)
            baseline_direction = np.asarray(baseline_payload[label]["unit_direction"], dtype=np.float32)
            center = np.asarray(payload["center"], dtype=np.float32)
            baseline_projected = ((x - center) @ baseline_direction).astype(np.float32)
            cosine = float(
                np.dot(np.asarray(payload["unit_direction"], dtype=np.float32), baseline_direction)
            )
            direction_rows.append(
                {
                    "variant": variant_key,
                    "variant_label": spec.label,
                    "label": label,
                    "fit_max": int(payload["fit_max"]),
                    "rho": float(payload["rho"]),
                    "delta": float(payload["delta"]),
                    "direction_cosine": cosine,
                    "direction_abs_cosine": abs(cosine),
                    "self_projection_corr_fit": base.pearson(
                        self_projected[fit_mask], counts_array[fit_mask].astype(np.float32)
                    ),
                    "self_projection_corr_all": base.pearson(
                        self_projected, counts_array.astype(np.float32)
                    ),
                    "baseline_transfer_corr_fit": base.pearson(
                        baseline_projected[fit_mask], counts_array[fit_mask].astype(np.float32)
                    ),
                    "baseline_transfer_corr_all": base.pearson(
                        baseline_projected, counts_array.astype(np.float32)
                    ),
                }
            )

    feature_rows: list[dict[str, object]] = []
    for label in FEATURE_HOOKS:
        print(f"Encoding motif features for {label}...")
        sae = base.GemmaScopeSAE(args.scope_dir / Path(base.HOOK_SPECS[label]["config"]), device)
        baseline_encoded = base.encode_hook_features(
            sae,
            raw_by_variant[BASELINE_VARIANT][label],
            device,
            args.batch_size,
        )
        baseline_correlations = base.pearson_by_feature(baseline_encoded, counts_array.astype(np.float32))
        baseline_table = base.top_feature_table(
            baseline_encoded,
            counts_array,
            top_k=TOP_FEATURE_SAVE_K,
            rank_mask=feature_rank_mask,
        )
        base.save_json(feature_dir / f"{BASELINE_VARIANT}__{base.safe_label(label)}.json", baseline_table)
        np.savez_compressed(
            feature_trajectory_path(output_dir, BASELINE_VARIANT, label),
            counts=counts_array.astype(np.int32),
            rank_mask=feature_rank_mask.astype(np.int8),
            feature_ids=np.array([int(row["feature_id"]) for row in baseline_table[:TOP_FEATURE_SAVE_K]], dtype=np.int32),
            rank_window_correlations=np.array(
                [float(row["rank_window_correlation"]) for row in baseline_table[:TOP_FEATURE_SAVE_K]],
                dtype=np.float32,
            ),
            full_correlations=np.array(
                [float(row["full_correlation"]) for row in baseline_table[:TOP_FEATURE_SAVE_K]],
                dtype=np.float32,
            ),
            values=baseline_encoded[:, [int(row["feature_id"]) for row in baseline_table[:TOP_FEATURE_SAVE_K]]].astype(np.float32),
        )
        baseline_top64 = {int(row["feature_id"]) for row in baseline_table[:64]}
        baseline_top10 = {int(row["feature_id"]) for row in baseline_table[:10]}
        baseline_best_feature = int(baseline_table[0]["feature_id"])
        baseline_best_corr = float(next(row for row in baseline_table if int(row["feature_id"]) == baseline_best_feature)["rank_window_correlation"])

        for spec in VARIANTS:
            if spec.key == BASELINE_VARIANT:
                encoded = baseline_encoded
                correlations = baseline_correlations
                feature_table = baseline_table
            else:
                encoded = base.encode_hook_features(
                    sae,
                    raw_by_variant[spec.key][label],
                    device,
                    args.batch_size,
                )
                correlations = base.pearson_by_feature(encoded, counts_array.astype(np.float32))
                rank_window_correlations = base.pearson_by_feature(
                    encoded[feature_rank_mask],
                    counts_array[feature_rank_mask].astype(np.float32),
                )
                feature_table = base.top_feature_table(
                    encoded,
                    counts_array,
                    top_k=TOP_FEATURE_SAVE_K,
                    rank_mask=feature_rank_mask,
                )
                base.save_json(feature_dir / f"{spec.key}__{base.safe_label(label)}.json", feature_table)
                np.savez_compressed(
                    feature_trajectory_path(output_dir, spec.key, label),
                    counts=counts_array.astype(np.int32),
                    rank_mask=feature_rank_mask.astype(np.int8),
                    feature_ids=np.array([int(row["feature_id"]) for row in feature_table[:TOP_FEATURE_SAVE_K]], dtype=np.int32),
                    rank_window_correlations=np.array(
                        [float(row["rank_window_correlation"]) for row in feature_table[:TOP_FEATURE_SAVE_K]],
                        dtype=np.float32,
                    ),
                    full_correlations=np.array(
                        [float(row["full_correlation"]) for row in feature_table[:TOP_FEATURE_SAVE_K]],
                        dtype=np.float32,
                    ),
                    values=encoded[:, [int(row["feature_id"]) for row in feature_table[:TOP_FEATURE_SAVE_K]]].astype(np.float32),
                )

            variant_top64 = {int(row["feature_id"]) for row in feature_table[:64]}
            variant_top10 = {int(row["feature_id"]) for row in feature_table[:10]}
            if spec.key == BASELINE_VARIANT:
                rank_window_correlations = base.pearson_by_feature(
                    encoded[feature_rank_mask],
                    counts_array[feature_rank_mask].astype(np.float32),
                )
            feature_rows.append(
                {
                    "variant": spec.key,
                    "variant_label": spec.label,
                    "label": label,
                    "baseline_best_feature_id": baseline_best_feature,
                    "baseline_best_feature_corr_on_baseline": baseline_best_corr,
                    "variant_best_feature_id": int(feature_table[0]["feature_id"]),
                    "variant_best_feature_corr": float(feature_table[0]["rank_window_correlation"]),
                    "same_best_feature": int(int(feature_table[0]["feature_id"]) == baseline_best_feature),
                    "baseline_best_feature_corr_on_variant": float(
                        base.pearson(
                            encoded[feature_rank_mask, baseline_best_feature],
                            counts_array[feature_rank_mask].astype(np.float32),
                        )
                    ),
                    "baseline_best_feature_full_corr_on_variant": float(correlations[baseline_best_feature]),
                    "baseline_best_feature_rank_on_variant": int(
                        top_feature_rank(rank_window_correlations, baseline_best_feature)
                    ),
                    "top10_jaccard": jaccard(baseline_top10, variant_top10),
                    "top64_jaccard": jaccard(baseline_top64, variant_top64),
                }
            )

        del baseline_encoded
        del sae
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()

    behavior_rows_sorted = sorted(behavior_rows, key=lambda row: str(row["variant"]))
    direction_rows_sorted = sorted(direction_rows, key=lambda row: (str(row["variant"]), str(row["label"])))
    feature_rows_sorted = sorted(feature_rows, key=lambda row: (str(row["variant"]), str(row["label"])))

    base.save_json(output_dir / "motif_behavior_summary.json", behavior_rows_sorted)
    base.write_csv(output_dir / "motif_behavior_summary.csv", behavior_rows_sorted)
    base.save_json(output_dir / "direction_alignment_summary.json", direction_rows_sorted)
    base.write_csv(output_dir / "direction_alignment_summary.csv", direction_rows_sorted)
    base.save_json(output_dir / "feature_alignment_summary.json", feature_rows_sorted)
    base.write_csv(output_dir / "feature_alignment_summary.csv", feature_rows_sorted)

    print(f"Motif invariance analysis complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
