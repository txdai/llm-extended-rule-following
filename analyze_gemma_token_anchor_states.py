#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from analyze_gemma_counting import (
    FEATURE_STRUCTURE_K,
    HOOK_ORDER,
    HOOK_SPECS,
    RESIDUAL_HOOKS,
    TOP_FEATURE_SAVE_K,
    ActivationRecorder,
    GemmaScopeSAE,
    fit_counter_direction,
    fit_principal_components,
    load_json,
    pearson,
    render_prompts,
    resolve_device,
    safe_label,
    save_json,
    sequence_token_masks,
    stream_and_layer,
    tokenize_prompts,
    top_feature_table,
    write_csv,
)

ANCHOR_ORDER = ["assistant_prefix", "last_item", "last_separator"]
ANCHOR_TITLES = {
    "assistant_prefix": "Assistant Prefix Token",
    "last_item": "Last Repeated Item Token",
    "last_separator": "Last Separator Token",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma anchor-state analysis for assistant-prefix, last-item, and last-separator tokens."
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
        "--source-analysis-dir",
        type=Path,
        default=Path("data/gemma_counting_mechanistic_analysis"),
        help="Existing Gemma counting analysis directory to reuse count sweep records from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gemma_token_anchor_state_analysis"),
        help="Directory for anchor-specific outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for activation collection and SAE encoding.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use. Defaults to auto -> mps, cuda, then cpu.",
    )
    parser.add_argument(
        "--feature-rank-max",
        type=int,
        default=26,
        help="Rank sparse features on counts 1..N, then record their full trajectories over the entire anchor-valid sweep.",
    )
    return parser.parse_args()


def make_anchor_subdir(root: Path, anchor: str) -> Path:
    path = root / anchor
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"features_{safe_label(label)}.json"


def feature_trajectory_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"feature_trajectories_{safe_label(label)}.npz"


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    exact_counts = [int(row["count"]) for row in records if bool(row["exact"])]
    failed_rows = [row for row in records if not bool(row["exact"])]
    failed_predictions = [
        int(row["parsed_prediction"])
        for row in failed_rows
        if row.get("parsed_prediction") is not None
    ]
    first_failure = next((int(row["count"]) for row in records if not bool(row["exact"])), None)
    initial_exact_prefix = 0
    for row in records:
        if not bool(row["exact"]):
            break
        initial_exact_prefix = int(row["count"])
    top_failed = {}
    for value in failed_predictions:
        top_failed[value] = top_failed.get(value, 0) + 1
    attractors = sorted(top_failed.items(), key=lambda item: (-item[1], item[0]))[:8]
    late_exact = [count for count in exact_counts if first_failure is not None and count > first_failure]
    return {
        "initial_exact_prefix": initial_exact_prefix,
        "first_failure": first_failure,
        "max_exact_count": max(exact_counts, default=0),
        "late_exact_pockets": late_exact,
        "accuracy": float(sum(bool(row["exact"]) for row in records) / max(len(records), 1)),
        "top_failed_attractors": [[int(value), int(freq)] for value, freq in attractors],
    }


def anchor_positions_for_prompt(
    prompt: str,
    count: int,
    tokenizer,
) -> dict[str, int | None]:
    sequence = ", ".join("a" for _ in range(count))
    seq_mask, item_mask, sep_mask = sequence_token_masks(prompt, sequence, tokenizer)
    if seq_mask.size == 0:
        raise ValueError(f"Unexpected empty prompt tokenization for count {count}")
    return {
        "assistant_prefix": int(seq_mask.shape[0] - 1),
        "last_item": int(np.flatnonzero(item_mask)[-1]) if item_mask.any() else None,
        "last_separator": int(np.flatnonzero(sep_mask)[-1]) if sep_mask.any() else None,
    }


def collect_anchor_raw_activations(
    model: Gemma3ForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer,
    counts: list[int],
    batch_size: int,
    device: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[dict[str, int | None]]]]:
    prompts = render_prompts(processor, counts)
    prompt_rows = [
        {
            "count": int(count),
            "positions": anchor_positions_for_prompt(prompt, int(count), tokenizer),
        }
        for count, prompt in zip(counts, prompts)
    ]
    prompt_map = {int(row["count"]): row["positions"] for row in prompt_rows}

    recorder = ActivationRecorder(model)
    recorder.register()
    activations: dict[str, dict[str, list[np.ndarray]]] = {
        anchor: {label: [] for label in HOOK_ORDER} for anchor in ANCHOR_ORDER
    }
    valid_counts: dict[str, list[int]] = {anchor: [] for anchor in ANCHOR_ORDER}
    try:
        for batch_start in range(0, len(counts), batch_size):
            count_batch = counts[batch_start : batch_start + batch_size]
            prompt_batch = prompts[batch_start : batch_start + batch_size]
            encoded = tokenize_prompts(tokenizer, prompt_batch, device)
            recorder.clear()
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            for anchor in ANCHOR_ORDER:
                batch_rows: list[int] = []
                batch_positions: list[int] = []
                batch_counts: list[int] = []
                for batch_index, count in enumerate(count_batch):
                    position = prompt_map[int(count)][anchor]
                    if position is None:
                        continue
                    batch_rows.append(batch_index)
                    batch_positions.append(int(position))
                    batch_counts.append(int(count))
                if not batch_rows:
                    continue
                row_tensor = torch.tensor(batch_rows, dtype=torch.long, device=device)
                pos_tensor = torch.tensor(batch_positions, dtype=torch.long, device=device)
                for label in HOOK_ORDER:
                    tensor = recorder.cache[label]
                    expected_hidden = int(HOOK_SPECS[label]["hidden_size"])
                    if int(tensor.shape[-1]) != expected_hidden:
                        raise ValueError(
                            f"{label} produced hidden size {tensor.shape[-1]}, expected {expected_hidden}"
                        )
                    selected = tensor[row_tensor, pos_tensor]
                    activations[anchor][label].append(selected.detach().float().cpu().numpy())
                valid_counts[anchor].extend(batch_counts)
    finally:
        recorder.close()

    final_activations = {
        anchor: {
            label: np.concatenate(chunks, axis=0)
            for label, chunks in by_hook.items()
            if chunks
        }
        for anchor, by_hook in activations.items()
    }
    position_rows = []
    for row in prompt_rows:
        payload = {"count": int(row["count"])}
        payload.update({key: value for key, value in row["positions"].items()})
        position_rows.append(payload)
    return final_activations, {"position_rows": position_rows, "valid_counts": valid_counts}


def build_anchor_summary(
    anchor: str,
    records: list[dict[str, object]],
    run_summary: dict[str, object],
    hook_leaderboard: list[dict[str, object]],
    residual_counter_metrics: list[dict[str, object]],
) -> dict[str, object]:
    best_hook = max(hook_leaderboard, key=lambda row: abs(float(row["best_feature_pearson"])))
    best_residual = max(
        residual_counter_metrics,
        key=lambda row: float(row["successful_holdout_projection_corr"]),
    )
    return {
        "anchor": anchor,
        "anchor_title": ANCHOR_TITLES[anchor],
        "count_span": [int(records[0]["count"]), int(records[-1]["count"])],
        "count_total": len(records),
        "run_summary": run_summary,
        "best_hook": best_hook,
        "best_residual": best_residual,
        "residual_counter_metrics": residual_counter_metrics,
    }


def build_root_summary(anchor_results: dict[str, dict[str, object]]) -> dict[str, object]:
    assistant = anchor_results["assistant_prefix"]
    item = anchor_results["last_item"]
    separator = anchor_results["last_separator"]
    return {
        "anchors": anchor_results,
        "anchor_order": ANCHOR_ORDER,
        "anchor_titles": ANCHOR_TITLES,
        "best_sparse_feature_anchor": max(
            anchor_results.values(),
            key=lambda result: abs(float(result["best_hook"]["best_feature_pearson"])),
        ),
        "best_residual_anchor": max(
            anchor_results.values(),
            key=lambda result: float(result["best_residual"]["successful_holdout_projection_corr"]),
        ),
        "assistant_prefix_best_residual": assistant["best_residual"],
        "last_item_best_residual": item["best_residual"],
        "last_separator_best_residual": separator["best_residual"],
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    count_sweep_records = load_json(args.source_analysis_dir / "count_sweep_records.json")
    record_map = {int(row["count"]): row for row in count_sweep_records}
    counts = sorted(record_map)

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    print(f"Using device: {device}")
    print("Loading Gemma 3 27B IT for anchor-state extraction...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    print("Collecting anchor-specific raw activations...")
    anchor_activations, anchor_metadata = collect_anchor_raw_activations(
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

    save_json(output_dir / "anchor_positions.json", anchor_metadata["position_rows"])
    save_json(output_dir / "anchor_valid_counts.json", anchor_metadata["valid_counts"])

    anchor_results: dict[str, dict[str, object]] = {}
    for anchor in ANCHOR_ORDER:
        anchor_dir = make_anchor_subdir(output_dir, anchor)
        anchor_counts = [int(value) for value in anchor_metadata["valid_counts"][anchor]]
        anchor_records = [record_map[count] for count in anchor_counts]
        run_summary = summarize_records(anchor_records)
        counts_array = np.array(anchor_counts, dtype=np.int32)
        exact_mask = np.array([bool(row["exact"]) for row in anchor_records], dtype=bool)
        exact_prefix_mask = counts_array <= int(run_summary["initial_exact_prefix"])
        post_failure_mask = (
            counts_array >= int(run_summary["first_failure"])
            if run_summary["first_failure"] is not None
            else np.zeros_like(counts_array, dtype=bool)
        )
        late_exact_pockets = set(int(item) for item in run_summary["late_exact_pockets"])
        late_exact_mask = np.array([count in late_exact_pockets for count in counts_array], dtype=bool)
        feature_rank_mask = counts_array <= min(int(args.feature_rank_max), int(counts_array[-1]))
        rank_window_label = f"counts 1..{int(np.max(counts_array[feature_rank_mask]))}"
        exact_prefix_indices = np.flatnonzero(exact_prefix_mask)
        holdout_train_mask = np.zeros_like(exact_prefix_mask, dtype=bool)
        holdout_eval_mask = np.zeros_like(exact_prefix_mask, dtype=bool)
        split_index = max(1, exact_prefix_indices.size // 2)
        holdout_train_mask[exact_prefix_indices[:split_index]] = True
        holdout_eval_mask[exact_prefix_indices[split_index:]] = True
        if not np.any(holdout_eval_mask):
            holdout_eval_mask = exact_prefix_mask.copy()

        save_json(anchor_dir / "count_sweep_records.json", anchor_records)
        save_json(anchor_dir / "run_summary.json", run_summary)

        hook_leaderboard: list[dict[str, object]] = []
        stream_layer_summary: list[dict[str, object]] = []
        residual_counter_metrics: list[dict[str, object]] = []
        exact_failed_comparison: list[dict[str, object]] = []
        analysis_arrays: dict[str, np.ndarray] = {
            "counts": counts_array,
            "exact_mask": exact_mask,
            "exact_prefix_mask": exact_prefix_mask,
            "post_failure_mask": post_failure_mask,
        }

        for label in HOOK_ORDER:
            print(f"[{anchor}] Encoding features for {label}...")
            safe = safe_label(label)
            stream, layer = stream_and_layer(label)
            sae = GemmaScopeSAE(args.scope_dir / Path(HOOK_SPECS[label]["config"]), device)
            activations = anchor_activations[anchor][label]
            chunks: list[np.ndarray] = []
            for start in range(0, activations.shape[0], args.batch_size):
                stop = start + args.batch_size
                batch = torch.from_numpy(activations[start:stop]).to(device=device)
                with torch.inference_mode():
                    encoded = sae.encode(batch)
                chunks.append(encoded.detach().float().cpu().numpy())
            encoded_features = np.concatenate(chunks, axis=0)

            feature_rows = top_feature_table(
                encoded_features,
                counts_array,
                top_k=TOP_FEATURE_SAVE_K,
                rank_mask=feature_rank_mask,
            )
            save_json(feature_path(anchor_dir, label), feature_rows)

            best_row = feature_rows[0]
            top5_mean_abs = float(np.mean([abs(float(row["correlation"])) for row in feature_rows[:5]]))
            best_feature_values = encoded_features[:, int(best_row["feature_id"])]
            exact_vs_failed_sep = (
                float(best_feature_values[exact_prefix_mask].mean() - best_feature_values[post_failure_mask].mean())
                if np.any(exact_prefix_mask) and np.any(post_failure_mask)
                else float("nan")
            )
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

            top_ids = [int(row["feature_id"]) for row in feature_rows[:TOP_FEATURE_PLOT_K]]
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
                feature_trajectory_path(anchor_dir, label),
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
                coords, explained = fit_principal_components(activations)
                center, direction, projected, rho, delta = fit_counter_direction(
                    activations,
                    counts_array,
                    exact_prefix_mask,
                )
                holdout_center, holdout_direction, holdout_projected, _holdout_rho, _holdout_delta = fit_counter_direction(
                    activations,
                    counts_array,
                    holdout_train_mask,
                )
                unit_direction = direction / np.clip(np.linalg.norm(direction), 1e-8, None)
                exact_prefix_projection = ((activations - center) @ unit_direction).astype(np.float32)
                exact_prefix_std = float(np.std(exact_prefix_projection[exact_prefix_mask]))
                projection_count_corr = pearson(projected, counts_array.astype(np.float32))
                exact_prefix_projection_corr = pearson(
                    projected[exact_prefix_mask],
                    counts_array[exact_prefix_mask].astype(np.float32),
                )
                post_failure_projection_corr = (
                    pearson(projected[post_failure_mask], counts_array[post_failure_mask].astype(np.float32))
                    if np.sum(post_failure_mask) >= 2
                    else float("nan")
                )
                successful_holdout_projection_corr = pearson(
                    holdout_projected[holdout_eval_mask],
                    counts_array[holdout_eval_mask].astype(np.float32),
                )
                adjacent_exact = np.diff(projected[exact_prefix_mask])
                eta_projection_threshold = None
                if run_summary["first_failure"] is not None and int(run_summary["first_failure"]) >= 2:
                    failure_count = int(run_summary["first_failure"])
                    if failure_count in anchor_counts and (failure_count - 1) in anchor_counts:
                        boundary_index = anchor_counts.index(failure_count) - 1
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
                        "successful_holdout_projection_corr": float(successful_holdout_projection_corr),
                        "successful_holdout_train_counts": counts_array[holdout_train_mask].astype(int).tolist(),
                        "successful_holdout_eval_counts": counts_array[holdout_eval_mask].astype(int).tolist(),
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
        save_json(anchor_dir / "hook_leaderboard.json", hook_leaderboard_sorted)
        write_csv(anchor_dir / "hook_leaderboard.csv", hook_leaderboard_sorted)
        save_json(anchor_dir / "stream_layer_summary.json", stream_layer_summary)
        save_json(anchor_dir / "residual_counter_metrics.json", residual_counter_metrics)
        save_json(anchor_dir / "exact_failed_comparison.json", exact_failed_comparison)
        np.savez_compressed(anchor_dir / "residual_analysis_arrays.npz", **analysis_arrays)
        save_json(
            anchor_dir / "anchor_summary.json",
            build_anchor_summary(anchor, anchor_records, run_summary, hook_leaderboard_sorted, residual_counter_metrics),
        )

        anchor_results[anchor] = {
            "best_hook": hook_leaderboard_sorted[0],
            "best_residual": max(
                residual_counter_metrics,
                key=lambda row: float(row["successful_holdout_projection_corr"]),
            ),
            "run_summary": run_summary,
            "num_counts": len(anchor_counts),
            "valid_span": (anchor_counts[0], anchor_counts[-1]),
        }

    save_json(output_dir / "anchor_comparison_summary.json", build_root_summary(anchor_results))
    save_json(
        output_dir / "anchor_metadata.json",
        {
            "anchors": ANCHOR_ORDER,
            "anchor_titles": ANCHOR_TITLES,
            "feature_rank_max": int(args.feature_rank_max),
            "hook_order": HOOK_ORDER,
            "residual_hooks": RESIDUAL_HOOKS,
        },
    )
    print(f"Anchor-state analysis complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
