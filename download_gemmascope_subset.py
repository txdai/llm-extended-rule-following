#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "google/gemma-scope-2-27b-it"

# The 27B IT subset layers exposed by Gemma Scope 2 are 16, 31, 40, 53.
COUNTING_EXTENDED_PREFIXES: list[str] = [
    "resid_post/layer_16_width_65k_l0_medium",
    "resid_post/layer_31_width_65k_l0_medium",
    "resid_post/layer_40_width_65k_l0_medium",
    "resid_post/layer_53_width_65k_l0_medium",
    "attn_out/layer_31_width_65k_l0_medium",
    "attn_out/layer_40_width_65k_l0_medium",
    "mlp_out/layer_31_width_65k_l0_medium",
    "mlp_out/layer_40_width_65k_l0_medium",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Gemma Scope 2 counting-extended SAE subset "
            "(same layer list as analyze_gemma_counting.py / analyze_gemma_motif_invariance.py)."
        )
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("model") / "gemma-scope-2-27b-it",
        help="Destination directory for downloaded artifacts.",
    )
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="Also download examples.safetensors. Off by default to save disk.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel download workers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matched patterns and exit without downloading.",
    )
    return parser.parse_args()


def build_patterns(prefixes: list[str], include_examples: bool) -> list[str]:
    patterns: list[str] = []
    for prefix in prefixes:
        patterns.append(f"{prefix}/config.json")
        patterns.append(f"{prefix}/params.safetensors")
        if include_examples:
            patterns.append(f"{prefix}/examples.safetensors")
    return patterns


def main() -> None:
    args = parse_args()
    prefixes = COUNTING_EXTENDED_PREFIXES
    allow_patterns = build_patterns(prefixes, include_examples=args.include_examples)

    print(f"Repo: {REPO_ID}")
    print("Subset: counting-extended (mechanistic analysis default)")
    print(f"Local dir: {args.local_dir}")
    print("Artifacts:")
    for prefix in prefixes:
        print(f"  - {prefix}")
    print(f"Include examples: {args.include_examples}")

    if args.dry_run:
        return

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(args.local_dir),
        allow_patterns=allow_patterns,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
