#!/usr/bin/env bash
# Run the round-2 experiments for one model in sequence.
#
#   scripts/round2_chain.sh <model> [steps]
#
# steps: comma list from accumulate,copy,retrieval,logits (default all).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; STEPS="${2:-accumulate,copy,retrieval,logits}"
for step in ${STEPS//,/ }; do
  case "$step" in
    accumulate) scripts/round2_suite.sh "$MODEL" accumulate ;;
    copy)       scripts/round2_suite.sh "$MODEL" output --task copy ;;
    retrieval)  scripts/round2_suite.sh "$MODEL" output --task retrieval ;;
    logits)     scripts/round2_suite.sh "$MODEL" logits ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done
echo "chain done $MODEL"
