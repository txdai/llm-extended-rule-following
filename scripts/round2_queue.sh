#!/usr/bin/env bash
# Two-lane queue for the round-2 runs on one 192-GB workstation: at most two
# models resident at once, llama-3.3-70b alone at the end.
#
#   scripts/round2_queue.sh <logdir>
#
# Assumes the E4 (accumulate) runs for gpt-oss-20b and gemma were launched
# separately and waits for their logs to report done before continuing.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="$1"

wait_for() { until grep -qE "$2" "$1" 2>/dev/null; do sleep 60; done; }

(
  wait_for "$LOG/r2-acc-20b.log" "done gpt-oss-20b|Traceback|Killed"
  scripts/round2_chain.sh gpt-oss-20b copy,retrieval,logits > "$LOG/r2-chain-20b.log" 2>&1
  scripts/round2_chain.sh muse-glimmer-30b > "$LOG/r2-chain-muse.log" 2>&1
  scripts/round2_chain.sh qwen3.6-35b-a3b > "$LOG/r2-chain-qwen.log" 2>&1
  echo "lane A done"
) &
(
  wait_for "$LOG/r2-acc-gemma.log" "done gemma|Traceback|Killed"
  scripts/round2_chain.sh gemma-4-26b-a4b-it copy,retrieval,logits > "$LOG/r2-chain-gemma.log" 2>&1
  scripts/round2_chain.sh gpt-oss-120b > "$LOG/r2-chain-120b.log" 2>&1
  scripts/round2_suite.sh gpt-oss-120b output --task counting --groups layers --layers 20-27 > "$LOG/r2-120b-layers.log" 2>&1
  echo "lane B done"
) &
wait
scripts/round2_chain.sh llama-3.3-70b-instruct > "$LOG/r2-chain-llama.log" 2>&1
echo "queue done"
