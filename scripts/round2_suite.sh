#!/usr/bin/env bash
# Round-2 mechanistic controls for one model (docs/plans/mechanistic-controls-round2.md).
#
#   scripts/round2_suite.sh <model> <experiment> [extra args...]
#
# experiment: accumulate (E4), output (E1), logits (E2). E3 is offline
# (`python -m counting.probes.suite.round2_report`). Per-model prefill, chat
# kwargs and scan directory are the suite's; K is the trajectory dimension
# that carried 90% of transfer at the last site in the subspace controls.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; EXPERIMENT="$2"; shift 2
PY=".venv/bin/python -u"
export PYTHONPATH=src

case "$MODEL" in
  gpt-oss-20b)    SCAN=scan-final;   PREFILL='<|channel|>final<|message|>'; KW=''; K=16 ;;
  gpt-oss-120b)   SCAN=scan-final;   PREFILL='<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>'; KW=''; K=16 ;;
  gemma-4-26b-a4b-it) SCAN=scan-nothink; PREFILL=''; KW=''; K=8 ;;
  muse-glimmer-30b) SCAN=scan-final; PREFILL=' to=user<|message|>'; KW=''; K=32 ;;
  qwen3.6-35b-a3b) SCAN=scan-nothink; PREFILL=''; KW='{"enable_thinking": false}'; K=16 ;;
  llama-3.3-70b-instruct) SCAN=scan-final; PREFILL=''; KW=''; K=16 ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac

R="results/suite/$MODEL"
CFG="configs/probes/$MODEL.yaml"
COMMON=(--scan-dir "$R/$SCAN" --prefill "$PREFILL")
[ -n "$KW" ] && COMMON+=(--chat-kwargs "$KW")

case "$EXPERIMENT" in
  accumulate)
    $PY -m counting.probes.suite.accumulate "$CFG" --output-dir "$R/round2" --k "$K" "${COMMON[@]}" "$@" ;;
  output)
    $PY -m counting.probes.suite.output_channel "$CFG" --output-dir "$R/round2" \
        --collect-dir "$R/collect" --interventions-dir "$R/interventions" --subspace-dir "$R/subspace" "${COMMON[@]}" "$@" ;;
  logits)
    $PY -m counting.probes.suite.logit_interp "$CFG" --output-dir "$R/round2" \
        --collect-dir "$R/collect" --interventions-dir "$R/interventions" --subspace-dir "$R/subspace" "${COMMON[@]}" "$@" ;;
  *) echo "unknown experiment $EXPERIMENT"; exit 1 ;;
esac
echo "done $MODEL $EXPERIMENT"
