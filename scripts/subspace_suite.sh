#!/usr/bin/env bash
# Subspace and periodic-feature controls for one model of the mechanistic suite.
#
#   scripts/subspace_suite.sh <model> [groups] [extra subspace_intervene args...]
#
# Steps: multi-prompt collect (skipped if present) -> offline readouts and
# bases -> causal groups on the GPU -> offline readouts again with the decoder
# subspace (principal angles, logit lens). Run `python -m
# counting.probes.suite.report --suite-dir results/suite` afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; shift
# (not GROUPS: that name is reserved by bash and assignments to it are ignored)
RUN_GROUPS="${1:-unembed,localize,transplant,interp,ablate,periodic}"
if [ $# -gt 0 ]; then shift; fi
PY=".venv/bin/python -u"
export PYTHONPATH=src

case "$MODEL" in
  gpt-oss-20b)    SCAN=scan-final;   PREFILL='<|channel|>final<|message|>'; KW='' ;;
  gpt-oss-120b)   SCAN=scan-final;   PREFILL='<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>'; KW='' ;;
  gemma-4-26b-a4b-it) SCAN=scan-nothink; PREFILL=''; KW='' ;;
  muse-glimmer-30b) SCAN=scan-final; PREFILL=' to=user<|message|>'; KW='' ;;
  qwen3.6-35b-a3b) SCAN=scan-nothink; PREFILL=''; KW='{"enable_thinking": false}' ;;
  llama-3.3-70b-instruct) SCAN=scan-final; PREFILL=''; KW='' ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac

R="results/suite/$MODEL"
CFG="configs/probes/$MODEL.yaml"
COMMON=(--scan-dir "$R/$SCAN" --prefill "$PREFILL")
[ -n "$KW" ] && COMMON+=(--chat-kwargs "$KW")

if [ ! -f "$R/collect-multi/activations.npz" ]; then
  $PY -m counting.probes.suite.collect_multi "$CFG" --output-dir "$R/collect-multi" "${COMMON[@]}"
fi

OFFLINE=(--collect-dir "$R/collect-multi" --single-collect-dir "$R/collect" --scan-dir "$R/$SCAN"
         --interventions-dir "$R/interventions" --output-dir "$R/subspace")
$PY -m counting.probes.suite.subspace "${OFFLINE[@]}"

$PY -m counting.probes.suite.subspace_intervene "$CFG" --collect-dir "$R/collect" \
    --interventions-dir "$R/interventions" --subspace-dir "$R/subspace" \
    --groups "$RUN_GROUPS" "${COMMON[@]}" "$@"

$PY -m counting.probes.suite.subspace "${OFFLINE[@]}" --unembed "$R/subspace/unembed.npz"
echo "done $MODEL"
