#!/usr/bin/env bash
# Dump teacher hidden states for offline EAGLE3 training.
# gpt-oss-120b is MXFP4; use one process with device_map=auto across all GPUs.
# Do NOT use 8-way DP here — each worker would need a full 120B copy.
set -euo pipefail

ROOT="${EAGLE3_ROOT:-$HOME/eagle3-gptoss}"
MODEL="${MODEL_ROOT:-$HOME/models}/gpt-oss-120b"
MODELOPT="${MODELOPT_ROOT:-/tmp/Model-Optimizer}"
HS="${ROOT}/hidden_states/gpt-oss-120b"
TRAIN_JSONL="${1:-${ROOT}/01_data/train.jsonl}"
SUB="${2:-}"  # optional: head -n N subset for dry-run

MO="${MODELOPT}/examples/speculative_decoding"
DUMP="${MO}/collect_hidden_states/compute_hidden_states_hf.py"

if [[ ! -f "${DUMP}" ]]; then
  echo "Model-Optimizer not found at ${MODELOPT}. Clone and pip install -e '.[hf]' first."
  exit 1
fi

INPUT="${TRAIN_JSONL}"
if [[ -n "${SUB}" ]]; then
  INPUT="${ROOT}/01_data/.hs_subset.jsonl"
  head -n "${SUB}" "${TRAIN_JSONL}" > "${INPUT}"
  echo "Using ${SUB}-row subset -> ${INPUT}"
fi

mkdir -p "${HS}"
echo "Dumping hidden states -> ${HS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  python3 "${DUMP}" \
  --model "${MODEL}" \
  --input-data "${INPUT}" \
  --output-dir "${HS}" \
  --trust_remote_code

count=$(find "${HS}" -name '*.pt' | wc -l)
echo "wrote ${count} .pt files under ${HS}"
test "${count}" -gt 0
