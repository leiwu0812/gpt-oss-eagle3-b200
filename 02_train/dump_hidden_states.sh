#!/usr/bin/env bash
# Dump teacher hidden states for offline EAGLE3 training.
# gpt-oss-120b is MXFP4: pin transformers 4.56 + kernels; 8-way DP (1 GPU / worker).
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

# MXFP4 on B200: transformers 5.x + hub_kernels breaks; keep container torch, pin 4.56.
pip install -q "transformers==4.56.2" kernels accelerate safetensors tokenizers huggingface-hub

INPUT="${TRAIN_JSONL}"
if [[ -n "${SUB}" ]]; then
  INPUT="${ROOT}/01_data/.hs_subset.jsonl"
  head -n "${SUB}" "${TRAIN_JSONL}" > "${INPUT}"
  echo "Using ${SUB}-row subset -> ${INPUT}"
fi

mkdir -p "${HS}"
echo "Dumping hidden states -> ${HS}"

DP="${DP_SIZE:-8}"
TMP="${ROOT}/01_data/.hs_part_"
split -n "l/${DP}" --numeric-suffixes=0 -d --additional-suffix=.jsonl "${INPUT}" "${TMP}"

pids=()
for i in $(seq 0 $((DP - 1))); do
  part="${TMP}$(printf '%02d' "$i").jsonl"
  lines=$(wc -l < "${part}")
  [[ "${lines}" -gt 0 ]] || continue
  echo "worker ${i}: ${lines} rows on GPU ${i}"
  CUDA_VISIBLE_DEVICES="${i}" python3 "${DUMP}" \
    --model "${MODEL}" \
    --input-data "${part}" \
    --output-dir "${HS}" \
    --trust_remote_code \
    --dp-rank 0 \
    --dp-world-size 1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
rm -f "${TMP}"*.jsonl

count=$(find "${HS}" -name '*.pt' | wc -l)
echo "wrote ${count} .pt files under ${HS}"
du -sh "${HS}"
test "${count}" -gt 0
test "${status}" -eq 0
