#!/usr/bin/env bash
# Step 4 (inside TRAINING container): dump teacher hidden states via HF backend.
# Data-parallel across all visible GPUs (one full teacher copy per GPU; fits on B200).
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/data/models/gpt-oss-120b}"
INPUT="${INPUT:-/data/synthetic/train.jsonl}"
HS_DIR="${HS_DIR:-/data/hidden_states/gpt-oss-120b}"
DP_SIZE="${DP_SIZE:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"

mkdir -p "$HS_DIR"
if [ ! -s "$INPUT" ]; then
    echo "ERROR: input file missing or empty: $INPUT" >&2
    exit 1
fi
if [ "$DP_SIZE" -lt 1 ]; then
    echo "ERROR: no GPUs visible (DP_SIZE=$DP_SIZE)" >&2
    exit 1
fi

cd /workspace/Model-Optimizer/examples/speculative_decoding

TMP_PREFIX="/tmp/hs_part-"
split -n "l/${DP_SIZE}" --numeric-suffixes=0 -d --additional-suffix=.jsonl \
    "$INPUT" "${TMP_PREFIX}"

pids=()
for i in $(seq 0 $((DP_SIZE - 1))); do
    part="${TMP_PREFIX}$(printf '%02d' "$i").jsonl"
    CUDA_VISIBLE_DEVICES="$i" python3 collect_hidden_states/compute_hidden_states_hf.py \
        --model "$BASE_MODEL" \
        --input-data "$part" \
        --output-dir "$HS_DIR" \
        --trust_remote_code \
        --dp-rank "$i" \
        --dp-world-size "$DP_SIZE" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
rm -f "${TMP_PREFIX}"*.jsonl

if [ "$status" -ne 0 ]; then
    echo "ERROR: one or more hidden-state workers failed." >&2
    exit "$status"
fi

du -sh "$HS_DIR"
find "$HS_DIR" -name '*.pt' | wc -l
