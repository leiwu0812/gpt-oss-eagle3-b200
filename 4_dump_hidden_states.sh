#!/usr/bin/env bash
# Step 4 (inside TRAINING container): dump teacher hidden states.
# We use the HF backend here because the training container intentionally does
# NOT ship tensorrt_llm. (TRT-LLM-backed dumping is faster but requires its own
# container; switch to that for production runs.)
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/data/models/gpt-oss-120b}"
INPUT="${INPUT:-/data/synthetic/train.jsonl}"
HS_DIR="${HS_DIR:-/data/hidden_states/gpt-oss-120b}"
mkdir -p "$HS_DIR"

cd /workspace/Model-Optimizer/examples/speculative_decoding

# DP wrapper around compute_hidden_states_hf.py, uses every visible GPU.
bash collect_hidden_states/run_hf_compute_hiddens_dp.sh \
    --model "$BASE_MODEL" \
    --input-file "$INPUT" \
    --output-dir "$HS_DIR"

du -sh "$HS_DIR"
