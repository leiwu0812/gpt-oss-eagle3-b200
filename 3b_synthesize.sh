#!/usr/bin/env bash
# Step 3b (inside TRAINING container): regenerate assistant turns via the vLLM
# server started in 3a. Training container uses --network=host, so localhost
# reaches the vLLM container directly.
set -euo pipefail

PROMPTS="${PROMPTS:-/data/prompts/active/train.jsonl}"
OUT="${OUT:-/data/synthetic/train.jsonl}"
PORT="${PORT:-8000}"
BASE_MODEL="${BASE_MODEL:-/data/models/gpt-oss-120b}"
mkdir -p "$(dirname "$OUT")"

# Health check before burning the dataset on a dead server.
echo "Checking vLLM at 127.0.0.1:${PORT} ..."
curl -fsS -H "Authorization: Bearer token-abc123" \
    "http://127.0.0.1:${PORT}/v1/models" >/dev/null \
    || { echo "vLLM not reachable. Run 3a_start_vllm_container.sh on the host and wait for startup."; exit 1; }
echo "vLLM is up."

cd /workspace/Model-Optimizer/examples/speculative_decoding
python scripts/server_generate.py \
    --data_path "$PROMPTS" \
    --output_path "$OUT" \
    --model "$BASE_MODEL" \
    --api_key token-abc123 \
    --port "$PORT" \
    --max_tokens 2048

echo "Synthetic data written to: $OUT"
echo "Now stop vLLM on the host to free HBM:  docker stop vllm-gpt-oss"
