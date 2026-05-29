#!/usr/bin/env bash
# Step 3b (inside TRAINING container): regenerate assistant turns via the vLLM
# server started in 3a. Training container uses --network=host, so localhost
# reaches the vLLM container directly.
set -euo pipefail

PROMPTS="${PROMPTS:-/data/prompts/active/train.jsonl}"
OUT="${OUT:-/data/synthetic/train.jsonl}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-token-abc123}"
BASE_MODEL="${BASE_MODEL:-/data/models/gpt-oss-120b}"
mkdir -p "$(dirname "$OUT")"

PROMPTS_CONV="${PROMPTS%.jsonl}.conv.jsonl"

echo "Checking vLLM at 127.0.0.1:${PORT} ..."
MODEL_ID="$(
    curl -fsS -H "Authorization: Bearer ${API_KEY}" \
        "http://127.0.0.1:${PORT}/v1/models" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])"
)" || { echo "vLLM not reachable. Run 3a_start_vllm_container.sh on the host and wait for startup."; exit 1; }
echo "vLLM is up. Served model id: ${MODEL_ID}"

# make_dataset.py writes OpenAI-style "messages"; server_generate.py expects "conversations".
python3 - <<'PY' "$PROMPTS" "$PROMPTS_CONV"
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
    for line in fi:
        if not line.strip():
            continue
        row = json.loads(line)
        if "conversations" not in row:
            if "messages" not in row:
                raise KeyError(f"row missing messages/conversations: {list(row)[:5]}")
            row["conversations"] = row["messages"]
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Wrote {dst}")
PY

cd /workspace/Model-Optimizer/examples/speculative_decoding
python scripts/server_generate.py \
    --data_path "$PROMPTS_CONV" \
    --output_path "$OUT" \
    --model "$MODEL_ID" \
    --url "http://127.0.0.1:${PORT}/v1" \
    --api_key "$API_KEY" \
    --max_tokens 2048

echo "Synthetic data written to: $OUT"
echo "Now stop vLLM on the host to free HBM:  docker stop vllm-gpt-oss"
