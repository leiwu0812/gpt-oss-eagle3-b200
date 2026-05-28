#!/usr/bin/env bash
# Step 6 (host): deploy the trained Eagle3 draft with TRT-LLM, in its own
# container. Run AFTER 5_train_and_export.sh has produced the HF-format draft.
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:latest}"
NAME="${NAME:-trtllm-gpt-oss}"
DATA_HOST="${DATA_HOST:-$HOME/gpt-oss-eagle3/data}"
PORT="${PORT:-8000}"

cat > "$DATA_HOST/extra-llm-api-config.yml" <<'EOF'
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 3
  speculative_model_dir: /data/ckpts/gpt-oss-120b-eagle3-hf
kv_cache_config:
  enable_block_reuse: false
EOF

docker pull "$IMAGE"
docker run -d --rm --name "$NAME" \
    --gpus all --ipc=host --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p ${PORT}:${PORT} \
    -v "$DATA_HOST":/data \
    "$IMAGE" \
    trtllm-serve /data/models/gpt-oss-120b \
        --host 0.0.0.0 --port ${PORT} --backend pytorch \
        --max_batch_size 32 --max_num_tokens 8192 --max_seq_len 8192 \
        --extra_llm_api_options /data/extra-llm-api-config.yml

echo "TRT-LLM container '$NAME' starting on port ${PORT}. Tail logs:"
echo "  docker logs -f $NAME"
