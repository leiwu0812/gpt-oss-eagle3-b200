#!/usr/bin/env bash
# Step 6 (host): deploy the trained Eagle draft with TRT-LLM.
# Config aligned with nvidia/gpt-oss-120b-Eagle3-long-context model card.
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:latest}"
NAME="${NAME:-trtllm-gpt-oss}"
DATA_HOST="${DATA_HOST:-$HOME/gpt-oss-eagle3/data}"
PORT="${PORT:-8000}"
TP="${TP:-8}"

mkdir -p "$DATA_HOST"
cat > "$DATA_HOST/extra-llm-api-config.yml" <<'EOF'
enable_attention_dp: false
disable_overlap_scheduler: true
enable_autotuner: false

cuda_graph_config:
    max_batch_size: 1

speculative_config:
    decoding_type: Eagle
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
        --tp_size ${TP} \
        --extra_llm_api_options /data/extra-llm-api-config.yml

echo "TRT-LLM container '$NAME' starting on port ${PORT}. Tail logs:"
echo "  docker logs -f $NAME"
