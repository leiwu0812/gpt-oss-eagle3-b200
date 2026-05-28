#!/usr/bin/env bash
# Step 3a (host): launch the vLLM serving container for SDDD.
# Runs SIDE BY SIDE with the training container, sharing /data and HF cache.
# vLLM here is the upstream image, which has its own matched torch/cuda — do
# NOT pip-install vllm into the training container.
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
NAME="${NAME:-vllm-gpt-oss}"
DATA_HOST="${DATA_HOST:-$HOME/gpt-oss-eagle3/data}"
HF_CACHE_HOST="${HF_CACHE_HOST:-$HOME/.cache/huggingface}"
TP="${TP:-8}"
PORT="${PORT:-8000}"

docker pull "$IMAGE"

# Sanity: model must already be downloaded by step 1.
if [ ! -f "$DATA_HOST/models/gpt-oss-120b/config.json" ]; then
    echo "ERROR: $DATA_HOST/models/gpt-oss-120b not found. Finish step 1 first." >&2
    exit 1
fi

docker run -d --rm --name "$NAME" \
    --gpus all \
    --ipc=host --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p ${PORT}:${PORT} \
    -v "$DATA_HOST":/data \
    -v "$HF_CACHE_HOST":/root/.cache/huggingface \
    "$IMAGE" \
        --model /data/models/gpt-oss-120b \
        --api-key token-abc123 \
        --host 0.0.0.0 --port ${PORT} \
        --tensor-parallel-size ${TP} \
        --enable-expert-parallel \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16

echo "vLLM container '$NAME' starting on port ${PORT}. Tail logs:"
echo "  docker logs -f $NAME"
echo "Wait for 'Application startup complete', then run 3b_synthesize.sh inside the training container."
