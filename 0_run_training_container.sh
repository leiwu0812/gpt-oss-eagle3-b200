#!/usr/bin/env bash
# Step 0 (host): start the TRAINING container.
# Only modelopt + HF stack lives here. vLLM and TRT-LLM run in their own
# containers (see 3a_start_vllm_server.sh and 6_deploy_trtllm.sh) because
# NGC pytorch:25.08-py3 ships a custom libtorch that is ABI-incompatible
# with the public vllm / tensorrt_llm wheels.
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
NAME="${NAME:-gpt-oss-train}"
WORKDIR_HOST="${WORKDIR_HOST:-$HOME/gpt-oss-eagle3}"
HF_CACHE_HOST="${HF_CACHE_HOST:-$HOME/.cache/huggingface}"
DATA_HOST="${DATA_HOST:-$HOME/gpt-oss-eagle3/data}"

mkdir -p "$HF_CACHE_HOST" "$DATA_HOST"
docker pull "$IMAGE"

docker run -d --rm \
    --name "$NAME" \
    --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size=64g \
    --network=host \
    -e HF_HOME=/hf-cache \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -v "$WORKDIR_HOST":/workspace/gpt-oss-eagle3 \
    -v "$HF_CACHE_HOST":/hf-cache \
    -v "$DATA_HOST":/data \
    -w /workspace/gpt-oss-eagle3 \
    "$IMAGE" sleep infinity

echo "Training container '$NAME' is up. Enter with:"
echo "  docker exec -it $NAME bash"
