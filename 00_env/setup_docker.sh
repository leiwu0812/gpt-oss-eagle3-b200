#!/usr/bin/env bash
# Docker-based environment for DGX nodes without python3-venv (Ubuntu 24.04).
# Creates container eagle3-train and installs ModelOpt + vLLM inside it.
set -euo pipefail

CONTAINER="${EAGLE3_CONTAINER:-eagle3-train}"
IMAGE="${EAGLE3_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"
REPO="${EAGLE3_REPO:-$HOME/gpt-oss-eagle3-b200}"
ROOT="${EAGLE3_ROOT:-$HOME/eagle3-gptoss}"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running setup}"

mkdir -p "${ROOT}"/{01_data,configs,ckpt,logs} "${HOME}/models"

# Symlink repo scripts into workspace
for f in 01_data/prepare.py 01_data/regenerate.py 01_data/convert_regen.py \
         configs/draft_config.json configs/eagle3_gpt_oss_offline.yaml \
         configs/eagle3_gpt_oss_online.yaml \
         02_train/train.sh 02_train/dump_hidden_states.sh \
         03_eval/eval_acceptance.py; do
  ln -sfn "${REPO}/${f}" "${ROOT}/${f}"
done

# Reuse NFS model if present
GPT_OSS_SNAP="$(find /mnt/models/models--openai--gpt-oss-120b/snapshots -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1 || true)"
if [[ -n "${GPT_OSS_SNAP}" ]]; then
  ln -sfn "${GPT_OSS_SNAP}" "${HOME}/models/gpt-oss-120b"
  echo "Linked gpt-oss-120b -> ${GPT_OSS_SNAP}"
else
  echo "WARN: gpt-oss-120b not found under /mnt/models; run huggingface-cli download inside container"
fi

docker rm -f "${CONTAINER}" 2>/dev/null || true
docker run -d --name "${CONTAINER}" \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "${ROOT}:/workspace/eagle3-gptoss" \
  -v "${HOME}/models:/workspace/models" \
  -v "${REPO}:/workspace/gpt-oss-eagle3-b200" \
  -v /mnt/models:/mnt/models:ro \
  -w /workspace/eagle3-gptoss \
  "${IMAGE}" sleep infinity

docker exec -e HF_TOKEN="${HF_TOKEN}" "${CONTAINER}" bash -lc '
set -euo pipefail
pip install -U pip
pip install "transformers>=4.55.2,<4.57" accelerate datasets
git clone --depth 1 https://github.com/NVIDIA/Model-Optimizer.git /tmp/Model-Optimizer 2>/dev/null || true
pip install -e "/tmp/Model-Optimizer[hf]"
pip install "vllm>=0.6.4" hf_transfer huggingface_hub
pip install -r /tmp/Model-Optimizer/examples/speculative_decoding/requirements.txt
huggingface-cli login --token "$HF_TOKEN"
python -c "import modelopt, vllm; print(\"modelopt ok\", \"vllm ok\")"
'

echo "Container ${CONTAINER} ready. Run: docker exec -it ${CONTAINER} bash"
echo "Inside container: export EAGLE3_ROOT=/workspace/eagle3-gptoss MODEL_ROOT=/workspace/models"
