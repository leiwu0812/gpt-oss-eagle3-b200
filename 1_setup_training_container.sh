#!/usr/bin/env bash
# Step 1 (inside TRAINING container): install ONLY what training needs.
# No vllm, no tensorrt_llm — those live in their own containers.
#   docker exec -it gpt-oss-train bash /workspace/gpt-oss-eagle3/1_setup_training_container.sh
set -euo pipefail

if ! mountpoint -q /data; then
    echo "ERROR: /data is not a bind mount. Run 0_run_training_container.sh first." >&2
    exit 1
fi
mkdir -p /data/models /data/prompts /data/synthetic /data/hidden_states /data/ckpts

cd /workspace

# 1. Clone Model-Optimizer.
if [ ! -d Model-Optimizer ]; then
    git clone --depth=1 https://github.com/NVIDIA/Model-Optimizer.git
fi

# 2. modelopt + example deps. Pin transformers below the version that modelopt
#    warns about, and reinstall without deps so pip doesn't bump torch again.
pip install -U "nvidia-modelopt[hf]>=0.35.0"
pip install -r Model-Optimizer/examples/speculative_decoding/requirements.txt
pip install "transformers<4.57" "hf_transfer"

# 3. HF login (the same token must have access to openai/gpt-oss-120b).
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# 4. Pre-download the teacher model into the mounted volume.
huggingface-cli download openai/gpt-oss-120b \
    --local-dir /data/models/gpt-oss-120b \
    --local-dir-use-symlinks False
# MXFP4 checkpoint is ~61 GiB on disk (not full bf16 weights).

nvidia-smi
python -c "import modelopt, torch, transformers; print('modelopt', modelopt.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
echo "Training container ready."
