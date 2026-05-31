#!/usr/bin/env bash
# Environment setup for EAGLE3 draft training on gpt-oss-120b (8xB200)
set -euo pipefail

# CUDA 12.4+ assumed. Use a fresh venv.
python -m venv ~/eagle3-gptoss/.venv
source ~/eagle3-gptoss/.venv/bin/activate
pip install -U pip wheel

# Core stack
pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.55.2" "accelerate>=1.0" "datasets>=3.0"
pip install "nvidia-modelopt[torch,hf]>=0.35"
pip install "vllm>=0.6.4"           # for response regeneration
pip install hf_transfer huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1

# Target model (~240 GB on disk)
huggingface-cli download openai/gpt-oss-120b \
  --local-dir ~/models/gpt-oss-120b --local-dir-use-symlinks False

# Reference EAGLE3 ckpt (for sanity-eval baseline and config reference)
huggingface-cli download nvidia/gpt-oss-120b-Eagle3-long-context \
  --local-dir ~/models/gpt-oss-120b-Eagle3-ref --local-dir-use-symlinks False

echo "Setup complete."
