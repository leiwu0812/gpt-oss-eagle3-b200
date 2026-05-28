#!/usr/bin/env bash
# Step 5: Offline EAGLE3 training on the dumped hidden states, then export
# a deployment-ready checkpoint.
set -euo pipefail

cd /workspace/Model-Optimizer/examples/speculative_decoding

CFG=/workspace/gpt-oss-eagle3/eagle3_gpt_oss.yaml
OUT=/data/ckpts/gpt-oss-120b-eagle3
EXPORT=/data/ckpts/gpt-oss-120b-eagle3-hf

# launch_train.sh is torchrun under the hood and picks up all visible GPUs.
# OmegaConf dotlist overrides on the command line beat anything in the YAML.
./launch_train.sh \
    --config "$CFG" \
    training.output_dir="$OUT"

# Export to a HF-style checkpoint that trtllm-serve / SGLang / vLLM understands.
python scripts/export_hf_checkpoint.py \
    --model_path "$OUT" \
    --export_path "$EXPORT"

echo "Draft model exported to: $EXPORT"
echo "Deploy from the host with the TRT-LLM container: bash 6_deploy_trtllm_container.sh"
