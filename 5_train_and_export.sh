#!/usr/bin/env bash
# Step 5: Offline EAGLE3 training on dumped hidden states, then export for deployment.
set -euo pipefail

cd /workspace/Model-Optimizer/examples/speculative_decoding

CFG=/workspace/gpt-oss-eagle3/eagle3_gpt_oss.yaml
OUT=/data/ckpts/gpt-oss-120b-eagle3
EXPORT=/data/ckpts/gpt-oss-120b-eagle3-hf

if [ ! -f "$CFG" ]; then
    echo "ERROR: missing training recipe: $CFG" >&2
    exit 1
fi

./launch_train.sh \
    --config "$CFG" \
    training.output_dir="$OUT"

python scripts/export_hf_checkpoint.py \
    --model_path "$OUT" \
    --export_path "$EXPORT" \
    --trust_remote_code

echo "Draft model exported to: $EXPORT"
echo "Deploy from the host with the TRT-LLM container: bash 6_deploy_trtllm_container.sh"
