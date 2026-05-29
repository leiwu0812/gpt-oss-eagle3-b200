#!/usr/bin/env bash
# Step 2: Build prompt skeletons (user turns only) for SDDD.
# Matches nvidia/gpt-oss-120b-Eagle3-long-context: UltraChat + Magpie-300K prompts.
set -euo pipefail

DATA_SCALE="${DATA_SCALE:-smoke}"   # smoke | full
REPO=/workspace/gpt-oss-eagle3

case "$DATA_SCALE" in
  smoke) CFG="$REPO/prompts_data_config.smoke.yaml" ;;
  full)  CFG="$REPO/prompts_data_config.full.yaml" ;;
  *)
    echo "ERROR: DATA_SCALE must be 'smoke' or 'full', got: $DATA_SCALE" >&2
    exit 1
    ;;
esac

mkdir -p /data/prompts/active

cd /workspace/Model-Optimizer/examples/speculative_decoding
echo "Building prompt skeletons with $CFG (DATA_SCALE=$DATA_SCALE) ..."
python ../dataset/make_dataset.py -f "$CFG"

if [ ! -s /data/prompts/active/train.jsonl ]; then
    echo "ERROR: /data/prompts/active/train.jsonl was not created or is empty." >&2
    exit 1
fi

wc -l /data/prompts/active/train.jsonl
head -1 /data/prompts/active/train.jsonl | python -m json.tool
