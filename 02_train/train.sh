#!/usr/bin/env bash
# EAGLE3 draft training for gpt-oss-120b, on-the-fly hidden states.
# 8 x B200. Dry-run defaults; for full 500k run bump --epochs/--data and resources.
set -euo pipefail

ROOT=~/eagle3-gptoss
DATA=${ROOT}/01_data/regen.jsonl
OUT=${ROOT}/ckpt/dry-run
TARGET=~/models/gpt-oss-120b
DRAFT_CFG=${ROOT}/configs/draft_config.json

mkdir -p "${OUT}" "${ROOT}/logs"

# ModelOpt's EAGLE3 launcher.  Module path matches modelopt>=0.35.
# Layer ids [1,17,32] match nvidia/gpt-oss-120b-Eagle3-long-context.
accelerate launch \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  -m modelopt.torch.speculative.eagle.train \
  --target_model "${TARGET}" \
  --draft_config "${DRAFT_CFG}" \
  --train_data "${DATA}" \
  --output_dir "${OUT}" \
  --aux_hidden_state_layer_ids 1 17 32 \
  --on_the_fly_hidden_states true \
  --target_tp 4 \
  --draft_dp 8 \
  --epochs 3 \
  --global_batch_size 128 \
  --per_device_batch_size 1 \
  --grad_accum 16 \
  --seq_len 8192 \
  --optimizer adamw \
  --lr 3e-4 --min_lr 1e-4 \
  --lr_scheduler cosine --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --save_steps 500 --logging_steps 10 \
  2>&1 | tee "${ROOT}/logs/train_$(date +%Y%m%d_%H%M%S).log"
