#!/usr/bin/env bash
# EAGLE3 draft training for gpt-oss-120b via NVIDIA Model-Optimizer.
# Default: offline path (hidden states on disk + draft-only training).
# Online/on-the-fly training OOMs on 120B with standard 8-GPU DDP — see README.
set -euo pipefail

ROOT="${EAGLE3_ROOT:-$HOME/eagle3-gptoss}"
MODEL="${MODEL_ROOT:-$HOME/models}/gpt-oss-120b"
MODELOPT="${MODELOPT_ROOT:-/tmp/Model-Optimizer}"
OUT="${ROOT}/ckpt/dry-run"
MO="${MODELOPT}/examples/speculative_decoding"
CFG="${ROOT}/configs/eagle3_gpt_oss_offline.yaml"
HS_SUBSET="${HS_SUBSET:-500}"  # dry-run: dump hidden states for first N rows

mkdir -p "${OUT}" "${ROOT}/logs"

# Ensure ModelOpt is installed from git (PyPI wheel lacks hf_eagle plugin).
if [[ ! -d "${MODELOPT}/.git" ]]; then
  git clone --depth 1 https://github.com/NVIDIA/Model-Optimizer.git "${MODELOPT}"
fi
pip install -q -e "${MODELOPT}[hf]"
pip install -q -r "${MO}/requirements.txt"

# Convert vLLM regen output -> ModelOpt conversation JSONL
python "${ROOT}/01_data/convert_regen.py" \
  --in "${ROOT}/01_data/regen.jsonl" \
  --out "${ROOT}/01_data/train.jsonl"

# Step 1: dump teacher hidden states (single process, all GPUs)
bash "$(dirname "$0")/dump_hidden_states.sh" \
  "${ROOT}/01_data/train.jsonl" "${HS_SUBSET}"

# Step 2: offline EAGLE3 training (draft only; base is fake/offline)
cd "${MO}"
bash launch_train.sh --config "${CFG}" \
  model.model_name_or_path="${MODEL}" \
  data.offline_data_path="${ROOT}/hidden_states/gpt-oss-120b" \
  training.output_dir="${OUT}" \
  2>&1 | tee "${ROOT}/logs/train_$(date +%Y%m%d_%H%M%S).log"

# Step 3: export HF checkpoint for vLLM / TRT-LLM deployment
python scripts/export_hf_checkpoint.py \
  --input_dir "${OUT}" \
  --output_dir "${OUT}-hf" \
  --trust_remote_code

echo "Draft checkpoint: ${OUT}-hf"
