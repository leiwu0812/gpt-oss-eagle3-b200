#!/usr/bin/env bash
set -euo pipefail
export HF_TOKEN="${HF_TOKEN:-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}"
N="${1:-80}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
REF_ONLY="${REF_ONLY:-0}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.21.0}"
REF_SNAP="633caf45f31288cbb70ee237f7c939db707ecc94"
LOG=/tmp/eagle3_eval_compare.log
exec > >(tee "$LOG") 2>&1
echo "=== EAGLE3 eval compare n=${N} $(date) ==="

run_eval() {
  local label="$1"
  local draft_src="$2"
  local copy_tok="${3:-0}"
  docker run --rm --gpus all --ipc=host --shm-size=64g --entrypoint bash \
    -e HF_TOKEN="$HF_TOKEN" \
    -v ~/eagle3-gptoss:/workspace/eagle3-gptoss \
    -v ~/models:/workspace/models \
    -v ~/gpt-oss-eagle3-b200:/workspace/gpt-oss-eagle3-b200 \
    -v /mnt/models:/mnt/models:ro \
    -v /mnt/home/amd_user/models:/mnt/ref:ro \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "$IMAGE" \
    -lc "
set -euo pipefail
pip install -q datasets
DRAFT=/tmp/draft-${label}
rm -rf \${DRAFT} && mkdir -p \${DRAFT}
cp -aL ${draft_src}/* \${DRAFT}/
if [ ${copy_tok} -eq 1 ]; then
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja; do
    cp -a /workspace/eagle3-gptoss/ckpt/dry-run/\${f} \${DRAFT}/ 2>/dev/null || true
  done
fi
python3 - <<PY
import json
from pathlib import Path
p = Path('/tmp/draft-${label}/config.json')
cfg = json.loads(p.read_text())
if cfg.get('use_cache') is None:
    cfg['use_cache'] = True
p.write_text(json.dumps(cfg, indent=4) + '\\n')
PY
python3 /workspace/gpt-oss-eagle3-b200/03_eval/eval_acceptance.py \
  --target /workspace/models/gpt-oss-120b \
  --draft \${DRAFT} \
  --label ${label} \
  --n ${N} \
  --draft_len 3
"
}

if [ "$REF_ONLY" != "1" ]; then
  echo "--- dry-run draft ---"
  run_eval "dry-run" "/workspace/eagle3-gptoss/ckpt/dry-run-hf" 1
fi

if [ "$DRY_RUN_ONLY" != "1" ]; then
  echo "--- reference Eagle3 ---"
  run_eval "ref-eagle3" "/mnt/ref/models--nvidia--gpt-oss-120b-Eagle3-long-context/snapshots/${REF_SNAP}" 1
fi

echo COMPARE_DONE
