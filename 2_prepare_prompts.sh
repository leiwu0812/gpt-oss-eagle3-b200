#!/usr/bin/env bash
# Step 2: Build prompt skeletons (user turns only) that we will feed to gpt-oss-120b
# during SDDD (Synthetic-Data-Driven Distillation). The README calls this
# "data synthesis": replacing dataset assistant turns with base-model-generated
# responses, so the draft model's distribution aligns with the target.
set -euo pipefail

cd /workspace/Model-Optimizer/examples/speculative_decoding

# Option A (smaller, faster smoke-test): UltraChat-200k, drop assistant turns.
# make_dataset.py reads the output path from the yaml, not the CLI. Copy the
# example config and rewrite its output path before running.
mkdir -p /data/prompts/ultrachat
cp ../dataset/example_data_config.yaml /tmp/ultrachat.yaml
python - <<'PY'
import re, pathlib
p = pathlib.Path("/tmp/ultrachat.yaml")
txt = p.read_text()
# Replace whichever key the upstream yaml uses for the output directory.
txt = re.sub(r'^(output_dir|save_path|output_path)\s*:.*$',
             r'\1: /data/prompts/ultrachat', txt, flags=re.M)
p.write_text(txt)
print(txt)
PY
# Skeletons only (no --full-conversations) so step 3 can regenerate assistant turns.
python ../dataset/make_dataset.py -f /tmp/ultrachat.yaml

# Option B (recommended for full-scale training): Nemotron Post-Training V2.
# It's a GATED dataset — you must first visit the HF page and click "Agree" with
# the same account as $HF_TOKEN. Leave commented until access is granted.
# python ../dataset/make_nemotron_ptv2_dataset.py \
#     --mode generate \
#     --output-dir /data/prompts/ptv2

# Canonical prompt set for the rest of the pipeline. Switch this symlink to
# /data/prompts/ptv2 once Nemotron access is granted.
ln -sfn /data/prompts/ultrachat /data/prompts/active
ls /data/prompts/active
