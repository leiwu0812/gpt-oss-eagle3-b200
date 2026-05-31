# Training notes

## Default: offline EAGLE3 (recommended for gpt-oss-120b)

`02_train/train.sh` runs three steps:

1. `01_data/convert_regen.py` — add `conversation_id` for ModelOpt
2. `02_train/dump_hidden_states.sh` — teacher hidden states (single process, `device_map=auto` on 8 GPUs)
3. `launch_train.sh --config configs/eagle3_gpt_oss_offline.yaml` — draft-only training

Set `HS_SUBSET=500` (default) for dry-run hidden-state dump size. Unset or
raise for full 5k/500k runs.

Install ModelOpt from git inside the training environment:
`pip install -e /tmp/Model-Optimizer[hf]`. PyPI wheels may lack `hf_eagle`.

## Online / on-the-fly (experimental)

`configs/eagle3_gpt_oss_online.yaml` uses `data.data_path` without offline
hidden states. Standard 8-GPU DDP loads the full 120B per rank and typically
OOMs on B200. Reserved for future ModelOpt TP mesh support.

## Aux layer ids

`[1, 17, 32]` matches `nvidia/gpt-oss-120b-Eagle3-long-context`. Do not change
unless re-collecting hidden states.

## RoPE

Draft uses llama3 RoPE scaling (`factor=8`, `original_max=8192`), not YaRN.
Keep `configs/draft_config.json` aligned with the HF reference.

## Dry-run -> full

- Dry-run: `HS_SUBSET=500`, `num_train_epochs: 1` in offline yaml
- Full: all rows, `num_train_epochs: 3`, expect multi-day HS dump + training

## Resume

ModelOpt writes `checkpoint-{step}/` under `training.output_dir`; pass
`--resume_from_checkpoint <path>` via `launch_train.sh` CLI overrides.
