# Training notes

## On-the-fly vs offline hidden states
We use ModelOpt's on-the-fly mode: the target model lives in GPU memory alongside
the draft. On 8xB200 (180GB each), a reasonable split is:
- Target gpt-oss-120b: TP=4 across 4 GPUs (~60GB bf16 per GPU)
- Draft (215M): replicated DP=8

`modelopt.torch.speculative.eagle.train` handles the topology — the flags
`--target_tp 4 --draft_dp 8` are passed through to its mesh planner.

If your installed modelopt version doesn't expose these exact flags, drop the
`--target_tp/--draft_dp` lines and let it auto-plan, or check
`python -m modelopt.torch.speculative.eagle.train --help`.

## Aux layer ids
`[1, 17, 32]` is what nvidia/gpt-oss-120b-Eagle3-long-context shipped with.
gpt-oss-120b has 36 transformer layers; this is "early/mid/late" rather than
even thirds. Do not change unless you have a reason.

## RoPE
The draft inherits llama3 RoPE scaling (factor=8, original_max=8192) — NOT YaRN.
This matches the long-context reference checkpoint. Make sure
`configs/draft_config.json` and any `--rope_*` overrides agree.

## Dry-run -> full
- Dry-run:  `--train_data regen_5k.jsonl --epochs 1`  (sanity check, ~30 min)
- Full:     500k regenerated, `--epochs 3`, expect multi-day on 8xB200.

## Resume
ModelOpt writes `checkpoint-{step}/` under `--output_dir`; pass
`--resume_from_checkpoint <path>` to continue.
