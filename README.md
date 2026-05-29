# Training EAGLE3 on gpt-oss-120b (SDDD) on an 8×B200 Node

End-to-end runbook. Sources:
- TensorRT-LLM blog11 — gpt-oss-120b + Eagle3 deployment on GB200/B200
- NVIDIA/Model-Optimizer `examples/speculative_decoding` — training pipeline,
  offline hidden states, data synthesis

> **SDDD = Synthetic-Data-Driven Distillation.** The Model-Optimizer README
> calls it *data synthesis*: serve the teacher (gpt-oss-120b), regenerate the
> assistant turns of your dataset with it, then train the EAGLE3 draft on those
> synthetic conversations so the draft's distribution matches the teacher.

---

## 1. Hardware & prerequisites

- 1 node, 8 × NVIDIA B200 (192 GB HBM3e each). Driver ≥ R570.
- Docker with NVIDIA Container Toolkit (`docker info | grep -i runtime` shows `nvidia`).
- **NVMe scratch** mounted at the host path you'll bind to `/data` — plan for
  several TB. Hidden states for a 120B teacher on millions of rows reach tens of TB.
- Hugging Face account with **two access approvals**:
  - `openai/gpt-oss-120b` — visit the model page, click "Agree".
  - (Optional, full-scale) `nvidia/Nemotron-Post-Training-Dataset-v2` — gated.
- An `HF_TOKEN` from that account: `https://huggingface.co/settings/tokens` (Read scope).

## 2. Why three containers

NGC `pytorch:25.08-py3` ships a custom libtorch that is **not ABI-compatible**
with the public vLLM / tensorrt_llm wheels. Pip-installing them into the
training image leads to `undefined symbol` / flashinfer / TransformerEngine
errors. So we split the pipeline across three vendor images, each with its own
matched stack. All three share the same `/data` volume.

| Container | Image | Role |
|---|---|---|
| `gpt-oss-train` | `nvcr.io/nvidia/pytorch:25.08-py3` | modelopt training + HF hidden-state dump |
| `vllm-gpt-oss`  | `vllm/vllm-openai:latest`            | SDDD synthetic-data generation |
| `trtllm-gpt-oss`| `nvcr.io/nvidia/tensorrt-llm/release:latest` | Eagle3 deployment |

## 3. Files in this directory

```
0_run_training_container.sh     # host  : start gpt-oss-train
1_setup_training_container.sh   # train : install modelopt + HF, download gpt-oss-120b
2_prepare_prompts.sh            # train : build prompt skeletons → /data/prompts/active
prompts_data_config.smoke.yaml  #       smoke-test mix (UltraChat, 10K)
prompts_data_config.full.yaml   #       full mix (UltraChat + Magpie 300K, ~503K)
3a_start_vllm_container.sh      # host  : start vllm-gpt-oss (TP=8, EP on)
3b_synthesize.sh                # train : server_generate.py → /data/synthetic/train.jsonl
4_dump_hidden_states.sh         # train : HF-backend teacher hidden states → /data/hidden_states
5_train_and_export.sh           # train : launch_train.sh + export_hf_checkpoint.py
6_deploy_trtllm_container.sh    # host  : start trtllm-gpt-oss with Eagle3 config
eagle3_gpt_oss.yaml             # offline EAGLE3 training recipe
```

All shell scripts honor these env vars (defaults in parens):

| Var | Default | Notes |
|---|---|---|
| `DATA_HOST` | `$HOME/gpt-oss-eagle3/data` | NVMe path, bind-mounted at `/data` |
| `HF_CACHE_HOST` | `$HOME/.cache/huggingface` | Shared by all three containers |
| `TP` | `8` | vLLM tensor parallel; drop to 4 on a 4-GPU box |
| `PORT` | `8000` | vLLM / TRT-LLM HTTP port |
| `BASE_MODEL` | `/data/models/gpt-oss-120b` | Teacher path inside containers |
| `PROMPTS` | `/data/prompts/active/train.jsonl` | Switch to `train.small.jsonl` for smoke tests |
| `DATA_SCALE` | `smoke` | `smoke` (10K UltraChat) or `full` (~503K, matches NVIDIA official) |

---

## 4. Step-by-step

### Step 0 — Host: start the training container

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
bash 0_run_training_container.sh
docker exec -it gpt-oss-train bash
```

Inside: confirm `/data` is a mount, not an empty dir:
```bash
mountpoint /data && ls -ld /data
```

### Step 1 — Train container: install deps + download teacher

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
bash /workspace/gpt-oss-eagle3/1_setup_training_container.sh
```

What it does:
- Clones `NVIDIA/Model-Optimizer`.
- Installs `nvidia-modelopt[hf]` + the speculative_decoding `requirements.txt`.
- Pins `transformers<4.57` (modelopt warns on newer).
- `huggingface-cli download openai/gpt-oss-120b` → `/data/models/gpt-oss-120b` (~61 GiB MXFP4 checkpoint; incremental on rerun).

Final line should print modelopt/torch/transformers versions with no traceback.

### Step 2 — Train container: prompt skeletons

```bash
# smoke (default): 10K UltraChat prompt skeletons
bash /workspace/gpt-oss-eagle3/2_prepare_prompts.sh

# full-scale (matches nvidia/gpt-oss-120b-Eagle3-long-context ~503K mix):
DATA_SCALE=full bash /workspace/gpt-oss-eagle3/2_prepare_prompts.sh

ls /data/prompts/active/        # expect train.jsonl
head -1 /data/prompts/active/train.jsonl | python -m json.tool
```

Defaults to **smoke** (`DATA_SCALE=smoke`, UltraChat 10K). For the official
~503K mix (UltraChat + Magpie-300K), rerun with `DATA_SCALE=full`.

**Smoke-test slice** (recommended for the first end-to-end run):
```bash
head -n 1000 /data/prompts/active/train.jsonl > /data/prompts/active/train.small.jsonl
export PROMPTS=/data/prompts/active/train.small.jsonl
```

Exit the training container shell (`exit`) — the next step happens on the host.

### Step 3a — Host: start vLLM serving container (SDDD)

```bash
bash 3a_start_vllm_container.sh
docker logs -f vllm-gpt-oss
```

Wait for `Application startup complete` (5–15 minutes for 120B with TP=8).
Health check:
```bash
curl -H "Authorization: Bearer token-abc123" http://127.0.0.1:8000/v1/models
```

### Step 3b — Train container: synthesize

```bash
docker exec -it gpt-oss-train bash
bash /workspace/gpt-oss-eagle3/3b_synthesize.sh
```

Walks the prompt JSONL turn-by-turn, calls the vLLM server, writes
`/data/synthetic/train.jsonl` with teacher-generated assistant turns.

When it finishes:
```bash
exit                              # leave training container
docker stop vllm-gpt-oss          # free HBM for step 4
nvidia-smi                        # confirm GPUs idle
```

### Step 4 — Train container: dump teacher hidden states (offline)

```bash
docker exec -it gpt-oss-train bash
bash /workspace/gpt-oss-eagle3/4_dump_hidden_states.sh
du -sh /data/hidden_states/gpt-oss-120b
```

Uses a data-parallel HF hidden-state dump (8 workers, one teacher per GPU).
This is the slowest step on a 120B teacher — use the smoke slice first.

### Step 5 — Train container: EAGLE3 training + export

```bash
bash /workspace/gpt-oss-eagle3/5_train_and_export.sh
```

What it does:
- `launch_train.sh --config eagle3_gpt_oss.yaml` runs offline EAGLE3 training
  reading from `/data/hidden_states/gpt-oss-120b`, writing to `/data/ckpts/gpt-oss-120b-eagle3/`.
- `export_hf_checkpoint.py` converts the modelopt checkpoint to deployment
  format at `/data/ckpts/gpt-oss-120b-eagle3-hf/`.

Monitor:
```bash
tail -f /data/ckpts/gpt-oss-120b-eagle3/trainer_log*.txt
```

Tunables in `eagle3_gpt_oss.yaml` (aligned with
[nvidia/gpt-oss-120b-Eagle3-long-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context)):
- `eagle.eagle_architecture_config.intermediate_size` (official: 17280)
- `eagle.eagle_architecture_config.eagle_aux_hidden_state_layer_ids` (official: [1, 17, 32])
- `training.per_device_train_batch_size`, `gradient_accumulation_steps`
- `training.num_train_epochs`, `learning_rate`
- `training.cp_size` (context parallel for long context)

Exit the training container.

### Step 6 — Host: deploy with TRT-LLM

```bash
bash 6_deploy_trtllm_container.sh
docker logs -f trtllm-gpt-oss
```

The script writes `/data/extra-llm-api-config.yml` matching the
[NVIDIA model card](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context):
```yaml
speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model_dir: /data/ckpts/gpt-oss-120b-eagle3-hf
kv_cache_config:
  enable_block_reuse: false
```

Smoke test:
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/data/models/gpt-oss-120b","messages":[{"role":"user","content":"hello"}]}'
```

---

## 5. Order summary (copy-paste)

```bash
# === HOST ===
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
bash 0_run_training_container.sh

# === gpt-oss-train ===
docker exec -it gpt-oss-train bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
bash /workspace/gpt-oss-eagle3/1_setup_training_container.sh
bash /workspace/gpt-oss-eagle3/2_prepare_prompts.sh
head -n 1000 /data/prompts/active/train.jsonl > /data/prompts/active/train.small.jsonl
export PROMPTS=/data/prompts/active/train.small.jsonl
exit

# === HOST ===
bash 3a_start_vllm_container.sh
docker logs -f vllm-gpt-oss        # wait for "Application startup complete"

# === gpt-oss-train ===
docker exec -it gpt-oss-train bash
export PROMPTS=/data/prompts/active/train.small.jsonl
bash /workspace/gpt-oss-eagle3/3b_synthesize.sh
exit

# === HOST ===
docker stop vllm-gpt-oss

# === gpt-oss-train ===
docker exec -it gpt-oss-train bash
bash /workspace/gpt-oss-eagle3/4_dump_hidden_states.sh
bash /workspace/gpt-oss-eagle3/5_train_and_export.sh
exit

# === HOST ===
bash 6_deploy_trtllm_container.sh
docker logs -f trtllm-gpt-oss
```

---

## 6. Troubleshooting cheatsheet

| Symptom | Cause | Fix |
|---|---|---|
| `mkdir: cannot create '/data/...'` inside container | `/data` not bind-mounted; you exec'd into a stale container or ran on the host | `docker inspect gpt-oss-train --format '{{range .Mounts}}{{.Source}}->{{.Destination}}{{"\n"}}{{end}}'` ; if missing, `docker rm -f gpt-oss-train && bash 0_run_training_container.sh` |
| `huggingface-cli ... FileNotFoundError '/data/models'` | `/data` exists but model parent dirs don't | `mkdir -p /data/models /data/prompts /data/synthetic /data/hidden_states /data/ckpts` |
| `DatasetNotFoundError: ... is a gated dataset` | Need to "Agree" on HF dataset page with the same account as `HF_TOKEN` | Approve on HF, or use UltraChat-only path (default in step 2) |
| `make_dataset.py: unrecognized arguments: --output_dir` | `make_dataset.py` reads output path from the yaml, not CLI | Use `prompts_data_config.*.yaml` (`filename` points at `/data/prompts/active/train.jsonl`) |
| `server_generate.py: unrecognized arguments: --port` | Upstream uses `--url`, not `--port` | Fixed in `3b_synthesize.sh` |
| `KeyError: 'conversations'` during synthesis | `make_dataset.py` writes `messages`, `server_generate.py` reads `conversations` | Fixed in `3b_synthesize.sh` (auto-converts) |
| `ModuleNotFoundError: No module named 'tensorrt'` | `tensorrt_llm` wheel didn't pull `tensorrt` as a hard dep | Don't install trt-llm into the training container — it's only needed in step 6's separate container |
| vLLM `undefined symbol: _ZN3c104cuda...` | vLLM wheel compiled against a different libtorch than the 25.08 image | Don't pip-install vllm into the training container; use `3a_start_vllm_container.sh` |
| `flashinfer-cubin version (X) does not match flashinfer version (Y)` | Same ABI hell as above | Same fix — use the dedicated vLLM container |
| vLLM startup hangs > 20 min | Model still loading (120B with TP=8 takes time); or HBM OOM | `docker logs -f vllm-gpt-oss` ; if OOM, drop `--max-model-len` or `--gpu-memory-utilization` in `3a_start_vllm_container.sh` |
| curl to port 8000 connection refused | vLLM crashed during startup | `docker logs vllm-gpt-oss` and read the bottom of the log |
| `Architecture 'GptOss...' not supported` | vLLM image too old | `docker pull vllm/vllm-openai:latest` or pin a known-good tag (≥ 0.10.2) |
| `compute_hidden_states_hf.py` errors on gpt-oss arch | modelopt's draft loader doesn't recognize the model | Try `pip install -U --pre nvidia-modelopt`; otherwise open an issue, or pin an `eagle_decoder_type` override in the yaml |
| `/data` fills up during step 4 | Hidden states for 120B teacher are huge | Use the smoke slice (`train.small.jsonl`); attach more NVMe before scaling |
| Eagle not actually speculating in step 6 | `extra-llm-api-config.yml` typo or wrong path | Verify `decoding_type: Eagle` and `speculative_model_dir` points at the `-hf` export directory |

---

## 7. Caveats for full-scale runs

1. **gpt-oss-120b is not in modelopt's official support matrix.** It's a
   Llama-family MoE, so the EAGLE3 recipe applies, but treat the first
   end-to-end run as a smoke test and watch for arch-specific failures in
   steps 4 and 5.
2. **Disk planning.** Estimate hidden-state size:
   `≈ 2 (bf16) × hidden_size × avg_seq_len × num_rows × 3 (Eagle3 taps)`.
   Verify before kicking off step 4 on a multi-million-row dataset.
3. **TP/EP topology.** TP=8 + expert parallel works on 8 × B200; on a 4-GPU
   box pass `TP=4` to `3a_start_vllm_container.sh`. Don't enable EP without
   enough ranks to spread experts.
4. **Timing.** On the smoke slice (1k rows): minutes. On the full dataset:
   SDDD is hours-to-days, hidden-state dump is days, training is hours.
   Get the pipeline to "exported draft model" before scaling.
5. **Validation.** `scripts/ar_validate.py` only works on online training
   checkpoints. For the offline flow here, evaluate via the deployed
   trt-llm endpoint (acceptance rate / tokens-per-second).
