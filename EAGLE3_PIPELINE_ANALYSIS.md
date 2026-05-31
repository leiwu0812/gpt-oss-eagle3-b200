# gpt-oss-120b EAGLE3 蒸馏训练流程对比分析

对比两条复现路线：
- **NVIDIA 路线**：`nvidia/gpt-oss-120b-Eagle3-long-context` 在 8×B200 上的复现（本仓库 `gpt-oss-eagle3-b200`）
- **Lumen-RL 路线**：Lumen-RL 在 8×MI308 上训练 gpt-oss-120b EAGLE3 draft（`/home/leiwu/Lumen-RL/examples/GPT_OSS_120b_MI308_vLLM`）

---

## 一、NVIDIA B200 路线总览

### 训练范式：SDDD（Synthetic-Data-Driven Distillation）+ 离线 EAGLE3

```
准备 prompt → vLLM 老师生成答案 → HF 老师 dump hidden states → 训练 EAGLE3 draft → 导出 → TRT-LLM 部署
```

所有阶段都在**同一台 8×B200 节点**上跑，**没有 train/inference 同时占卡** —— 是**串行**复用 8 卡。

### 三个容器、三套技术栈

NGC `pytorch:25.08` 自带的 libtorch 跟公开的 vLLM / tensorrt_llm wheel ABI 不兼容，所以拆成 3 个容器，共享同一个 `/data` 卷：

| 阶段 | 容器 | 框架 | GPU 用法 |
|---|---|---|---|
| 训练 + HF hidden state dump | `nvcr.io/nvidia/pytorch:25.08-py3` | nvidia-modelopt + HF transformers | 8 卡 DP（dump）/ DDP（train）|
| 合成数据（步骤 3）| `vllm/vllm-openai:latest` | **vLLM**，OpenAI HTTP API | TP=8 + expert-parallel |
| 部署（步骤 6）| `nvcr.io/nvidia/tensorrt-llm/release` | **TRT-LLM 1.1.0rc1** trtllm-serve | TP=8 |

### 各步骤的卡分配 + 通信方式

**Step 3 — 合成数据 (SDDD)**
- vLLM 容器吃 8 卡，把 gpt-oss-120b 全量加载为老师，跑 OpenAI 兼容 HTTP 服务
- 训练容器里的 `server_generate.py` 通过 `--url http://127.0.0.1:8000/v1` 走 **HTTP REST** 调用老师重新生成 assistant turn（`3b_synthesize.sh:42-48`）
- 通信是**进程间 HTTP**，不是 NCCL。两个容器靠 `--network=host` + 共享 `/data` 协作
- 这一步完成必须 `docker stop vllm-gpt-oss`，否则下一步显存不够

**Step 4 — Dump teacher hidden states**
- 走纯 HF 后端，**数据并行**：把合成数据用 `split -n l/8` 切 8 份，每张卡一个 worker（`CUDA_VISIBLE_DEVICES=$i`），各自独立跑 `compute_hidden_states_hf.py`（`4_dump_hidden_states.sh:28-38`）
- **无 GPU 间通信**，纯 embarrassingly parallel；输出 `.pt` 文件到 `/data/hidden_states/gpt-oss-120b`
- 这是全流程最慢、最吃磁盘的一步（120B 老师×百万行 hidden state 容易到几十 TB）

**Step 5 — 离线 EAGLE3 训练**
- `launch_train.sh`（modelopt 提供）通常用 `torchrun` 起 8 卡 **DDP**，老师**冻结**（`eagle_freeze_base_model: true`），只训练 draft head（~0.2B）
- 训练读取的是 dump 好的 hidden state，**不再前向跑老师** —— 这就是"offline EAGLE3"，比 online 蒸馏便宜得多
- 通信是标准 torchrun + NCCL allreduce
- 关键超参（`eagle3_gpt_oss.yaml`）：`hidden_size=2880, intermediate_size=17280, aux_hidden_state_layer_ids=[1,17,32], training_seq_len=2048, bs=4×ga=4, lr=1e-4 cosine, epochs=1, bf16`，long-context 用 llama3 rope scaling 8×
- 完成后 `export_hf_checkpoint.py` 转成部署格式

**Step 6 — TRT-LLM 部署**
- `trtllm-serve` 起一个 TP=8 服务，老师 + EAGLE3 draft 一起加载，`speculative_config.decoding_type=Eagle, max_draft_len=3`，模型卡报告平均接受率 ~2.4 tokens/step

### 参数怎么传

主要靠 **环境变量 + YAML 配置**：
- Shell 环境变量统一：`DATA_HOST`, `HF_CACHE_HOST`, `TP`, `PORT`, `BASE_MODEL`, `PROMPTS`, `DATA_SCALE` —— 通过 `docker run -e`/`-v` 注入容器
- 容器间状态通过 `/data` 卷传递（prompt → 合成 jsonl → hidden state `.pt` → ckpt）
- 训练超参全在 `eagle3_gpt_oss.yaml` 里，`launch_train.sh --config ... training.output_dir=...` 这种 Hydra 风格 CLI override

---

## 二、Lumen-RL @ MI308 训练链路

**单容器 + 8 卡在线协同（4+4 切分）**，不像 NVIDIA 把全卡串行复用。

### 卡分配（`run_gpt_oss_120b.sh:48-49`）
```
GPU 0-3 ── torchrun FSDP2 训 EAGLE3 draft（每 rank 1 卡）
GPU 4-7 ── vLLM 老师 TP=4（gpt-oss-120b MXFP4 MoE）
```
**训练和推理同时在线**，不是 NVIDIA 那种"先全卡 dump，再全卡训"。

### 训练框架
- **Lumen-RL 自研 trainer**（不是 modelopt 也不是 SpecForge）：`lumenrl/trainer/spec_distill_trainer.py:280`
- 后端：**FSDP2 + aiter-patched composable replicate**（`:505-530`）
- Loss：默认 **forward KL** + position decay 0.8（`configs/phase1_ultrachat.yaml:45-48`），不是 modelopt 的交叉熵
- Draft 架构：1 层 Llama block，hidden=2880、ffn=17280、heads=64/kv=8、head_dim=64、aux_layers=`[1,17,32]` —— 跟 NVIDIA 官方一致

### 老师框架
- **vLLM 0.19.1 + ATOM 插件**（`vllm_teacher_engine.py:1-291`）
- 通过 **subprocess + FIFO** 起独立 vLLM 进程
- 后端：ROCm `ROCM_AITER_UNIFIED_ATTN`，MoE 走 Triton MXFP4

### Trainer ↔ Teacher 通信（核心差异点）
**Mooncake TCP 传输引擎**（`configs/phase1_ultrachat.yaml:86-90`）：
- rank 0 起 Mooncake master，监听 TCP 51111 + HTTP metadata 8129
- vLLM 老师通过 `MooncakeHiddenStatesConnector` **把 3 个 aux 层 hidden states 实时写进共享 store**（BF16, `[B,T,hidden]`）
- 训练侧 `_TeacherPrefetcher` CPU 线程拉取 → `_ShmWriterThread` 双缓冲 SHM → GPU（绕过 ROCm VM_L2_PROTECTION_FAULT 问题）（`spec_distill_trainer.py:41-150`）
- **不落盘**，纯内存+TCP 流式

### 数据
- Phase 1：UltraChat 200K（仅 prompt 字段）
- Phase 2：Magpie-Llama-3.1-Pro-300K
- 老师对 prompt 在线前向产生 hidden states 喂给 draft —— 严格说是 **online off-policy 蒸馏**，不需要先 SDDD 合成 + 再 dump

### 超参 & 训练规模
- global_bs=32（micro=1 × accum=8 × 4 ranks），lr 5e-5（P1）/ 2e-5（P2），warmup=0.015，grad_clip=0.5
- max_seq_len=8192（训练），131K（推理时靠 YaRN factor 32 外推）
- Phase 1: 19,488 steps × 3 epoch；Phase 2: 28,125 steps × 3 epoch

### MI308 特有的坑
- gfx942 ISA，容器 `lumenrl-vllm-mi308:latest`
- `HIP_FORCE_DEV_KERNARG=1`、`expandable_segments` 不支持
- **HSA aperture fault 每 500-700 步偶发崩** → `run_with_retry.sh` 监听日志静默 >600s 就 kill+从最近 ckpt resume
- NCCL/RCCL 超时设 7200s

---

## 三、整体路线对比

| 维度 | NVIDIA B200 | Lumen-RL MI308 |
|---|---|---|
| **GPU 复用模式** | **串行**：8 卡先全跑 vLLM → 全跑 HF dump → 全跑 DDP 训 → 全跑 TRT-LLM | **并行**：4 卡训 + 4 卡老师同时在线 |
| **训练框架** | NVIDIA Model-Optimizer 的 `launch_train.sh`（DDP）| 自研 trainer + **FSDP2** |
| **老师 serving** | vLLM (CUDA)，TP=8 + EP | vLLM (ROCm) + **ATOM/AITER** 插件，TP=4 |
| **蒸馏范式** | **离线 SDDD**：① vLLM 合成答案 jsonl → ② HF dump hidden state `.pt` → ③ 离线读盘训 | **在线流式**：老师边算 hidden state 边喂 trainer，**不落盘** |
| **trainer↔teacher 通信** | **HTTP REST**（OpenAI 兼容）+ **磁盘 `.pt`** | **Mooncake TCP** 共享内存 store + SHM 双缓冲 |
| **磁盘需求** | 巨量（百万行 × 120B hidden state 可达数十 TB）| 极小（不落 hidden state，只存 prompt 数据+ckpt）|
| **数据生成** | SDDD：用老师重新生成 assistant turn 替换原始答案 | 直接用 UltraChat / Magpie 原文，老师只前向取 hidden |
| **Loss** | 默认交叉熵 / self-distill logit | **forward KL + position decay 0.8** |
| **容器** | 3 个（pytorch / vllm / trtllm 各一套，ABI 隔离）| **1 个**（lumenrl-vllm-mi308）|
| **部署** | TRT-LLM `trtllm-serve` + `extra-llm-api-config.yml` | vLLM (ROCm) 原生 spec decoding |
| **稳定性兜底** | 无特殊（流程串行，崩了重跑某一步）| `run_with_retry.sh` watchdog + resume（应对 HSA fault）|
| **长上下文** | 训练即用 long-context rope scaling（131K）| **训 8K，推理靠 YaRN factor 32 外推** 到 131K |

### 一句话总结
- **NVIDIA 路线**：把 SDDD 拆成"合成→dump→训→部署"四个**离线阶段**，每段全卡满载、对磁盘要求极高，胜在每步可独立 debug、可断点续跑、对训练框架要求低。
- **Lumen-RL 路线**：用 **Mooncake 把 vLLM 老师和 FSDP2 trainer 拼成一个在线 pipeline**，**省掉几十 TB 的 hidden state 落盘**，代价是 8 卡得切两半（训练吞吐少了 4 卡）、调度更复杂、还要扛 ROCm 自己的硬件 fault。本质上是**用通信换磁盘**。

---

## 四、每一步每张卡的详细分工

### 4.1 NVIDIA B200 路线（不同时刻 8 卡角色切换）

#### Step 0–2：准备阶段（**纯 CPU**）
- 拉容器、安装 modelopt、下载 `gpt-oss-120b`（~61 GiB MXFP4）、生成 prompt jsonl
- **8 张 GPU 全部空闲**

#### Step 3 —— SDDD 合成数据（vLLM 容器）

| GPU | 角色 |
|---|---|
| 0–7 | **vLLM tensor-parallel rank 0–7**：每张装 1/8 的 gpt-oss-120b 权重 + `--enable-expert-parallel`（MoE 专家也切到 8 路）|

- **通信**：GPU 间走 **NCCL all-reduce / all-to-all**（attention TP + MoE EP）
- **外部 IO**：训练容器里的 `server_generate.py` 通过 **HTTP REST**（`localhost:8000/v1/chat/completions`）把 prompt 喂进来，vLLM 流式返回 token，写 `/data/synthetic/train.jsonl`
- **CPU 侧**：训练容器是个客户端进程，单线程拉 prompt → POST → 写文件
- 结束后 `docker stop vllm-gpt-oss`，**8 卡全部释放显存**

#### Step 4 —— Dump 老师 hidden states（HF 容器）

8 张卡**完全独立**，数据并行：
```bash
split -n l/8 train.jsonl → 8 个 shard
for i in 0..7:
   CUDA_VISIBLE_DEVICES=$i python compute_hidden_states_hf.py --dp-rank $i ...
```

| GPU | 角色 |
|---|---|
| 0 | 进程 0：自己装一份完整 gpt-oss-120b（BF16/MXFP4），喂 shard 0 → 写 `.pt` |
| 1 | 进程 1：同上，shard 1 |
| ... | ... |
| 7 | 进程 7：shard 7 |

- **通信**：**没有 GPU 间通信**，8 个进程互不感知（embarrassingly parallel）
- 之所以能这么干，是因为 B200 单卡 192 GB HBM 装得下整个 120B（MXFP4 大概 60 GB）
- **唯一瓶颈是磁盘**：3 个 aux 层 × BF16 × seq_len × 数据条数，可以到几十 TB
- 结束：所有 hidden state 落盘到 `/data/hidden_states/gpt-oss-120b/*.pt`

#### Step 5 —— 离线 EAGLE3 训练

`launch_train.sh` 内部起 `torchrun --nproc_per_node=8`：

| GPU | 角色 |
|---|---|
| 0–7 | **DDP rank 0–7**：每张都装 ① **冻结**的老师（只为 embed/lm_head 前向，不算 hidden state）+ ② **完整的 draft 头（0.2B）** |

- 老师**只跑 embedding + lm_head**，中间层的 hidden state 直接从磁盘读（这就是"offline"的意义）
- Draft 前向 → 跟磁盘里的老师 hidden state 算 forward KL/CE → 反向 → **NCCL all-reduce 梯度** → 优化器 step
- **数据并行**，每个 rank 拿不同 batch，hidden state 文件按 rank 分配
- 通信：纯 GPU 间 NCCL，**没有外部 HTTP / Mooncake**

#### Step 6 —— TRT-LLM 部署

| GPU | 角色 |
|---|---|
| 0–7 | **TRT-LLM TP=8**：每张装 1/8 老师 + 1/8 draft，做投机解码（draft 出 3 token，老师并行 verify）|

- 客户端走 HTTP `:8000/v1/chat/completions` 进来

---

### 4.2 Lumen-RL MI308 路线（**同时**训练 + 推理，4+4 切分）

8 张 MI308 **从训练开始到结束分工固定不变**：4 张训、4 张推理，**全程并发跑**。

#### 启动阶段（rank 0 一次性）
GPU 0 上的训练进程额外起一个 **Mooncake master**（TCP 51111，metadata HTTP 8129），不占 GPU 显存，只是 host 上一个守护进程。

#### 卡上的常驻角色

| GPU | 进程 | 装的东西 |
|---|---|---|
| 0 | torchrun rank 0 + Mooncake master | FSDP2 shard of draft (0.2B / 4) ≈ 50M 参 + 优化器状态 + 激活；CPU 上跑 `_TeacherPrefetcher` + `_ShmWriterThread` |
| 1 | torchrun rank 1 | FSDP2 shard of draft |
| 2 | torchrun rank 2 | FSDP2 shard of draft |
| 3 | torchrun rank 3 | FSDP2 shard of draft |
| 4 | vLLM TP rank 0 | 1/4 的 gpt-oss-120b 权重（含 expert shard）|
| 5 | vLLM TP rank 1 | 1/4 老师 |
| 6 | vLLM TP rank 2 | 1/4 老师 |
| 7 | vLLM TP rank 3 | 1/4 老师 |

注意：**GPU 0-3 跟 4-7 之间没有 NCCL/RCCL 通信**，它们属于两个独立的进程组，靠 host 上的 Mooncake/TCP/SHM 协作。

#### 一个 step 内的时序（流水线化执行）

```
            GPU 0-3 (训练侧)                    GPU 4-7 (老师侧)
            ─────────────────                   ────────────────────
[t0] rank0 从 dataloader 拿 prompt batch
     → 通过 FIFO 推给 vLLM worker
                                          [t1] vLLM 收到 batch
                                               TP=4 前向：
                                               GPU4↔5↔6↔7 之间 RCCL all-reduce
                                               (attention + MoE EP)
                                               捕获 layer [1,17,32] + 最后层
                                               hidden state（BF16）
                                          [t2] 通过 MooncakeHiddenStatesConnector
                                               把 hidden state 写进 Mooncake store
                                               （TCP，落到 host 内存池）
[t3] CPU 上 _TeacherPrefetcher 线程从
     Mooncake 拉到 host 内存
     → _ShmWriterThread 双缓冲 SHM
     → H2D 拷到 GPU 0-3
[t4] 每个 rank 0-3：
       - draft 前向（1 层 Llama block）
       - 用 teacher hidden + lm_head 算 forward KL
       - position_decay=0.8
[t5] 反向：FSDP2 在 GPU 0-3 之间 reduce-scatter 梯度
[t6] 优化器 step（lr=5e-5, AdamW）

  与此同时 vLLM 已经在处理下一个 batch ──────────┘ (流水线)
```

#### 通信路径汇总

| 通道 | 走哪 | 干啥 |
|---|---|---|
| 训练侧 GPU 0↔1↔2↔3 | **RCCL** | FSDP2 参数/梯度/优化器状态 reduce-scatter / all-gather |
| 老师侧 GPU 4↔5↔6↔7 | **RCCL** | vLLM TP attention all-reduce + MoE expert dispatch/combine |
| 训练 rank0 → vLLM | **FIFO（host pipe）** | 推送 prompt batch、控制命令 |
| vLLM → 训练 | **Mooncake TCP（host RAM 池）** | hidden state tensor 传递 |
| Mooncake → 训练 GPU | **SHM 双缓冲 + cudaMemcpy H2D** | 喂进 GPU 0-3 |

**关键点**：训练 GPU 跟老师 GPU 之间**完全不共享 CUDA context**，它们各自跑自己的 NCCL group，所有跨组数据靠 **CPU-mediated 通信**（FIFO + Mooncake + SHM）。

---

## 五、关键差异图示

```
NVIDIA B200（不同时刻 8 卡角色切换）：
─────────────────────────────────────────────────────────────
Step3:  [G0..G7] = 全部 vLLM TP=8 老师
Step4:  [G0..G7] = 8 个独立 dump 进程（每卡装满老师）
Step5:  [G0..G7] = DDP 8 ranks 训 draft（老师冻结）
Step6:  [G0..G7] = TRT-LLM TP=8 部署
        │
        └── 每一步内：所有 GPU 干同一件事，通信靠 NCCL

Lumen-RL MI308（同一时刻 8 卡角色固定 4+4）：
─────────────────────────────────────────────────────────────
持续运行:
  [G0..G3] = FSDP2 训 draft  ────RCCL────┐
                                          ├── Mooncake TCP + SHM ──┐
  [G4..G7] = vLLM TP=4 老师  ────RCCL────┘                         │
                                                                    │
  host CPU: Mooncake master + prefetcher + SHM writer ←─────────────┘
```

## 六、副作用对比

| 维度 | NVIDIA | Lumen-RL |
|---|---|---|
| 训练吞吐（draft 反向能用几张卡）| **8 张** | **4 张** |
| 老师服务吞吐（多少卡跑老师）| Step3/6 用 8 张；Step5 不需要老师在线 | 全程 4 张 |
| 单卡显存压力 | Step4 最紧张（每卡一份完整 120B 老师），靠 192GB HBM 撑住 | 始终轻松（老师 TP=4 分担，draft 才 0.2B/4）|
| 磁盘 | **几十 TB**（hidden state）| **接近 0**（不落 hidden state，只存 ckpt）|
| 故障域 | 各步串行，崩了重跑当前步即可 | 8 卡两组进程 + Mooncake 都不能挂，任一挂了整链路停 → 必须 `run_with_retry.sh` 兜底 |
| 调度复杂度 | 低（一个一个脚本跑）| 高（FIFO/Mooncake/SHM/prefetcher 串起来的流水线）|
