# PR #1851 消融实验计划 — Stage 2

**Created:** 2026-04-28
**Hardware:** 8×H100 SXM
**Script:** `parameter-golf-dev-lite/run_ablation_1851_v2.sh`
**Primary training file:** `parameter-golf-dev-lite/train_gpt_s9_caseops_lqer.py`
**Diagnostic training files:** `parameter-golf-dev-lite/train_gpt_s0_pr1851_mod_diag.py`, `parameter-golf-dev-lite/train_gpt_s0_pr1851_mod_diag_ttt.py`

## 为什么需要 Stage 2

Stage 1 (`run_ablation_1851.sh`) 已经覆盖了 LQER、SmearGate、SparseAttnGate、TTT phasing 和若干 S9/#1797/#1801 组合分支。Stage 2 不再扩大搜索树，而是做一个最小闭环，补足三个直接影响解释的问题：

- **真实 #1851 baseline 锚点。** Stage 1 的 A0 使用 `FUSED_CE_ENABLED=0`，而 #1851 原始 log 中 `fused_ce_enabled: True`。Stage 2 用 `Z0` 固定 `FUSED_CE_ENABLED=1`，作为真实 #1851 口径的锚点。
- **torch.compile 是否在 600s cap 下浪费训练时间。** P 组只回答 compile on/off/dynamic 的 throughput 问题，不改变模型设计。
- **激进量化和 TTT 小 batch 噪声是否解释剩余现象。** Q 组只保留 4-bit floor/rescue 与 embed floor；R 组只保留能说明 TTT gradient sample 噪声的 rank×batch_size 小网格。

这份计划刻意删除了 matrix_bits=7、SpinQuant、GPTQ calibration=64、LQER_TOP_K=11、attn_clip、vanilla sp8192、low-rank TTT 和 high-rank bs128 等探索支线。它们可能有价值，但不是这次必须回答的问题。

## Per-run 公共配置

- 单 seed = 42
- `MAX_WALLCLOCK_SECONDS=600`
- `GPTQ_RESERVE_SECONDS=0.5`
- `CASEOPS_ENABLED=1`
- `FUSED_CE_ENABLED=1`
- `torchrun --nproc_per_node=8 --standalone <per-cell script>`
- 12 cells：7 个完整训练/GPTQ/TTT cells + 5 个 TTT eval-only diagnostic cells
- 按 Stage 1 实测，完整训练 cells 约 7 × 22-24 分钟 ≈ **2.7 小时**；R 组 eval-only TTT runtime 取决于 TTT 配置，可中途停止

## 12 个 Cell

### Group Z — 真实 #1851 baseline（1 cell）

| Cell | 改动 (env override) | 在测什么 |
|---|---|---|
| **Z0** | (none beyond common env) | 真实 #1851 口径 baseline：`FUSED_CE_ENABLED=1` |

### Group P — torch.compile / throughput tracing（3 cells）

| Cell | Script | 改动 | 在测什么 |
|---|---|---|---|
| P0 | `train_gpt_s0_pr1851_mod_diag.py` | `DIAG_COMPILE_TRACE=1` | compile 开启时每个 `torch.compile` site 的首次调用耗时，以及完整 600s cap 下的训练表现 |
| P1 | `train_gpt_s0_pr1851_mod_diag.py` | `DIAG_COMPILE_TRACE=1 DIAG_COMPILE_DISABLE=1` | 把 `torch.compile` patch 成 identity，估计不用 compile 能多跑多少训练 |
| P2 | `train_gpt_s0_pr1851_mod_diag.py` | `DIAG_COMPILE_TRACE=1 TRAIN_COMPILE_DYNAMIC=1` | 主训练 compile 从 `dynamic=False` 改成 `dynamic=True` 是否影响 compile/throughput |

P 组输出 side-channel：`logs/diag_<RUN_ID>.compile_trace.jsonl`。

### Group Q — 最小激进量化闭环（3 cells）

| Cell | 改动 | 在测什么 |
|---|---|---|
| Q1 | `MATRIX_BITS=4` | matrix quantization 的 4-bit floor 是否直接破坏 BPB |
| Q3 | `MATRIX_BITS=4 LQER_RANK=8` | 如果 Q1 变差，增加 LQER rank 是否能 rescue 4-bit matrix |
| Q4 | `EMBED_BITS=6 EMBED_CLIP_SIGMAS=10.0` | embedding quantization 的低 bit-rate floor |

Q 组和 Stage 1 的 D 组互补：Stage 1 已有 `matrix_bits=5` 和 `embed_bits=8`，Stage 2 只补下探端点和一个 LQER rescue 配对。

### Group R — TTT gradient-noise story（5 cells）

所有 R cells 使用：

- `train_gpt_s0_pr1851_mod_diag_ttt.py`
- `TTT_EVAL_ONLY=1`
- `QUANTIZED_MODEL_PATH=models/20260427_232132-A0-s42-38fe4988.int6.ptz`
- `DIAG_TTT_NOISE=1`
- `DIAG_TTT_NOISE_K=4`

| Cell | 额外改动 | 在测什么 |
|---|---|---|
| R0 | (rank=96, bs=64 default) | TTT diagnostic anchor |
| R1 | `TTT_BATCH_SIZE=16` | 小 batch 是否导致 LoRA grad 方向更噪、更不一致 |
| R2 | `TTT_BATCH_SIZE=128` | 大 batch 是否让 gradient sample 更稳定 |
| R4 | `TTT_LORA_RANK=192` | 高 rank 在默认 batch 下的 control |
| R5 | `TTT_LORA_RANK=192 TTT_BATCH_SIZE=16` | 高 rank + 小 batch 的 SMT-style failure mode |

R 组输出 side-channel：`logs/diag_<RUN_ID>.ttt_grad_cos.rank*.jsonl` 以及 summary。核心读数是每个 TTT inner step 的 batch-slice LoRA grad pairwise cosine、norm 和 per-rank record count。

## 关键预期

- **Z0** 是后续所有 Stage 2 cells 的唯一比较锚点；不要直接拿 Stage 1 A0 做同轴差分，因为 Stage 1 A0 的 fused CE 口径不同。
- **P1 vs P0** 给出 compile overhead 的实证口径：如果 P1 多跑明显更多训练步且 BPB 更好，说明 600s cap 下 compile 时间是有效训练时间损失。
- **P2 vs P0** 判断 dynamic compile 是否值得保留；如果 compile trace/throughput 无改善，就不需要继续展开。
- **Q1 vs Z0** 判断 int4 matrix 是否过激；**Q3 vs Q1** 判断 LQER rank 是否能补偿 int4 损失。
- **Q4 vs Z0** 判断 embedding 低 bit floor 是否存在明显质量损失。
- **R1/R2 vs R0** 是 TTT batch-size 主线：小 batch 若 pairwise cosine 更低、norm 更不稳定，同时 BPB 变差，就支持“小 batch gradient sample 不准”的解释。
- **R5 vs R4** 是 high-rank 下的小 batch stress test：若 R5 明显坏于 R4，说明 failure 更像 batch-sampling/noise 问题，而不是单纯 rank 大小问题。

## 风险点

- R 组依赖 `models/20260427_232132-A0-s42-38fe4988.int6.ptz` 已经在 H100 host 上存在；缺失会直接失败。
- R 组是 eval-only，但 TTT runtime 仍可能很长；这些 cells 放在 sweep 末尾，可在拿到足够 diagnostic 后中止。
- P 组使用 diagnostic fork，会写 side-channel JSONL；主要 metrics 仍从 per-run log 和 TSV 中读取。
- Stage 2 默认不回答 tokenizer/dataset story，也不回答所有 quantization trick；这些可以单独开后续短实验，不应混进本轮最小闭环。
