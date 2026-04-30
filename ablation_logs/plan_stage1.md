# PR #1851 消融实验计划

**Created:** 2026-04-27
**Hardware:** 8×H100 SXM
**Script:** `parameter-golf-dev-lite/run_ablation_1851.sh`
**Training file:** `parameter-golf-dev-lite/train_gpt_s9_caseops_lqer.py`

## Per-run 公共配置

- 单 seed = 42
- `MAX_WALLCLOCK_SECONDS=600`
- `GPTQ_RESERVE_SECONDS=0.5`
- `CASEOPS_ENABLED=1`
- `FUSED_CE_ENABLED=0`（training file 默认；Stage 1 使用 s9_caseops baseline 口径）
- `torchrun --nproc_per_node=8 --standalone train_gpt_s9_caseops_lqer.py`
- 实测单 cell 约 **22-24 分钟**，26 cells ≈ **10 小时** 全套 wallclock

## 26 个 Cell

### Group A — LQER 量化补偿（9 cells）

| Cell | 改动 (env override) | 在测什么 |
|---|---|---|
| **A0** | (none) | Stage 1 baseline（s9_caseops / fused CE off 口径），目标 post-TTT BPB 仍参考 #1851 附近 |
| A1 | `LQER_ENABLED=0` | LQER 整体贡献 |
| A2 | `LQER_ASYM_ENABLED=0` | sym vs asym |
| A3 | `LQER_RANK=2` | rank 砍半 |
| A4 | `LQER_RANK=8` | rank 翻倍 |
| A5 | `LQER_TOP_K=1` | 只补 1 个张量 |
| A6 | `LQER_TOP_K=5` | 补 5 个张量 |
| A7 | `LQER_FACTOR_BITS=2` | INT2 极致压缩 |
| A8 | `LQER_ASYM_GROUP=128` | group 翻倍 |

### Group B — SmearGate（2 cells）

| Cell | 改动 | 在测什么 |
|---|---|---|
| B1 | `SMEAR_GATE_ENABLED=0` | SmearGate 整体贡献 |
| B2 | `SMEAR_GATE_BOS_MASK=0` | #1851 BOS-fix 单独贡献（其他全保持 #1851） |

### Group C — Attention gate 选型（3 cells）

| Cell | 改动 | 在测什么 |
|---|---|---|
| C1 | `SPARSE_ATTN_GATE_ENABLED=0` | SparseAttnGate 关掉损失多少 |
| C2 | `GATED_ATTN_QUANT_GATE=1` | #1851 故意关 GateQuant，重开是否变差 |
| C3 | `SPARSE_ATTN_GATE_ENABLED=0 GATED_ATTN_ENABLED=1` | 用密集 GatedAttn 替代 sparse |

### Group D — 量化超参（3 cells）

| Cell | 改动 | 在测什么 |
|---|---|---|
| D1 | `EMBED_BITS=8 EMBED_CLIP_SIGMAS=20.0` | 退回 S9 嵌入量化默认 |
| D2 | `MATRIX_BITS=5` | int5 矩阵量化 |
| D3 | `MLP_CLIP_SIGMAS=10.0` | 退回 S9 MLP clip |

### Group E — TTT 调度（5 cells）

| Cell | 改动 | 在测什么 |
|---|---|---|
| E1 | `PHASED_TTT_NUM_PHASES=1` | 单阶段 |
| E2 | `PHASED_TTT_NUM_PHASES=5` | 五阶段 |
| E3 | `TTT_LORA_ALPHA=96` | 退回 S9 alpha |
| E4 | `TTT_WEIGHT_DECAY=0.5` | 退回 S9 wd |
| E5 | `TTT_ENABLED=0 PHASED_TTT_NUM_PHASES=1` | 关 TTT，给 pre-TTT 地板 |

### Group F — 跨支线组合（4 cells）

| Cell | 改动 | 在测什么 |
|---|---|---|
| F1 | `SMEAR_GATE_BOS_MASK=0 GATED_ATTN_QUANT_GATE=1` | 直接复现 #1797，目标 BPB ≈ 1.06181 |
| F2 | `RECUR_ALPHA_ENABLED=1` | 在 #1851 上叠 #1801 RecurAlpha |
| F3 | `RECUR_ALPHA_ENABLED=1 GATED_ATTN_QUANT_GATE=1 SMEAR_GATE_ENABLED=0` | 接近原版 #1801，目标 BPB ≈ 1.06366 |
| F4 | `LQER_ENABLED=0 SMEAR_GATE_ENABLED=0 SPARSE_ATTN_GATE_ENABLED=0` | 三大菜全关，给最低地板 |


## 关键预期

- **A0** 是 Stage 1 内部锚点；由于 `FUSED_CE_ENABLED=0`，它不是 byte-identical 的真实 #1851 log 口径（真实 fused-CE 口径由 Stage 2 `Z0` 补齐）
- **F1** ≈ 1.06181（直接复现 #1797）
- **F3** ≈ 1.06366（接近 #1801）
- **B2 − A0** = BOS-fix 单独贡献（预期约 -0.0005，1851 vs 1797 已知差距）
- **A1 − A0** = LQER 整体贡献（预期约 +0.001 ~ +0.003）
- **F4 − A0** = LQER + SmearGate + SparseGate 合计贡献（地板线）

## 风险点

- 8×H100 内存不会成为瓶颈（已被 #1851 验证）
- `MAX_WALLCLOCK_SECONDS=600` 只限制训练主循环；实测单 cell 端到端约 25-26 分钟（训练 + GPTQ + Phased TTT eval）
- 部分 cell（如 E2 `num_phases=5`）可能比均值更慢，需按 TSV 空项或 runner log 判断是否重跑
- 若某 cell wallclock 不够导致提前停训，TSV 中 `quant_bpb` / `ttt_bpb` 可能为空，需重跑或个别延长 wallclock
