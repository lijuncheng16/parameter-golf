# 04-28-ablation_stage6-muon-ddp

Muon / DDP 通信-计算 attribution，`run_6_ddp_trace.sh` +
`train_gpt_s0_pr1851_mod_ddp_trace.py`。

## Stage 6 归档特例

**实验产物**（`diag_*.ddp_trace.rank*.jsonl` + `diag_*.profile_step*.json`）
不在本目录，而是 sibling
`../../04-28-ablation_stage6-muon-ddp/`（checkpoint 目录）：

```
../../04-28-ablation_stage6-muon-ddp/
  diag_20260429_032508-DT-FULL-s42-e0ff45bc.ddp_trace.rank{0..7}.jsonl   (~5.3 MB total)
```

本目录（log dir）只放：
- `*.runner.txt` / `*.txt` — torchrun stderr/stdout + 训练日志
- `stage6_final.tsv` — 采纳 TSV
- `README.md` — 本文件

`DDP_TRACE_README.md` §"Archival" 已记录这个约定。

## 采纳 TSV

`stage6_final.tsv` — 与 stage1/2/3/4/5_final.tsv 同 13-col schema。共 1 行：

| 行 | sweep_id | cell | status | ttt_bpb | bytes |
|---|---|---|---|---:|---:|
| 1 | 20260429_032508 | DT-FULL | OK | 1.06197542 | 15,954,632 |

未列入 stage6_final.tsv：

- DT-PROF（Layer 3 chrome trace）尚未跑。
- 原 sweep TSV `sweep_6_ddp_trace_20260429_032508.tsv` 是空 header（只有 column 行没有数据行）——
  bash post-run 报告 block 在脚本中途被中断，但 JSONL flush 完整保留，所以
  数据从 JSONL + `.txt` 重建到 `stage6_final.tsv` 的 comment 列里。

跨 stage 合并见 `../all_ablation.tsv`（71 行 = 1 header + 4 stage-0
+ 26 stage-1 + 12 stage-2 + 14 stage-3 + 11 stage-4 + 2 stage-5 + 1 stage-6）。

## 主要结论速览（详见 `parameter-golf-res/docs/submission/results_stage6-muon-ddp.md`）

1. **Trace 仪器化是 BPB 与 throughput 中性的**：DT-FULL ttt_bpb 1.06197 vs
   Stage 4 CU_VERIFY 1.06212 = −0.00014（噪声），tok/s 6.59M vs 6.58M = +0.13%。
   所以 trace 数据可以直接代表生产训练 run 的真实 comm/compute 比例。
2. **Comm 不是瓶颈**：exposed comm wait = 2.3% of step.total（112 µs / 4820 µs）。
   Muon 流水线（reduce_scatter ↔ Newton-Schulz overlap、all_gather ↔ apply_prev_update
   overlap）几乎完全藏住了 collective 时间。
3. **真正的瓶颈是 `opt.ar_packed` 的 tensor pack/unpack**：phase 占 57.5%（2772 µs），
   但 `dist.all_reduce` 自身只占 56 µs。差 2716 µs **不是通信，而是 `_all_reduce_packed_grads`
   里 cat + copy_ 处理多个小梯度的 GPU 启动开销**。Optimization ROI 最高的方向。
4. **Newton-Schulz 是第二大单 phase**（655 µs，13.6%）。继续压 Muon
   compute 应从这里入手而非 momentum / apply_update。
5. **Rank skew 小**（comm.*.wait_us max-min 12-22%）：不存在显著 straggler，
   集群通信对称。

## 文件

- `stage6_final.tsv` — 采纳 TSV（13-col，1 行）。
- `20260429_032508-DT-FULL-s42-e0ff45bc.txt` — 训练日志（含 hyperparameters / train_loss / val_bpb / GPTQ / TTT eval；4947 step / 600s）。
- `20260429_032508-DT-FULL-s42-e0ff45bc.runner.txt` — torchrun stderr/stdout，记 ttt phase 进度（最后输出 `quantized_ttt_phased val_bpb:1.06197542`）。

实验产物 JSONL 在 `../../04-28-ablation_stage6-muon-ddp/`。
