# 04-28-ablation_stage5-tokenizer

Billy `<cap>` tokenizer 消融实验（vocab capacity sweep），
`run_billy_tokenizer.sh` + `train_gpt_s0_pr1851_mod_billy_tokenizer.py`。

## 状态

- **7/7 cells 完成 + 1 representative pre-fix failure**：T0 + TA + TB + TF +
  TC + TD + TE 全部 OK 含 ttt_bpb；TA pre-fix failure 保留作为 bug 修复
  前后对比的代表性记录。
- **数据全在远端** `/dev/shm/billy_data/datasets/`，可直接复用做 EXT 实验。
- **EXT runner 已就位**（commit `a6bea37`）：`run_billy_tokenizer_ext.sh`
  3 cell（T0_NORM / TB_NORM / TC_NORM），尚未运行。

## 采纳 TSV

`stage5_final.tsv` — 与 stage1/stage2/stage3/stage4_final.tsv 同 13-col schema
(`stage, sweep_id, cell_id, description, seed, run_id, status,
pre_quant_bpb, quant_bpb, ttt_bpb, artifact_bytes, overrides, comment`)。
共 8 行覆盖 5 轮 sweep 中的 7 个**有代表性**的成功 + 1 个**有代表性**的失败：

| # | sweep_id | cell | status | pre_quant | quant | ttt | bytes |
|---|---|---|---|---:|---:|---:|---:|
| 1 | 20260429_010238 | T0 | OK | 1.07063 | 1.08030 | **1.06929** | 15,950,599 |
| 2 | 20260429_010238 | TA | FAIL_compile_assert | 1.07361 | 1.08312 | — | 15,919,197 |
| 3 | 20260429_040050 | TA | OK | 1.07034 | 1.08011 | 1.06889 | 15,932,372 |
| 4 | 20260429_040050 | TB | OK | 1.06984 | 1.07946 | **1.06837** ← best BPB | 15,892,900 |
| 5 | 20260429_040050 | TF | OK | 1.07150 | 1.08120 | 1.06987 (refutes H5) | 15,828,127 |
| 6 | 20260429_085604 | TC | OK | 1.07028 | 1.07999 | 1.06879 | 15,804,730 |
| 7 | 20260429_093153 | TD | OK | 1.07066 | 1.08036 | 1.06893 (H2 control) | 15,817,466 |
| 8 | 20260429_093153 | TE | OK | 1.07176 | 1.08128 | 1.06909 | **15,650,823** ← best bytes |

第 2 行（pre-padding-fix TA failure）保留作为修复前的代表性失败记录；
第 3 行（post-fix TA）是同一 cell 的成功重跑。

## 已合并并删除的原始 sweep TSV

下列 TSV 在合并入 `stage5_final.tsv` 后从 `parameter-golf-dev-lite/logs/`
删除：

- `sweep_billy_tokenizer_20260429_025316.tsv`（TA pre-fix 第 2 次复现，未入 final TSV，仅 README 点名）
- `sweep_billy_tokenizer_20260429_040050.tsv`（TA post-fix + TB + TF）
- `sweep_billy_tokenizer_20260429_085604.tsv`（TC，单独 sweep）
- `sweep_billy_tokenizer_20260429_093153.tsv`（TD + TE，最后批次）

跨 stage 合并见 `../all_ablation.tsv`（83 行 = 1 header + 4 stage-0
+ 26 stage-1 + 12 stage-2 + 14 stage-3 + 11 stage-4 + 6 stage-4-patch
+ 8 stage-5 + 1 stage-6）。

## 主要结论速览

1. **TB (p=0.2 inverse_freq) 是 ttt_bpb 全局最优**：1.06837 vs T0 1.06929
   = −0.00093 (3× noise floor)。**推荐为 Billy `<cap>` 主结果**。
2. **TE (p=1.0) 是 bytes 全局最优**：−299,776 bytes vs T0，ttt_bpb 仍 net
   positive（−0.00020）；inflation 1.135× 不破坏 BPB net-positive。
3. **TD vs TF 是 sweep 最强反直觉信号**：uniform 删 552 random > fmax 删
   470 low-freq by −0.00094 ttt_bpb（10× noise）。**保护高频 cap token
   主动有害**——H5 不只是无用，是 actively harmful。
4. **H2 (inverse_freq > uniform) inconclusive at single seed**：TC vs TD
   = −0.00014（在 noise 内）。但间接证据强：uniform 距 best inverse_freq
   仅 0.00056 → freq-mass 不是 BPB 主要驱动。
5. **GPTQ 不偏向破坏 `<cap>`**：全 7 cells 的 quant−pre_quant Δ 都在
   0.00951–0.00977，与 T0 0.00967 同段。
6. **吞吐不受影响**：全部 7 cell tok/s 6.44–6.66M，step 4863–4988，与
   T0 (6.61M / 4962) 持平 ±3%。
7. **Inductor stride bug 已修复**：commit `37b5de6` 引入 `VOCAB_PAD_MULTIPLE=64`
   把 model-side vocab dim padding 解耦于 tokenizer vocab。

## Pareto 前沿

```
        ttt_bpb (lower better)
            ▲
   1.06987  ┤            ● TF ← refuted (worse than baseline)
            │
   1.06929  ┤  ● T0 (anchor)
            │
   1.06909  ┤                                  ● TE
            │
   1.06893  ┤                ● TD
            │
   1.06889  ┤  ● TA
            │
   1.06879  ┤              ● TC
            │
   1.06837  ┤        ● TB ← best BPB
            │
            └─────────────────────────────────────►
              0   −18  −58  −122 −133 −146 −300  bytes saved (KB)
```

真正在 Pareto frontier (lower-left envelope) 上的 cells：T0 → TA → TB →
TC → TE。TF/TD strictly dominated。

## 文件

- `stage5_final.tsv` — 采纳 TSV（13-col，8 行）。
- `*.txt` / `*.runner.txt` — 9 套 log（T0 1 套 + TA 3 套 retry + TB + TF + TC + TD + TE）。
