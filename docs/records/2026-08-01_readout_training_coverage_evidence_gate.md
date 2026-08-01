# Readout 训练覆盖与优化器步数证据门

日期：2026-08-01  
证据范围：正式 cached-readout 的样本覆盖、optimizer-step 预算、epoch 恢复和最终
checkpoint 可审计性；不构成方法效果结论。

## 审计发现

原执行路径本身满足完整训练语义：`ShardShuffleSampler` 在每个 epoch 对全部 cache
索引生成无重复全排列，`DataLoader` 默认 `drop_last=False`，并以 `seed + epoch` 固定
shuffle；last checkpoint 只在完整 epoch 的模型、optimizer、scheduler、RNG 和 history
全部持久化后写入，恢复从下一个完整 epoch 继续。

但原 training report 只记录数据集的静态 `train_samples`、目标 `epochs` 和
`batch_size`，没有记录每个 epoch 实际消费的样本数或实际执行的 optimizer steps。
因此，即使执行实现正确，最终 evidence audit 也不能独立拒绝“末批被丢弃、某个 epoch
少跑一步或报告只声明预算但未执行”的伪完整结果。这是正式长跑前必须关闭的证据缺口。

## 固定执行契约

本门注册并在 checkpoint/report 中持久化以下契约：

- sampler：`shard_shuffle_full_permutation_v1`；
- epoch seed：`run_seed_plus_zero_based_epoch_v1`；
- `drop_last=false`；
- 每个实际 batch 恰好执行一次 optimizer step。

每个 epoch 结束前，训练进程必须同时证明：

1. `train_samples == len(train_dataset)`；
2. `optimizer_steps == ceil(train_samples / batch_size)`。

任一条件不成立会在 checkpoint/report 写入前直接失败。每个 history entry 记录实际
`train_samples` 和 `optimizer_steps`；checkpoint/report 另记录每 epoch 与累计步数、累计
sample exposures。resume checkpoint 必须具有完全相同的覆盖契约和步数预算。

## 正式注册总量

| 数据集 | train 样本 | epochs | batch | 每 cell 步数 | 60 cells 总步数 |
|---|---:|---:|---:|---:|---:|
| ImageNet-100 | 116,455 | 90 | 128 | 81,900 | 4,914,000 |
| VOC 2012 | 1,318 | 80 | 4 | 26,400 | 1,584,000 |
| ADE20K | 18,189 | 80 | 2 | 727,600 | 43,656,000 |
| NYUv2 | 715 | 80 | 4 | 14,320 | 859,200 |

四任务、20 表示、3 seed 的注册总量为：

- `51,013,200` optimizer steps；
- `725,922,600` training sample exposures。

matrix report 汇总每个 cell 的实际累计值；最终 full evidence report 同时输出 required 与
reported totals，缺失或不相等均使裁决为 `incomplete`。

## 最终 checkpoint 独立核验

passed training report 必须绑定最终 last checkpoint 的路径和 SHA-256。evidence audit 会
重新计算 SHA-256、加载 checkpoint，并核验：

- 最终 epoch、目标 epochs、batch size、cache identity 和代码 revision；
- history 与 report 逐值一致；
- 覆盖契约、每 epoch 步数、累计步数和 sample exposures；
- AdamW 所有已初始化 parameter state 的内部 `step` 恰好等于该 cell 的注册总步数。

测试会同时篡改 report 中的末 epoch 步数和 checkpoint 内部 AdamW step-state，并在更新
文件 SHA 后确认 evidence audit 仍判为 `incomplete`，避免测试只命中哈希不匹配旁路。

## 本地验证

- `ruff check src tests scripts`：passed；
- `python -m pytest`：`137 passed`；
- toy smoke（2 steps）：passed，`cache_reload_equal=true`；
- `git diff --check`：passed；
- 3 样本、batch 2 的回归测试确认末批被保留，实际为 2 optimizer steps；
- 中断后 resume 与不间断训练的模型、optimizer、scheduler、RNG、history、累计步数和
  sample exposures 精确一致；
- serial 与多 seed worker 的 checkpoint 训练覆盖字段精确一致。

## 结论边界

本记录只关闭“正式 readout 是否完整执行注册训练预算以及最终报告能否证明该事实”的
工程证据缺口。真实 AuraFlow signal gate、四任务主矩阵、完整 VOC 无监督诊断、四项
因果/条件控制和最终机器可读裁决仍未完成，因此不能据此声称 FieldScope 有效。
