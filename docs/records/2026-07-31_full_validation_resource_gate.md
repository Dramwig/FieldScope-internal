# 全量验证资源门记录

日期：2026-07-31  
机器：`pro6000`  
代码：`da5c9efb`（运行时），配置固化提交为后续 revision  
证据范围：正确性与资源可行性；不构成方法效果结论

## 事实

- 本地与远端均通过 `ruff check src tests scripts` 和 27 项 pytest。
- 真实 AuraFlow contract smoke 通过；backbone 自检为完全冻结。
- cache v2 包含 768 维 `dit_hidden`、`dit_attention` 和
  `dit_attention_adjacency`，保存后精确回读。
- 256 px、三时间、R=2、中心差分、双噪声视图时，probe batch 从 1 增至 2：
  transformer 调用由 30 降至 18，抽取由 1.651 秒降至 1.201 秒，峰值已分配
  显存均约 14.03 GB。
- 512 px 最终设置（`R=8`、中心差分、双噪声、probe batch 8）单样本评估
  102 个状态、18 次 transformer 调用；抽取 5.190 秒，峰值已分配显存
  14.779 GB，cache（含 synthetic 四任务 target）7,593,893 字节。
- 相同设置的图像 batch 2 共抽取 9.944 秒，即 4.972 秒/样本；峰值已分配
  显存 14.803 GB。该设置小幅提升持续吞吐并将每个 shard 合并为两个样本，
  因此固化为最终配置。

完整机器可读数字与 cache fingerprint 见
`artifacts/reports/2026-07-31_auraflow_resource_gate.json`。

## 解释边界

这些结果只证明最终张量合约和单样本资源门可运行。它们没有测量对象边界、
分类、分割或深度有效性，不能写成 FieldScope 优于任何 baseline。

512 px 数字可用于估算抽取调度，但不能直接按单样本线性外推完整数据集：
真实数据加载、分片批量、target 类型和长期 GPU 热状态尚未计入。下一步必须在
真实 CIFAR-10/VOC cache 上测量持续吞吐，再决定完整任务的分片和监控策略。
