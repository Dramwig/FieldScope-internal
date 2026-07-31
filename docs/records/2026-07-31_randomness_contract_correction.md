# 抽取随机性合约修正记录

日期：2026-07-31  
证据范围：实现正确性；不构成方法效果结论

## 发现的问题

在全量验证可执行性审计中发现，旧实现于每次
`FieldResponseExtractor.extract()` 调用内重置 path-noise RNG，并按当前 batch
形状生成 probe。其后果是：

- 同一 batch 位置上的不同图像可能复用相同 path noise；
- 同一样本的 probe basis 可能随 batch 位置变化；
- 相同样本在不同 batch/shard 布局下没有显式的不变性保证。

这与预注册的“每张图固定噪声”及 response sketch 公共坐标系不一致。该问题在
正式 signal gate 和任何主任务全量缓存生成前发现；此前资源门报告只用于显存、
吞吐和体积估计，不作为效果证据。

## 修正

- path noise seed 固定由 `probe.seed` 与完整 `sample_id` 的 SHA-256 派生；
- noise 在 CPU/float32 以逐样本独立 generator 生成，再转换到模型 device/dtype；
- structured、Gaussian 和 spatially shuffled probe 均使用固定共享 basis，再沿
  batch 维广播；
- dataset cache manifest 记录随机性策略；
- 新代码树 hash 使旧缓存 extraction signature 自动失效。

## 验证

- 新增 extractor 级 batch 合并/拆分逐张一致性测试；
- 新增 dataset cache 在 batch `1/2`、shard `2/3` 间逐样本一致性测试；
- 新增三种 probe control 的跨 batch 共享 basis 测试；
- 本地 `ruff check src tests scripts` 通过；
- 本地 `python -m pytest`：43 项通过。

真实 AuraFlow 合约和资源数字需要在新 revision 上随下一轮远端 gate 复核。复核前，
不得把旧 revision 的 cache fingerprint 解释为当前正式实验资产。
