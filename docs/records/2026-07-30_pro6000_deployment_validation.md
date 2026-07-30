# pro6000 部署与验证记录

日期：2026-07-30。该记录只证明实现、资产和接口可运行，不证明 FieldScope
研究假设或任何下游性能结论。

## 被测实现

- 实现 revision：`a84b1ad2af30575dc0807210f1d7c00ccdb98263`。
- 远端代码：`/root/autodl-tmp/FieldScope/FieldScope-internal`。
- 条件与资产：见
  `docs/experiment_conditions/2026-07-30_pro6000.md`。

## 已通过检查

```text
ruff check src tests scripts
All checks passed

python -m pytest
19 passed
```

远端 toy smoke（`configs/eval/toy_smoke.yaml`，1 个优化步骤）：

```text
status: passed
state:    [2, 64, 14]
response: [2, 64, 96]
affinity: [2, 64, 64]
loss: 3.8963918685913086
cache_reload_equal: true
```

数据链路验证：

- CIFAR-10 与 VOC 2012 归档远端 SHA-256 均与固定条件记录一致。
- 安全解包脚本成功，prepared 目录分别约 178 MB 和 2.0 GB。
- CIFAR-10 train 取 2 样本、VOC 2012 train 取 1 样本，经 torchvision adapter、
  toy response extractor 和分片 cache 写入全部成功。
- 验证 cache 位于
  `/root/autodl-tmp/FieldScope/datasets/feature_cache/validation/`。

这些结果不包含真实任务指标；正式实验仍必须使用冻结主干、同容量任务头、
无监督诊断和预先记录的对照。
