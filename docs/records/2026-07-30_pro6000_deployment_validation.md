# pro6000 部署与验证记录

日期：2026-07-30 至 2026-07-31。该记录只证明实现、资产和接口可运行，不证明 FieldScope
研究假设或任何下游性能结论。

## 被测实现

- 被测 revision：`c5de8f69a4c4c02cce59bf8736f4364181763ad4`。
- 远端代码：`/root/autodl-tmp/FieldScope/FieldScope-internal`。
- 条件与资产：见
  `docs/experiment_conditions/2026-07-30_pro6000.md`。

## 已通过检查

```text
ruff check src tests scripts
All checks passed

python -m pytest
20 passed
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

真实主干合约 smoke（`configs/eval/auraflow_contract_smoke.yaml`）：

```text
status: passed
backend: auraflow
variant: fp16
dtype: bfloat16
frozen: true
state:    [1, 16, 14]
response: [1, 16, 4]
affinity: [1, 16, 16]
loss: 2.9897477626800537
cache_reload_equal: true
```

- VAE 编码、空文本 UMT5、两分片 AuraFlow transformer、clean-time/速度方向转换、
  前向响应差分、关系图、FP32 readout 反向传播和 cache 回读均实际执行。
- smoke 时另有既存 `pf-vlm` 进程占用约 21–23 GB 显存且 GPU 利用率较高；
  本记录不使用耗时或吞吐作为结论。
- PyTorch 对 CUDA median 与二维交叉熵发出 `warn_only` 非确定性警告；
  固定 seed 不应被表述为跨运行逐 bit 一致。

## 验证中修正

- 数据目录忽略规则改为根路径锚定，补入数据配置和安全解包脚本。
- 运行脚本增加 `FIELDSCOPE_PYTHON`，避免后台任务依赖交互式环境激活。
- AuraFlow 强制 safetensors，并为上游 legacy FP16 transformer index 增加
  37,663 字节的 Diffusers 0.38 兼容索引别名。
- 冻结 BF16 特征进入任务头前统一转换为 FP32；cache readout 使用同一规则。

这些结果不包含真实任务指标；正式实验仍必须使用冻结主干、同容量任务头、
无监督诊断和预先记录的对照。
