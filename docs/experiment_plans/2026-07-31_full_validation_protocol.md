# FieldScope 全量有效性验证协议

日期：2026-07-31  
状态：预注册计划，尚无方法有效性结论

## 目标与判定边界

本协议检验：冻结 AuraFlow-v0.3 时，图像平面锚定的局部扰动响应是否提供了
静态潜变量、速度场和 DiT 内部特征不能等价解释的关系结构，并能以同容量任务头
支持分类、分割和深度。接口 smoke、toy 结果、小样本调试和单 seed 结果均不构成
该假设成立的证据。

方法有效性至少同时满足：

1. 无监督对象/边界指标上，响应图稳定优于 VAE、原始速度、DiT hidden 和
   DiT attention 对照；
2. 同容量、同训练预算 readout 下，`response` 或 `full` 相对静态/hidden 对照
   在全局分类和至少两个密集任务上有多 seed 稳定增益；
3. 主结论使用完全冻结的 VAE、文本编码器和 Flow Transformer；
4. 核心结论经三 seed、置信区间和配对检验支持，且失败条件完整记录。

若任一条件未满足，论文只能报告相应否定或限定结果，不得写成通用视觉表示结论。

## 固定方法

- backbone：`fal/AuraFlow-v0.3`，revision 与权重哈希见远端条件记录；
- 公共时间：clean-time `t=0` 为噪声、`t=1` 为图像；
- 最终主配置：512 px、时间 `[0.2, 0.5, 0.8]`、`R=8`、中心差分、
  antithetic noise、16×16 patch 图、local radius 1、global top-k 16；
- 条件：空文本为主设置；中性文本和类别无关文本仅作条件消融；
- 容量匹配：各表示先经固定、无训练参数的 768 维高斯 sketch，再共享一个
  768→hidden 投影；`full` 在该投影前无参数融合，DiT hidden cache 也固定为
  768 维；
- 特征按样本 ID 分片缓存；任务头训练不重复调用生成主干；
- 主 seed：`4121`、`7319`、`104729`。调试 seed 不进入主表。

资源门可使用 256 px、`R=2`、8×8 图，但结果只决定执行可行性，不进入最终主张。

## 数据集、划分与主指标

| 层级 | 数据集 | 划分 | 主指标 | 角色 |
|---|---|---|---|---|
| 调试 | CIFAR-10 | 官方 train/test；train 内固定 5k val | top-1 | 管线与低成本消融 |
| 无监督/分割 | PASCAL VOC 2012 | train/val | boundary AP、pairwise AUROC、mIoU | 第一密集任务 |
| 分类主任务 | ImageNet-100 | 完整选中类的 train/val | top-1/top-5 | 全局任务 |
| 分类扩展 | ImageNet-1k | 官方 train/val | top-1/top-5 | 通过 ImageNet-100 门后执行 |
| 分割扩展 | ADE20K | 官方 train/validation | mIoU/pixel accuracy | 第二语义密集任务 |
| 几何密集 | NYUv2 | 官方 train/test | AbsRel、RMSE、δ1/δ2/δ3 | 深度任务 |

ImageNet-100 类集合固定为可用 ImageNet-1k train 目录中 WordNet ID
字典序前 100 类，并逐项写入 cache manifest。现有
`imagenet_256_10pct` 只能用于规模门；最终 ImageNet-100 结果必须使用这些类的
完整训练样本。ImageNet-1k、ADE20K、NYUv2 在资产版本、划分和哈希完成记录前
保持 TODO。

## 必做对照

所有监督对照共享 tokenizer 深度、hidden width、任务头、增强、优化器、
epoch 数和选模规则：

- `z0`、`zt`、`velocity`、`mismatch`、`endpoint`；
- `state`：场状态节点 + 固定局部图；
- `response_local`：响应节点 + 固定局部图；
- `state_graph`：状态节点 + 响应图；
- `response`：响应节点 + 响应图；
- `full`：状态与响应节点融合 + 响应图；
- DiT hidden + 固定局部图；
- DiT hidden + DiT attention 图；
- 参数量匹配的随机特征/随机 Flow 和标签无关 probe 对照。

其中 `response_local` 对 `response` 隔离图贡献，`state` 对 `state_graph`
隔离节点响应与关系图贡献。DiT hidden/attention 捕获实现完成并通过冻结与形状
测试前，不得宣称排除了内部特征解释。

## 核心消融

- 时间：单时间 `{0.2, 0.5, 0.8}` 对多时间；
- probe：结构化/随机、`R={1,2,4,8}`、forward/central、eta 稳定性；
- 图：无图、固定局部图、响应图、DiT attention 图，top-k 与 radius 扫描；
- 输入状态：无 state、无 response、无 graph；
- 因果破坏：跨图像打乱响应、空间打乱 probe、随机 Flow；
- 条件：空文本、中性文本、类别无关文本；
- 适配：冻结主干为主；LoRA 仅作为“是否必须适配”的诊断，不替代冻结结论。

## 训练与选模

- train/validation/test 严格分离；test 不参与超参数选择；
- 分类使用交叉熵，分割使用 ignore-index 255 的像素交叉熵，深度使用有效像素损失；
- AdamW、cosine schedule、梯度裁剪 1.0；每任务的学习率、weight decay、
  epoch 和增强在首次主实验前冻结；
- 仅根据 validation 主指标选择 checkpoint；每 seed 只在最终 test 评估一次；
- 记录 trainable parameter count、吞吐、峰值显存、cache 大小和失败重试；
- 断点恢复必须保持 epoch 级 shuffle 可复现，cache 恢复必须校验样本 ID 与 target。

## 统计与报告

- 对每个主任务和主对照报告三 seed 的均值、标准差和 95% t 区间；
- 对同一 seed/划分的主要方法差异做配对检验，并同时报告原始差值；
- 无监督指标按图像 bootstrap 95% 区间；类别指标另报 macro 分布；
- 主表之外的多重比较使用 Holm 校正；
- 同时报告最好、最差和中位 seed，不隐去非有限值、OOM 或退化图；
- 所有数字回链到代码 revision、配置、cache manifest、checkpoint 和结果 JSON。

## 执行门

1. **正确性门**：单元测试、toy smoke、真实 AuraFlow contract smoke；
2. **资源门**：记录单样本 wall time、峰值显存、缓存体积并估算全量成本；
3. **信号门**：VOC 无监督图和 CIFAR-10 同容量 readout，不作论文结论；
4. **主任务门**：完整 ImageNet-100、VOC、ADE20K、NYUv2 三 seed；
5. **扩展门**：前四门满足后执行 ImageNet-1k 与高成本消融；
6. **结论门**：逐条核对本文件四项有效性条件，再更新论文。

任何资源缩减、数据缺失或协议偏离都写入 `docs/records/`，并在结果表标明，
不得静默替换最终设置。
