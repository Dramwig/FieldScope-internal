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
- 每张图使用由样本 ID 与 probe seed 派生的固定 path noise；同一样本在不同
  batch/shard/恢复布局下必须逐张一致，不同样本不得静默复用同一噪声。
  所有图像共享固定 probe basis，使 response signature 处于同一随机草图坐标系；
- 条件：空文本为主设置；中性文本和类别无关文本仅作条件消融；
- 容量匹配：各表示先经固定、无训练参数的 768 维高斯 sketch，再共享一个
  768→hidden 投影；`full` 在该投影前无参数融合，DiT hidden cache 也固定为
  768 维；
- 特征按样本 ID 分片缓存；模型 batch 固定为 2，cache shard 固定为 64，训练按 shard
  分组洗牌并使用固定容量 LRU；pro6000 的同一 readout matrix 进程允许最多 160 GiB
  只读内存 cache，任务头训练不重复调用生成主干；
- 非方形图像及密集标签保持纵横比，将短边缩放到目标分辨率后做配对中心方形裁剪；
- 无监督图诊断使用 `dense` cache；监督 readout 使用 `readout_sparse`，后者仅无损
  打包实际使用的加权 adjacency，不保留未被监督 tokenizer 读取的 dense affinity。
  两种策略的表示与职责必须在 manifest 中显式记录，不得用 sparse cache 计算无监督指标；
  NYUv2 主结果使用该完整中心裁剪而非 Eigen crop，因此只在本协议内比较，不直接与
  使用不同 crop 的公开数字横比；
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
字典序前 100 类，并逐项写入 cache manifest。每类 train 样本按 seed 4121
固定抽取 10% 作为 internal validation，官方 validation 仅作最终 test。现有
`imagenet_256_10pct` 只能用于规模门；最终 ImageNet-100 结果必须使用这些类的
完整训练样本。ImageNet-1k、ADE20K、NYUv2 在资产版本、划分和哈希完成记录前
保持 TODO。

VOC 2012 与 ADE20K 同样从官方 training split 以 seed 4121 固定抽取 10%
internal validation，官方 validation 仅作最终 test。NYUv2 使用官方 795/654
划分，并从 795 张官方 train 中固定抽取 10% internal validation。

## 必做对照

所有监督对照共享 tokenizer 深度、hidden width、任务头、增强、优化器、
epoch 数和选模规则：

- `z0`、`zt`、多时间 `trajectory`、`velocity`、`mismatch`、`endpoint`；
- `random_feature_local`：按 sample ID 与训练 seed 固定生成的 768 维随机
  patch 特征 + 固定局部图，用于排除任务头仅凭容量拟合的解释；
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
- 密集 readout 在冻结的 16×16 patch 网格上训练，监督目标以配对的 nearest
  （分割）、仅有效像素的 area average（深度）或 bilinear（法向）规则降采样；
  正式 test 指标仍在 512×512 目标网格上计算，其中分割只上采样离散 argmax
  标签，避免构造 512×512×150 的无意义 logits。该实现约束在任何新 signal
  或正式结果产生前固定；
- AdamW、cosine schedule、梯度裁剪 1.0；每任务的学习率、weight decay、
  epoch 和增强在首次主实验前冻结；
- 仅根据 validation 主指标选择 checkpoint；每 seed 只在最终 test 评估一次；
- 记录 trainable parameter count、吞吐、峰值显存、cache 大小和失败重试；
- 断点恢复必须保持 epoch 级 shuffle 可复现，cache 恢复必须校验样本 ID 与 target。
- cache manifest 还必须记录 path-noise 与 probe-basis 策略；任何旧随机性策略缓存
  不得与当前主实验混用。

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
   固定晋级规则如下：
   - VOC：`response` 的 boundary AP 同时高于 response-shuffled、state、
     DiT hidden 与 DiT attention，且 pairwise AUROC 高于 response-shuffled 和 state；
   - CIFAR：`response` 与 `full` 各自相对 response-shuffled 和每 seed 最佳的
     state/DiT hidden/DiT attention 对照，三 seed 平均 top-1 增益至少 0.5 个百分点，
     且至少 2/3 seed 同向；
   - 任一 cache 不完整、revision 不一致、随机性合约不匹配或指标非有限时，
     判为 `incomplete`；全部满足为 `proceed`，信号完整但任一阈值不满足为
     `stop_or_redesign`。最初该门用于节省全量算力；在用户明确要求必须完成全量
     训练测试后，`stop_or_redesign` 改为必须保留的负向预诊断，不再停止四个正式
     主任务。只有 `incomplete`（输入、revision、cache 或指标不完整）会阻止继续；
     这一预注册变更发生在任何新 signal 或正式任务结果产生之前。该门不作为论文
     有效性证据；
4. **主任务门**：完整 ImageNet-100、VOC、ADE20K、NYUv2 三 seed；
5. **扩展门**：前四门满足后执行 ImageNet-1k 与高成本消融；
6. **结论门**：逐条核对本文件四项有效性条件，再更新论文。

任何资源缩减、数据缺失或协议偏离都写入 `docs/records/`，并在结果表标明，
不得静默替换最终设置。

## 实现澄清：关系图归因对照

正式矩阵加入参数量匹配的 identity-adjacency（无跨 patch 消息传递）与固定图像网格
对照：`state_nograph`、`response_nograph`、`full_nograph`、`response_local` 和
`full_local`。`response` 或 `full` 只有在同一 seed、同一任务和同一训练预算下，
同时稳定超过对应的无图版本与固定局部图版本，才能计为“响应诱导关系图”带来的证据；
仅超过静态 latent、velocity 或 DiT hidden 对照不足以完成图结构归因。

## 实现澄清：最终因果与条件结论门

主任务审计最多输出 `main_tasks_supported_pending_causal_audits`，不得直接输出核心假设
成立。只有在完整 VOC 2012 test 的 paired-image 审计中，空文本、预训练、结构化 probe
的 response 同时以 bootstrap 95% 区间下界大于零超过随机 Flow 与空间打乱 probe，且
在固定的 `a neutral photograph` 和 `an unrelated scene` 两种条件下继续超过
response-shuffled、state、DiT hidden 与 DiT attention，联合审计才可输出
`supports_core_hypothesis`。否则报告因果归因失败或证据不完整，不得以主任务增益替代。
跨图像打乱响应固定为按 seed 随机生成的 pooled derangement：每个样本恰有一个 donor、
无 self-donor，pool 由全局随机 shard 组成并在 pool 内逐样本随机成单环，从而避免按类别
排序缓存上的可逆标签映射，同时把随机访问限制在至多 32 个 shard 的局部工作集内。
无论主任务审计为完整正向还是 `limited_or_negative`，Random Flow、空间打乱 probe、
中性文本和无关文本四类已注册对照都必须执行并进入统一最终判定；只有 `incomplete`
可以阻止该阶段。完整负向主结果的最终 verdict 保持 `limited_or_negative`；完整正向
结果再由因果门区分 `supports_core_hypothesis` 与
`main_task_gain_not_causally_attributed`。该修正在任何真实 signal 或正式结果产生前固定。
