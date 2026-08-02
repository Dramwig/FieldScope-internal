# FieldScope 监督主任务逐样本误差分析协议

日期：2026-08-02
状态：前瞻性次要分析协议；在四任务正式主矩阵结果产生前锁定

## 角色与边界

本协议用于解释完整 ImageNet-100、VOC2012、ADE20K 和 NYUv2 主任务的正向、
混合或负向结果，不修改
`docs/experiment_plans/2026-07-31_full_validation_protocol.md` 中的主指标、表示、
seed、统计门或最终 verdict。正式矩阵的平均指标、三 seed 配对检验、完整 VOC
无监督诊断和四项因果/条件控制仍是方法有效性的唯一注册主证据。

本协议是在完整 VOC 无监督诊断已经完成、但四任务监督主矩阵尚未产生结果时提出。
因此，“响应可能更接近粗区域耦合而非精确边界”的动机属于后验观察；下面针对尚未
观察的监督结果所固定的分层与方向是前瞻性次要分析，整体仍不得改写为确认性主结论。

## 问题与固定预测

问题：如果 `response` 或 `full` 含有不同于静态或 DiT hidden 的有效信息，其相对
对照的逐样本增益是否随空间复杂度系统变化？

固定预测如下：

1. 若响应主要编码粗区域耦合，分类的逐类增益可以为正；在分割中，增益应在低边界
   密度、少类别或高主导类别占比图像上更大，并随边界密度与类别数增加而下降。
2. 在深度中，若该信号偏向区域几何，低深度不连续密度样本的 AbsRel 改善应大于
   高不连续密度样本。
3. 如果增益不随这些固定 strata 变化，或在所有 strata 均低于对照，则不使用粗区域
   耦合解释；如果只有事后新增切分为正，也不得作为本协议支持。

## 固定输入与比较

只读取正式主矩阵已经冻结的 held-out test cache、最佳 validation checkpoint、训练
报告、test 报告和 matrix report。不得重新选 checkpoint、调整 readout、修改 test
样本或按本分析选择超参数。

每个任务、seed 和 sample ID 固定比较：

- `response` 对 `response_shuffled`、`state`、`dit_hidden_local`、
  `dit_hidden_attention`、`response_local` 和 `response_nograph`；
- `full` 对 `full_shuffled`、`state`、`dit_hidden_local`、
  `dit_hidden_attention`、`full_local` 和 `full_nograph`。

所有差值统一为“候选更好时为正”：分类使用 top-1 正确指示差，分割使用逐图 mIoU
差，深度使用 `control_abs_rel - candidate_abs_rel`。同一比较必须使用相同 seed、test
cache 和完全对齐的 sample ID；缺失、重复、非有限或错位样本使该比较 `incomplete`，
不得静默删样本。

## 固定逐样本量与 strata

### ImageNet-100

- top-1 与 top-5 正确指示；
- 真实类别概率、负对数似然和 top-1 margin；
- 每类样本数、候选减对照的 top-1 差及 macro 类别分布。

不从分类标签臆造空间复杂度 strata。

### VOC2012 与 ADE20K

在原始 held-out target 分辨率、忽略 label 255 后计算：

- 逐图 mIoU：对 target 或 prediction 中 union 非零的类别取 IoU 宏平均；
- pixel accuracy；
- 有效像素占比；
- target 中不同有效语义类别数；
- 主导类别占有效像素的比例；
- 四邻域水平/垂直相邻有效像素中标签不同的边界密度。

固定 strata：边界密度四分位、主导类别比例四分位，以及类别数 `1`、`2`、`3+`。
四分位边界由整个固定 test split 的 target 描述量决定，与表示、seed 和预测无关。

### NYUv2

只在有限且大于零的 target 深度像素上计算：

- 逐图 AbsRel、RMSE 和 delta-1；
- 有效像素占比；
- target 深度的 5%–95% 分位范围；
- 四邻域相邻有效像素中，相对深度差
  `abs(d1-d2) / max(min(d1,d2), 1e-6) > 0.1` 的不连续密度。

固定 strata：深度不连续密度、有效像素占比和 5%–95% 深度范围的四分位。

## 固定汇总与统计

- 每个 candidate/control/task/seed 报告全部逐样本值和原始配对差；
- 按三个 seed 分别报告，再对同一 sample 的三个 seed 差取均值，形成跨 seed 样本差；
- 对总体均值差和每个固定 stratum 使用 seed 4121、2,000 次 paired-sample bootstrap
  95% 区间；
- 报告候选优于对照的样本比例；
- 分割报告边界密度、类别数、主导类别比例与逐样本差的 Pearson 相关；深度报告三项
  固定描述量与 AbsRel 改善的 Pearson 相关；
- 这些区间和相关仅用于解释，不进入主门，不据此触发 ImageNet-1k 或高成本消融。

不得只报告正向 comparison、seed 或 stratum。任何新增切分必须明确标为 post-hoc，
与本协议固定输出分开。

## provenance 与 fail-closed 契约

分析报告必须记录：

- 正式 checkpoint revision、code-tree SHA-256、checkpoint 路径与 SHA-256；
- matrix/training/test report 路径与 SHA-256；
- test cache identity、manifest SHA-256、样本数与 sample-ID SHA-256；
- analyzer revision、code-tree SHA-256、命令、batch size 和运行时间；
- checkpoint 内 task、representation、seed、config、mode 和维度。

由于 analyzer 可以在正式矩阵完成后加入，允许 analyzer revision 晚于 checkpoint
revision，但必须 fail-closed 地证明两 revision 间以下重放核心没有变化：配置解析、
cache/`CachedFeatureDataset`、表示选择、图与固定 sketch、tokenizer、heads、model、
任务 target 变换和正式 readout 推理。任一核心路径有差异时，不得直接重放旧 checkpoint；
必须在正式 revision 的独立 worktree 中运行经 SHA-256 固定的 analyzer，或另立兼容性
验证记录。

报告的 `status=passed` 只表示输入完整、重放与分层计算成功；它不表示方法有效、预测
成立或某个 stratum 显著。
