# Readout seed 并行运行门

日期：2026-08-01
证据范围：正式 cached readout 的调度等价性、资源安全性与运行时选择；不构成方法效果结论。

## 工程动机

注册的四任务、20 表示、3 seeds 训练预算不变：

- ImageNet-100：90 epochs、batch 128，共 `4,914,000` optimizer steps；
- VOC 2012：80 epochs、batch 4，共 `1,584,000` optimizer steps；
- ADE20K：80 epochs、batch 2，共 `43,656,000` optimizer steps；
- NYUv2：80 epochs、batch 4，共 `859,200` optimizer steps；
- 合计：`51,013,200` optimizer steps。

串行执行的预计墙钟时间是正式验证的主要工程风险。新增路径只允许并行调度彼此独立的
seed，不减少任务、样本、表示、seed、epoch、batch size、优化器步骤或证据阈值。

## 固定运行契约

运行门注册 1/2/3 个 spawned seed workers，固定 seeds 为
`4121, 7319, 104729`，在 signal 阶段已经生成的 CIFAR-10 train/val/test cache 上训练
`full` classification readout 20 epochs、batch 128。`full` 包含双固定输入投影和图消息，
并用注册 workload envelope 验证其 activation proxy 上界覆盖 ImageNet-100 batch 128、VOC
batch 4、ADE20K batch 2/150 类和 NYUv2 batch 4。选择非串行候选必须同时满足：

- best/last checkpoint 中 model、optimizer、scheduler、RNG、epoch、去除墙钟字段后的
  history、best epoch 与 best metric 精确相等；
- held-out metric 精确相等；
- CUDA reserved 上界不超过总显存 70%；
- 每 worker 160 GiB cache 上界之外仍保留至少 64 GiB 可用 RAM；
- 相对串行至少快 5%。

若 2/3-worker 候选失败、OOM、不等价、资源不安全或加速不足，则标记为不合格并回退到
1-worker；1-worker 本身失败则运行门失败。所有候选完成后选择墙钟时间最短的合格候选。

## 证据绑定

- 运行门只接受 clean worktree，并记录 revision、tree hash 和 readout execution contract；
- 正式四任务矩阵必须嵌入同一份 content-addressed profile identity；
- 主证据审计重新验证 profile 的 provenance、config contract、候选速度/内存/资格、最快
  候选选择及矩阵身份；缺失、过期或篡改均判为 `incomplete`；
- profile 的 evidence scope 明确为运行时等价性与吞吐，不得包含方法效果结论。

## 本地回归事实

CPU 小数据回归中，串行与 3 个 spawned workers 的 model、optimizer、scheduler、RNG、
去除墙钟字段后的 history、best/last 选择及 held-out metric 精确相等。Windows 上多个 worker
同时替换共享中间 `matrix_report.json` 曾触发 `WinError 5`；现改为每 seed 独立可丢弃进度
报告，worker 成功后删除，父进程再通过原始全矩阵校验路径生成唯一正式报告。该修复不改变
任何 cell 的训练语义。

真实 CUDA 候选的速度、显存和最终 worker 选择仍待远端运行门执行，不能由以上 CPU 回归
推出。方法有效性仍需真实 AuraFlow signal、四任务主矩阵和注册因果控制共同裁决。
