# Readout 热路径精确等价门

日期：2026-08-01
证据范围：正式 cached-readout 的运行可行性、失败语义与执行等价性；不构成方法效果结论。

## 动机

注册的四任务、20 表示、3 seed 全量矩阵包含 `51,013,200` 个 optimizer
steps，其中 ADE20K 为主要长跑负载。原实现会在每个 step 重复扫描已由 cache
加载边界完整验证的 feature 张量、搬运当前表示不消费的 state/response/graph，并重复
构造固定位置与固定图。这些操作不属于模型数学，但可能使注册协议在合理时间内无法完成。

本次修改不减少数据、表示、seed、epoch、batch size、优化器步骤、任务头容量或证据阈值。

## 执行契约

- `load_features` 仍在 cache 加载边界执行完整结构与 finite-value 校验；损坏或非有限 cache
  会在进入训练前失败。
- 已完整验证的 shard 在只读 slice、representation selection 和 stack 后只重复检查结构；
  模型公共 forward 默认仍支持完整校验。
- representation-aware collate 对注册的 20 个表示只物化 tokenizer 实际消费的
  state、response 与 content graph；未消费字段使用零存储/常数存储 view 占位。
- 正式 fast path 只把消费张量传到设备，并缓存固定二维位置、单位邻接、局部邻接及其
  归一化结果。
- loss 与 gradient norm 的非有限检查在 CUDA fast path 使用同 stream 的异步断言；
  strict reference 使用 host-synchronous 检查。
- checkpoint 恢复、cache/control contract、RNG、optimizer、scheduler、best/last 选择和
  held-out test 规则不变。

## Runtime profile schema v2

真实 CUDA readout runtime gate 先运行 strict validated/synchronous serial reference，
再运行 fast serial。二者必须满足：

1. 三个注册 seed 的 best/last checkpoint 持久化语义逐值精确相等；
2. held-out metric 精确相等；
3. fast serial 相对 strict reference 的实测加速至少为 `5%`。

随后才比较 1/2/3 seed workers。非串行候选必须同时满足逐 seed 精确等价、相对 fast
serial 至少 `5%` 加速、并发 CUDA reserved 上界不超过 `70%`、RAM 保留不少于
`64 GiB`。所有候选失败时串行只在 fast-path 门本身通过后作为保底。

门使用 `full` representation、batch 128、三个正式 seed 和 signal 阶段 CIFAR cache；
注册的保守 activation proxy 必须覆盖 ImageNet-100、VOC 2012、ADE20K 与 NYUv2
正式 workload。最终四个 matrix report 必须嵌入同一份 content-addressed runtime
profile；缺失、旧 schema、过期 revision、篡改或身份不一致均使正式证据不完整。

## 本地验证

- `ruff check src tests scripts`：passed；
- `python -m pytest`：`136 passed`；
- toy smoke（2 steps）：passed，`cache_reload_equal=true`；
- `git diff --check`：passed；
- 20 个注册表示的 compact collate 消费张量与标准 stack 精确相等；
- strict/fast checkpoint 与 held-out evaluation 精确相等；
- Windows spawn seed worker 使用互不重叠的临时 matrix report，父进程生成唯一正式矩阵。

本机没有可用 WSL 发行版，因此四个正式 Bash runbook 的 `bash -n` 留到远端标准 Bash
环境执行。真实 CUDA 的速度、显存、RAM 与候选 worker 选择仍待新固定 revision 部署后
由 runtime gate 实测；在该门完成前，不记录任何吞吐改善为既成事实。

## 结论边界

本记录只说明运行实现通过本地工程与精确性门。真实 AuraFlow signal、四任务主矩阵、
完整 VOC 无监督诊断、四项因果/条件控制和最终机器可读裁决尚未完成，因此不能据此声称
FieldScope 有效。
