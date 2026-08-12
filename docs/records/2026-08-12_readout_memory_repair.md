# Readout 内存故障与修复记录

日期：2026-08-12
状态：工程修复已实现并完成隔离回归；正式续跑尚未启动。本文不是方法有效性结论，
也不改变已记录的 signal-gate 负结果。

## 触发事实

固定 revision `020c1de567edd88e0eda245fd085335ffe678f47` 的正式 ImageNet-100
readout worker 在五次恢复尝试中均被 shell 报告为 `Killed`。远端容器的
`memory.max` 为 110 GiB，而宿主机视角的 `/proc/meminfo` 约为 1 TiB。原正式主配置
登记 `readout_memory_cache_gib: 160`；资源门只看宿主机可用内存，因此没有阻止这一
不安全组合。正式日志未见 CUDA OOM、Traceback、磁盘满或 non-finite。

进一步审计发现，原 LRU 以 shard 的 `.pt` 文件大小估算常驻 RAM。`readout_sparse`
shard 在加载后会把稀疏邻接恢复成 dense tensor，因此磁盘大小会显著低估进程实际
保留的 tensor storage。

## 修复内容

- 正式主配置与 `random_flow` 对照配置的 readout LRU 上限降为 8 GiB；其余样本、
  表示、seed、epoch、batch、优化器和矩阵范围不变。
- LRU 改为递归计算反序列化后所有唯一 tensor storage 的实际字节数；按真实账本
  驱逐 shard，单个 shard 超过预算时不将其固定在共享缓存中。
- readout runtime gate 读取 cgroup v1/v2 的内存上限、当前占用和可回收页缓存；
  对 shmem 不计入可回收 file cache。有有限 cgroup 上限但读不到 `memory.current` 时
  fail-closed。
- 新 runtime profile 使用 schema 3 并记录可审计内存账本；旧 schema 2 profile
  视为过期，不能复用。
- runtime gate 在启动昂贵候选运行前先验证至少一个 worker 的缓存预算加 64 GiB
  保留量可由当前可用 RAM 满足。

## 验证与部署边界

隔离副本验证结果：`ruff check src tests scripts`、完整 `pytest`（181 tests）、针对性
内存/runtime/evidence 测试（60 tests）和 toy smoke 均通过。另在现有正式
ImageNet-100 shard 上只读测得，约 37.4 MB 的 `readout_sparse` 文件加载后保留约
58.1 MB tensor storage，验证了磁盘大小低估常驻 RAM 的问题。

本修复只处理资源安全和可审计性，不对 signal gate 的 `failed/stop_or_redesign`
结果作正向重解释，也不写入论文效果数字。远端正式链路尚未重启。

完整部署路径使用修复 revision、新 cache/output tag，并重新生成 provenance、正式缓存
manifest、schema 3 runtime profile 和后续完整证据链；旧 revision 的缓存、日志和负结果
须原样保留。

固定旧 revision `020c1de` 还有一条更窄的执行恢复路径，但它不是上述代码修复的替代：

- 特征抽取与现有 cache manifest 继续使用原注册配置（其中 readout cache 字段为
  `160 GiB`），因此现有 ImageNet-100 train/val/test cache 的 extraction signature、
  shard SHA-256 和 provenance 均不改写；
- 仅 cached-readout 训练使用仓库外、内容寻址的 `8 GiB` 运行配置，并重新执行旧
  revision 自身的 schema-2 readout runtime gate；矩阵、checkpoint 和测试报告必须嵌入
  这份新 profile 的 identity；
- 主证据审计会按矩阵中的 readout 配置验证新 profile，同时按 cache identity 验证旧
  manifest。它不会允许篡改旧 manifest，也不能复用原来基于宿主机 RAM 生成的 160 GiB
  profile；
- 现有旧代码在 readout cache 为 `0` 时只保留每个 dataset 最多 4 个 shard。对正式
  ImageNet-100 train cache 做的隔离只读探针连续加载 20 个 shard 后，进程峰值 RSS 约
  `903368 KiB`，共享 cache 仍为 0；这只证明内存路径可控，不构成运行门或方法效果证据；
- 若主证据正向触发扩展，ImageNet-1k 同样必须把特征抽取的原注册配置与 readout-only
  的 8 GiB 配置分开编排；不得直接复用把同一 `FIELDSCOPE_CONFIG` 同时用于抽取与训练的
 旧脚本路径。

固定 revision 路径只有在 GPU 空闲、唯一 worker、全新 runtime-profile/log 标识和所有
fail-closed 审计均满足时才能启动。本文记录的是经代码审计确认的恢复设计，不表示该路径
已经执行或通过。
