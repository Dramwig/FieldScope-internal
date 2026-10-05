# dsw-h200 训练前就绪记录

日期：2026-08-18（Asia/Shanghai）。  
状态：`passed`；H200 watchdog、analyzer 与物理 GPU 6 正式训练 worker 已启动。  
范围：数据、权重、环境、固定 revision、资源门和恢复入口；不构成方法效果或论文结论。

## 固定入口

- 服务器：`dsw-h200`；
- 项目根：`/mnt/omni_ssd/user_workspace/wangzixi/FieldScope`；
- 环境：`$FIELDSCOPE_ROOT/.venv`；
- 正式 worktree：`$FIELDSCOPE_ROOT/recovery/worktrees/formal-020c1de`；
- 分析 worktree：`$FIELDSCOPE_ROOT/recovery/worktrees/analysis-020c1de`；
- revision：`020c1de567edd88e0eda245fd085335ffe678f47`；
- code tree SHA-256：
  `6ef1305effef91b29d432c9d2c1505b4726645531adad72f60299168d7119c4d`。

两个 worktree 已核验为 clean。根 `.venv` 的 editable 安装不作为固定 revision 的
源码选择依据；所有正式命令必须设置：

```bash
export FIELDSCOPE_ROOT=/mnt/omni_ssd/user_workspace/wangzixi/FieldScope
export FIELDSCOPE_DATASETS_ROOT=$FIELDSCOPE_ROOT/datasets
export FIELDSCOPE_CHECKPOINTS_ROOT=$FIELDSCOPE_ROOT/checkpoints
export HF_HOME=$FIELDSCOPE_ROOT/hf_home
export FIELDSCOPE_PYTHON=$FIELDSCOPE_ROOT/.venv/bin/python
export FIELDSCOPE_REPOSITORY=$FIELDSCOPE_ROOT/recovery/worktrees/formal-020c1de
export PYTHONPATH=$FIELDSCOPE_REPOSITORY/src
cd $FIELDSCOPE_REPOSITORY
```

## 环境与实现验证

- Python 3.10.16；
- PyTorch 2.7.1+cu126；torchvision 0.22.1；
- diffusers 0.38.0；transformers 4.56.2；accelerate 1.10.1；
- 8 张 NVIDIA H200 可见；doctor 输出 `ready=true`；
- `ruff check src tests scripts`：passed；
- `python -m pytest`：`173 passed in 107.05s`；
- 两步 toy smoke：passed，`cache_reload_equal=true`。

这些检查只证明实现与环境可执行，不证明真实任务效果。

## 已通过的主任务资产门

固定 preflight 根：
`$FIELDSCOPE_REPOSITORY/outputs/full_validation/auraflow_v03/preflight`。

- CIFAR-10：45,000 / 5,000 / 10,000；
- VOC 2012：1,318 / 146 / 1,449；
- ImageNet-100：116,455 / 12,940 / 5,000；
- ADE20K：18,189 / 2,021 / 2,000；
- NYUv2：715 / 80 / 654。

五份 split audit 均为 `status=passed`、`problems=[]`、`code_dirty=false`。AuraFlow-v0.3
FP16 asset audit 同样为 `passed`，注册的权重与配置文件 SHA-256 全部匹配。

## H200 资源门

- AuraFlow runtime profile：
  `$FIELDSCOPE_REPOSITORY/outputs/runtime_gate/auraflow_runtime_profile_020c1de567edd88e0eda245fd085335ffe678f47_dsw-h200.json`；
- SHA-256：
  `8e36dc71a2110b48ee05e05b12e8fe38bd88105a992d923ead053c37e4d5e37f`；
- 状态：passed；选择 `image_batch_size=2`、`probe_batch_size=8`。

- readout runtime profile：
  `$FIELDSCOPE_ROOT/recovery/fixed-revision-readout/readout_runtime_profile_020c1de567edd88e0eda245fd085335ffe678f47_0be38fa17ba13c1a9e0b248642c85832cd8cf7e63afe2b799fe89f0f0ea98ed1_30c6073ce73f6de74803eb29f497ceedf1ee3cafc488340cf236395824ea42e3.json`；
- SHA-256：
  `878ae28a98a08c66c4dec54243c4fbd8861ee3aba0ee04fd726d6aa3d35cb28e`；
- 状态：passed；1/2/3 worker 均与串行结果精确等价，选择 `seed_workers=1`。

## 恢复入口

- H200 recovery wrapper：
  `$FIELDSCOPE_ROOT/recovery/orchestration/020c1de/wrappers/run_fixed_revision_readout_recovery_h200.sh`；
- wrapper SHA-256：
  `98fc2baea7203d3c225a890deb6eb26307f4164deb8d94f1b73997b7b662f3e9`；
- H200 watchdog：
  `$FIELDSCOPE_ROOT/recovery/orchestration/020c1de/watchdog_h200.sh`；
- watchdog SHA-256：
  `23a2a22c6fa6a28c368094a9536989c38ad574178a4736e11f66660aea691963`；
- watchdog recovery tag：`020c1de-readout0g-cifar-gate-v3-h200`；
- watchdog state：
  `$FIELDSCOPE_ROOT/recovery/orchestration/020c1de/watchdog-h200`。

上列 wrapper SHA 是 `2026-08-23T14:50:08Z` recovery-handoff 加固后的当前权威值。
训练前 readiness 通过时的 wrapper SHA 为
`fef17407a2b261335183ae17f9dd23dedf4446aa36eb41f32f3776a22d21c251`，原件仍按该 SHA
只读备份。当前版本只为未来 ADE20K watchdog 恢复增加已审计 all-ignore shim SHA
`f95bbb0ed70ebbc127e6e08cd8d47c7bb18e6d394c2233a5677dc9cdfbf7bac4` 的 fail-closed
验证与注入；当前 recovery 和训练 worker 均未重启。完整审计见
`docs/records/2026-08-23_dsw_h200_gpu67_recovery_handoff_hardening.md`。

H200 watchdog 与 readout/final/extension wrapper 均已通过 `bash -n`，引用文件存在。
watchdog 于 2026-08-18 02:39:07 UTC 首次启动，初始 PID 34701；初始 analyzer PID
30573。目标机使用
cgroup v1，watchdog 已兼容读取 cgroup v2 的 `memory.current` / `memory.max` 和 cgroup v1
的 `memory/memory.usage_in_bytes` / `memory/memory.limit_in_bytes`。修复前 watchdog
SHA-256 为 `6bfff14618752015bb5c0cb076372f6db4a7d8e06103f40782c95b1fffdef823`，
原件已按 SHA 命名备份。wrapper 同时保留了 `/proc/<pid>/cmdline` 短生命周期进程竞态修复。

2026-08-19 16:35 UTC 的后续资源边界审计发现：等待中的 analyzer 虽继承了
`FIELDSCOPE_GPU_INDEX=6`，但未继承 `CUDA_VISIBLE_DEVICES=6`；它尚未开始任何回放或
CUDA 工作，因此没有使用物理 GPU 0--5，也没有产生或修改科学产物。外部 watchdog 的
`start_analyzer` 环境仅增加一行 `CUDA_VISIBLE_DEVICES="$gpu_index"`，修改前 SHA-256 为
`f8c46f0abbdf950a23b32ab4bccaced7a674f59832af1cab98b94fcfb0a04390`，原件保存在
`watchdog_h200.sh.f8c46f0abbdf950a.pre-analyzer-gpu6-binding`；修改后 SHA-256 即上列
`23a2a22c...691963`，并重新通过 `bash -n`。只重启了等待中的 analyzer 与 watchdog；
正式 readout PID 107681、recovery PID 92034 和五个 cache worker 均未重启。当前
watchdog PID 106745、analyzer PID 105935，后者进程环境已直接核验包含
`CUDA_VISIBLE_DEVICES=6`；watchdog state 为 `active`、`analyzer.alive=true`、
`changes_scientific_verdict=false`。
初次训练前预检在 GPU 7 上通过，作为当时资源 provenance 保留；随后 FieldScope 的唯一
正式卡固定为物理 GPU 6。2026-08-18 显式设置 `CUDA_VISIBLE_DEVICES=6`、
`FIELDSCOPE_GPU_INDEX=6`，使用生产默认的两轮、每轮 5 次资源检查重新执行
`FIELDSCOPE_RECOVERY_PREFLIGHT_ONLY=1`；可用显存为 136,353–141,738 MiB，无竞态
警告，最终退出码为 0，并输出：

```text
H200 fixed-revision recovery preflight passed; training not started
```

GPU 6 preflight state：
`$FIELDSCOPE_ROOT/logs/full_validation_recovery_020c1de-readout0g-cifar-gate-v3-h200-gpu6-preflight.state.json`，
SHA-256 为
`765a1ff8776ae972f04cf185ccb8d2e2e975635814a093436c7895ad00bef760`，stage 为 `ready`。
对应日志 SHA-256 为
`fb46aa5e740b30eb68f2b0fd7a269b98e4f876171941b01fcd442c7b3cd781ad`。

GPU 6 上另执行了一个短时 BF16 CUDA 探针：进程只看到一张 NVIDIA H200，逻辑
`cuda:0` 对应物理 GPU 6，探针成功完成且退出后显存释放。机器上的低显存占卡任务会在
真实负载进入时让出算力，因此其空闲期 100% utilization 读数不单独作为拒绝条件；正式
入口仍以 GPU 6、显存阈值、固定 revision 和恢复合约共同约束。

首次非 preflight 恢复在完整 GPU 6 资源门后，没有启动训练 worker，而是在尝试
`extract-dataset --resume` 复用迁移来的 ImageNet-100 cache 时按设计拒绝：cache 的
immutable extraction signature 正确记录迁移前 `pro6000` 的绝对数据路径和 runtime
profile 身份，H200 路径与 runtime profile 因而不能产生同一 signature。三份 manifest、
所有 shard SHA-256、样本覆盖、固定 revision 和 code tree 审计均已通过；cache 与 epoch-20
checkpoint 未被改写。H200 外部 recovery wrapper 已改为：完整 cache 先通过现有逐 shard
审计，审计通过则保持原 manifest 并直接复用；缺失、不完整或损坏的 cache 才进入正常
提取/恢复路径。修复前 wrapper SHA-256 为
`ef931bb9e4450fff506cd7d85f1062b5d95bade3dbbc8a8464f79923017ff976`，备份路径为
`run_fixed_revision_readout_recovery_h200.sh.ef931bb9e4450fff.pre-cache-reuse-fix`。
修复后 wrapper 已通过 `bash -n`，watchdog 已用新版本重新启动恢复链。该修复只改变
迁移后完整 cache 的恢复控制流，不改变固定 worktree、cache manifest、特征 shard、训练
checkpoint 或科学裁决规则。

cache 复用门通过后，固定 revision 的严格训练报告合约又正确拒绝了迁移前绝对路径：
seed 4121 / 7319 的报告与 checkpoint 只在 `config.backend.model_path`、
`config.runtime.cache_dir` 和 train/validation cache identity 的 `path` 字段上仍指向
`/root/autodl-tmp/FieldScope`；训练超参、manifest SHA-256、extraction signature、控制
合约、revision、code tree、epoch history、模型、优化器、调度器和 RNG 状态均未变化。
为避免放宽固定 revision 的严格比较，已执行一次可审计的 path-only rebind：

- 脚本：`$FIELDSCOPE_ROOT/recovery/migration-rebind/rebind_imagenet100_recovery_paths.py`；
- 脚本 SHA-256：
  `9bb64ecab4758e1fdeb744908c87623483ed465a9688c429a95b2b53802d94c2`；
- 审计：
  `$FIELDSCOPE_ROOT/recovery/migration-rebind/2026-08-18_imagenet100_pro6000_to_h200/audit.json`；
- 审计 SHA-256：
  `2373110a8f08b51c5ceefdf80e52ed0c5fb3ccda5d700f2fac80b7cb3c7b547a`；
- 审计状态：`passed`，`changes_scientific_verdict=false`、
  `checkpoint_scientific_payload_unchanged=true`、`original_artifacts_preserved=true`；
- 8 份原始 artifact 已逐 SHA 保存在审计目录的 `original/` 下。

该操作没有改 cache manifest 或 shard，也没有改变 checkpoint 的模型/优化器/调度器/RNG
语义载荷；只把迁移前绝对路径元数据绑定到 H200 项目根，并同步其派生文件 SHA。完成后
使用固定 revision 原实现重新验证了报告、train/validation/test cache identity 和四份
checkpoint 合约。watchdog 已在下一轮继续执行完整 GPU 6 启动门。

严格 checkpoint 合约通过后，固定 revision 在 PyTorch 2.7.1 下暴露出一个与迁移无关的
恢复时设备放置缺陷：`torch.load(..., map_location=cuda)` 会把 checkpoint 内用于
`torch.set_rng_state` / `torch.cuda.set_rng_state_all` 的 RNG ByteTensor 一并移到 CUDA，
而两个 setter 要求 CPU ByteTensor，因此原始 checkpoint 和 path-only rebound checkpoint
都会在 `restore_rng_state` 报 `TypeError`。固定 worktree 未修改；H200 外部恢复入口增加了
一个内容寻址的 `sitecustomize.py`：

- 路径：`$FIELDSCOPE_ROOT/recovery/orchestration/020c1de/sitecompat/sitecustomize.py`；
- SHA-256：
  `2f834fc37d66f66d7bd47104d56bc203071b50d91d2b6aba865a0755a2531c89`；
- 行为：仅当 `torch.load` 返回含 `rng_state` 的 checkpoint 时，将 `torch_cpu` 与
  `torch_cuda` RNG byte tensors 放回 CPU；模型、优化器和调度器 tensor 继续服从原
  `map_location=cuda`；
- GPU 6 探针：RNG setter 均成功，模型 tensor 仍为 `cuda:0`；
- wrapper 会在任何正式 Python 进程前验证 shim SHA-256，并通过 `PYTHONPATH` 加载；
- 启用 shim 前 wrapper SHA-256 为
  `7e8065f758d1342d28abd0d2abf98aff25faa10598572af377f6779bcde8681f`，原件已备份。

该兼容层实现固定 revision 原代码已经表达的意图——精确恢复 CPU/CUDA RNG 状态——不
改变任何 RNG 值、模型/优化器状态、训练顺序、指标或科学裁决规则。watchdog 已被安全
唤醒并用新 wrapper 重新执行完整两轮 GPU 6 启动门。

2026-08-18 03:48:53 UTC，修复后的恢复链完成两轮 GPU 6 资源门、完整 ImageNet-100
cache 审计、runtime/readout profile 复核、迁移路径合约和 RNG 精确恢复后，启动正式
worker PID 107681。核验结果：

- cwd：`$FIELDSCOPE_ROOT/recovery/worktrees/formal-020c1de`；
- `CUDA_VISIBLE_DEVICES=6`，进程只授权物理 GPU 6；
- `PYTHONPATH` 首项为 fixed revision `src`，并包含内容寻址的 H200 sitecompat；
- 仅一个 `fieldscope.cli run-readout-matrix` worker；
- 完整 20 表示 × 3 seed ImageNet-100 命令，`readout_memory_cache_gib=0`；
- seed 4121 的 epoch 90 结果保持 `passed`；seed 7319 从 epoch 20、18,200 optimizer
  steps、2,329,100 training sample exposures 无损恢复；
- epoch 21 已提交，worker 正在执行 epoch 22。

worker 越过了报告、cache、checkpoint、optimizer/scheduler 和 RNG 恢复门，因此正式训练
已经开始；方法有效性仍未形成结论，必须等待后续主矩阵、因果/条件控制、按 main verdict
决定的扩展、132 份回放和最终机器可读裁决。

epoch 21 的首次 H200 提交已完成并通过连续性核验：history 20→21，optimizer steps
18,200→19,110（+910），training sample exposures 2,329,100→2,445,555
（+116,455）；last checkpoint 的 epoch/history 均为 21。epoch 21 训练耗时约 352.00 秒，
吞吐约 330.84 samples/s；这只是执行性能与负结果训练轨迹，不构成方法有效性证据。
last checkpoint SHA-256 为
`b1aeabc0e2eec96c5f80473de92ce673868eccb7b431c07bad42586535d74062`；best checkpoint
SHA-256 为 `f02ccf556b7cf34d54775c09a4703e93ffd727855ab3a6fc9ee9af062fcbbf1b`。
watchdog 已注册 101,010 optimizer steps 与 12,926,505 training sample exposures，状态仍为
`active`、`changes_scientific_verdict=false`。signal 与 VOC unsupervised 两份既有证据 SHA
仍分别为 `0378867d09a99c99b69ec7179a8ac7d20523106cc3f7b53236177e37a0e76a1e`
和 `4382c9e53be7fa4a77b5d5a6409011cd8ce21ff4dcb3f4422bb4f668c6aa840e`。

## ImageNet-1k

已传输 tar 的注册事实：

- 源快照：`1bd0400450249a7fe90c0aece37d0d03e7ea956a`；
- train：1,281,167；官方 val：50,000；类别：1,000；
- manifest bytes：405,484,553；
- manifest SHA-256：
  `9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0`；
- 传输 tar bytes：19,802,286,080；
- 传输 tar SHA-256：
  `5ef0f1a0f14ebd8864297d4be631ada508465b16ccf43ffa0b5eafdd6dbe8885`。

最终资产根与训练入口：

```text
/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/datasets/prepared/imagenet1k
/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/datasets/prepared/imagenet1k/extracted
```

固定 revision 正式报告：

- `outputs/full_validation/auraflow_v03/extension/preflight/imagenet1k_asset_audit.json`，
  SHA-256
  `5a6078faa29cc582d8803625530d1555817278af055953ed0db31c3663c4c597`；
- `outputs/full_validation/auraflow_v03/extension/preflight/imagenet1k_split_audit.json`，
  SHA-256
  `45dff214eac59def2c653ea30f406f08b415a6f93101b3b8bc7d6e81ed7cc0c2`。

两份报告均为 `status=passed`、`problems=[]`、`code_dirty=false`。资产审计确认
1,331,167 条唯一路径无缺失，source train/val 为 1,281,167 / 50,000，两个 split
均为 1,000 类；FieldScope train/val/test 为 1,153,049 / 128,118 / 50,000。split
审计确认各 split 内无重复且三者两两零重叠。

临时 transfer tar 在 pro6000 与 H200 删除前再次复算为相同 SHA-256；审计通过后，
两端临时 tar、H200 64 个 shard 和空 staging 目录均已永久删除。原始 CoFiTok 数据集
和正式 H200 ImageNet-1k 保留。

## 正式训练启动命令

正式 watchdog 已按下列命令启动；恢复链仍会在每次重试时重新确认两个 worktree clean、
机器可读 readiness 报告为 `passed`，且 GPU 6 满足 110,000 MiB 空闲阈值：

```bash
export FIELDSCOPE_ROOT=/mnt/omni_ssd/user_workspace/wangzixi/FieldScope
cd $FIELDSCOPE_ROOT/recovery/orchestration/020c1de
nohup env \
  FIELDSCOPE_GPU_INDEX=6 \
  FIELDSCOPE_GPU_MIN_FREE_MIB=110000 \
  FIELDSCOPE_WATCHDOG_POLL_SECONDS=600 \
  bash watchdog_h200.sh \
  </dev/null >>$FIELDSCOPE_ROOT/logs/watchdog_h200-launch.log 2>&1 &
```

watchdog 自身会写入
`$FIELDSCOPE_ROOT/recovery/orchestration/020c1de/watchdog-h200/{watchdog.pid,watchdog.log,state.json}`，
并只在固定 revision、资源门和恢复合约保持有效时启动恢复链。

## 证据边界

迁移、环境、资产、runtime 和恢复门通过，且正式 worker 已从迁移 checkpoint 启动。
当前仅新增提交了 seed 7319 epoch 21，且该表示仍处于约 1% top-1 的负结果轨迹；真实任务
主证据、因果证据和论文结论均未因此形成。
