# FieldScope 项目入口

每次进入本项目先读本文件。它是根目录下两个 Git 仓库的统一入口；进入子仓库后，再读该仓库自己的 `AGENTS.md`。

## 当前状态

- 项目来源：`FieldScope_repositioned_idea_zh.md`。
- `FieldScope-internal/` 已形成可运行 MVP，并已注册真实信号门、四任务
  20 表示 × 3 seed 全量矩阵、四项因果/条件控制和机器可读结论门。
- 全量项目、数据集、权重、cache、日志与恢复状态已迁移到 `dsw-h200`；旧
  `pro6000` 上的 FieldScope 进程已停止，项目目录已在校验迁移后删除。当前正式
  验证需从新远端按固定 revision 恢复或重启；真实任务效果与研究假设仍待最终
  主证据及因果证据完成，接口或 smoke 通过不等于论文结论成立。
- 2026-08-18 的 H200 训练前就绪门已经通过：五个主任务数据集、完整 ImageNet-1k、
  AuraFlow 权重、项目内 `.venv`、固定 revision、runtime/readout gate、Ruff、173 项
  测试、toy smoke 和 recovery preflight 均已核验。原 recovery/watchdog 链持续在物理
  GPU 6 执行；2026-08-22 用户新增授权物理 GPU 7 用于不同任务间并行，任务内部仍固定
  `seed_workers=1`，不得并发写同一 dataset/cell 输出。2026-08-23 用户进一步授权在显存
  足够时继续跨任务并行。2026-08-24 跨主机 ownership handoff 已审计通过：H200 物理
  GPU 6/7 分别承载 ImageNet-100/ADE20K，`dsw-M3Call` A800 物理 GPU 0/1 分别承载
  NYUv2/VOC2012；共享 CephFS、任务锁、固定 revision/clean worktree、A800 worker identity
  registry、独立进程组与共享 H200 watchdog heartbeat monitor 均已核验。ADE20K 首次
  `random_feature_local / seed 4121` 在 epoch 1 内因非有限 loss/CUBLAS 失败且无原子提交，
  失败日志与机器可读审计已完整保留；完整标签扫描定位到固定 sampler 下的整批
  all-ignore target，retry 1 在确定性复现前停止且无正式输出。经内容寻址 ADE-only
  零监督 batch 兼容门通过后已启动 retry 2；固定 worktree、全部样本、batch、sampler、
  LR、step/exposure 账本均不变，仍不得把进程存活写成原子进度。迁移后强审计持续通过；
  NYUv2/VOC2012 的实测 A800 对 H200 吞吐分别为 `2.76x/1.07x`，主要收益是两主机四条
  独立任务 lane。2026-08-24T14:40:50Z 的全局重建为 `20/240` terminal cells、
  308 份接受的中间审计、5 份已分类且完整保留的 audit races、0 个 unresolved failures，
  活动证据为 ImageNet-100 epoch 45、ADE20K epoch 9、NYUv2 epoch 59 与 VOC2012 epoch 75。
  watcher 已内容寻址升级并保留 VOC epoch 55 checkpoint-ahead 失败原件；账本仍为
  `status=passed`、`execution_complete=false`、`changes_scientific_verdict=false`。精确实时进度以 H200 watchdog
  heartbeat、两端 worker PID/lock、
  A800 registry/monitor 和原子 report 的合并审计为准，阶段记录见
  `FieldScope-internal/docs/records/2026-08-19_dsw_h200_formal_validation_progress.md` 与
  `FieldScope-internal/docs/records/2026-08-24_dsw_m3call_a800_cross_host_handoff.md`。
- 任何假设、计划或待验证结果都必须明确标注，不得写成论文结论。

## 固定布局

```text
FieldScope/
├── AGENTS.md                         # 本文件：统一入口与边界
├── FieldScope_repositioned_idea_zh.md # 当前中文 idea 原稿
├── FieldScope-internal/              # 独立 Git：代码、实验、记录
└── paper/                            # 独立 Git：论文、图表、参考文献
```

- 代码、配置、测试、实验脚本和记录只放 `FieldScope-internal/`。
- LaTeX 正文、图、表和 bib 只放 `paper/`。
- 根目录只保留入口说明和 idea 原稿；不要在根目录新增临时代码、输出或下载物。
- 两个子目录是相互独立的 Git 仓库；不要在一个仓库里提交另一个仓库的 `.git/`。

## 研究边界

FieldScope 的待验证核心假设是：对冻结的预训练 Flow Matching 生成向量场施加图像平面锚定的结构化局部扰动，读取输入—输出响应签名，并将 patch 间的响应相似性组织成关系图，用于统一视觉表示和下游分类/密集预测。

写作与实验必须保持以下边界：

- Jacobian、生成轨迹、速度残差和中间特征都是需要控制的对照，不单独构成创新。
- 不声称“首次使用生成模型做视觉理解”“首次使用 Flow Matching backbone”或“首次使用 diffusion Jacobian”。
- “对象边界/区域耦合可由场响应恢复”是核心经验假设，必须由无监督诊断、同容量任务头、跨任务一致性和冻结主干实验共同检验。
- 在证据形成前，不使用“通用视觉理解”“已证明”“显式编码”等强结论。

## 工作规则

1. 开始工作前检查两个仓库：

   ```powershell
   git -C FieldScope-internal status --short --branch
   git -C paper status --short --branch
   ```

2. 代码改动进入 `FieldScope-internal/`；论文改动进入 `paper/`；跨仓库结论必须用记录或论文备注明确关联。
3. 实验条件写入 `FieldScope-internal/docs/experiment_conditions/`，结果与失败记录写入 `FieldScope-internal/docs/records/`，可复用的小型汇总写入 `FieldScope-internal/artifacts/reports/`。
4. 数据集、模型权重、完整输出、hidden cache、checkpoint、adapter、LoRA 和私密配置不进入 Git；只记录外部位置、版本、校验和与获取方式。
5. 远端实验先检查 GPU/磁盘并记录机器、环境、代码 revision、数据/权重版本和命令。
6. 论文中的数字、图表和结论必须能回溯到内部记录；没有记录就保持为 TODO 或计划。

## 固定远端

- 入口：`ssh dsw-h200`；连接细节只保存在本机 `~/.ssh/config`，不要写入仓库。
- A800 side-worker 入口：`ssh dsw-M3Call`；仅承载已注册的 NYUv2/VOC2012 worker，
  不得脱离共享任务锁、A800 registry 和 H200 heartbeat monitor 独立启动正式写入。
- 项目根：`/mnt/omni_ssd/user_workspace/wangzixi/FieldScope`；数据、权重、环境、
  cache、日志和完整输出必须留在该项目根及其数据盘，不得写入系统盘或用户目录下
  的其他临时位置。
- 代码：`FieldScope-internal/`；环境：`.venv/`；日志：`logs/`。
- 数据：`datasets/{raw,prepared,feature_cache}`；权重：`checkpoints/`；Hub 缓存：`hf_home/`。
- 默认资产：`checkpoints/AuraFlow-v0.3`（FP16 variant）、CIFAR-10 与 PASCAL VOC 2012。
- 当前 `.venv` 使用目标机 `/opt/conda/bin/python3.10` 创建，并启用
  `--system-site-packages` 复用目标机 CUDA PyTorch；任何正式运行仍须记录当次可见
  GPU、环境版本和 runtime gate，不能沿用旧机器的资源数字。
- 固定正式 worktree：`recovery/worktrees/formal-020c1de`；分析 worktree：
  `recovery/worktrees/analysis-020c1de`。二者均固定在
  `020c1de567edd88e0eda245fd085335ffe678f47`。根 `.venv` 的 editable 安装指向文档
  仓库，因此固定 revision 命令必须显式设置
  `PYTHONPATH=$FIELDSCOPE_ROOT/recovery/worktrees/formal-020c1de/src`，不得依赖 editable
  安装的默认源码位置。
- H200 runtime profile、readout profile、恢复 wrapper 和 watchdog 的权威路径与 SHA-256
  见 `FieldScope-internal/docs/records/2026-08-18_dsw_h200_pretrain_readiness.md`。
  readiness 记录及机器可读报告现为 `passed`；原 watchdog、analyzer 和 GPU 6 正式
  worker 已启动。2026-08-22 起当前授权正式卡为物理 GPU 6 和 7；2026-08-23 起两卡均可
  承载输出目录不重叠的跨任务 worker。GPU 6 的原 recovery 链保持不变，额外任务不得加入
  其进程组；所有额外 worker 均须有独立锁、guard pause、固定 revision/clean worktree 检查、
  独立任务进程组和主 recovery 退出联动保护。任何恢复重试仍须复核资源门、两个 worktree
  clean、报告状态未变化，并保持每任务注册 `seed_workers=1`；不得把进程存活或 GPU 占用
  写成原子训练进度或科学证据。
- 2026-08-24 A800 handoff 的权威路径、worker identity、锁与首批强审计见
  `FieldScope-internal/docs/records/2026-08-24_dsw_m3call_a800_cross_host_handoff.md`。
  原 A800 readiness 报告是迁移前不可变快照，不得因 worker 已启动而回写；当前状态以
  handoff 审计和最新原子账本为准。
- 旧 `pro6000:/root/autodl-tmp/FieldScope` 已退役并删除。历史条件、记录和报告中的
  旧机器名与绝对路径属于实验 provenance，不得批量改写为 H200；当前操作只以本节
  和 `FieldScope-internal/docs/experiment_conditions/2026-08-17_dsw_h200.md` 为准。
- 资产校验和、环境版本和验证结果见
  `FieldScope-internal/docs/experiment_conditions/` 与 `FieldScope-internal/docs/records/`。

```bash
export FIELDSCOPE_ROOT=/mnt/omni_ssd/user_workspace/wangzixi/FieldScope
export FIELDSCOPE_DATASETS_ROOT=$FIELDSCOPE_ROOT/datasets
export FIELDSCOPE_CHECKPOINTS_ROOT=$FIELDSCOPE_ROOT/checkpoints
export FIELDSCOPE_IMAGENET1K_ROOT=$FIELDSCOPE_DATASETS_ROOT/prepared/imagenet1k/extracted
export HF_HOME=$FIELDSCOPE_ROOT/hf_home
export FIELDSCOPE_PYTHON=$FIELDSCOPE_ROOT/.venv/bin/python
export FIELDSCOPE_REPOSITORY=$FIELDSCOPE_ROOT/recovery/worktrees/formal-020c1de
export PYTHONPATH=$FIELDSCOPE_REPOSITORY/src
cd $FIELDSCOPE_REPOSITORY
```

## 常用位置

```text
FieldScope-internal/src/fieldscope/       # Python 包
FieldScope-internal/configs/              # 可审计配置
FieldScope-internal/scripts/              # smoke/eval/train/paper 脚本
FieldScope-internal/docs/design/          # 方法设计
FieldScope-internal/docs/records/         # 实验事实与失败记录
FieldScope-internal/artifacts/reports/    # 轻量汇总
paper/sections/                           # 论文分节
paper/floats/figures/                     # 论文图
paper/floats/tables/                      # 论文表
paper/output/                             # 本地编译输出，不作为源文件编辑
```

2026-08-24T14:51:00Z latest cross-host atomic reconstruction remains 20/240 terminal cells,
310 accepted intermediate audits, 5 classified transient races and 0 unresolved failures;
active strong-audit boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 45/9/63/78. Ledger SHA-256:
eef69d202d730315fe3cc768dee116ccdf5cc60179833f9214b4a762198959b5.
status=passed, execution_complete=false and changes_scientific_verdict=false remain unchanged.
The matching pretrigger audit is
FieldScope-internal/artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T145107Z.json;
all downstream/final artifacts remain absent and the gate waits for all 240 main cells.

- 2026-08-24T15:01:07Z watcher/reconstructor update: 311 accepted intermediate audits, 20/240 terminal cells, 5 classified transient races and 0 unresolved failures; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 45/9/65/78. Ledger SHA-256 is `44e8ed2cf2c705907ffc00b3f30f96eeb488b37590541ea8e35f35a8cf41ed25`; `status=passed`, `execution_complete=false`, and `changes_scientific_verdict=false`. The pretrigger audit `FieldScope-internal/artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T150107Z.json` passed and all downstream/final artifacts remain absent.

- 2026-08-24T15:11:11Z: 314 accepted intermediate audits, 21/240 terminal cells, 5 classified transient races, 0 unresolved failures; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/68/zt-seed4121-epoch2. Ledger SHA-256 `0fda8f7fa3dc15e2d2d79478f19a9786496ad04eb7b5999b82ae20a7c6623426`, `status=passed`, `execution_complete=false`, `changes_scientific_verdict=false`; pretrigger gate remains closed.

- 2026-08-24T15:21:26Z: 316 accepted intermediate audits, 21/240 terminal cells, 5 classified transient races and 0 unresolved failures; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/72/zt-seed4121-epoch6. Ledger SHA-256 `31fe7b51081fb79a7b8d27d2e91a778901d4911bb49c57272c144f3cdd24e4ee`, `status=passed`, `execution_complete=false`, `changes_scientific_verdict=false`; pretrigger gate remains closed.

- 2026-08-24T15:31:34Z: 318 accepted intermediate audits, 21/240 terminal cells, 5 classified transient races and 0 unresolved failures; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/75/zt-seed4121-epoch9. Ledger SHA-256 `8cdc397be10aa262dd40065aef9fa9f1ddc791fd7ee8bac9da2c2a2a473d69a4`, `status=passed`, `execution_complete=false`, `changes_scientific_verdict=false`; pretrigger gate remains closed.

- 2026-08-24T15:41:41Z: 320 accepted intermediate audits, 21/240 terminal cells, 5 classified transient races and 0 unresolved failures; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/79/zt-seed4121-epoch12. Ledger SHA-256 `5fe709c9341d33cf7a8ca6a068db5f61677b6e2d0484bd52cf0e9adc982075e5`, `status=passed`, `execution_complete=false`, `changes_scientific_verdict=false`; pretrigger gate remains closed.

## 2026-10-06 释放前归档

本次实际连接发现 `pro6000:/root/autodl-tmp/FieldScope` 存在；上述迁移和运行快照均作为
历史记录保留，不能当作当前运行状态。代码已合并并推送至
`https://github.com/Dramwig/FieldScope-internal` 的 `main`。
本次代码、必要记录、raw 数据及权重归档已完成，资产完整性校验通过；未释放服务器，
也未删除服务器原项目。完整 pytest 以退出码 137 中止，不能写为本次测试通过。
本次临时目录 `FieldScope-internal/tmp/retirement-20261006/` 的删除被自动审批阻止，
仍需本机用户手动清理；模型与实验传输分片已清理。
完整范围、资产位置、测试限制和最终完成状态见
`FieldScope-internal/docs/records/2026-10-06_pro6000_local_archive.md`。
本次不恢复正式训练，也不改变科学结论。
