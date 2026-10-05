# FieldScope-internal

本目录是 FieldScope 的独立代码与实验记录 Git 仓库。项目级规则以根目录 `../AGENTS.md` 为准；这里仅补充代码仓库范围的约定。

- Python 包位于 `src/fieldscope/`，配置位于 `configs/`，脚本位于 `scripts/`，测试位于 `tests/`。
- 方法设计写入 `docs/design/`，实验条件写入 `docs/experiment_conditions/`，结果/失败写入 `docs/records/`。
- 数据集、模型权重、完整输出、checkpoint、cache、adapter 和私密配置放在仓库外；仓库只保存小型可审计样例与报告。
- 公共时间统一为 clean-time：`t=0` 是噪声，`t=1` 是图像；backend 负责转换原生 scheduler 时间与速度方向。
- 默认真实 backbone 是冻结的 `fal/AuraFlow-v0.3`；本地/CI 使用 `fieldscope/toy-coupled-field-v1`。
- 当前固定远端是 `dsw-h200`，项目根为
  `/mnt/omni_ssd/user_workspace/wangzixi/FieldScope`；旧 `pro6000` 条件与路径只作为
  历史 provenance 保留。具体环境与迁移记录见根目录 `../AGENTS.md`、
  `docs/experiment_conditions/2026-08-17_dsw_h200.md` 和
  `docs/records/2026-08-17_dsw_h200_full_migration.md`。
- 正式运行使用 `$FIELDSCOPE_ROOT/recovery/worktrees/formal-020c1de`，固定 revision 为
  `020c1de567edd88e0eda245fd085335ffe678f47`；必须显式设置
  `PYTHONPATH=$FIELDSCOPE_ROOT/recovery/worktrees/formal-020c1de/src`。训练前就绪状态见
  `docs/records/2026-08-18_dsw_h200_pretrain_readiness.md`；该记录和对应机器可读报告
  已标记 `passed`。H200 watchdog、analyzer 与物理 GPU 6 上的原正式 worker 已启动；
  2026-08-22 用户新增授权 GPU 7 做不同任务间并行，2026-08-23 进一步授权 GPU 6/7
  在显存足够时继续跨任务并行。2026-08-24 跨主机 handoff 已通过：H200 GPU 6/7 分别
  承载 ImageNet-100/ADE20K，`dsw-M3Call` A800 GPU 0/1 分别承载 NYUv2/VOC2012；
  A800 worker 使用共享任务锁、精确身份 registry、独立进程组和 H200 heartbeat monitor。
  ADE20K 首次 `random_feature_local / seed 4121` 在 epoch 1 内因
  非有限 loss/CUBLAS 失败且无原子提交，失败审计已完整保留；完整标签扫描定位到固定
  sampler 下的整批 all-ignore target，retry 1 在确定性复现前停止且无正式输出。经
  内容寻址 ADE-only 零监督 batch 兼容门通过后已启动 retry 2；固定 worktree、全部样本、
  batch、sampler、LR 与 step/exposure 账本均不变。任务内部仍保持注册
  `seed_workers=1`，不得并发写同一
  dataset/cell 输出。额外 worker 必须有独立锁、进程组和主 recovery/heartbeat 联动；进程
  存活不等于原子训练进度。交接后的首个 2026-08-24T10:47:30Z 快照为 `20/240`
  terminal cells、261 份接受的中间审计和 0 个 unresolved failures；活动强审计为
  ImageNet-100 epoch 43、ADE20K epoch 7、NYUv2 epoch 7、VOC2012 epoch 26。
  精确实时进度以 watchdog state、两端 worker/lock、A800 registry/monitor 和原子 report
  的合并审计为准，阶段记录见
  `docs/records/2026-08-19_dsw_h200_formal_validation_progress.md` 与
  `docs/records/2026-08-24_dsw_m3call_a800_cross_host_handoff.md`。
  2026-08-24T14:40:50Z 的已审计原子状态为 `20/240` terminal cells、308 份接受的
  中间审计、5 份已分类且完整保留的 audit races 和 0 个 unresolved failures；活动强审计为
  ImageNet-100 epoch 45、ADE20K epoch 9、NYUv2 epoch 59、VOC2012 epoch 75。内容寻址
  watcher 已将 VOC epoch 55 的稳定 report/checkpoint-ahead 竞态由后续 epoch 58 强审计确认，
  原失败 artifact 保持不变；账本仍为 `status=passed`、`execution_complete=false`、
  `changes_scientific_verdict=false`。任何重试只能使用记录中的 H200
  watchdog 命令，不得恢复旧 `pro6000` wrapper 或路径。
- 修改后至少运行：`ruff check src tests scripts`、`python -m pytest` 和 toy smoke。
- 远端、环境和资产规范以根目录 `../AGENTS.md` 为准；不得凭空填写实验结果。

- 2026-08-24T14:51:00Z cross-host ledger snapshot: 310 accepted intermediate audits, 20/240 terminal cells, five classified races, zero unresolved failures; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 45/9/63/78. Ledger SHA-256 is `eef69d202d730315fe3cc768dee116ccdf5cc60179833f9214b4a762198959b5`; `execution_complete=false` and `changes_scientific_verdict=false` remain unchanged. Matching pretrigger report: `artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T145107Z.json`.

2026-08-24T15:01:07Z watcher/reconstructor update: 311 accepted intermediate audits, 20/240 terminal cells, five classified races, zero unresolved failures; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 45/9/65/78. Ledger SHA-256 is `44e8ed2cf2c705907ffc00b3f30f96eeb488b37590541ea8e35f35a8cf41ed25`; `execution_complete=false` and `changes_scientific_verdict=false` remain unchanged. The matching pretrigger report is `artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T150107Z.json`.

2026-08-24T15:11:11Z watcher/reconstructor update: 314 accepted intermediate audits, 21/240 terminal cells, five classified races, zero unresolved failures; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/68/zt-seed4121-epoch2. Ledger SHA-256 `0fda8f7fa3dc15e2d2d79478f19a9786496ad04eb7b5999b82ae20a7c6623426`; `execution_complete=false` and `changes_scientific_verdict=false` remain unchanged.

2026-08-24T15:21:26Z watcher/reconstructor update: 316 accepted intermediate audits, 21/240 terminal cells, five classified races, zero unresolved failures; active boundaries ImageNet-100/ADE20K/NYUv2/VOC2012 = 46/9/72/zt-seed4121-epoch6. Ledger SHA-256 `31fe7b51081fb79a7b8d27d2e91a778901d4911bb49c57272c144f3cdd24e4ee`; `execution_complete=false` and `changes_scientific_verdict=false` remain unchanged.
