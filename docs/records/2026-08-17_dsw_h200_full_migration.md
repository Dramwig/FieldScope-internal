# FieldScope 全量迁移到 dsw-h200 记录

日期：2026-08-17（Asia/Shanghai）。  
状态：项目迁移、活动文件补同步、目标环境重建和源目录删除已完成。本文只记录运维
事实，不表示全量实验完成，也不改变已有正向、负向或待验证科学结论。

> 后续：原本位于 FieldScope 根目录外的完整 ImageNet-1k 资产已于 2026-08-18 单独
> 传输到 H200 并通过固定 revision 审计；见
> `../experiment_conditions/2026-08-18_dsw_h200_imagenet1k.md`。这不改变本文记录的
> 2026-08-17 主目录迁移边界。

## 源与目标

- 源：`pro6000:/root/autodl-tmp/FieldScope`。
- 目标：`dsw-h200:/mnt/omni_ssd/user_workspace/wangzixi/FieldScope`。
- 目标保持相同的项目内目录结构，包括两个 Git 仓库、`datasets/`、`checkpoints/`、
  `hf_home/`、`logs/`、`recovery/`、cache、权重和既有输出。
- 迁移审计保存在目标机
  `/mnt/omni_ssd/user_workspace/wangzixi/.fieldscope_transfer`，不进入 Git。

## 完整性核验

重建目标 `.venv` 之前，源与目标清点均为：

- 240,951 个常规文件；
- 3,395 个目录；
- 4 个符号链接；
- 路径、对象类型与符号链接目标树无差异。

全量 SHA-256 清单包含相同的 240,951 条路径。源端实验停止前仍变化的以下 4 个文件
在主清单后单独补同步，并以活动清单再次核验；`active.source.sha256z` 与
`active.dest.sha256z` 逐字节相同：

```text
e2980d303b53290ea4c57f77b2f445236b095e26a61a56f201a0565b69d19f7b  logs/full_validation_recovery_020c1de-readout0g-cifar-gate-v3.log
021bdb1320c4bec122d541e00c71426f2b043d9da0cd17869984b48db1985d42  logs/full_validation_recovery_020c1de-watchdog-launch.log
2012e2096e1acc5af2bdecee37a56e8499c5d386291f39e71841b95236e2ec4b  recovery/orchestration/020c1de/analysis-output/formal-supervised-errors-020c1de/watchdog-launch.log
44c15d1c861fe0f47eec2726bede89d84806b0b9182f654585eb7de2f4f8df73  recovery/orchestration/020c1de/watchdog/state.json
```

因此迁移时的项目文件可由“全量清单 + 上述 4 文件活动清单”完整覆盖。目标 `.venv`
随后按 H200 本机 CUDA 环境重新创建，这是有意的机器本地差异；从该时点起，源/目标
整棵目录不再要求同一 SHA-256 树，但 `.venv` 外的已迁移项目内容不因重建环境而改写。

## 停机与环境重建

- 源端 FieldScope 实验进程组已使用 `SIGTERM` 正常停止；恢复脚本和 watchdog 未再拉起。
- 删除前按命令行和工作目录复查，FieldScope 相关进程数为 0；无关 CoFiTok 训练未停止。
- 目标 `.venv` 位于项目根，使用 `/opt/conda/bin/python3.10`、
  `--system-site-packages` 与目标机 PyTorch 2.7.1+cu126 重建。
- `pip install -e ".[backbone,dev,data]"` 完成；核心模块导入、H200 CUDA 张量和
  FieldScope doctor 均通过。
- 环境安装日志：
  `/mnt/omni_ssd/user_workspace/wangzixi/.fieldscope_transfer/venv-rebuild.log`。

## 源目录删除

删除前再次确认源目录：

- 真实路径精确为 `/root/autodl-tmp/FieldScope`；
- 是普通目录而非符号链接；
- 没有嵌套挂载点；
- 占用约 138 GiB；
- FieldScope 相关进程为 0；
- 目标项目与 `.venv` 可用，活动文件校验通过。

随后永久删除 `pro6000:/root/autodl-tmp/FieldScope`，并确认该路径不再存在。
`pro6000` 不再是 FieldScope 的运行入口；当前入口、路径和环境以
`../experiment_conditions/2026-08-17_dsw_h200.md` 与项目根 `AGENTS.md` 为准。

## 证据边界

文件迁移、SHA-256 一致、CUDA 可用和 doctor 通过只证明资产与执行环境已转移。正式
全量验证尚须在 H200 上按固定 revision 恢复或重启，并为新机器重新记录 runtime gate、
资源使用、命令和结果；不得把 pro6000 历史结果改写为 H200 复现结果。
