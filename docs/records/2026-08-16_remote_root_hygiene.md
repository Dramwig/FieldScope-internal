# 远端 FieldScope 根外残留收拢记录

> 迁移注记（2026-08-17）：本文记录的是 `pro6000` 上迁移前的目录治理事实；该机的
> FieldScope 项目根随后已删除。下文“后续约束”中的旧绝对路径不再是当前操作入口，
> H200 上的现行路径见 `../experiment_conditions/2026-08-17_dsw_h200.md`。

日期：2026-08-16  
状态：目录收拢完成，固定 revision 恢复链已从新路径恢复监督。本文仅记录运维事实，
不改变实验配置、既有负结果或科学结论。

## 问题与来源

`/root/autodl-tmp` 曾存在多批 `FieldScope-*` 顶层项：七个 Git bundle、三个版本的
fixed-revision wrapper、分析 checkout、分析输出目录，以及 watchdog 脚本和状态目录。
它们来自分阶段上传 revision、构造隔离分析 checkout 和多轮恢复编排；不是模型缓存自动
生成的内容。总量不足 5 MiB，因此主要问题是命名空间和所有权不清，而不是这批文件本身
造成磁盘容量压力。

仓库脚本还存在一个可复发入口：
`run_supervised_error_analysis_after_final.sh` 的默认输出路径使用
`$FIELDSCOPE_ROOT/../FieldScope-analysis-output`，会主动越出项目根。该默认值已改为
`$FIELDSCOPE_ROOT/recovery/analysis-output`，并添加 runbook 回归断言。

## 保留位置

当前固定 revision 为 `020c1de567edd88e0eda245fd085335ffe678f47`。仍需使用的资产已收拢到：

- 分析 worktree：`/root/autodl-tmp/FieldScope/recovery/worktrees/analysis-020c1de`；
- wrapper：`/root/autodl-tmp/FieldScope/recovery/orchestration/020c1de/wrappers`；
- watchdog 脚本与状态：
  `/root/autodl-tmp/FieldScope/recovery/orchestration/020c1de/{watchdog.sh,watchdog/}`；
- 监督后处理输出：
  `/root/autodl-tmp/FieldScope/recovery/orchestration/020c1de/analysis-output`。

保留的 wrapper SHA-256 分别为：readout
`24ac9a3e8bfff301b407234b89420642b29c272921d4a2ba574169d944d52f3e`、final
`8e918f0053768e50025ddb7d281b329c1de2e71288547bbc574a6af1db95e9b1`、extension
`ebd52059cc65de0562b2446847c0e4bce82212251102730695f281a105280233`。迁移后的 watchdog
SHA-256 为 `ce5b139b9e84af48f924ea578da958afc151416e964a3e66b05dc853374e813f`。

## 删除依据与运行连续性

七个 bundle 中的 HEAD 均已存在于本地 `main` 和远端 Git 对象库；v1/v2 wrapper 已被
上述 v3 内容取代。因此删除这些可重建传输物和旧 wrapper，没有删除唯一提交、实验输出、
cache、checkpoint 或日志。清理期间另一个既有部署会话曾按旧路径一次性重建干净的分析
clone 和同 SHA-256 wrapper；确认无进程引用且与保留副本完全相同后也已移除。

迁移发生在 recovery state 为 `preflight_passed` 的边界：cache SHA-256 审计已完成，尚无
训练、特征抽取或分析 Python worker。原监督进程组终止后，分析 checkout 通过
`git worktree move` 迁移，控制脚本完成路径替换、shell 语法检查和 manifest JSON 校验；
watchdog 于 `2026-08-16T21:07:02+08:00` 从新路径重启，并拉起相同 revision、配置、日志
标识和 checkpoint 的 recovery 与分析 waiter。记录时 recovery 因其他项目占用 GPU 而按
原门禁等待，这不表示 FieldScope 运行失败或实验完成。

## 后续约束

- 远端传输 bundle 只允许暂存于 `/root/autodl-tmp/FieldScope/.incoming/`，导入后删除；
- 临时或固定 revision checkout 只放 `FieldScope/recovery/worktrees/`；
- wrapper、watchdog、PID、状态和后处理输出只放 `FieldScope/recovery/` 或 `FieldScope/logs/`；
- 不再在 `/root/autodl-tmp` 顶层创建任何 `FieldScope-*` 文件或目录；
- 顶层清洁检查使用
  `find /root/autodl-tmp -mindepth 1 -maxdepth 1 -name 'FieldScope-*'`，预期无输出。
