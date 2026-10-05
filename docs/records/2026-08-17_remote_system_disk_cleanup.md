# 远端系统盘清理与数据盘迁移记录

> 历史运维记录：本文描述 `pro6000` 在 FieldScope 迁移前的系统盘状态和路径。
> FieldScope 当前固定远端是 `dsw-h200`，项目根为
> `/mnt/omni_ssd/user_workspace/wangzixi/FieldScope`；不要把下文旧路径用于新任务。

日期：2026-08-17  
状态：系统盘清理与已迁移内容校验完成；仍被运行中任务引用的临时路径按保护清单保留。本文仅记录运维事实，不改变实验配置、实验结果或科学结论。

## 清理前状态与成因

远端根文件系统为 30 GiB overlay，清理前 `df -h /` 显示已用 28 GiB、可用
2.7 GiB、使用率 92%。主要来源不是 FieldScope 模型资产，而是以下几类没有在阶段任务结束后回收的工作文件：

- `/tmp` 中累计的 CoFiTok checkout、rehearsal、bundle、测试日志和恢复脚本；
- `/tmp/xmoe_eta_images` 与 `/tmp/alignledger_pro6000_verify` 两个跨项目暂存目录；
- `/root/.cache`、`/root/.vscode-server`、`/root/.nv`、`/root/.triton` 和
  `/root/nltk_data` 使用应用默认位置写入的缓存或运行时数据；
- 33 个旧 FieldScope 修复 checkout、patch、bundle、smoke 输出和审计日志；
- 一个错误转义命令在 `/root` 下创建的多行目录名及其空目录层级。

FieldScope 自身另有一个可复发入口：
`scripts/eval/run_supervised_error_analysis_after_final.sh` 原先默认把分析输出写到
`$FIELDSCOPE_ROOT/../FieldScope-analysis-output`。该默认值已改为
`$FIELDSCOPE_ROOT/recovery/analysis-output`，并由 runbook 测试约束；详细的 FieldScope
顶层收拢过程见 `2026-08-16_remote_root_hygiene.md`。

## 已迁移并保留的数据

所有跨文件系统迁移均先复制，再以 `rsync --dry-run --delete --itemize-changes` 确认零差异，之后才删除系统盘源副本。

- 416 个非活动 CoFiTok 临时项，共 13,867,253,760 bytes，迁到
  `/root/autodl-tmp/CoFiTok/system-disk-recovery/2026-08-17/tmp/`；选择、保护、迁移和重试清单位于同级 `manifests/`。
- `/tmp/xmoe_eta_images`，7,377,752,064 bytes，迁到
  `/root/autodl-tmp/system-disk-recovery/2026-08-17/tmp/xmoe_eta_images/`。
- `/tmp/alignledger_pro6000_verify`，548,151,296 bytes，迁到
  `/root/autodl-tmp/system-disk-recovery/2026-08-17/tmp/alignledger_pro6000_verify/`。
- 33 个 FieldScope 临时项，共 11,362,304 bytes，迁到
  `/root/autodl-tmp/FieldScope/recovery/system-disk-recovery-2026-08-17/tmp/`；逐项清单位于上一级 `tmp-preservation.tsv`。
- 782 个其余非活动、非隐藏且在当日之前生成的通用 `/tmp` 项，共
  108,052,480 bytes，迁到
  `/root/autodl-tmp/system-disk-recovery/2026-08-17/tmp-other/`；选择、保护和迁移清单位于同级 `manifests/`。其中包括旧日志、bundle、JSON、Python 临时目录和一个旧 embeddings 文件，没有直接删除未知项目资产。
- `/root/outputs` 的既有内容迁到
  `/root/autodl-tmp/system-disk-recovery/2026-08-17/root-outputs/`。

## 已重定向到数据盘的缓存与运行时目录

以下原路径现为软链接；应用继续使用原路径时，实际写入位于 `/root/autodl-tmp`：

- `/root/.cache/huggingface` → `/root/autodl-tmp/huggingface`
- `/root/.cache/torch` → `/root/autodl-tmp/system-home/cache/torch`
- `/root/.vscode-server` → `/root/autodl-tmp/system-home/vscode-server`
- `/root/.nv` → `/root/autodl-tmp/system-home/cache/nv`
- `/root/.triton` → `/root/autodl-tmp/system-home/cache/triton`
- `/root/nltk_data` → `/root/autodl-tmp/system-home/nltk_data`
- `/root/tf-logs` → `/root/autodl-tmp/system-home/tf-logs`
- `/root/outputs` → `/root/autodl-tmp/system-disk-recovery/2026-08-17/root-outputs`
- `/tmp/torchinductor_root` → `/root/autodl-tmp/system-home/cache/torchinductor_root`

Hugging Face、Torch、VS Code Server、NVIDIA、Triton、NLTK 和 TorchInductor 的系统盘备份均在迁移后再次通过零差异检查，随后删除；最终审计没有发现
`*.system-disk-backup*` 或迁移中间链接。

## 已删除的可再生或无效内容

- `/tmp/pytest-of-root`、`/root/.pytest_cache`、`/root/.ruff_cache`；
- 7-byte 的旧 `/root/.tmp.444668`；
- 41 个无人监听、且没有 VS Code Server 进程使用的五月至六月旧 VS Code/Git Unix socket；
- 错误转义命令创建的唯一异常目录树。删除前确认其 inode，且树内只有空目录、没有文件、链接或挂载点。

## 运行中任务的保护例外

清理期间没有停止或改写运行中的实验。以下 CoFiTok 路径仍被进程直接引用，因此继续保留在 `/tmp`：

- `/tmp/cofitok-quality-bridge-execution-cf0e5fa`
- `/tmp/cofitok-quality-bridge-followup-decision-9b02fa8`
- `/tmp/cofitok-capacity-probe-22a5994`
- `/tmp/cofitok-capacity-reference-c7424ed`
- `/tmp/cofitok_generation_archive_parallel_20260817`
- 当前 seeding/archive 的日志与守护脚本

完整的 12 项保护清单位于
`/root/autodl-tmp/CoFiTok/system-disk-recovery/2026-08-17/manifests/cofitok-protected.tsv`。
记录时 `/tmp` 剩余 1,614,700,544 bytes（约 1.50 GiB），主要就是这些活动目录、当日生成项和少量隐藏系统项；它们不是 FieldScope 资产，必须在对应任务退出后由 CoFiTok 运维再做同样的迁移或删除。

`/root/miniconda3` 仍被 JupyterLab 和 TensorBoard 的常驻进程使用，因此本次没有移动或删除。迁移该运行时需要单独的停机窗口，不能与在线清理混做。

## 清理后验证

- `df -h /`：已用 2.9 GiB、可用 28 GiB、使用率 10%；
- `df -h /root/autodl-tmp`：已用 832 GiB、可用 169 GiB、使用率 84%；
- `/root/autodl-tmp` 顶层不存在 `FieldScope-*` 残留；
- FieldScope watchdog 与 fixed-revision readout recovery wrapper 仍从
  `/root/autodl-tmp/FieldScope/recovery/orchestration/020c1de/` 运行；
- 清理后的本地两步 toy smoke 通过，`cache_reload_equal=true`。

上述 smoke、接口或运维连续性检查不构成论文结果，也不表示真实任务主证据或因果证据已经完成。
