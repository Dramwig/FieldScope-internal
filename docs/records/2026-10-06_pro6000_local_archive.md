# pro6000 释放前本地归档

日期：2026-10-06（Asia/Shanghai）。本记录仅描述代码和资产保存，不改变实验科学结论。

## 来源与代码保存

本次实际检查发现 `pro6000:/root/autodl-tmp/FieldScope` 存在，且包含后续 H200/A800
审计记录。这是本次观察；2026-08-17 的迁移与删除历史记录保持原样，不倒写历史。
连接参数仅保存在本机 SSH 配置/本次连接命令中，不写入仓库。

- 服务器原代码基于 `020c1de567edd88e0eda245fd085335ffe678f47`，原 origin 是历史 bundle。
- 本地原 HEAD：`649bf9f97fa9b62f09add16185db0d27da1f600f`，有后续修复及未提交记录。
- 服务器未提交内容保存为 `5811c4ace5e8671e71dd55ff0b1e82aa796763a4`。
- 本地未提交内容先保存为 `cd46723`，再合并为 `4b9997a45f6d03de80760aabee22276b34a5c6c8`。
- 发布目标：<https://github.com/Dramwig/FieldScope-internal>，默认分支 `main`。
  服务器原始保存提交另保留在 `archive/pro6000-retirement-20261006`。
- 两端对比：服务器独有 647 个文件、本地独有 28 个文件、19 个同名文件内容不同
  （忽略纯 CRLF/LF 差异）。本地后续代码修复保留，服务器独有记录合并。
- 三份冲突审计 JSON 不拼接为新实验结果：原路径使用服务器版本，本地合并前版本
  保存在 `artifacts/reports/pro6000_retirement_20261006/local_before_merge/`。
- 服务器根入口原件保存于 `pro6000_archive_20261006/server_root_AGENTS.md`。

## 数据集边界

本地目标为 `D:/datasets_raw_hub/registry`。没有复制服务器解压图片树、深度数组或
`datasets/feature_cache`。条目元数据记录原始资产、校验和及配方。

| 条目 | 保存范围 |
| --- | --- |
| `cifar10` | 本地既有 `raw/cifar-10-python.tar.gz`，与既有 SHA-256 清单核对通过 |
| `ade20k` | 本地既有 `raw/ADEChallengeData2016.zip`，与既有 SHA-256 清单核对通过 |
| `pascal_voc_2012` | 本地既有 `raw/VOCtrainval_11-May-2012.tar`，与既有 SHA-256 清单核对通过 |
| `imagenet100_full` | 复用本地既有 `imagenet100_full.tar`，保存类别列表与重建配方；服务器与本地完整 SHA-256 一致 |
| `nyuv2` | 下载并核验 `raw/splits.mat`；服务器缺失的原始 labeled MAT 已从 NYU 官方来源补齐，并与历史 SHA-256 核对一致 |
| `imagenet_1k_256x256_hf` | CoFiTok 提供的 HF 原始依赖，共 45 个 Parquet 分片及原始 README；46/46 个文件已按服务器原始 SHA-256 清单通过校验；不与官方 ILSVRC2012 原包混同 |

ImageNet-100 的选择为 `lexicographically_first_100_wordnet_ids`，不代表所有常见
ImageNet-100 定义。NYUv2 缺失原始 MAT 的历史预期大小为 `2972037809` 字节，历史
SHA-256 为 `2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa`；
该参考值来自准备记录；本次随后从
`https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat`
补回原文件，经本地完整 SHA-256 校验一致后纳入 raw 归档。

## 权重与实验文件

权重条目位于 `D:/checkpoints_hub/registry/fal/AuraFlow-v0.3`，固定上游 revision
为 `2cd8588f04c886002be4571697d84654a50e3af3`、FP16 variant。保留权重、配置、
tokenizer、许可证、模型卡及两种兼容索引文件名。逐文件 SHA-256 与服务器比对后才
将条目标记为已验证；不保存重复 Hub cache。

本次 18 个模型源文件（共 `16837515416` 字节）全部校验通过；模型清单、下载脚本及
校验文件已落盘。实验整包也已完成服务器与本地 SHA-256 比对，并正式移入下述位置。
两个条目已追加到 `D:/checkpoints_hub/registry/checkpoints.yaml`，没有覆盖其他项目条目。

实验归档最终位置与校验和如下（完成状态由机器可读清单记录）：

- `D:/checkpoints_hub/registry/FieldScope/experiments/pro6000-20261006.tar`：
  日志、恢复状态、历史失败归档、readout checkpoints 和实验输出；
  `16223567872` 字节，SHA-256
  `a7626a5ebb000029200c0aa37b7a25ca80fb5065e880a1e7fa01ddaa8d53fc00`。
- 本代码仓库被 Git 忽略的 `outputs/retirement-20261006/dataset-provenance.tar`：
  数据转换清单、标签映射、划分及来源说明；`431104000` 字节，SHA-256
  `a21c813d0a6199bba2b5acb6b99b137282588dfb6cad8a69ee138a0dab45f649`。
- 同目录 `source-records-precommit.tar`：1410 份服务器提交前的代码/记录原始文件，
  保留原始行尾且不含 `.git`；`29726720` 字节，SHA-256
  `76e969ce3ede573f65f6dd1bc62661c577a16f03ae05c1f2e520ad7ab9797fd2`。
- 资产清单与最终校验结果另保存在 `artifacts/reports/pro6000_retirement_20261006/`。

大文件不进入 Git。本次不启动训练，不恢复 watchdog，不修改历史实验判断。

## 验证与当前完成状态

合并后的 Ruff 已通过。服务器旧 `.venv` 的解释器链接不可用，首次使用现有
Python 3.12 验证时因缺少 SciPy 在 pytest 收集阶段失败；后续验证依赖安装在
项目 `.incoming/retirement-validation-deps`，临时目录也限制在项目内。

补齐依赖后完整 pytest 以退出码 `137` 中止，已捕获日志中没有断言失败；终止原因
未确认，不能推断为已确认的 OOM（当次 cgroup `oom_kill=0`）。无卡容器当次限制为
0.5 CPU、2 GiB RAM。toy smoke 应用退出码为 0，输出完整；Git fsck 通过。
对应命令、环境与原始日志见 `validation_summary.json` 及其引用文件。完整 pytest
没有通过，不能把历史测试成功计入本次验证。后续只修改归档文档与资产元数据。

恢复时先核对各清单 SHA-256，将实验 tar 解压到独立项目目录，再从 Git 恢复正式
revision `020c1de567edd88e0eda245fd085335ffe678f47` 的 worktree，并重建环境。
归档内历史 `.git` 指针中的绝对路径不能作为本地有效 worktree 注册；它们仅保留
来源信息。raw 数据、权重分别从两个 registry 绑定或重新准备，旧 `.venv`、重复 Hub
cache、解压图像树和 feature cache 未复制。

`paper/` 原有未跟踪论文草稿保持原样；服务器没有对应 paper 目录。

本次资产归档已完成。机器可读总清单位于
`artifacts/reports/pro6000_retirement_20261006/archive_summary.json`，逐项验证报告与
测试限制均随代码保存。共享 ImageNet 条目已有的 raw 文件直接复用并重新校验；
连接中断后本任务用独立临时文件补齐 7 个缺失分片，未操作其他任务的续传文件。

本次模型/实验下载分片已清理。临时克隆、工具依赖和迁移脚本仍保留在
`tmp/retirement-20261006/`：自动审批拒绝了批量清理及收窄到该精确目录后的删除请求，
仅返回 `blocked by policy`；未绕过该限制。该目录已经没有活动进程，必要内容已经
保存到正式归档，仍需本机用户手动删除。Git worktree 登记只保留两个主仓库。
两端代码和
GitHub `main` 的最终提交号核对回执位于被 Git 忽略的
`outputs/retirement-20261006/git_sync.json`（仅在三端一致后生成）。根目录入口说明的
本次快照另保存在 `pro6000_archive_20261006/local_root_AGENTS.md`。

没有释放服务器或删除服务器原项目、数据、权重。本记录确认本次范围内文件保存完成，
不确认其他项目的释放条件，也不改变任何科学结论：`changes_scientific_verdict=false`。
