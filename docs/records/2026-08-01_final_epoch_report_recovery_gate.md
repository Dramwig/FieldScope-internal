# 最终 epoch 报告恢复门

日期：2026-08-01  
证据范围：全量 readout 训练的崩溃恢复与远端运行就绪性；不构成方法效果结论。

## 问题

在父 revision `9136bd5e1629b48cd5ab8154ea9e531c0bee2162` 上，readout 每个
epoch 先原子写入 best/last checkpoint，再原子写入 training report。若进程恰在最终
epoch 的 last checkpoint 已提交、`status=passed` report 尚未提交时丢失，恢复逻辑会
从完整 checkpoint 正确重建模型、优化器、scheduler、RNG、history 与验证结果，但只在
内存中构造 passed report，没有将其写回磁盘。已有的上一 epoch `status=running` report
会因此残留，使最终证据审计错误地把已完成训练判为 `incomplete`。

## 修复

- 当 `start_epoch == target_epochs`、即恢复 checkpoint 已包含所有目标 epoch 时，恢复
  分支原子写回标准 training report；
- 不改变模型、数据、表示、seed、epoch、batch size、优化器、scheduler、采样顺序、
  指标或 checkpoint 选择；
- 新增 2-epoch 回归：保留第 1 epoch 的 `status=running` report，在第 2 epoch last
  checkpoint 写完后模拟进程丢失，要求恢复后磁盘 report 为 `passed`，且 history、
  best epoch、best primary metric 与 last checkpoint 一致。

## 本地验证

- 针对性 readout resume 测试：2 passed；
- `ruff check src tests scripts`：passed；
- `python -m pytest -q`：99 passed（加入强化测试前的首次全套；强化后将随最终 revision
  再次运行）；
- toy smoke：passed，cache reload exact equality 为 true；
- `git diff --check`：passed。

## 同 revision 远端就绪审计

正式 GPU 阶段启动前，使用 FieldScope 正式 Python 环境和 512 px 数据适配器提前执行：

- AuraFlow v0.3 backbone 资产审计：passed；四个权重文件及配置/tokenizer 文件哈希
  匹配注册值；
- CIFAR-10、VOC 2012、ImageNet-100、ADE20K、NYUv2 split 审计：5/5 passed；固定
  train/val/test 数量匹配，split 内无重复，split 间无重叠；
- 六份审计均记录 parent revision `9136bd5...`、`code_dirty=false`、零 problems。

这些预审只排除资产与 split 就绪性问题。正式 runbook 仍会在最终固定 revision 上重跑
相同门，并在真实 AuraFlow runtime gate、信号诊断、四任务矩阵和因果控制全部完成后才
生成有效性裁决。

## 磁盘预留

正式缓存、checkpoint/report 与 10 GiB reserve 的保守需求约为 147.66 GiB。为吸收外部
配对训练后续 checkpoint 增长，在逐文件核对已记录 SHA-256 后，删除了不再被正式
runbook 读取、且 prepared split 已通过审计的四个可恢复原始资产：CIFAR-10、VOC 2012、
ADE20K 归档和 NYUv2 labeled MAT，共回收 `6107546894` 字节。ImageNet-100 原始归档和
全部 prepared 数据保留。恢复路径、字节数与 SHA-256 记录在服务器仓库外：

`/root/autodl-tmp/FieldScope/logs/recoverable_raw_cleanup_9136bd5.tsv`

