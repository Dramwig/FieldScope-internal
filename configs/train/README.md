# Training configs

记录 tokenizer、任务头和损失的可复现实验配置。默认冻结生成主干；任何解冻或 LoRA 设置必须显式标注。

`readout_main.yaml` 是全量验证的预注册训练预算；在 signal gate 后冻结具体
超参数。命令行训练必须显式提供独立 validation cache，最终 test 只由
`evaluate-cache` 读取最佳 checkpoint 一次。
