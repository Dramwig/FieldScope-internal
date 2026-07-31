# 全量验证数据准备记录

日期：2026-07-31  
证据范围：资产获取、校验和划分准备；不构成方法效果结论。

## ADE20K

- 远端原始包：
  `/root/autodl-tmp/FieldScope/datasets/raw/ade20k/ADEChallengeData2016.zip`；
- 字节数：`965371974`；
- SHA-256：
  `a4a2860390141240c3c05981b5c0084c1d773b2a8c39ac14209da40602cc11ea`；
- `scripts/prepare_ade20k.py` 校验并解包通过；
- 官方 training 图像 `20210` 张，validation 图像 `2000` 张；
- 远端准备目录：
  `/root/autodl-tmp/FieldScope/datasets/prepared/ade20k`。

训练适配器的成对图像/标签检查和内部 train/validation 划分仍需在本次代码同步后执行。

## ImageNet-100

使用完整 ILSVRC2012 train、validation 和 devkit 原始包，在本地数据盘按 WordNet ID
字典序选择前 100 类。`scripts/prepare_imagenet100.py` 正在逐文件写入 SHA-256 manifest；
在脚本完成、计数通过并记录三个源包哈希前，资产状态仍为准备中。

## NYUv2

官方 `nyu_depth_v2_labeled.mat` 与 `splits.mat` 已在远端启动下载。下载完成后必须记录
字节数与 SHA-256，并由 `scripts/prepare_nyuv2.py` 验证官方 `795/654` 划分及固定内部
validation 划分，之后才可用于主实验。
