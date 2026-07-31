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

训练适配器在真实目录上完成成对图像/标签检查。固定 seed 4121 的内部划分为
`18189/2021`，官方 validation 的 `2000` 张仅作为 test；抽样标签值域包含有效类及
`ignore=255`。

## ImageNet-100

使用完整 ILSVRC2012 train、validation 和 devkit 原始包，在本地数据盘按 WordNet ID
字典序选择前 100 类。`scripts/prepare_imagenet100.py` 已完成逐文件 SHA-256 manifest
和计数断言：

- 完整选中训练集 `129395` 张，官方 validation `5000` 张；
- 图像总字节数 `17191430050`；
- 图像 manifest SHA-256：
  `26e5f44f25b3d8283265064d11d92e91db560aff9799d035916252fdbb8e57a1`；
- 真实目录适配器划分为 train `116455`、internal validation `12940`、test `5000`，
  类别标签连续映射到 `0..99`。

本地资产已经通过，但远端传输及三个源包自身的 SHA-256 尚未完成，因此仍不能将远端
ImageNet-100 标记为可运行。

## NYUv2

官方资产已经在远端下载并通过 `scripts/prepare_nyuv2.py`：

- `nyu_depth_v2_labeled.mat`：`2972037809` 字节，SHA-256
  `2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa`；
- `splits.mat`：`2626` 字节，SHA-256
  `6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee`；
- 1,449 对 RGB/depth 转换完成，depth 单位为米；
- 官方 train 的固定内部划分为 `715/80`，官方 test 为 `654`；
- 真实适配器样本的 RGB/depth 形状、有限值和正深度检查通过。

## Cache v3 容量复核

用已记录的最终 512 px、`R=8`、batch 2 AuraFlow 特征离线重存：

- 包含 synthetic 四任务 target 的 v2 文件为 `15180709` 字节；
- 去除冗余邻接后的 v3 文件为 `10723857` 字节；
- 仅保留分类 target 的 v3 文件为 `1810351` 字节，即约 `905176` 字节/样本。

据此，完整 ImageNet-100 cache 的静态估算约为 118 GB。该数字只用于存储调度；必须由
真实 dataset cache manifest 的累计字节数复核，不构成效果或最终吞吐结论。
