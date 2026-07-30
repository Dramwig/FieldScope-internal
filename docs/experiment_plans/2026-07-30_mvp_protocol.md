# FieldScope MVP protocol

日期：2026-07-30

## 阶段顺序

1. toy backend 验证张量、有限差分、图、缓存和训练闭环。
2. AuraFlow 单图 smoke：验证 VAE、时间/速度转换、显存和响应非退化。
3. CIFAR-10 小样本分类：比较所有 baseline，检查数值与计算成本。
4. PASCAL VOC 小样本无监督图诊断与分割 readout。
5. 通过前两关后再扩大到 ImageNet-100/ADE20K；深度数据集另立条件记录。

## 固定 MVP 配置

- backbone：`fal/AuraFlow-v0.3`，完全冻结；
- clean-times：`[0.2, 0.5, 0.8]`；
- probes：接口合约 smoke 为 `R=1` 前向差分，扩展 smoke 为 `R=4`，正式为 `R=8`；
- difference：中心差分；
- noise：固定 seed 的 antithetic pair；
- graph：局部半径 1 + global top-k；
- condition：空文本；
- 特征先缓存，再训练 readout。

## 必报对照

`z0`、`zt`、`velocity`、`mismatch`、`endpoint`、`state`、`response`、`full`。

## 早停

- 单图响应近零、非有限或对 probe seed 不敏感；
- 无监督图不优于 VAE/velocity affinity；
- 匹配 readout 下 `response/full` 没有稳定增益；
- 必须解冻或 LoRA 主干才能产生信号。
