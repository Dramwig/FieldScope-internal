# dsw-h200 ImageNet-1k 资产条件

日期：2026-08-18（Asia/Shanghai）。  
范围：ImageNet-1k 资产身份、文件存在性与 split 身份；不构成方法效果证据。

## 固定资产

```bash
export FIELDSCOPE_ROOT=/mnt/omni_ssd/user_workspace/wangzixi/FieldScope
export FIELDSCOPE_DATASETS_ROOT=$FIELDSCOPE_ROOT/datasets
export FIELDSCOPE_IMAGENET1K_ROOT=$FIELDSCOPE_DATASETS_ROOT/prepared/imagenet1k/extracted
```

- 资产根：`$FIELDSCOPE_DATASETS_ROOT/prepared/imagenet1k`；
- manifest：`$FIELDSCOPE_DATASETS_ROOT/prepared/imagenet1k/metadata/image_manifest.jsonl`；
- export summary：
  `$FIELDSCOPE_DATASETS_ROOT/prepared/imagenet1k/metadata/export_summary.json`；
- 来源：Hugging Face `benjamin-paine/imagenet-1k-256x256`；
- source snapshot：`1bd0400450249a7fe90c0aece37d0d03e7ea956a`；
- 预处理：复制已有 256×256 JPEG bytes，不 resize、不 recode；不得描述为与官方
  ILSVRC tar 字节相同。

## 传输证据

- source tar 与 H200 receive tar bytes：19,802,286,080；
- 两端最终复算 tar SHA-256：
  `5ef0f1a0f14ebd8864297d4be631ada508465b16ccf43ffa0b5eafdd6dbe8885`；
- manifest bytes：405,484,553；
- manifest SHA-256：
  `9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0`；
- export summary SHA-256：
  `8e86a7e86cd865971cacd0b64476acfada2a892f5644c47ce84d5b44d8b8f1f3`。

审计通过后，pro6000 临时 tar、H200 receive tar、64 个临时 shard 与空 staging 目录
均已永久删除；原始 CoFiTok 数据集未删除，正式 H200 ImageNet-1k 未改写。

## 固定 revision 审计

- revision：`020c1de567edd88e0eda245fd085335ffe678f47`；
- code dirty：false；
- code tree SHA-256：
  `6ef1305effef91b29d432c9d2c1505b4726645531adad72f60299168d7119c4d`。

正式报告：

- asset audit：
  `$FIELDSCOPE_ROOT/recovery/worktrees/formal-020c1de/outputs/full_validation/auraflow_v03/extension/preflight/imagenet1k_asset_audit.json`；
- asset audit SHA-256：
  `5a6078faa29cc582d8803625530d1555817278af055953ed0db31c3663c4c597`；
- split audit：
  `$FIELDSCOPE_ROOT/recovery/worktrees/formal-020c1de/outputs/full_validation/auraflow_v03/extension/preflight/imagenet1k_split_audit.json`；
- split audit SHA-256：
  `45dff214eac59def2c653ea30f406f08b415a6f93101b3b8bc7d6e81ed7cc0c2`。

资产审计结果：

- source train：1,281,167；source val：50,000；
- train/val 类目录：1,000 / 1,000；
- manifest 唯一路径：1,331,167；
- missing、invalid、problems：均为空；
- FieldScope train/val/test：1,153,049 / 128,118 / 50,000。

split 审计确认三个 split 各自无重复且两两零重叠：

- train sample IDs SHA-256：
  `11bc32e33b04d76f333acb815ab9d6dc524f1eae76e8aba4ce35dad0c56cbe98`；
- val sample IDs SHA-256：
  `264369d39681d78d18833254a1cb2474418c951822b4345d66b0ea6ff67bbf8a`；
- test sample IDs SHA-256：
  `d7bad5f1bb03abe85e646986640c2b1ce4b9fafe35820afcbc1acce6b7d177de`。

## 使用边界

后续 extension run 必须显式设置 `FIELDSCOPE_IMAGENET1K_ROOT`，并使用上述正式报告。
资产审计通过只证明数据身份与 split 合约，不证明 ImageNet-1k 扩展效果。
