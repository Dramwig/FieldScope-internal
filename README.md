# FieldScope-internal

FieldScope 的代码、配置、实验脚本、测试和研究记录仓库。根目录
[`AGENTS.md`](../AGENTS.md) 是跨代码/论文仓库的统一入口。

## 已实现范围

- 冻结向量场统一接口，以及无需权重的空间耦合 toy backend。
- `fal/AuraFlow-v0.3` adapter：VAE 图像编码、空文本条件、clean-time/原生时间转换和速度符号转换。
- 线性 rectified-flow 图像锚定状态、中心/单边有限差分和四类结构化扰动。
- 多时间、多噪声视图的响应签名、余弦关系图、局部 + global top-k 稀疏化。
- State–Geometry Fusion Tokenizer 与分类、分割、深度、法线轻量任务头。
- 无监督 spectral partition、边界强度诊断、特征缓存、分片数据集提取和离线 readout 训练。
- `z0/zt/velocity/mismatch/endpoint/state/response/full` 匹配 baseline 接口。

生成 VAE、Flow Transformer 和文本编码器始终冻结；训练只发生在 tokenizer 与任务头。

## 目录

```text
src/fieldscope/backends/  toy 与 AuraFlow 场接口
src/fieldscope/           path/probes/response/graph/tokenizer/heads/cache/CLI
configs/                  模型、数据、评估和消融配置
scripts/                  数据准备、smoke、正式实验与远端同步
docs/                     设计、计划、环境和事实记录
tests/                    单元、集成与端到端测试
```

## 本地环境与验证

```powershell
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe numpy Pillow PyYAML tqdm pytest pytest-cov ruff
uv pip install --python .venv\Scripts\python.exe -e . --no-deps

.\.venv\Scripts\ruff.exe check src tests scripts\data
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m fieldscope.cli smoke --config configs\eval\toy_smoke.yaml --steps 2
```

## 真实 backbone

默认真实模型为 `fal/AuraFlow-v0.3` 的 FP16 Diffusers variant。权重必须放在仓库外，并通过
`FIELDSCOPE_CHECKPOINTS_ROOT` 指向：

```text
${FIELDSCOPE_CHECKPOINTS_ROOT}/AuraFlow-v0.3
```

```bash
python -m fieldscope.cli doctor --config configs/eval/auraflow_contract_smoke.yaml
bash scripts/smoke/run_auraflow_smoke.sh
```

`auraflow_contract_smoke.yaml` 只验证真实主干加载与响应契约；正式探测使用
`auraflow_probe_smoke.yaml` 或审计后的实验配置。

## 离线实验流程

先预计算冻结场特征，再训练小型 readout：

```bash
python -m fieldscope.cli extract-dataset \
  --config configs/model/auraflow_v03.yaml \
  --dataset cifar10 \
  --root "$FIELDSCOPE_DATASETS_ROOT/prepared/cifar10" \
  --split train \
  --output "$FIELDSCOPE_DATASETS_ROOT/feature_cache/auraflow_v03/cifar10_train"

python -m fieldscope.cli train-cache \
  --config configs/model/auraflow_v03.yaml \
  --cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/auraflow_v03/cifar10_train" \
  --task classification \
  --representation full \
  --epochs 20
```

大数据、权重、完整输出和 feature cache 不进入 Git。
