# FieldScope-internal

本目录是 FieldScope 的独立代码与实验记录 Git 仓库。项目级规则以根目录 `../AGENTS.md` 为准；这里仅补充代码仓库范围的约定。

- Python 包位于 `src/fieldscope/`，配置位于 `configs/`，脚本位于 `scripts/`，测试位于 `tests/`。
- 方法设计写入 `docs/design/`，实验条件写入 `docs/experiment_conditions/`，结果/失败写入 `docs/records/`。
- 数据集、模型权重、完整输出、checkpoint、cache、adapter 和私密配置放在仓库外；仓库只保存小型可审计样例与报告。
- 公共时间统一为 clean-time：`t=0` 是噪声，`t=1` 是图像；backend 负责转换原生 scheduler 时间与速度方向。
- 默认真实 backbone 是冻结的 `fal/AuraFlow-v0.3`；本地/CI 使用 `fieldscope/toy-coupled-field-v1`。
- 修改后至少运行：`ruff check src tests scripts`、`python -m pytest` 和 toy smoke。
- 远端、环境和资产规范以根目录 `../AGENTS.md` 为准；不得凭空填写实验结果。
