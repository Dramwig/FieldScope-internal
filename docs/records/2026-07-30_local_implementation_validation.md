# Local implementation validation

日期：2026-07-30

## 检查

```text
ruff: passed
pytest: 18 passed
coverage: 76% total
toy smoke: passed
cache reload: exact equality
```

toy smoke 使用 `configs/eval/toy_smoke.yaml`，输出：

```text
state:    [2,64,14]
response: [2,64,96]
affinity: [2,64,64]
edge density: 0.1989
```

两个训练 step 的总损失从 `3.8964` 降到 `1.9525`。这只证明实现闭环和梯度可用，
不构成 FieldScope 研究假设的实验支持。

