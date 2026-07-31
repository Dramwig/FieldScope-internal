# Formal readout non-finite fail-fast gate

Date: 2026-08-01  
Evidence scope: pre-result training-integrity gate only; no effectiveness conclusion

Before any new real signal-gate or formal-task metric was produced, the formal
readout loop was audited for non-finite behavior. The final evidence auditor
already rejects non-finite test metrics, but the training loop could otherwise
continue after a non-finite loss, gradient norm, or validation primary metric
and only fail at the final evidence audit.

The loop now raises immediately before `backward`, before `optimizer.step`, or
before best-checkpoint selection, respectively. This preserves the registered
datasets, samples, representations, seeds, epochs, batch sizes, optimizer,
scheduler, losses, metrics, checkpoint-selection rule, and evidence thresholds.
For finite runs, the optimization trajectory is unchanged.

Regression tests cover:

- non-finite training loss before any checkpoint is written;
- finite loss with a non-finite gradient norm before the optimizer step;
- non-finite validation primary metric before best-checkpoint selection;
- non-finite held-out test primary metric before a test report is returned.

Local verification on the pre-result revision candidate:

- `ruff check src tests scripts`: passed;
- `python -m pytest`: 98 passed;
- toy smoke (`configs/eval/toy_smoke.yaml`, 2 steps): passed;
- `git diff --check`: passed.

This gate is a failure-localization and compute-preservation correction. It is
not evidence that FieldScope is effective and must not be cited as a paper
result.
