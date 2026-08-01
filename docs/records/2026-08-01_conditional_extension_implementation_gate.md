# Conditional extension implementation gate

Date: 2026-08-01
Evidence scope: implementation, provenance, and execution gating only. No real
FieldScope signal, task, causal, ImageNet-1k, or ablation metric existed when this
gate was implemented.

## Implemented scope

- The full ImageNet-1k server export has a fail-closed asset audit for the fixed
  405,484,553-byte manifest, registered SHA-256, 1,281,167/50,000 source split
  counts, 1,000 WNID directories, contiguous labels, unique relative paths, and
  existence of every referenced JPEG.
- `imagenet` is enabled in the metadata-only split audit. Its exact internal
  train/validation counts are derived from the registered per-class seed-4121
  holdout rule and then supplied back as exact expected counts.
- A conditional 512 px ImageNet-1k runbook executes all 20 registered
  representations at all three seeds for 90 epochs and 1,000 classes. The existing
  full matrix audit was parameterized without making ImageNet-1k an unconditional
  requirement of the four-task core gate.
- The high-cost extension materializes exactly 15 one-factor configs from the
  tracked main config. Every generated config and runtime profile is hashed and
  tied to the clean code revision. Every variant runs the label-free runtime gate,
  a complete 1,449-image dense VOC test extraction, and the per-image graph
  diagnostic.
- The extension audit verifies every ImageNet-1k readout training epoch, optimizer
  step, sample exposure, best/last checkpoint hash, and test report. It also loads
  and hashes every ablation cache shard, checks target/feature shapes and sample
  identity, and reports paired-image bootstrap intervals for boundary AP and
  pairwise AUROC relative to the main configuration.
- A combined resource gate budgets the complete ImageNet-1k sparse cache, all 120
  retained ImageNet-1k checkpoints, and all 15 dense VOC caches. Failure is
  `resource_blocked`; it does not silently reduce pixels, classes, samples,
  representations, seeds, or ablation variants.
- The final conclusion audit requires causal evidence for every complete main
  result. It additionally requires extension evidence after a positive main-task
  result, but does not run the conditional extension after a complete negative
  main result.

## Local validation facts

- `ruff check src tests scripts`: passed.
- `python -m pytest`: 141 tests passed.
- toy smoke with two optimization steps: passed; cache reload exact equality was
  true.
- `git diff --check`: passed.
- The five new/changed Bash runbooks passed `bash -n` on `pro6000` by streaming
  their local contents to the server shell; this did not deploy or execute them.

These facts establish that the conditional execution and audit path is available.
They do not establish that the extension fits the final free disk, that any formal
run has started, or that the FieldScope hypothesis is supported.
