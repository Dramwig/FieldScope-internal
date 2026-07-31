# Dataset identity and cache-budget gates

Date: 2026-07-31
Evidence scope: pre-experiment correctness, asset, and resource gates only. This
record is not evidence that the FieldScope method is effective.

## Implemented gates

- Formal dataset splits now use source-stable sample IDs. CIFAR-10 uses official
  array indices; VOC and ImageNet use source split plus class/file name; ADE20K
  uses the official file stem; NYUv2 uses its explicit manifest ID.
- `fieldscope audit-dataset-splits` checks within-split uniqueness and pairwise
  train/val/test disjointness without calling `Dataset.__getitem__` or decoding
  images and targets. Its JSON includes counts, order-independent ID SHA-256,
  overlap examples, registered expected-count checks, and code provenance. A failed
  audit returns non-zero.
- `fieldscope plan-cache-budget` accepts only a completed cache of the requested
  storage policy. It verifies every shard's existence and actual byte count,
  plus SHA-256 when present, before measuring bytes/sample. The default policy
  applies a 1.15 safety factor and reserves 10 GiB. Insufficient space returns
  `resource_blocked` without reducing formal sample counts.
- The formal sparse-cache budget explicitly adds dense target payloads before the
  safety factor: 262,144 bytes/sample for VOC and ADE20K uint8 masks, and
  1,048,576 bytes/sample for NYUv2 float32 depth maps. This avoids treating the
  four-byte CIFAR classification target as representative of dense tasks.
- `scripts/eval/run_full_validation_after_signal_gate.sh` requires a same-revision
  complete signal verdict (`proceed` or `stop_or_redesign`), all five metadata-only
  split audits, and a combined
  disk budget for formal sparse readout caches plus the full VOC dense diagnostic
  cache. Only when `fits=true` does it run the VOC unsupervised diagnostic and the
  complete three-seed readout matrices. It does not change registered thresholds,
  shrink datasets, or delete caches automatically.
- `scripts/data/verify_prepare_imagenet100_remote.sh` is the explicit post-transfer
  asset gate. It verifies archive bytes, SHA-256, and member count before safe
  extraction, then checks 100 classes and 129,395/5,000 image counts and runs the
  exact-count split audit. Machine reports go to ignored
  `outputs/asset_verification/` so the committed-revision worktree remains clean;
  verified facts are copied into a tracked record afterward. The script is not
  invoked while upload is incomplete.
- `scripts/eval/run_full_validation_when_ready.sh` is the persistent CPU-side
  orchestrator. It validates the live signal-gate PID and revision, waits for the
  transfer archive to reach the exact byte count in three consecutive checks,
  runs the asset gate, and requires a same-revision complete signal decision. A
  complete negative signal is retained as a diagnostic but no longer truncates the
  explicitly required full four-task validation. The formal
  runbook then requires five consecutive GPU-free checks before extraction.
- `fieldscope audit-full-evidence` is the formal conclusion gate. It rejects stale
  revisions, dirty worktrees, incomplete caches, non-finite metrics, unmatched
  readout parameter counts, missing checkpoints, or missing representation/seed
  cells. A candidate must beat the strongest per-seed static/hidden control, its
  no-graph/fixed-local controls, and its shuffled-response control on ImageNet-100
  plus at least two dense tasks. VOC structure additionally requires paired-image
  bootstrap intervals above zero against every registered control. At this
  stage the only verdicts are `main_tasks_supported_pending_causal_audits`,
  `limited_or_negative`, and `incomplete`; the main-task audit never emits the
  final strong-claim verdict before the registered causal controls run.
- Formal cache manifests now carry the SHA-256 of the exact source sample-ID
  multiset. The conclusion gate matches those hashes against same-revision split
  audits, verifies all cache-shard contents, exact readout budgets, checkpoint
  hashes, and the registered AuraFlow FP16 weights, component configs, model index,
  scheduler, and tokenizer hashes. The separate causal
  gate is the only component allowed to emit `supports_core_hypothesis`.
- The formal disk gate excludes CIFAR-10 after promotion because CIFAR is a
  signal-only task, budgets the four registered main tasks, five full VOC dense
  caches (main plus causal/condition variants), and reserves an additional
  20 GiB for 240 runs retaining both best and last checkpoints (480 checkpoint
  files) plus machine-readable reports. This bound was corrected before any real
  signal or task metric after measuring post-AdamW-state checkpoints at about
  29 MB for classification and 38 MB for segmentation/depth; the prior 13 GiB
  bound understated the four-matrix worst case.
- The formal supervised matrix includes a deterministic sample-ID random-feature
  control with the same tokenizer and active task head; its randomness is derived
  from the readout seed and never consumes labels or image content.
- The shuffled-response control now uses a seeded random pooled derangement rather
  than an index offset. It is one-to-one and fixed-point-free, mixes globally
  randomized shards within cache-local pools of at most 32 shards, and therefore
  does not preserve a reversible class-block mapping on sorted classification
  caches. This correction was made before any real signal or formal result.
- Registered Random Flow, spatially shuffled probe, neutral-prompt, and unrelated-
  prompt controls now run after every complete main-task result, including
  `limited_or_negative`. The final audit preserves that negative verdict or, after
  a positive main result, distinguishes causal support from failed attribution.
  Only incomplete evidence may truncate the registered final stage.

## Local validation facts

- `ruff check src tests scripts`: passed.
- `python -m pytest`: 84 tests passed after the formal evidence, causal runbook,
  checkpoint-lineage, exact-count, dense-target budget, shuffled-control contract,
  and complete-negative final-stage checks.
- Bash syntax checks passed for the signal gate, full-validation waiter,
  formal full-validation runbook, and final causal-validation runbook.
- Toy end-to-end smoke: passed, including exact cache reload.
- Local ImageNet-100 prepared asset: 100 classes, 129,395 train images, 5,000
  official validation images, and 134,600 transfer-archive members.
- Local ImageNet-100 metadata-only split audit:
  - internal train: 116,455 IDs, SHA-256
    `c912291cd4087ba8422275741175dccec93d9ae7ee6d81e7c01a3768876cd2b0`;
  - internal validation: 12,940 IDs, SHA-256
    `af36634ff50addecbd525c0384302667e92e9187d19600fb3a2d76f3982e0d69`;
  - official validation used as test: 5,000 IDs, SHA-256
    `6817c39f4c6319b0c5e2199f1c325cfa108ef51c25d09404180b059884401ef1`;
  - all pairwise overlaps and all within-split duplicate counts were zero.

The ImageNet audit above ran from an uncommitted code tree and therefore reported
`code_dirty=true`. It validates the local implementation only. The audit must be
rerun on the committed revision after the remote asset is fully verified.

On remote revision `02867e3b6453817cb956ecc7c70254ba9b88221f`, with
`code_dirty=false`, exact-count metadata audits passed for all currently deployed
datasets: CIFAR-10 `45000/5000/10000`, VOC 2012 `1318/146/1449`, ADE20K
`18189/2021/2000`, and NYUv2 `715/80/654`. Every audited split had zero duplicate
IDs and every pairwise train/val/test overlap was zero. The final runbook reruns
these audits after any code revision change; this fact does not waive that gate.

## ImageNet-100 sources and transfer

Completed local source hashes:

- ILSVRC2012 train tar:
  `b08200a27a8e34218a0e58fde36b0fe8f73bc377f4acea2d91602057c3ca45bb`;
- ILSVRC2012 validation tar:
  `c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0`;
- devkit:
  `b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953`;
- ImageNet-100 transfer tar: 17,319,391,232 bytes, SHA-256
  `c5b57e1f6b6994d709ba5842952e7c669d0dd9b6d5ab2390e603e5fc2ffb0e6d`.

At 2026-07-31 16:25 +0800, the remote resumable file was 1,341,849,600 bytes.
Transfer was still active. Remote tar SHA-256, member count, extracted class/image
counts, and split audit were not yet complete, so the asset remains
`pending_remote_verification`. Formal ImageNet-100 extraction must not start yet.

## Interpretation boundary

在任何真实信号或任务指标运行前，revision 还注册了一个纯性能 runtime gate：
固定 image/probe 候选为 2/8、2/16、4/32、8/64，只有固定合成图上的所有 cache
张量按字节等价、峰值 reserved 显存不超过 70%、且吞吐至少提升 5% 的候选
才可替代 2/8。gate 产物及哈希进入 cache manifest，并由最终证据审计验证。
它不读取标签，不减少样本、任务、seed 或表示，也不提供方法有效性结论。

These gates only reduce data-leakage, identity-instability, and disk-exhaustion
risks. They do not measure classification, boundary, segmentation, or depth
quality. Formal training still requires all of the following:

1. complete signal gate verdict (`proceed` or `stop_or_redesign`); only
   `incomplete` blocks formal execution;
2. complete remote ImageNet-100 asset verification;
3. split audits for all five datasets on the committed revision;
4. combined sparse+dense cache budget with `fits=true`.
