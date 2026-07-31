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
- `scripts/eval/run_full_validation_after_signal_gate.sh` requires a same-revision
  signal verdict of `proceed`, all five metadata-only split audits, and a combined
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

## Local validation facts

- `ruff check src tests scripts`: passed.
- `python -m pytest`: 61 tests passed after the runbook and exact-count gates.
- Bash syntax checks passed for the signal gate, dataset extraction helper, and
  formal full-validation runbook.
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

These gates only reduce data-leakage, identity-instability, and disk-exhaustion
risks. They do not measure classification, boundary, segmentation, or depth
quality. Formal training still requires all of the following:

1. signal gate verdict `proceed`;
2. complete remote ImageNet-100 asset verification;
3. split audits for all five datasets on the committed revision;
4. combined sparse+dense cache budget with `fits=true`.
