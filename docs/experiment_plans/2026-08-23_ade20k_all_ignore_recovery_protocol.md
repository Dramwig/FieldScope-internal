# ADE20K all-ignore batch recovery protocol

Date: 2026-08-23. Scope: implementation recovery for the fixed base revision
`020c1de567edd88e0eda245fd085335ffe678f47`; this is not method-effectiveness
evidence and does not change any scientific decision threshold.

## Deterministic blocker

The first ADE20K formal attempt failed inside epoch 1 with non-finite
segmentation loss. A complete read-only scan of all 18,189 cached train targets
found 16 official samples whose resized label mask contains only ignore label
255. The registered `ShardShuffleSampler(seed=4121)` and batch size 2 place
samples `ADE_train_00005913` (cache index 5336) and
`ADE_train_00005905` (cache index 5328) together at zero-based batch 1697.
PyTorch 2.7 mean cross-entropy divides the empty valid-pixel sum by zero and
returns NaN. An unchanged retry is therefore deterministically invalid and was
stopped before it reached the same batch; its log and exit evidence are kept.

## Locked recovery semantics

The formal recovery must satisfy all of the following:

1. Keep the fixed worktree clean at revision `020c1de...f47`.
2. Keep all 18,189 samples, batch size 2, seed order, epoch sampler, LR,
   weight decay, epoch count, representation order, optimizer-step count, and
   sample-exposure count unchanged.
3. Do not fabricate labels or turn ignore pixels into a semantic class.
4. For a segmentation batch containing at least one target pixel other than
   255, call the original PyTorch cross-entropy implementation unchanged and
   require exact loss and gradient equality in the compatibility gate.
5. Only when every target pixel in the entire batch is 255, return
   `logits.sum() * 0.0`: a finite differentiable zero contribution with exactly
   zero gradient. Still perform the registered optimizer step so the coverage
   ledger remains unchanged.
6. Preserve the fixed finite-value gate: non-finite logits must not be hidden
   by the empty-batch rule.
7. Load the rule only for the ADE20K worker from a content-addressed external
   `sitecustomize.py`; existing ImageNet-100, VOC2012, and NYUv2 workers remain
   untouched. Verify the shim SHA-256 before every ADE launch.
8. Retain the original failed run, retry-1 stop, complete target scan, shim
   source, self-tests, launch audit, and every later negative result.

This recovery implements the already registered meaning of ignore-index pixel
cross-entropy: pixels with label 255 contribute no supervised loss. It repairs
an undefined empty mean without changing information, task weighting, data
coverage, model capacity, optimization hyperparameters, metric direction, or
scientific gates. `execution_complete=false`,
`method_effectiveness_conclusion=null`, `changes_scientific_verdict=false`.

The complete registered training schedule contains 22 such steps across all
240 seed-epochs: 7 for seed 4121, 10 for seed 7319, and 5 for seed 104729. This
is `22 / 43,656,000` registered optimizer steps; the compatibility layer must
therefore remain active for the whole ADE matrix rather than only epoch 1.
The validation split has one all-ignore sample but no all-ignore batch at the
registered batch size; the test split has none. These facts are recorded in
`artifacts/reports/ade20k_all_ignore_coverage_audit_20260823.json`.
