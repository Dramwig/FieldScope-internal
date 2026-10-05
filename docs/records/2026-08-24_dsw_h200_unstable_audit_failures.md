# DSW-H200 Unstable Audit Artifacts

At `2026-08-24T08:17:50.491725Z`, the authoritative watcher state was still
`active` with 229 accepted intermediate audits, 18 terminal audits, and two
previously classified transient races. Two new audit attempts returned
`status=failed`; neither is counted in the scientific ledger:

- NYUv2 `random_feature_local / seed 7319 / epoch 68`: audit SHA
  `93914d8138ba7072b9e8ecaa90648160176b6ea339da3997440b40effe280c64`.
  Problems were `stable_report_double_read` plus epoch/history/step/exposure
  mismatches while the report advanced during training.
- VOC2012 `z0 / seed 7319 / epoch 66`: audit SHA
  `abb4bfe5b1612c7ef971b20c3ab81203e4210309ebab945c52a3a0a61b1d04eb`.
  The sole problem was `stable_checkpoint_during_hash_and_load`.

The raw failed artifacts are preserved. They are not yet classified as
transient races because the current classifier requires stable double-read and
stable checkpoint evidence. Later successful audits must establish the next
consistent atomic point before any classification is reconsidered. The clean
scientific ledger remains at 229 accepted intermediate audits and 18 terminal
audits; no downstream gate is triggered.

## Later stable confirmation

At `2026-08-24T08:50:58.418160Z`, both failures were closed as observation
races by a hardened external watcher. The classifier requires a later strong
audit for the same revision, task, representation, seed, target epochs, and
runtime profile; every later check must be true and optimizer-step and exposure
counts must be monotonic. NYUv2 epoch 68 was confirmed by the passed epoch 70
audit, and VOC2012 epoch 66 was confirmed by the passed epoch 69 audit.

The failed artifacts and their original SHA-256 values remain unchanged. They
are recorded with `failed_artifact_preserved=true` and
`unresolved_scientific_failure=false`; neither failed attempt is counted as
accepted progress. The watcher state now has 236 accepted intermediate audits,
18 terminal cells, four preserved transient races, and `failures=[]`. The
watcher-only change did not restart any training worker or modify formal
training outputs. The machine-readable hardening report is
`artifacts/reports/atomic_audit_watcher_later_stable_hardening_20260824.json`.
