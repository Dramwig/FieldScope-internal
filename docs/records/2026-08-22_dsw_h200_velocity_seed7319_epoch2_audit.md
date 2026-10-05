# H200 ImageNet-100 velocity seed-7319 epoch-2 audit

- Observed at: `2026-08-21T21:30:43Z` (`2026-08-22T05:30:43+08:00`).
- Cell: ImageNet-100 / `velocity` / seed `7319`; epoch 2/90 committed at `2026-08-21T21:29:51Z`; status remains `running`.
- Host/device/revision: `dsw-h200`, physical GPU 6, `020c1de567edd88e0eda245fd085335ffe678f47`.
- Strong audit: `passed`; `problems=[]`; contiguous history `1..2`.
- Cumulative optimizer steps/exposures: `1820 / 232910`; epoch increment exactly `910 / 116455`; AdamW step `1820`; scheduler `last_epoch=2/T_max=90`.
- Train/validation seconds: `3214.2878253739327 / 357.2727726493031`; validation top-1/top-5 `0.010046367851622875 / 0.05023183925811438`; best remains epoch 1.
- Report SHA-256: `b10b9302c5b7fdd1083367b7098f55a431e0b89e40b74b83f2e2122ab938c656`.
- Last/best checkpoint SHA-256: `76d167d90ef7d5ba1729c8071ca5edf19d7aefd284054c4063d7834e1de233ba` / `ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`.
- Report/checkpoint, RNG, cache/config/control/coverage, runtime profile, fixed batches, stable double-read and both clean worktrees passed. No external M3Call compute was observed.
- This remains early intermediate evidence only; `execution_complete=false`, `method_effectiveness_conclusion=null`, `changes_scientific_verdict=false`.

Machine-readable audit: `artifacts/reports/imagenet100_velocity_seed7319_epoch2_strong_audit_20260821.json`.
