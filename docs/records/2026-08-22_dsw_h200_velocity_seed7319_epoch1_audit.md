# H200 ImageNet-100 velocity seed-7319 epoch-1 audit

- Observed at: `2026-08-21T20:31:28Z` (`2026-08-22T04:31:28+08:00`).
- Host/device/revision: `dsw-h200`, physical GPU 6, `020c1de567edd88e0eda245fd085335ffe678f47`.
- Cell: ImageNet-100 / `velocity` / seed `7319`; first atomic report commit `2026-08-21T20:30:19Z`; epoch 1/90, status `running`.
- Strong audit: `passed`; `problems=[]`; optimizer steps/exposures `910 / 116455`.
- AdamW step `910`; scheduler `last_epoch=1/T_max=90`; history `[1]`, report/checkpoint history, four RNG states, cache/config/control/coverage, fixed batches/runtime profile and both clean worktrees passed.
- Train/validation seconds: `3192.316484350711 / 345.9408985329792`; validation top-1/top-5 `0.010046367851622875 / 0.05023183925811438`.
- Report SHA-256: `60e5875608fa6e517d33a5c02169102b3cfe449332cc4f4d6d695f1c5193d78f`.
- Last/best checkpoint SHA-256: `9d42747e48118a43529e4f8306673a9d0023213b80edb6f65fd58d77a3424d3e` / `ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`.
- No external M3Call compute was observed. This is an early intermediate value only; `execution_complete=false`, `method_effectiveness_conclusion=null`, `changes_scientific_verdict=false`.

Machine-readable audit: `artifacts/reports/imagenet100_velocity_seed7319_epoch1_strong_audit_20260821.json`.
