# H200 formal readout epoch-89 audit

- Observed at: `2026-08-21T18:24:53Z` (`2026-08-22T02:24:53+08:00`).
- Host/device/revision: `dsw-h200`, physical GPU 6, `020c1de567edd88e0eda245fd085335ffe678f47`.
- Cell: ImageNet-100 / `velocity` / seed `4121`; atomic commit `2026-08-21T18:23:59Z`; epoch 89/90; status `running`.
- Audit: `passed`; `problems=[]`; cumulative optimizer steps/exposures `80990 / 10364495`.
- AdamW step `80990`; scheduler `last_epoch=89/T_max=90`; report/checkpoint histories `1..89` are equal after JSON tuple normalization.
- Train/validation seconds: `4430.68719429709 / 378.47353957686573`; validation top-1/top-5 `0.13709428129829984 / 0.375193199381762`; best remains epoch 75.
- Report SHA-256: `eccdae654ddc4c00686931b792f4ab623d7be2e55f403d19fae0f8dcdcb72b05`.
- Last/best checkpoint SHA-256: `ae03f4f801f3e9f96e0e079989e4f9bcb317a0f490a16c16aa6d995e767ce13b` / `ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`.
- Cache/config/control/coverage, RNG, runtime profile, fixed batches, stable double-read, guard presence, and both clean worktrees passed. No external M3Call compute was observed; overlap effects remain `not_inferred`.
- This is intermediate evidence only; `changes_scientific_verdict=false`.

Machine-readable audit: `artifacts/reports/imagenet100_velocity_seed4121_epoch89_strong_audit_20260821.json`.
