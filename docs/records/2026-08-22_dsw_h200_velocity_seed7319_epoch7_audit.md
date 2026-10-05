# H200 ImageNet-100 velocity seed-7319 epoch-7 audit

- Observed at: `2026-08-22T02:14:01.287424Z` (`2026-08-22T10:14:01.287424+08:00`).
- Cell: ImageNet-100 / `velocity` / seed `7319`; epoch 7/90 committed at `2026-08-22T02:13:02.702558Z`; status remains `running`.
- Host/device/revision: `dsw-h200`, physical GPU 6, `020c1de567edd88e0eda245fd085335ffe678f47`.
- Strong audit: `passed`; `problems=[]`; contiguous history `1..7`.
- Cumulative optimizer steps/exposures: `6370 / 815185`; epoch increment exactly `910 / 116455`; AdamW step `6370`; scheduler `last_epoch=7/T_max=90`.
- Train/validation seconds: `2831.149357547052 / 352.2701987242326`; validation top-1/top-5 `0.010046367851622875 / 0.05023183925811438`; best remains epoch 1.
- Report SHA-256: `ee8c003b73c4e2a8607a1ecd60fd68c528e7fedf9197da4caec92e95b3226eea`.
- Last/best checkpoint SHA-256: `199c518ef09ac2108cb1512c32e96947f529891c459b19dbfcd16209f3923337` / `ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`.
- Report/checkpoint history under JSON tuple normalization, RNG, cache/config/control/coverage, runtime profile, fixed batches, stable double-read and both clean worktrees passed. The running report correctly retains `last_checkpoint_sha256=null`.
- GPU 6 contained only the FieldScope worker and paused resource guard at audit time; no external M3Call process was observed. Resource-overlap scientific effect remains `not_inferred`.
- The unchanged early negative metric is preserved without interpretation. This remains intermediate evidence only; `execution_complete=false`, `method_effectiveness_conclusion=null`, `changes_scientific_verdict=false`.

Machine-readable audit: `artifacts/reports/imagenet100_velocity_seed7319_epoch7_strong_audit_20260822.json` (SHA-256 `cc2c1cb7ba0fb261b1dbc674a061ae5cc6fcc2028bcaf63367f48cc83f7a889a`).
