# H200 ImageNet-100 velocity seed-4121 terminal audit

- Terminal training report committed at `2026-08-21T19:28:52Z`; terminal training audit observed at `19:30:01Z`.
- Held-out test and matrix registration completed by `2026-08-21T19:31:21Z`; full cell audit observed at `19:32:17Z` (`2026-08-22T03:32:17+08:00`).
- Host/device/revision: `dsw-h200`, physical GPU 6, `020c1de567edd88e0eda245fd085335ffe678f47`.
- Cell: ImageNet-100 / `velocity` / seed `4121`.
- Terminal training audit: `passed`, `problems=[]`; 90 contiguous epochs, `81900` optimizer steps, `10480950` training-sample exposures, AdamW step `81900`, scheduler `last_epoch=90/T_max=90`.
- Terminal epoch train/validation seconds: `3501.412796163 / 391.5171627141535`; validation top-1/top-5: `0.13709428129829984 / 0.3750386398763524`; best remains epoch 75 with top-1 `0.14049459041731066`.
- Training report SHA-256: `9191693ba9bc6a0a53f2cfa82cad72f529e60cac2585546a663dd331e829e9ee`.
- Last/best checkpoint SHA-256: `694d808bcdb6b074b62d9ad819fc6153911fff707d82b097261909978fe66643` / `ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`.
- Held-out test audit: `passed`; official test cache has 5000 samples; top-1/top-5 `0.1384 / 0.3706`; test report SHA-256 `c94a08f9b088641751d10ef7f068c591267346a59f7fa1c5a26cf0f4019057e7`.
- The matrix registered the cell in the exact representation-outer/seed-inner order, increasing ImageNet-100 from 12 to 13 completed runs. Matrix SHA-256 was `cf26e15d07864c41d5ba8b12ddb053614c808831e2d3a4688c512fd9f6febf38`; cumulative completed steps/exposures are `1064700 / 136252350`.
- Stable training/test/matrix double reads, checkpoint/cache/provenance contracts and both clean worktrees passed; `problems=[]`.
- The same worker PID `102910` continued reading after matrix registration and entered `velocity / seed 7319`; no first epoch had committed at audit time.
- The cell is complete, but the four-task execution is not: `full_execution_complete=false`, `method_effectiveness_conclusion=null`, `changes_scientific_verdict=false`.

Machine-readable audits:

- `artifacts/reports/imagenet100_velocity_seed4121_epoch90_terminal_training_audit_20260821.json`
- `artifacts/reports/imagenet100_velocity_seed4121_terminal_cell_strong_audit_20260821.json`
