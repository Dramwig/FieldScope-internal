# H200 formal readout epoch-85 audit

- Observed at: 2026-08-21T13:43:09Z
- Host: `dsw-h200`; formal GPU remains physical GPU 6.
- Fixed revision: `020c1de567edd88e0eda245fd085335ffe678f47`.
- Cell: ImageNet-100 / `velocity` / seed `4121`.
- Report commit: 85/90; report status remains `running` because the enclosing matrix is continuing.
- Audit status: `passed`; `problems=[]`.
- Cumulative optimizer steps: `77350`; cumulative training sample exposures: `9898675`.
- Epoch-85 validation top-1/top-5: `0.13655332302936632` / `0.37302936630602784`.
- Best epoch remains 75. This intermediate cell result is not a scientific verdict.
- Report SHA-256: `defa79b1573253e023625c92bbd3f5fc8cb1e8856cacb1ff088db654d68e84fc`.
- Last-checkpoint SHA-256: `c433747ee86543d098257a52b0da79846df4098e839d9c8eeead9d2a53a2f25b`.
- Best-checkpoint SHA-256: `ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`.
- Semantic checks passed: stable double-read; contiguous history `1..85`; AdamW step `77350`; scheduler `last_epoch=85`, `T_max=90`; RNG keys; report/checkpoint history equality; cache/config/control/coverage contracts; runtime worker count; readout batch `128`; formal cache batch `2`; memory cache `0`; both fixed-revision worktrees clean.
- External overlap provenance is retained; scientific effect remains `not_inferred`. `changes_scientific_verdict=false`.

Machine-readable audit: `artifacts/reports/imagenet100_velocity_seed4121_epoch85_strong_audit_20260821.json`; remote copy: `/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/logs/imagenet100_velocity_seed4121_epoch85_strong_audit_20260821.json`.
