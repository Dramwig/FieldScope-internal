# H200 formal readout epoch-86 audit

- Observed at: 2026-08-21T14:51:04Z.
- Host: `dsw-h200`; formal device remains physical GPU 6.
- Fixed revision: `020c1de567edd88e0eda245fd085335ffe678f47`.
- Cell: ImageNet-100 / `velocity` / seed `4121`.
- Atomic report commit: 2026-08-21T14:50:10Z; committed epoch 86/90; report status remains `running`.
- Audit status: `passed`; `problems=[]`.
- Cumulative optimizer steps: `78260`; cumulative training sample exposures: `10015130`.
- Epoch-86 train/validation seconds: `3905.7324223620817` / `385.7801098274067`.
- Epoch-86 validation top-1/top-5: `0.13655332302936632` / `0.37619783616692426`.
- Best epoch remains 75. This is an intermediate cell result, not a scientific verdict.
- Report SHA-256: `08838b717b8412987af7cc31145b8fde09321e78a3f976c24415ff8393d310fb`.
- Last-checkpoint SHA-256: `12c493b47714e1de4fec47eebde4c9ba6e7884adcb417e5ba86bfea11ad32b1a`.
- Best-checkpoint SHA-256: `ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`.
- Stable double-read, contiguous history, AdamW step, scheduler, RNG, report/checkpoint history, cache/config/control/coverage, batch/runtime and clean-worktree checks all passed.
- Two preceding M3Call overlaps ended naturally before the atomic commit. Their scientific effect remains `not_inferred`; `changes_scientific_verdict=false`.

Machine-readable audit: `artifacts/reports/imagenet100_velocity_seed4121_epoch86_strong_audit_20260821.json`; remote copy: `/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/logs/imagenet100_velocity_seed4121_epoch86_strong_audit_20260821.json`.
